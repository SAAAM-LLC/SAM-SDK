"""
Disaster Recovery System for SAM-SDK

Provides robust backup, state replication, and recovery capabilities
for SAM nodes and the broader hive network.
"""

import os
import time
import json
import logging
import threading
import shutil
import gzip
import hashlib
import pickle
from typing import Dict, List, Optional, Union, Any, Callable

logger = logging.getLogger("sam_sdk.recovery")

class StateSnapshot:
    """
    Represents a point-in-time snapshot of a SAM node's state
    
    Snapshots enable recovery from failures, migration between hosts,
    and state analysis.
    """
    
    def __init__(self, node_id: str, snapshot_type: str = "full"):
        """
        Initialize a state snapshot
        
        Args:
            node_id: ID of the node
            snapshot_type: Type of snapshot (full, partial, delta)
        """
        self.node_id = node_id
        self.snapshot_type = snapshot_type
        self.timestamp = time.time()
        self.snapshot_id = f"{node_id}_{int(self.timestamp)}_{snapshot_type}"
        
        # State components
        self.state = {}
        self.metadata = {
            "version": "0.1.0",
            "created_at": self.timestamp,
            "node_id": node_id,
            "type": snapshot_type,
            "components": []
        }
        
        # Metrics
        self.size_bytes = 0
        self.checksum = None
    
    def add_component(self, name: str, state: Any, compress: bool = True) -> None:
        """
        Add a component to the snapshot
        
        Args:
            name: Component name
            state: Component state
            compress: Whether to compress the state
        """
        try:
            # Serialize the state
            serialized = pickle.dumps(state)
            
            # Compress if requested
            if compress and len(serialized) > 1024:  # Only compress if > 1KB
                compressed = gzip.compress(serialized)
                if len(compressed) < len(serialized):  # Only use compression if it helps
                    serialized = compressed
                    is_compressed = True
                else:
                    is_compressed = False
            else:
                is_compressed = False
            
            # Calculate checksum
            component_checksum = hashlib.sha256(serialized).hexdigest()
            
            # Store component
            self.state[name] = {
                "data": serialized,
                "compressed": is_compressed,
                "checksum": component_checksum,
                "size_bytes": len(serialized)
            }
            
            # Update metadata
            self.metadata["components"].append({
                "name": name,
                "compressed": is_compressed,
                "checksum": component_checksum,
                "size_bytes": len(serialized)
            })
            
            # Update snapshot size
            self.size_bytes += len(serialized)
            
            logger.debug(f"Added component '{name}' to snapshot (size: {len(serialized)} bytes, compressed: {is_compressed})")
        except Exception as e:
            logger.error(f"Error adding component '{name}' to snapshot: {e}")
            raise
    
    def get_component(self, name: str) -> Any:
        """
        Get a component from the snapshot
        
        Args:
            name: Component name
            
        Returns:
            Any: Component state
        """
        if name not in self.state:
            raise KeyError(f"Component '{name}' not found in snapshot")
            
        component = self.state[name]
        serialized = component["data"]
        
        # Decompress if needed
        if component["compressed"]:
            serialized = gzip.decompress(serialized)
        
        # Deserialize
        return pickle.loads(serialized)
    
    def finalize(self) -> None:
        """Finalize the snapshot and calculate overall checksum"""
        # Create a combined checksum from all components
        checksums = [c["checksum"] for c in self.metadata["components"]]
        checksum_str = "".join(checksums)
        self.checksum = hashlib.sha256(checksum_str.encode()).hexdigest()
        
        # Update metadata
        self.metadata["checksum"] = self.checksum
        self.metadata["size_bytes"] = self.size_bytes
        self.metadata["finalized_at"] = time.time()
    
    def save(self, directory: str) -> str:
        """
        Save the snapshot to disk
        
        Args:
            directory: Directory to save the snapshot
            
        Returns:
            str: Path to the saved snapshot
        """
        if not self.checksum:
            self.finalize()
            
        # Create directory
        os.makedirs(directory, exist_ok=True)
        
        # Create snapshot directory
        snapshot_dir = os.path.join(directory, self.snapshot_id)
        os.makedirs(snapshot_dir, exist_ok=True)
        
        # Save metadata
        metadata_path = os.path.join(snapshot_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save each component
        for name, component in self.state.items():
            component_path = os.path.join(snapshot_dir, f"{name}.bin")
            with open(component_path, "wb") as f:
                f.write(component
