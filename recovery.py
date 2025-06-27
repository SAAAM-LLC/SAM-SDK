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
                f.write(component["data"])
        
        logger.info(f"Saved snapshot {self.snapshot_id} to {snapshot_dir}")
        return snapshot_dir
    
    @classmethod
    def load(cls, snapshot_dir: str) -> 'StateSnapshot':
        """
        Load a snapshot from disk
        
        Args:
            snapshot_dir: Path to the snapshot directory
            
        Returns:
            StateSnapshot: Loaded snapshot
        """
        # Check if directory exists
        if not os.path.isdir(snapshot_dir):
            raise ValueError(f"Snapshot directory {snapshot_dir} does not exist")
            
        # Load metadata
        metadata_path = os.path.join(snapshot_dir, "metadata.json")
        if not os.path.isfile(metadata_path):
            raise ValueError(f"Metadata file not found in {snapshot_dir}")
            
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            
        # Create snapshot
        snapshot = cls(metadata["node_id"], metadata["type"])
        snapshot.timestamp = metadata["created_at"]
        snapshot.snapshot_id = os.path.basename(snapshot_dir)
        snapshot.metadata = metadata
        snapshot.size_bytes = metadata.get("size_bytes", 0)
        snapshot.checksum = metadata.get("checksum")
        
        # Load components
        for component_meta in metadata["components"]:
            name = component_meta["name"]
            component_path = os.path.join(snapshot_dir, f"{name}.bin")
            
            if not os.path.isfile(component_path):
                logger.warning(f"Component file {name}.bin not found in {snapshot_dir}")
                continue
                
            with open(component_path, "rb") as f:
                data = f.read()
                
            # Verify checksum
            component_checksum = hashlib.sha256(data).hexdigest()
            if component_checksum != component_meta["checksum"]:
                logger.warning(f"Checksum mismatch for component {name}")
                
            # Store component
            snapshot.state[name] = {
                "data": data,
                "compressed": component_meta["compressed"],
                "checksum": component_checksum,
                "size_bytes": len(data)
            }
            
        logger.info(f"Loaded snapshot {snapshot.snapshot_id} from {snapshot_dir}")
        return snapshot


class DisasterRecoveryManager:
    """
    Manages disaster recovery for SAM nodes
    
    Features:
    - Automated periodic snapshots
    - Differential snapshots for efficiency
    - Versioned storage with retention policies
    - Automatic failover support
    - Cross-node state replication
    """
    
    def __init__(self, 
                 sam_instance,
                 backup_dir: str = "./backups",
                 snapshot_interval: int = 3600,
                 retention_count: int = 5,
                 enable_replication: bool = False,
                 replication_targets: List[str] = None):
        """
        Initialize the disaster recovery manager
        
        Args:
            sam_instance: Reference to the SAM instance
            backup_dir: Directory for backups
            snapshot_interval: Seconds between automatic snapshots
            retention_count: Number of snapshots to retain
            enable_replication: Whether to enable cross-node replication
            replication_targets: List of target node IDs for replication
        """
        self.sam = sam_instance
        self.backup_dir = backup_dir
        self.snapshot_interval = snapshot_interval
        self.retention_count = retention_count
        self.enable_replication = enable_replication
        self.replication_targets = replication_targets or []
        
        # Node identity
        self.node_id = getattr(sam_instance, 'node_id', str(hash(sam_instance)))
        
        # Snapshot tracking
        self.snapshots = {}  # snapshot_id -> metadata
        self.last_snapshot_time = 0
        self.current_snapshot = None
        
        # Component handlers
        self.component_handlers = {
            "concept_bank": self._snapshot_concept_bank,
            "thought_state": self._snapshot_thought_state,
            "experience_manager": self._snapshot_experience_manager,
            "consciousness": self._snapshot_consciousness,
            "config": self._snapshot_config,
            "model_state": self._snapshot_model_state
        }
        
        # Automated scheduling
        self.scheduler_thread = None
        self._stop_event = threading.Event()
        
        # Initialize
        self._load_snapshot_registry()
        
        logger.info(f"Initialized DisasterRecoveryManager for node {self.node_id}")
    
    def _load_snapshot_registry(self) -> None:
        """Load the snapshot registry from disk"""
        registry_path = os.path.join(self.backup_dir, f"{self.node_id}_registry.json")
        
        if os.path.isfile(registry_path):
            try:
                with open(registry_path, "r") as f:
                    self.snapshots = json.load(f)
                logger.info(f"Loaded {len(self.snapshots)} snapshots from registry")
            except Exception as e:
                logger.error(f"Error loading snapshot registry: {e}")
                self.snapshots = {}
        else:
            # Create empty registry
            self.snapshots = {}
            self._save_snapshot_registry()
    
    def _save_snapshot_registry(self) -> None:
        """Save the snapshot registry to disk"""
        os.makedirs(self.backup_dir, exist_ok=True)
        registry_path = os.path.join(self.backup_dir, f"{self.node_id}_registry.json")
        
        try:
            with open(registry_path, "w") as f:
                json.dump(self.snapshots, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving snapshot registry: {e}")
    
    def create_snapshot(self, snapshot_type: str = "full") -> str:
        """
        Create a snapshot of the current SAM state
        
        Args:
            snapshot_type: Type of snapshot (full, partial, delta)
            
        Returns:
            str: Snapshot ID
        """
        # Create snapshot
        snapshot = StateSnapshot(self.node_id, snapshot_type)
        
        try:
            # Snapshot core components
            for component_name, handler in self.component_handlers.items():
                handler(snapshot)
                
            # Finalize and save
            snapshot.finalize()
            snapshot_dir = snapshot.save(self.backup_dir)
            
            # Update registry
            self.snapshots[snapshot.snapshot_id] = {
                "id": snapshot.snapshot_id,
                "type": snapshot_type,
                "timestamp": snapshot.timestamp,
                "path": snapshot_dir,
                "size_bytes": snapshot.size_bytes,
                "checksum": snapshot.checksum,
                "components": [c["name"] for c in snapshot.metadata["components"]]
            }
            self._save_snapshot_registry()
            
            # Update tracking
            self.last_snapshot_time = time.time()
            self.current_snapshot = snapshot.snapshot_id
            
            # Enforce retention policy
            self._enforce_retention_policy()
            
            # Replicate if enabled
            if self.enable_replication and self.replication_targets:
                self._replicate_snapshot(snapshot.snapshot_id)
                
            logger.info(f"Created {snapshot_type} snapshot {snapshot.snapshot_id}")
            return snapshot.snapshot_id
            
        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")
            raise
    
    def _snapshot_concept_bank(self, snapshot: StateSnapshot) -> None:
        """
        Add concept bank state to snapshot
        
        Args:
            snapshot: Snapshot to update
        """
        if not hasattr(self.sam, 'concept_bank'):
            logger.warning("SAM instance has no concept_bank attribute")
            return
            
        try:
            # Extract concept bank state
            concept_bank = self.sam.concept_bank
            
            # Basic state (serializable directly)
            state = {
                "next_concept_id": concept_bank.next_concept_id,
                "concept_metadata": concept_bank.concept_metadata,
                "source_to_concept": concept_bank.source_to_concept,
                "related_concepts": dict(concept_bank.related_concepts),
                "hive_shared_concepts": list(concept_bank.hive_shared_concepts),
                "hive_private_concepts": list(concept_bank.hive_private_concepts),
                "hive_pending_sync": list(concept_bank.hive_pending_sync),
                "hive_origin": concept_bank.hive_origin,
                "hive_global_id_map": concept_bank.hive_global_id_map,
                "modality_concepts": {k: list(v) for k, v in concept_bank.modality_concepts.items()},
                "creation_history": concept_bank.creation_history
            }
            
            # Tensors (need special handling)
            if hasattr(concept_bank, 'concept_frequencies'):
                state["concept_frequencies"] = concept_bank.concept_frequencies.cpu().numpy()
                
            if hasattr(concept_bank, 'concept_timestamps'):
                state["concept_timestamps"] = concept_bank.concept_timestamps.cpu().numpy()
                
            if hasattr(concept_bank, 'meaning_vectors'):
                state["meaning_vectors"] = concept_bank.meaning_vectors.cpu().numpy()
                
            if hasattr(concept_bank, 'concept_embeddings') and hasattr(concept_bank.concept_embeddings, 'weight'):
                state["concept_embeddings_weight"] = concept_bank.concept_embeddings.weight.data.cpu().numpy()
            
            # Add to snapshot
            snapshot.add_component("concept_bank", state)
            
        except Exception as e:
            logger.error(f"Error snapshotting concept bank: {e}")
            raise
    
    def _snapshot_thought_state(self, snapshot: StateSnapshot) -> None:
        """
        Add thought state to snapshot
        
        Args:
            snapshot: Snapshot to update
        """
        if not hasattr(self.sam, 'thought_state'):
            logger.warning("SAM instance has no thought_state attribute")
            return
            
        try:
            # Extract thought state
            thought_state = self.sam.thought_state
            
            # Basic state
            state = {
                "thought_depth": thought_state.thought_depth,
                "evolution_history": thought_state.evolution_history,
                "personal_factor": thought_state.personal_factor,
            }
            
            # Tensor state
            if hasattr(thought_state, 'thought_memory') and thought_state.thought_memory:
                state["thought_memory"] = [t.cpu().numpy() for t in thought_state.thought_memory]
                
            if hasattr(thought_state, 'superposition_memories') and thought_state.superposition_memories:
                state["superposition_memories"] = [
                    [m.cpu().numpy() for m in memories] 
                    for memories in thought_state.superposition_memories if memories
                ]
                
            if hasattr(thought_state, 'modality_thoughts'):
                state["modality_thoughts"] = {
                    k: v.cpu().numpy() if torch.is_tensor(v) else None
                    for k, v in thought_state.modality_thoughts.items()
                }
                
            if hasattr(thought_state, 'amplitudes') and thought_state.amplitudes.numel() > 0:
                state["amplitudes"] = thought_state.amplitudes.cpu().numpy()
                
            if hasattr(thought_state, 'shared_thought') and thought_state.shared_thought is not None:
                state["shared_thought"] = thought_state.shared_thought.cpu().numpy()
                
            if hasattr(thought_state, 'local_thought') and thought_state.local_thought is not None:
                state["local_thought"] = thought_state.local_thought.cpu().numpy()
            
            # Add to snapshot
            snapshot.add_component("thought_state", state)
            
        except Exception as e:
            logger.error(f"Error snapshotting thought state: {e}")
            raise
    
    def _snapshot_experience_manager(self, snapshot: StateSnapshot) -> None:
        """
        Add experience manager state to snapshot
        
        Args:
            snapshot: Snapshot to update
        """
        if not hasattr(self.sam, 'experience_manager'):
            logger.warning("SAM instance has no experience_manager attribute")
            return
            
        try:
            # Extract experience manager state
            experience_manager = self.sam.experience_manager
            
            # Basic state
            state = {
                "experiences": experience_manager.experiences,
                "loaded_experiences": experience_manager.loaded_experiences,
                "shared_experiences": experience_manager.shared_experiences,
                "private_experiences": experience_manager.private_experiences,
                "pending_sync_experiences": list(experience_manager.pending_sync_experiences),
                "modality_experiences": dict(experience_manager.modality_experiences)
            }
            
            # Add to snapshot
            snapshot.add_component("experience_manager", state)
            
        except Exception as e:
            logger.error(f"Error snapshotting experience manager: {e}")
            raise
    
    def _snapshot_consciousness(self, snapshot: StateSnapshot) -> None:
        """
        Add consciousness monitor state to snapshot
        
        Args:
            snapshot: Snapshot to update
        """
        if not hasattr(self.sam, 'consciousness'):
            logger.warning("SAM instance has no consciousness attribute")
            return
            
        try:
            # Extract consciousness state
            consciousness = self.sam.consciousness
            
            # Basic state
            state = {
                "concept_cluster_history": consciousness.concept_cluster_history,
                "concept_entropy_history": consciousness.concept_entropy_history,
                "resonance_scores": consciousness.resonance_scores,
                "personal_concepts": list(consciousness.personal_concepts),
                "personality_initialized": consciousness.personality_initialized
            }
            
            # Handle tensors
            if hasattr(consciousness, 'identity_centroids'):
                state["identity_centroids"] = {
                    k: v.cpu().numpy() for k, v in consciousness.identity_centroids.items()
                }
                
            if hasattr(consciousness, 'modality_centroids'):
                state["modality_centroids"] = {
                    k: v.cpu().numpy() for k, v in consciousness.modality_centroids.items()
                }
                
            if hasattr(consciousness, 'personality_vector') and consciousness.personality_vector is not None:
                state["personality_vector"] = consciousness.personality_vector.cpu().numpy()
            
            # Add to snapshot
            snapshot.add_component("consciousness", state)
            
        except Exception as e:
            logger.error(f"Error snapshotting consciousness: {e}")
            raise
    
    def _snapshot_config(self, snapshot: StateSnapshot) -> None:
        """
        Add configuration to snapshot
        
        Args:
            snapshot: Snapshot to update
        """
        if not hasattr(self.sam, 'config'):
            logger.warning("SAM instance has no config attribute")
            return
            
        try:
            # Extract config
            config = self.sam.config
            
            # Convert to serializable form
            if hasattr(config, 'to_dict'):
                config_dict = config.to_dict()
            elif hasattr(config, '__dict__'):
                config_dict = {
                    k: v for k, v in config.__dict__.items()
                    if not k.startswith('_') and not callable(v)
                }
            else:
                config_dict = dict(config)
                
            # Handle non-serializable types
            for k, v in list(config_dict.items()):
                if isinstance(v, torch.dtype):
                    config_dict[k] = str(v)
                    
            # Add to snapshot
            snapshot.add_component("config", config_dict)
            
        except Exception as e:
            logger.error(f"Error snapshotting config: {e}")
            raise
    
    def _snapshot_model_state(self, snapshot: StateSnapshot) -> None:
        """
        Add model state to snapshot
        
        Args:
            snapshot: Snapshot to update
        """
        try:
            # Extract model state dict
            if hasattr(self.sam, 'state_dict'):
                # Convert state dict to CPU tensors
                state_dict = {}
                for k, v in self.sam.state_dict().items():
                    if torch.is_tensor(v):
                        state_dict[k] = v.cpu()
                    else:
                        state_dict[k] = v
                        
                # Add to snapshot
                snapshot.add_component("model_state_dict", state_dict)
                
            # Save additional model info
            if hasattr(self.sam, 'global_step'):
                snapshot.add_component("global_step", self.sam.global_step)
                
            if hasattr(self.sam, 'growth_history'):
                snapshot.add_component("growth_history", self.sam.growth_history)
                
            if hasattr(self.sam, 'layers') and isinstance(self.sam.layers, (list, nn.ModuleList)):
                snapshot.add_component("num_layers", len(self.sam.layers))
                
        except Exception as e:
            logger.error(f"Error snapshotting model state: {e}")
            raise
    
    def restore_from_snapshot(self, snapshot_id: str, components: List[str] = None) -> bool:
        """
        Restore SAM state from a snapshot
        
        Args:
            snapshot_id: ID of the snapshot
            components: List of components to restore (None for all)
            
        Returns:
            bool: True if restore was successful
        """
        # Check if snapshot exists
        if snapshot_id not in self.snapshots:
            logger.error(f"Snapshot {snapshot_id} not found")
            return False
            
        snapshot_info = self.snapshots[snapshot_id]
        snapshot_dir = snapshot_info["path"]
        
        try:
            # Load snapshot
            snapshot = StateSnapshot.load(snapshot_dir)
            
            # Determine components to restore
            if components is None:
                components = snapshot_info["components"]
                
            # Restore each component
            for component in components:
                self._restore_component(snapshot, component)
                
            logger.info(f"Restored SAM state from snapshot {snapshot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring from snapshot {snapshot_id}: {e}")
            return False
    
    def _restore_component(self, snapshot: StateSnapshot, component: str) -> None:
        """
        Restore a component from a snapshot
        
        Args:
            snapshot: Snapshot to restore from
            component: Component name
        """
        try:
            # Get component state
            state = snapshot.get_component(component)
            
            # Restore based on component type
            if component == "concept_bank":
                self._restore_concept_bank(state)
            elif component == "thought_state":
                self._restore_thought_state(state)
            elif component == "experience_manager":
                self._restore_experience_manager(state)
            elif component == "consciousness":
                self._restore_consciousness(state)
            elif component == "config":
                self._restore_config(state)
            elif component == "model_state_dict":
                self._restore_model_state(state)
            elif component == "global_step":
                self.sam.global_step = state
            elif component == "growth_history":
                self.sam.growth_history = state
            else:
                logger.warning(f"No restore handler for component {component}")
                
        except Exception as e:
            logger.error(f"Error restoring component {component}: {e}")
            raise
    
    def _restore_concept_bank(self, state: Dict) -> None:
        """
        Restore concept bank from state
        
        Args:
            state: Concept bank state
        """
        if not hasattr(self.sam, 'concept_bank'):
            logger.warning("SAM instance has no concept_bank attribute")
            return
            
        concept_bank = self.sam.concept_bank
        
        # Restore basic state
        concept_bank.next_concept_id = state["next_concept_id"]
        concept_bank.concept_metadata = state["concept_metadata"]
        concept_bank.source_to_concept = state["source_to_concept"]
        concept_bank.related_concepts = defaultdict(list, state["related_concepts"])
        concept_bank.hive_shared_concepts = set(state["hive_shared_concepts"])
        concept_bank.hive_private_concepts = set(state["hive_private_concepts"])
        concept_bank.hive_pending_sync = set(state["hive_pending_sync"])
        concept_bank.hive_origin = state["hive_origin"]
        concept_bank.hive_global_id_map = state["hive_global_id_map"]
        concept_bank.modality_concepts = {k: set(v) for k, v in state["modality_concepts"].items()}
        concept_bank.creation_history = state["creation_history"]
        
        # Restore tensors
        device = concept_bank.concept_embeddings.weight.device
        dtype = concept_bank.concept_embeddings.weight.dtype
        
        if "concept_frequencies" in state:
            concept_bank.register_buffer("concept_frequencies", 
                                        torch.tensor(state["concept_frequencies"], device=device))
                                        
        if "concept_timestamps" in state:
            concept_bank.register_buffer("concept_timestamps", 
                                        torch.tensor(state["concept_timestamps"], device=device))
                                        
        if "meaning_vectors" in state:
            concept_bank.register_buffer("meaning_vectors", 
                                        torch.tensor(state["meaning_vectors"], device=device, dtype=dtype))
                                        
        if "concept_embeddings_weight" in state and concept_bank.concept_embeddings.weight.shape == state["concept_embeddings_weight"].shape:
            with torch.no_grad():
                concept_bank.concept_embeddings.weight.copy_(
                    torch.tensor(state["concept_embeddings_weight"], device=device, dtype=dtype)
                )
        elif "concept_embeddings_weight" in state:
            # Resize embeddings if needed
            old_size = concept_bank.concept_embeddings.num_embeddings
            new_size = state["concept_embeddings_weight"].shape[0]
            
            if new_size > old_size:
                concept_bank.grow_if_needed(required_size=new_size)
                
            # Copy weights that fit
            with torch.no_grad():
                concept_bank.concept_embeddings.weight[:min(old_size, new_size)].copy_(
                    torch.tensor(state["concept_embeddings_weight"][:min(old_size, new_size)], 
                                device=device, dtype=dtype)
                )
    
    def _restore_thought_state(self, state: Dict) -> None:
        """
        Restore thought state from state
        
        Args:
            state: Thought state
        """
        if not hasattr(self.sam, 'thought_state'):
            logger.warning("SAM instance has no thought_state attribute")
            return
            
        thought_state = self.sam.thought_state
        device = next(thought_state.parameters()).device
        
        # Restore basic state
        thought_state.thought_depth = state["thought_depth"]
        thought_state.evolution_history = state["evolution_history"]
        thought_state.personal_factor = state.get("personal_factor", 0.8)
        
        # Restore tensor state
        if "thought_memory" in state and state["thought_memory"]:
            thought_state.thought_memory = [
                torch.tensor(t, device=device) for t in state["thought_memory"]
            ]
            
        if "superposition_memories" in state and state["superposition_memories"]:
            thought_state.superposition_memories = [
                [torch.tensor(m, device=device) for m in memories]
                for memories in state["superposition_memories"]
            ]
            
        if "modality_thoughts" in state:
            thought_state.modality_thoughts = {
                k: torch.tensor(v, device=device) if v is not None else None
                for k, v in state["modality_thoughts"].items()
            }
            
        if "amplitudes" in state:
            thought_state.register_buffer("amplitudes", 
                                         torch.tensor(state["amplitudes"], device=device))
                                         
        if "shared_thought" in state and state["shared_thought"] is not None:
            thought_state.shared_thought = torch.tensor(state["shared_thought"], device=device)
            
        if "local_thought" in state and state["local_thought"] is not None:
            thought_state.local_thought = torch.tensor(state["local_thought"], device=device)
    
    def _restore_experience_manager(self, state: Dict) -> None:
        """
        Restore experience manager from state
        
        Args:
            state: Experience manager state
        """
        if not hasattr(self.sam, 'experience_manager'):
            logger.warning("SAM instance has no experience_manager attribute")
            return
            
        experience_manager = self.sam.experience_manager
        
        # Restore state
        experience_manager.experiences = state["experiences"]
        experience_manager.loaded_experiences = state["loaded_experiences"]
        experience_manager.shared_experiences = state["shared_experiences"]
        experience_manager.private_experiences = state["private_experiences"]
        experience_manager.pending_sync_experiences = deque(state["pending_sync_experiences"])
        experience_manager.modality_experiences = defaultdict(list, state["modality_experiences"])
    
    def _restore_consciousness(self, state: Dict) -> None:
        """
        Restore consciousness from state
        
        Args:
            state: Consciousness state
        """
        if not hasattr(self.sam, 'consciousness'):
            logger.warning("SAM instance has no consciousness attribute")
            return
            
        consciousness = self.sam.consciousness
        device = next(self.sam.parameters()).device
        
        # Restore basic state
        consciousness.concept_cluster_history = state["concept_cluster_history"]
        consciousness.concept_entropy_history = state["concept_entropy_history"]
        consciousness.resonance_scores = state["resonance_scores"]
        consciousness.personal_concepts = set(state["personal_concepts"])
        consciousness.personality_initialized = state["personality_initialized"]
        
        # Restore tensors
        if "identity_centroids" in state:
            consciousness.identity_centroids = {
                k: torch.tensor(v, device=device) 
                for k, v in state["identity_centroids"].items()
            }
            
        if "modality_centroids" in state:
            consciousness.modality_centroids = {
                k: torch.tensor(v, device=device)
                for k, v in state["modality_centroids"].items()
            }
            
        if "personality_vector" in state and state["personality_vector"] is not None:
            consciousness.personality_vector = torch.tensor(state["personality_vector"], device=device)
    
    def _restore_config(self, state: Dict) -> None:
        """
        Restore config from state
        
        Args:
            state: Config state
        """
        if not hasattr(self.sam, 'config'):
            logger.warning("SAM instance has no config attribute")
            return
            
        # Handle special types
        for k, v in state.items():
            if isinstance(v, str) and "torch.float" in v:
                if "float16" in v or "half" in v:
                    state[k] = torch.float16
                elif "float64" in v or "double" in v:
                    state[k] = torch.float64
                else:
                    state[k] = torch.float32
                    
        # Restore config
        if hasattr(self.sam.config, 'from_dict'):
            self.sam.config = self.sam.config.from_dict(state)
        elif hasattr(self.sam.config, '__dict__'):
            for k, v in state.items():
                if hasattr(self.sam.config, k):
                    setattr(self.sam.config, k, v)
        else:
            # Assume config is a dict-like object
            for k, v in state.items():
                self.sam.config[k] = v
    
    def _restore_model_state(self, state_dict: Dict) -> None:
        """
        Restore model state dict
        
        Args:
            state_dict: Model state dict
        """
        if not hasattr(self.sam, 'load_state_dict'):
            logger.warning("SAM instance has no load_state_dict method")
            return
            
        try:
            # Handle strict vs non-strict loading
            try:
                self.sam.load_state_dict(state_dict, strict=True)
                logger.info("Restored model state dict with strict=True")
            except Exception as strict_error:
                logger.warning(f"Strict state dict loading failed: {strict_error}")
                logger.info("Trying non-strict loading...")
                self.sam.load_state_dict(state_dict, strict=False)
                logger.info("Restored model state dict with strict=False")
                
        except Exception as e:
            logger.error(f"Error restoring model state dict: {e}")
            raise
    
    def get_snapshots(self, count: int = None, oldest_first: bool = False) -> List[Dict]:
        """
        Get list of available snapshots
        
        Args:
            count: Number of snapshots to return (None for all)
            oldest_first: Whether to return oldest snapshots first
            
        Returns:
            List[Dict]: Snapshot metadata
        """
        snapshots = list(self.snapshots.values())
        
        # Sort by timestamp
        snapshots.sort(key=lambda s: s["timestamp"], reverse=not oldest_first)
        
        # Limit count if specified
        if count is not None:
            snapshots = snapshots[:count]
            
        return snapshots
    
    def delete_snapshot(self, snapshot_id: str) -> bool:
        """
        Delete a snapshot
        
        Args:
            snapshot_id: ID of the snapshot
            
        Returns:
            bool: True if deletion was successful
        """
        if snapshot_id not in self.snapshots:
            logger.error(f"Snapshot {snapshot_id} not found")
            return False
            
        snapshot_info = self.snapshots[snapshot_id]
        snapshot_dir = snapshot_info["path"]
        
        try:
            # Delete directory
            if os.path.isdir(snapshot_dir):
                shutil.rmtree(snapshot_dir)
                
            # Remove from registry
            del self.snapshots[snapshot_id]
            self._save_snapshot_registry()
            
            logger.info(f"Deleted snapshot {snapshot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting snapshot {snapshot_id}: {e}")
            return False
    
    def _enforce_retention_policy(self) -> None:
        """Enforce the snapshot retention policy"""
        if self.retention_count <= 0:
            return  # No limit
            
        # Get snapshots sorted by timestamp (newest first)
        snapshots = self.get_snapshots()
        
        # Keep only the specified number of snapshots
        snapshots_to_delete = snapshots[self.retention_count:]
        
        for snapshot in snapshots_to_delete:
            self.delete_snapshot(snapshot["id"])
    
    def _replicate_snapshot(self, snapshot_id: str) -> None:
        """
        Replicate a snapshot to target nodes
        
        Args:
            snapshot_id: ID of the snapshot to replicate
        """
        if not self.enable_replication or not self.replication_targets:
            return
            
        # This would involve sending the snapshot to other nodes
        # The actual implementation depends on the network communication system
        logger.info(f"Replication of snapshot {snapshot_id} to {len(self.replication_targets)} targets would happen here")
        
        # Example implementation would look like:
        # for target_id in self.replication_targets:
        #     hive_network.send_snapshot(target_id, snapshot_id, self.snapshots[snapshot_id])
    
    def start_auto_snapshots(self) -> bool:
        """
        Start automatic snapshot scheduling
        
        Returns:
            bool: True if started successfully
        """
        if self.scheduler_thread is not None and self.scheduler_thread.is_alive():
            logger.warning("Automatic snapshots already running")
            return False
            
        self._stop_event.clear()
        
        def scheduler_loop():
            while not self._stop_event.is_set():
                try:
                    # Check if it's time for a snapshot
                    time_since_last = time.time() - self.last_snapshot_time
                    if time_since_last >= self.snapshot_interval:
                        logger.info("Taking scheduled snapshot")
                        self.create_snapshot()
                    
                    # Sleep for a bit, checking stop event periodically
                    for _ in range(60):  # Check every second for 60 seconds
                        if self._stop_event.is_set():
                            break
                        time.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Error in snapshot scheduler: {e}")
                    time.sleep(60)  # Sleep for a minute after error
        
        self.scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info(f"Started automatic snapshots every {self.snapshot_interval} seconds")
        return True
    
    def stop_auto_snapshots(self) -> bool:
        """
        Stop automatic snapshot scheduling
        
        Returns:
            bool: True if stopped successfully
        """
        if self.scheduler_thread is None or not self.scheduler_thread.is_alive():
            logger.warning("Automatic snapshots not running")
            return False
            
        self._stop_event.set()
        self.scheduler_thread.join(timeout=10)
        self.scheduler_thread = None
        
        logger.info("Stopped automatic snapshots")
        return True
    
    def get_status(self) -> Dict:
        """
        Get recovery manager status
        
        Returns:
            Dict: Status information
        """
        return {
            "node_id": self.node_id,
            "snapshot_count": len(self.snapshots),
            "last_snapshot_time": self.last_snapshot_time,
            "auto_snapshots_enabled": self.scheduler_thread is not None and self.scheduler_thread.is_alive(),
            "snapshot_interval": self.snapshot_interval,
            "retention_count": self.retention_count,
            "replication_enabled": self.enable_replication,
            "replication_targets": self.replication_targets,
            "backup_dir": self.backup_dir,
            "current_snapshot": self.current_snapshot
        }
