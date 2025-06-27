"""
Configuration classes for the SAM Hive Mind SDK
"""
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union

@dataclass
class HiveConfig:
    """Configuration for HiveNetwork and HiveNode behavior"""
    
    # Network parameters
    sync_interval: float = 60.0  # Seconds between syncs
    connection_timeout: float = 10.0  # Seconds for connection timeout
    sync_timeout: float = 30.0  # Seconds for sync timeout
    
    # Sync limits
    sync_concept_limit: int = 1000  # Max concepts per sync
    sync_experience_limit: int = 20  # Max experiences per sync
    
    # Behavior flags
    auto_sync: bool = True  # Automatically sync at intervals
    enable_compression: bool = True  # Compress network traffic
    compression_level: int = 6  # Zlib compression level (0-9)
    
    # Thought integration
    thought_blend_factor: float = 0.2  # How much to blend shared thoughts (0-1)
    
    # Security
    require_api_key: bool = False  # Require API key for connections
    api_keys: List[str] = field(default_factory=list)  # List of valid API keys
    security_policy: str = "standard"  # Security policy (standard, strict, open)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'HiveConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
    
    @classmethod
    def load(cls, file_path: str) -> 'HiveConfig':
        """Load config from JSON file"""
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            raise ValueError(f"Failed to load config from {file_path}: {e}")
    
    def save(self, file_path: str) -> bool:
        """Save config to JSON file"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            raise ValueError(f"Failed to save config to {file_path}: {e}")


@dataclass
class TaskConfig:
    """Configuration for task execution in the hive network"""
    
    # Task execution
    timeout_seconds: float = 120.0  # Max seconds for task execution
    max_retries: int = 3  # Maximum retry attempts for failed tasks
    
    # Task distribution
    distribution_strategy: str = "capability_match"  # Strategy for distributing tasks
    
    # Priorities
    priority_levels: Dict[str, int] = field(default_factory=lambda: {
        "high": 100,
        "medium": 50,
        "low": 10
    })
    
    # Resource management
    max_concurrent_tasks: int = 5  # Max tasks per node
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return asdict(self)
