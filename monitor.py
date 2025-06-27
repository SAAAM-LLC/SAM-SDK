"""
Monitoring components for the SAM Hive Mind SDK
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque

@dataclass
class SyncEvent:
    """Record of a synchronization event"""
    timestamp: float
    sent_concepts: int
    received_concepts: int
    sent_experiences: int
    received_experiences: int
    success: bool
    duration: float = 0.0


class NetworkStats:
    """
    Collects and aggregates statistics for a hive node
    """
    
    def __init__(self, node_id: str, max_history: int = 100):
        """
        Initialize statistics collector
        
        Args:
            node_id: ID of the node these stats belong to
            max_history: Maximum number of events to store
        """
        self.node_id = node_id
        self.max_history = max_history
        
        # Sync statistics
        self.sync_events: Deque[SyncEvent] = deque(maxlen=max_history)
        self.total_syncs = 0
        self.successful_syncs = 0
        self.failed_syncs = 0
        
        # Concept statistics
        self.concepts_sent = 0
        self.concepts_received = 0
        
        # Experience statistics
        self.experiences_sent = 0
        self.experiences_received = 0
        
        # Task statistics
        self.tasks_executed = 0
        self.tasks_successful = 0
        self.tasks_failed = 0
        
        # Performance statistics
        self.avg_sync_duration = 0.0
    
    def record_sync(self, 
                   sent_concepts: int = 0, 
                   received_concepts: int = 0,
                   sent_experiences: int = 0,
                   received_experiences: int = 0,
                   success: bool = True,
                   duration: float = 0.0):
        """
        Record a synchronization event
        
        Args:
            sent_concepts: Number of concepts sent
            received_concepts: Number of concepts received
            sent_experiences: Number of experiences sent
            received_experiences: Number of experiences received
            success: Whether the sync was successful
            duration: Duration of the sync in seconds
        """
        # Create event
        event = SyncEvent(
            timestamp=time.time(),
            sent_concepts=sent_concepts,
            received_concepts=received_concepts,
            sent_experiences=sent_experiences,
            received_experiences=received_experiences,
            success=success,
            duration=duration
        )
        
        # Add to history
        self.sync_events.append(event)
        
        # Update counters
        self.total_syncs += 1
        if success:
            self.successful_syncs += 1
        else:
            self.failed_syncs += 1
            
        self.concepts_sent += sent_concepts
        self.concepts_received += received_concepts
        self.experiences_sent += sent_experiences
