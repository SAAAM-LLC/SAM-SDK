"""
SAM Hive Mind SDK
-----------------
A powerful interface for creating, connecting and orchestrating 
networks of Synergistic Autonomous Machines.

Created by SAAAM LLC
"""

__version__ = "0.1.0"

from .hive import HiveNetwork, HiveNode, MasterNode
from .config import HiveConfig
from .sync import SyncManager, SyncStrategy
from .tasks import TaskRegistry, Task, TaskDistributor
from .monitor import HiveMonitor, NetworkStats
from .security import HiveSecurity, ConnectionPolicy
