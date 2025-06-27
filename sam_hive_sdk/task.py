"""
Task management system for SAM Hive Mind SDK
"""
import uuid
import time
import logging
import threading
from typing import Dict, List, Optional, Union, Callable, Any
from enum import Enum, auto

logger = logging.getLogger("sam_hive_sdk")

class TaskPriority(Enum):
    """Priority levels for tasks"""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class TaskStatus(Enum):
    """Status states for tasks"""
    CREATED = auto()
    QUEUED = auto()
    ASSIGNED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELED = auto()

class Task:
    """
    Represents a unit of work to be performed by a SAM instance
    
    Tasks are the primary mechanism for coordinating work across
    the hive network. They can represent generation requests,
    learning objectives, analysis work, or specialized processing.
    """
    
    def __init__(self, 
                 task_type: str,
                 parameters: Dict = None,
                 priority: TaskPriority = TaskPriority.MEDIUM,
                 task_id: str = None,
                 creator_id: str = None,
                 timeout_seconds: float = 120.0):
        """
        Initialize a new task
        
        Args:
            task_type: Type of task (generate, learn, analyze, dream, etc)
            parameters: Task-specific parameters
            priority: Priority level
            task_id: Unique ID (generated if None)
            creator_id: ID of node that created the task
            timeout_seconds: Max execution time in seconds
        """
        self.task_id = task_id or str(uuid.uuid4())
        self.task_type = task_type
        self.parameters = parameters or {}
        self.priority = priority
        self.creator_id = creator_id
        
        # Execution info
        self.status = TaskStatus.CREATED
        self.created_at = time.time()
        self.assigned_at = None
        self.started_at = None
        self.completed_at = None
        self.assigned_to = None
        self.timeout_seconds = timeout_seconds
        
        # Result storage
        self.result = None
        self.error = None
        
        # Metadata
        self.metadata = {}
        self.retry_count = 0
        self.max_retries = 3
        
        logger.debug(f"Created task {self.task_id} of type {task_type}")
    
    def to_dict(self) -> Dict:
        """Convert task to dictionary representation"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'parameters': self.parameters,
            'priority': self.priority.name,
            'creator_id': self.creator_id,
            'status': self.status.name,
            'created_at': self.created_at,
            'assigned_at': self.assigned_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'assigned_to': self.assigned_to,
            'timeout_seconds': self.timeout_seconds,
            'result': self.result,
            'error': self.error,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Task':
        """Create a task from dictionary representation"""
        task = cls(
            task_type=data['task_type'],
            parameters=data.get('parameters', {}),
            priority=TaskPriority[data.get('priority', 'MEDIUM')],
            task_id=data['task_id'],
            creator_id=data.get('creator_id'),
            timeout_seconds=data.get('timeout_seconds', 120.0)
        )
        
        # Restore state
        task.status = TaskStatus[data.get('status', 'CREATED')]
        task.created_at = data.get('created_at', time.time())
        task.assigned_at = data.get('assigned_at')
        task.started_at = data.get('started_at')
        task.completed_at = data.get('completed_at')
        task.assigned_to = data.get('assigned_to')
        task.result = data.get('result')
        task.error = data.get('error')
        task.retry_count = data.get('retry_count', 0)
        task.max_retries = data.get('max_retries', 3)
        task.metadata = data.get('metadata', {})
        
        return task
    
    def assign(self, node_id: str) -> bool:
        """
        Assign task to a node
        
        Args:
            node_id: ID of node to assign task to
            
        Returns:
            bool: True if assignment was successful
        """
        if self.status != TaskStatus.CREATED and self.status != TaskStatus.QUEUED:
            return False
            
        self.assigned_to = node_id
        self.assigned_at = time.time()
        self.status = TaskStatus.ASSIGNED
        
        logger.debug(f"Task {self.task_id} assigned to node {node_id}")
        return True
    
    def start(self) -> bool:
        """
        Mark task as started
        
        Returns:
            bool: True if status change was successful
        """
        if self.status != TaskStatus.ASSIGNED:
            return False
            
        self.started_at = time.time()
        self.status = TaskStatus.RUNNING
        
        logger.debug(f"Task {self.task_id} started execution")
        return True
    
    def complete(self, result: Any) -> bool:
        """
        Mark task as completed with result
        
        Args:
            result: Task result data
            
        Returns:
            bool: True if status change was successful
        """
        if self.status != TaskStatus.RUNNING:
            return False
            
        self.completed_at = time.time()
        self.status = TaskStatus.COMPLETED
        self.result = result
        
        logger.debug(f"Task {self.task_id} completed successfully")
        return True
    
    def fail(self, error: str) -> bool:
        """
        Mark task as failed with error
        
        Args:
            error: Error message or details
            
        Returns:
            bool: True if status change was successful
        """
        if self.status != TaskStatus.RUNNING and self.status != TaskStatus.ASSIGNED:
            return False
            
        self.completed_at = time.time()
        self.status = TaskStatus.FAILED
        self.error = error
        
        logger.debug(f"Task {self.task_id} failed: {error}")
        return True
    
    def cancel(self) -> bool:
        """
        Cancel a pending or running task
        
        Returns:
            bool: True if cancellation was successful
        """
        if self.status == TaskStatus.COMPLETED or self.status == TaskStatus.FAILED:
            return False
            
        self.completed_at = time.time()
        self.status = TaskStatus.CANCELED
        
        logger.debug(f"Task {self.task_id} canceled")
        return True
    
    def is_timed_out(self) -> bool:
        """
        Check if task has exceeded its timeout
        
        Returns:
            bool: True if task has timed out
        """
        if self.status != TaskStatus.RUNNING:
            return False
            
        if not self.started_at:
            return False
            
        return (time.time() - self.started_at) > self.timeout_seconds
    
    def should_retry(self) -> bool:
        """
        Check if failed task should be retried
        
        Returns:
            bool: True if task should be retried
        """
        return (
            self.status == TaskStatus.FAILED and
            self.retry_count < self.max_retries
        )
    
    def retry(self) -> bool:
        """
        Reset task for retry
        
        Returns:
            bool: True if retry was successful
        """
        if not self.should_retry():
            return False
            
        self.retry_count += 1
        self.status = TaskStatus.CREATED
        self.assigned_to = None
        self.assigned_at = None
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None
        
        logger.debug(f"Task {self.task_id} reset for retry #{self.retry_count}")
        return True


class TaskQueue:
    """
    Queue for managing and prioritizing tasks
    
    The task queue manages pending tasks and handles prioritization,
    timeout detection, and retry logic.
    """
    
    def __init__(self):
        """Initialize task queue"""
        self.tasks = {}  # task_id -> Task
        self.queued_tasks = {
            TaskPriority.LOW: [],
            TaskPriority.MEDIUM: [],
            TaskPriority.HIGH: [],
            TaskPriority.CRITICAL: []
        }
        self.running_tasks = {}  # task_id -> Task
        self.completed_tasks = {}  # task_id -> Task
        
        self.task_lock = threading.RLock()
        
        # Start monitoring thread
        self._stop_monitor = threading.Event()
        self.monitor_thread = threading.Thread(target=self._monitor_tasks)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Task queue initialized")
    
    def enqueue(self, task: Task) -> bool:
        """
        Add a task to the queue
        
        Args:
            task: Task to enqueue
            
        Returns:
            bool: True if task was enqueued
        """
        with self.task_lock:
            # Store task
            self.tasks[task.task_id] = task
            
            # Update status
            if task.status == TaskStatus.CREATED:
                task.status = TaskStatus.QUEUED
                
            # Add to priority queue
            self.queued_tasks[task.priority].append(task.task_id)
            
            logger.debug(f"Task {task.task_id} enqueued with {task.priority.name} priority")
            return True
    
    def dequeue(self) -> Optional[Task]:
        """
        Get highest priority task from queue
        
        Returns:
            Optional[Task]: Next task or None if queue is empty
        """
        with self.task_lock:
            # Check priority queues in order
            for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, 
                           TaskPriority.MEDIUM, TaskPriority.LOW]:
                if self.queued_tasks[priority]:
                    task_id = self.queued_tasks[priority].pop(0)
                    task = self.tasks[task_id]
                    return task
            
            return None
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get task by ID
        
        Args:
            task_id: ID of task to retrieve
            
        Returns:
            Optional[Task]: Task if found, None otherwise
        """
        return self.tasks.get(task_id)
    
    def update_task(self, task: Task) -> bool:
        """
        Update task in queue
        
        Args:
            task: Updated task
            
        Returns:
            bool: True if update was successful
        """
        with self.task_lock:
            if task.task_id not in self.tasks:
                return False
                
            # Update task
            self.tasks[task.task_id] = task
            
            # Update collections based on status
            if task.status == TaskStatus.QUEUED:
                # Ensure task is in queued_tasks
                for priority in self.queued_tasks:
                    if task.task_id in self.queued_tasks[priority]:
                        self.queued_tasks[priority].remove(task.task_id)
                self.queued_tasks[task.priority].append(task.task_id)
                
                # Remove from other collections
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
                if task.task_id in self.completed_tasks:
                    del self.completed_tasks[task.task_id]
                    
            elif task.status == TaskStatus.RUNNING:
                # Remove from queued
                for priority in self.queued_tasks:
                    if task.task_id in self.queued_tasks[priority]:
                        self.queued_tasks[priority].remove(task.task_id)
                
                # Add to running
                self.running_tasks[task.task_id] = task
                
                # Remove from completed
                if task.task_id in self.completed_tasks:
                    del self.completed_tasks[task.task_id]
                    
            elif task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED]:
                # Remove from queued
                for priority in self.queued_tasks:
                    if task.task_id in self.queued_tasks[priority]:
                        self.queued_tasks[priority].remove(task.task_id)
                
                # Remove from running
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
                
                # Add to completed
                self.completed_tasks[task.task_id] = task
            
            return True
    
    def remove_task(self, task_id: str) -> bool:
        """
        Remove task from queue
        
        Args:
            task_id: ID of task to remove
            
        Returns:
            bool: True if removal was successful
        """
        with self.task_lock:
            if task_id not in self.tasks:
                return False
                
            # Remove from all collections
            for priority in self.queued_tasks:
                if task_id in self.queued_tasks[priority]:
                    self.queued_tasks[priority].remove(task_id)
                    
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
                
            if task_id in self.completed_tasks:
                del self.completed_tasks[task_id]
                
            # Remove from tasks
            del self.tasks[task_id]
            
            return True
    
    def get_queue_stats(self) -> Dict:
        """
        Get statistics about the task queue
        
        Returns:
            Dict: Queue statistics
        """
        with self.task_lock:
            return {
                'total_tasks': len(self.tasks),
                'queued': {
                    'critical': len(self.queued_tasks[TaskPriority.CRITICAL]),
                    'high': len(self.queued_tasks[TaskPriority.HIGH]),
                    'medium': len(self.queued_tasks[TaskPriority.MEDIUM]),
                    'low': len(self.queued_tasks[TaskPriority.LOW]),
                    'total': sum(len(q) for q in self.queued_tasks.values())
                },
                'running': len(self.running_tasks),
                'completed': len(self.completed_tasks)
            }
    
    def _monitor_tasks(self):
        """Monitor running tasks for timeouts and handle retries"""
        while not self._stop_monitor.is_set():
            try:
                with self.task_lock:
                    # Check for timed out tasks
                    timed_out_tasks = []
                    for task_id, task in list(self.running_tasks.items()):
                        if task.is_timed_out():
                            timed_out_tasks.append(task_id)
                            
                    # Handle timed out tasks
                    for task_id in timed_out_tasks:
                        task = self.running_tasks[task_id]
                        task.fail("Task timed out")
                        
                        # Move to completed
                        self.completed_tasks[task_id] = task
                        del self.running_tasks[task_id]
                        
                        # Check for retry
                        if task.should_retry():
                            if task.retry():
                                # Re-enqueue
                                task.status = TaskStatus.QUEUED
                                self.queued_tasks[task.priority].append(task_id)
                                logger.info(f"Task {task_id} timed out and requeued for retry #{task.retry_count}")
                        else:
                            logger.info(f"Task {task_id} timed out and will not be retried")
            
            except Exception as e:
                logger.error(f"Error in task monitor: {e}")
                
            # Sleep for a bit
            for _ in range(10):  # Check every second, but allow for quick shutdown
                if self._stop_monitor.is_set():
                    break
                time.sleep(0.1)
    
    def shutdown(self):
        """Shutdown the task queue"""
        self._stop_monitor.set()
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("Task queue shutdown")


class TaskRegistry:
    """
    Registry of task handlers
    
    Maps task types to handler functions that execute them.
    """
    
    def __init__(self):
        """Initialize task registry"""
        self.handlers = {}  # task_type -> handler function
    
    def register_handler(self, task_type: str, handler: Callable[[Task], Any]):
        """
        Register a handler for a task type
        
        Args:
            task_type: Type of task
            handler: Function that executes the task
        """
        self.handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    def get_handler(self, task_type: str) -> Optional[Callable]:
        """
        Get handler for a task type
        
        Args:
            task_type: Type of task
            
        Returns:
            Optional[Callable]: Handler function if registered
        """
        return self.handlers.get(task_type)
    
    def execute_task(self, task: Task) -> Any:
        """
        Execute a task using its registered handler
        
        Args:
            task: Task to execute
            
        Returns:
            Any: Result of task execution
            
        Raises:
            ValueError: If no handler is registered for task type
        """
        handler = self.get_handler(task.task_type)
        if not handler:
            raise ValueError(f"No handler registered for task type: {task.task_type}")
            
        try:
            # Mark task as running
            task.start()
            
            # Execute handler
            result = handler(task)
            
            # Mark task as completed
            task.complete(result)
            
            return result
            
        except Exception as e:
            # Mark task as failed
            task.fail(str(e))
            
            # Re-raise
            raise


class TaskDistributor:
    """
    Distributes tasks across a network of SAM nodes
    
    Handles task routing, load balancing, and fault tolerance.
    """
    
    def __init__(self, network):
        """
        Initialize task distributor
        
        Args:
            network: HiveNetwork instance
        """
        self.network = network
        self.task_queue = TaskQueue()
        self.task_registry = TaskRegistry()
        
        # Node capabilities cache
        self.node_capabilities = {}  # node_id -> capabilities
        
        # Distribution settings
        self.max_tasks_per_node = 5
        self.node_load = {}  # node_id -> current task count
        
        # Worker thread
        self._stop_worker = threading.Event()
        self.worker_thread = threading.Thread(target=self._distribution_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        logger.info("Task distributor initialized")
    
    def submit_task(self, task: Task) -> str:
        """
        Submit a task for execution
        
        Args:
            task: Task to submit
            
        Returns:
            str: Task ID
        """
        # Enqueue task
        self.task_queue.enqueue(task)
        logger.info(f"Task {task.task_id} ({task.task_type}) submitted for execution")
        return task.task_id
    
    def get_task_status(self, task_id: str) -> Dict:
        """
        Get status of a task
        
        Args:
            task_id: ID of task to check
            
        Returns:
            Dict: Task status
        """
        task = self.task_queue.get_task(task_id)
        if not task:
            return {'error': 'Task not found'}
            
        return task.to_dict()
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task
        
        Args:
            task_id: ID of task to cancel
            
        Returns:
            bool: True if cancellation was successful
        """
        task = self.task_queue.get_task(task_id)
        if not task:
            return False
            
        # Cancel task
        if task.cancel():
            # Update in queue
            self.task_queue.update_task(task)
            
            # If assigned to a node, notify it
            if task.assigned_to:
                self._notify_node_cancel(task.assigned_to, task_id)
                
            logger.info(f"Task {task_id} canceled")
            return True
            
        return False
    
    def update_node_capabilities(self, node_id: str, capabilities: Dict):
        """
        Update cached capabilities for a node
        
        Args:
            node_id: ID of node
            capabilities: Node capabilities
        """
        self.node_capabilities[node_id] = capabilities
        logger.debug(f"Updated capabilities for node {node_id}")
    
    def _distribution_worker(self):
        """Worker thread for distributing tasks"""
        while not self._stop_worker.is_set():
            try:
                # Get next task from queue
                task = self.task_queue.dequeue()
                if not task:
                    # No tasks to process
                    time.sleep(0.1)
                    continue
                    
                # Find best node for task
                node_id = self._select_node_for_task(task)
                if not node_id:
                    # No suitable node found, requeue task
                    self.task_queue.enqueue(task)
                    logger.debug(f"No suitable node for task {task.task_id}, requeued")
                    time.sleep(1)  # Wait before trying again
                    continue
                    
                # Assign task to node
                if task.assign(node_id):
                    # Update task in queue
                    self.task_queue.update_task(task)
                    
                    # Update node load
                    self.node_load[node_id] = self.node_load.get(node_id, 0) + 1
                    
                    # Send task to node
                    self._send_task_to_node(node_id, task)
                    
                    logger.info(f"Task {task.task_id} assigned to node {node_id}")
                else:
                    # Failed to assign, requeue
                    self.task_queue.enqueue(task)
                    logger.warning(f"Failed to assign task {task.task_id}, requeued")
            
            except Exception as e:
                logger.error(f"Error in distribution worker: {e}")
                time.sleep(1)
    
    def _select_node_for_task(self, task: Task) -> Optional[str]:
        """
        Select best node for a task
        
        Args:
            task: Task to assign
            
        Returns:
            Optional[str]: ID of selected node
        """
        # Check if no nodes are available
        if not self.node_capabilities:
            return None
            
        # Collect candidate nodes
        candidates = []
        
        for node_id, capabilities in self.node_capabilities.items():
            # Skip nodes that are at capacity
            current_load = self.node_load.get(node_id, 0)
            if current_load >= self.max_tasks_per_node:
                continue
                
            # Skip nodes that don't match required capabilities
            if not self._node_can_handle_task(node_id, capabilities, task):
                continue
                
            # Calculate node score (lower is better)
            score = current_load
            
            # Add to candidates
            candidates.append((node_id, score))
            
        if not candidates:
            return None
            
        # Sort by score (lower is better)
        candidates.sort(key=lambda x: x[1])
        
        # Return best node
        return candidates[0][0]
    
    def _node_can_handle_task(self, node_id: str, capabilities: Dict, task: Task) -> bool:
        """
        Check if a node can handle a task
        
        Args:
            node_id: ID of node
            capabilities: Node capabilities
            task: Task to check
            
        Returns:
            bool: True if node can handle task
        """
        # Check task type requirements
        task_type = task.task_type
        
        # Example: Check if node supports multimodal for multimodal tasks
        if task_type == 'multimodal_generate':
            return capabilities.get('multimodal', False)
            
        # Example: Check for minimum model size for complex tasks
        if task_type in ['advanced_reasoning', 'complex_generation']:
            model_size = capabilities.get('model_size', {})
            param_count = model_size.get('parameter_count', 0)
            return param_count >= 100000000  # 100M parameters
            
        # Default: assume node can handle basic tasks
        return True
    
    def _send_task_to_node(self, node_id: str, task: Task):
        """
        Send a task to a node for execution
        
        Args:
            node_id: ID of node
            task: Task to send
        """
        try:
            # Get node connection info
            node_info = self.network.nodes.get(node_id)
            if not node_info:
                raise ValueError(f"Node {node_id} not found")
                
            # Prepare task data
            task_data = {
                'action': 'execute',
                'task': task.to_dict()
            }
            
            # Send task to node (actual implementation depends on network)
            # This is just a placeholder - in real implementation would use network to send
            logger.debug(f"Would send task {task.task_id} to node {node_id}")
            
            # In a real implementation:
            # response = requests.post(f"{node_info['api_url']}/task", json=task_data)
            # if response.status_code != 200:
            #     raise ValueError(f"Failed to send task: {response.text}")
            
        except Exception as e:
            logger.error(f"Error sending task to node {node_id}: {e}")
            
            # Mark task as failed
            task.fail(f"Failed to send to node: {str(e)}")
            
            # Update in queue
            self.task_queue.update_task(task)
    
    def _notify_node_cancel(self, node_id: str, task_id: str):
        """
        Notify a node that a task has been canceled
        
        Args:
            node_id: ID of node
            task_id: ID of canceled task
        """
        try:
            # Get node connection info
            node_info = self.network.nodes.get(node_id)
            if not node_info:
                raise ValueError(f"Node {node_id} not found")
                
            # Prepare cancel data
            cancel_data = {
                'action': 'cancel',
                'task_id': task_id
            }
            
            # Send cancel to node (actual implementation depends on network)
            # This is just a placeholder - in real implementation would use network to send
            logger.debug(f"Would send cancel for task {task_id} to node {node_id}")
            
            # In a real implementation:
            # response = requests.post(f"{node_info['api_url']}/task", json=cancel_data)
            # if response.status_code != 200:
            #     raise ValueError(f"Failed to send cancel: {response.text}")
            
        except Exception as e:
            logger.error(f"Error sending cancel to node {node_id}: {e}")
    
    def handle_task_update(self, node_id: str, task_id: str, status: str, result: Any = None, error: str = None):
        """
        Handle update from a node about a task
        
        Args:
            node_id: ID of node
            task_id: ID of task
            status: New task status
            result: Task result (if completed)
            error: Error message (if failed)
        """
        # Get task
        task = self.task_queue.get_task(task_id)
        if not task:
            logger.warning(f"Received update for unknown task {task_id}")
            return
            
        # Verify node assignment
        if task.assigned_to != node_id:
            logger.warning(f"Received update for task {task_id} from wrong node {node_id}")
            return
            
        # Update task based on status
        if status == 'completed':
            task.complete(result)
        elif status == 'failed':
            task.fail(error or "Unknown error")
        elif status == 'running':
            if task.status != TaskStatus.RUNNING:
                task.start()
        elif status == 'canceled':
            task.cancel()
            
        # Update task in queue
        self.task_queue.update_task(task)
        
        # Update node load
        if status in ['completed', 'failed', 'canceled']:
            self.node_load[node_id] = max(0, self.node_load.get(node_id, 0) - 1)
    
    def shutdown(self):
        """Shutdown the task distributor"""
        self._stop_worker.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        self.task_queue.shutdown()
        logger.info("Task distributor shutdown")
