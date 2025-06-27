"""
Intelligent Task Distribution System for SAM-SDK

Enables efficient and intelligent distribution of tasks across a network
of SAM nodes based on capabilities, specialization, and current load.
"""

import logging
import time
import uuid
import heapq
import asyncio
import threading
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Callable

logger = logging.getLogger("sam_sdk.distribution")

class Task:
    """
    Represents a task that can be distributed across the SAM network
    
    Tasks are the fundamental unit of work in the distributed intelligence
    system, with rich metadata for optimal routing and execution.
    """
    
    def __init__(self, 
                 task_type: str,
                 parameters: Dict,
                 priority: int = 5,
                 deadline: Optional[float] = None,
                 required_capabilities: Optional[Dict] = None,
                 preferred_domains: Optional[List[str]] = None):
        """
        Initialize a task
        
        Args:
            task_type: Type of task (generate, learn, analyze, etc.)
            parameters: Task-specific parameters
            priority: Priority level (1-10, higher is more important)
            deadline: Optional deadline timestamp
            required_capabilities: Optional capabilities required to execute this task
            preferred_domains: Optional list of preferred knowledge domains
        """
        self.task_id = str(uuid.uuid4())
        self.task_type = task_type
        self.parameters = parameters
        self.priority = max(1, min(10, priority))
        self.created_at = time.time()
        self.deadline = deadline
        self.required_capabilities = required_capabilities or {}
        self.preferred_domains = preferred_domains or []
        
        # Execution state
        self.assigned_to = None  # Node ID
        self.assigned_at = None
        self.started_at = None
        self.completed_at = None
        self.status = "pending"  # pending, assigned, in_progress, completed, failed
        self.result = None
        self.error = None
        
        # Metrics
        self.attempts = 0
        self.execution_time = None
        
        # Analysis
        self.complexity_score = self._calculate_complexity()
        self.urgency_score = self._calculate_urgency()
        
        logger.debug(f"Created task {self.task_id} of type {task_type}")
    
    def _calculate_complexity(self) -> float:
        """Calculate the complexity score of this task"""
        complexity = 1.0
        
        # Adjust based on task type
        complexity_weights = {
            "generate": 1.0,
            "learn": 1.2,
            "analyze": 1.5,
            "dream": 2.0,
            "evolve": 2.5,
            "research": 3.0
        }
        complexity *= complexity_weights.get(self.task_type, 1.0)
        
        # Adjust based on parameters
        if "max_length" in self.parameters:
            complexity *= (1.0 + 0.5 * min(1.0, self.parameters["max_length"] / 1000))
        
        if "temperature" in self.parameters:
            # Higher temperature = more creative = more complex
            complexity *= (1.0 + 0.3 * self.parameters.get("temperature", 0.7))
        
        if "content" in self.parameters and isinstance(self.parameters["content"], str):
            complexity *= (1.0 + 0.2 * min(1.0, len(self.parameters["content"]) / 5000))
        
        return complexity
    
    def _calculate_urgency(self) -> float:
        """Calculate the urgency score of this task"""
        # Base urgency from priority
        urgency = self.priority / 10.0
        
        # If deadline exists, factor it in
        if self.deadline:
            time_remaining = max(0, self.deadline - time.time())
            time_factor = 1.0 - min(1.0, time_remaining / 3600)  # 1 hour scale
            urgency += time_factor * 0.5
        
        return min(1.0, urgency)
    
    def update_status(self, status: str, node_id: Optional[str] = None) -> None:
        """
        Update the status of this task
        
        Args:
            status: New status
            node_id: Optional ID of the node updating status
        """
        old_status = self.status
        self.status = status
        
        if status == "assigned" and old_status == "pending":
            self.assigned_to = node_id
            self.assigned_at = time.time()
            
        elif status == "in_progress" and old_status in ["pending", "assigned"]:
            if not self.assigned_to and node_id:
                self.assigned_to = node_id
                self.assigned_at = time.time()
            self.started_at = time.time()
            
        elif status == "completed" and old_status in ["in_progress", "assigned"]:
            self.completed_at = time.time()
            if self.started_at:
                self.execution_time = self.completed_at - self.started_at
                
        elif status == "failed":
            self.completed_at = time.time()
            self.attempts += 1
    
    def set_result(self, result: Any, error: Optional[str] = None) -> None:
        """
        Set the result of this task
        
        Args:
            result: Task result
            error: Optional error message
        """
        self.result = result
        self.error = error
        self.update_status("completed" if error is None else "failed")
    
    def can_execute_on(self, node_capabilities: Dict) -> bool:
        """
        Check if this task can execute on a node with given capabilities
        
        Args:
            node_capabilities: Capabilities of the node
            
        Returns:
            bool: True if the node can execute this task
        """
        # Check each required capability
        for cap_name, cap_value in self.required_capabilities.items():
            # Handle nested capabilities (e.g., model_size.hidden_dim)
            if "." in cap_name:
                parts = cap_name.split(".")
                node_value = node_capabilities
                for part in parts:
                    if part not in node_value:
                        return False
                    node_value = node_value[part]
                
                # Check capability
                if isinstance(cap_value, (int, float)):
                    if node_value < cap_value:
                        return False
            elif cap_name not in node_capabilities:
                return False
            elif isinstance(cap_value, (int, float)) and node_capabilities[cap_name] < cap_value:
                return False
        
        return True
    
    def to_dict(self) -> Dict:
        """Convert task to dictionary representation"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "parameters": self.parameters,
            "priority": self.priority,
            "created_at": self.created_at,
            "deadline": self.deadline,
            "required_capabilities": self.required_capabilities,
            "preferred_domains": self.preferred_domains,
            "assigned_to": self.assigned_to,
            "assigned_at": self.assigned_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "attempts": self.attempts,
            "execution_time": self.execution_time,
            "complexity_score": self.complexity_score,
            "urgency_score": self.urgency_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Task':
        """Create task from dictionary representation"""
        task = cls(
            task_type=data["task_type"],
            parameters=data["parameters"],
            priority=data["priority"],
            deadline=data.get("deadline"),
            required_capabilities=data.get("required_capabilities"),
            preferred_domains=data.get("preferred_domains")
        )
        
        # Restore state
        task.task_id = data["task_id"]
        task.created_at = data["created_at"]
        task.assigned_to = data.get("assigned_to")
        task.assigned_at = data.get("assigned_at")
        task.started_at = data.get("started_at")
        task.completed_at = data.get("completed_at")
        task.status = data["status"]
        task.attempts = data["attempts"]
        task.execution_time = data.get("execution_time")
        task.complexity_score = data.get("complexity_score", task.complexity_score)
        task.urgency_score = data.get("urgency_score", task.urgency_score)
        
        return task


class TaskDistributor:
    """
    Manages the distribution of tasks across SAM nodes
    
    The TaskDistributor implements intelligent routing algorithms to match
    tasks with the most appropriate nodes based on capabilities, specialization,
    current load, and performance history.
    """
    
    def __init__(self, hive_network=None):
        """
        Initialize the task distributor
        
        Args:
            hive_network: Optional reference to the hive network
        """
        self.hive = hive_network
        
        # Task management
        self.tasks = {}  # task_id -> Task
        self.task_queue = []  # Priority queue
        self.failed_tasks = {}  # task_id -> Task (for retry)
        
        # Node tracking
        self.nodes = {}  # node_id -> node info
        self.node_loads = {}  # node_id -> current load
        self.node_performance = {}  # node_id -> performance metrics
        
        # Distribution strategies
        self.strategies = {
            "round_robin": self._distribute_round_robin,
            "load_balanced": self._distribute_load_balanced,
            "capability_match": self._distribute_capability_match,
            "domain_specialized": self._distribute_domain_specialized,
            "performance_optimized": self._distribute_performance_optimized
        }
        self.current_strategy = "capability_match"
        
        # Async task processing
        self.processing = False
        self.processing_thread = None
        self._stop_event = threading.Event()
        
        # Round-robin state
        self._next_node_index = 0
        
        # Statistics
        self.stats = {
            "tasks_created": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "avg_queue_time": 0.0
        }
        
        logger.info("Initialized TaskDistributor")
    
    def register_node(self, node_id: str, name: str, capabilities: Dict) -> None:
        """
        Register a node with the distributor
        
        Args:
            node_id: ID of the node
            name: Name of the node
            capabilities: Node capabilities
        """
        self.nodes[node_id] = {
            "id": node_id,
            "name": name,
            "capabilities": capabilities,
            "registered_at": time.time(),
            "last_seen": time.time(),
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "domains": capabilities.get("domains", [])
        }
        
        self.node_loads[node_id] = 0
        self.node_performance[node_id] = {
            "avg_execution_time": 0.0,
            "success_rate": 1.0,
            "last_results": []  # List of recent success/failure (True/False)
        }
        
        logger.info(f"Registered node {name} ({node_id})")
    
    def update_node_status(self, node_id: str, status: Dict) -> None:
        """
        Update a node's status
        
        Args:
            node_id: ID of the node
            status: Status information
        """
        if node_id not in self.nodes:
            logger.warning(f"Attempted to update unknown node: {node_id}")
            return
            
        # Update last seen
        self.nodes[node_id]["last_seen"] = time.time()
        
        # Update capabilities if provided
        if "capabilities" in status:
            self.nodes[node_id]["capabilities"] = status["capabilities"]
        
        # Update domains if provided
        if "domains" in status:
            self.nodes[node_id]["domains"] = status["domains"]
        
        # Update load if provided
        if "current_load" in status:
            self.node_loads[node_id] = status["current_load"]
    
    def create_task(self, 
                   task_type: str,
                   parameters: Dict,
                   priority: int = 5,
                   deadline: Optional[float] = None,
                   required_capabilities: Optional[Dict] = None,
                   preferred_domains: Optional[List[str]] = None) -> str:
        """
        Create a new task
        
        Args:
            task_type: Type of task
            parameters: Task parameters
            priority: Priority level (1-10)
            deadline: Optional deadline timestamp
            required_capabilities: Optional required capabilities
            preferred_domains: Optional preferred domains
            
        Returns:
            str: Task ID
        """
        task = Task(
            task_type=task_type,
            parameters=parameters,
            priority=priority,
            deadline=deadline,
            required_capabilities=required_capabilities,
            preferred_domains=preferred_domains
        )
        
        self.tasks[task.task_id] = task
        
        # Add to priority queue with negative priority (heap is min-heap)
        score = self._calculate_task_priority_score(task)
        heapq.heappush(self.task_queue, (-score, task.created_at, task.task_id))
        
        self.stats["tasks_created"] += 1
        
        logger.debug(f"Task {task.task_id} added to queue with priority score {score}")
        
        return task.task_id
    
    def _calculate_task_priority_score(self, task: Task) -> float:
        """
        Calculate the priority score for a task
        
        Args:
            task: The task
            
        Returns:
            float: Priority score
        """
        # Base priority
        score = task.priority * 10.0
        
        # Factor in deadline
        if task.deadline:
            time_remaining = max(0, task.deadline - time.time())
            if time_remaining <= 0:
                # Overdue tasks get highest priority
                score += 1000
            else:
                # Tasks with closer deadlines get higher priority
                score += 100 * (1.0 / max(0.1, time_remaining / 3600))
        
        # Factor in complexity
        score -= task.complexity_score * 5.0  # More complex tasks lower priority a bit
        
        # Factor in attempts (retry priority boost)
        score += task.attempts * 20.0
        
        return score
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID
        
        Args:
            task_id: ID of the task
            
        Returns:
            Optional[Task]: The task if found
        """
        return self.tasks.get(task_id)
    
    def get_node_tasks(self, node_id: str) -> List[Task]:
        """
        Get tasks assigned to a node
        
        Args:
            node_id: ID of the node
            
        Returns:
            List[Task]: Tasks assigned to the node
        """
        return [task for task in self.tasks.values() if task.assigned_to == node_id]
    
    def start_processing(self) -> bool:
        """
        Start the task processing thread
        
        Returns:
            bool: True if started successfully
        """
        if self.processing:
            return False
            
        self.processing = True
        self._stop_event.clear()
        
        def process_loop():
            while not self._stop_event.is_set():
                try:
                    self._process_task_queue()
                    time.sleep(0.1)  # Small delay to prevent CPU hogging
                except Exception as e:
                    logger.error(f"Error in task processing loop: {e}")
                    time.sleep(1.0)  # Longer delay after error
        
        self.processing_thread = threading.Thread(target=process_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("Started task processing thread")
        return True
    
    def stop_processing(self) -> bool:
        """
        Stop the task processing thread
        
        Returns:
            bool: True if stopped successfully
        """
        if not self.processing:
            return False
            
        self._stop_event.set()
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
            
        self.processing = False
        logger.info("Stopped task processing thread")
        return True
    
    def _process_task_queue(self) -> None:
        """Process the next task in the queue"""
        if not self.task_queue:
            return
            
        # Check if we have any active nodes
        active_nodes = self._get_active_nodes()
        if not active_nodes:
            return
            
        # Get highest priority task
        _, _, task_id = self.task_queue[0]
        task = self.tasks.get(task_id)
        
        if not task or task.status != "pending":
            # Task not found or not pending, remove from queue
            heapq.heappop(self.task_queue)
            return
            
        # Try to distribute the task
        distribute_func = self.strategies.get(self.current_strategy, self._distribute_capability_match)
        
        success, node_id = distribute_func(task, active_nodes)
        
        if success:
            # Task assigned successfully
            heapq.heappop(self.task_queue)  # Remove from queue
            task.update_status("assigned", node_id)
            
            # Update node load
            self.node_loads[node_id] = self.node_loads.get(node_id, 0) + 1
            
            logger.debug(f"Task {task_id} assigned to node {node_id}")
            
            # If we have a hive network, notify the node
            if self.hive:
                self._notify_node(node_id, task)
        else:
            # Couldn't assign, requeue with reduced priority
            heapq.heappop(self.task_queue)  # Remove from queue
            
            # Check if the task should be failed due to no matching nodes
            if not self._can_execute_on_any_node(task, active_nodes):
                logger.warning(f"No nodes can execute task {task_id}, marking as failed")
                task.update_status("failed")
                task.error = "No compatible nodes available"
                self.stats["tasks_failed"] += 1
                return
                
            # Requeue with slightly lower priority
            score = self._calculate_task_priority_score(task) * 0.95
            heapq.heappush(self.task_queue, (-score, task.created_at, task.task_id))
    
    def _get_active_nodes(self) -> List[str]:
        """
        Get IDs of active nodes
        
        Returns:
            List[str]: Node IDs
        """
        active_time = time.time() - 60.0  # Considered active if seen in last minute
        return [
            node_id for node_id, info in self.nodes.items()
            if info["last_seen"] >= active_time
        ]
    
    def _can_execute_on_any_node(self, task: Task, node_ids: List[str]) -> bool:
        """
        Check if any node can execute a task
        
        Args:
            task: The task
            node_ids: List of node IDs to check
            
        Returns:
            bool: True if at least one node can execute the task
        """
        for node_id in node_ids:
            node_info = self.nodes.get(node_id)
            if node_info and task.can_execute_on(node_info["capabilities"]):
                return True
        return False
    
    def _distribute_round_robin(self, task: Task, node_ids: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Distribute task using round-robin strategy
        
        Args:
            task: The task to distribute
            node_ids: List of active node IDs
            
        Returns:
            Tuple[bool, Optional[str]]: (success, node_id)
        """
        if not node_ids:
            return False, None
            
        # Filter nodes that can execute this task
        capable_nodes = [
            node_id for node_id in node_ids
            if task.can_execute_on(self.nodes[node_id]["capabilities"])
        ]
        
        if not capable_nodes:
            return False, None
            
        # Use round-robin to select node
        idx = self._next_node_index % len(capable_nodes)
        self._next_node_index = (self._next_node_index + 1) % len(capable_nodes)
        
        return True, capable_nodes[idx]
    
    def _distribute_load_balanced(self, task: Task, node_ids: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Distribute task based on current load
        
        Args:
            task: The task to distribute
            node_ids: List of active node IDs
            
        Returns:
            Tuple[bool, Optional[str]]: (success, node_id)
        """
        if not node_ids:
            return False, None
            
        # Filter nodes that can execute this task
        capable_nodes = [
            node_id for node_id in node_ids
            if task.can_execute_on(self.nodes[node_id]["capabilities"])
        ]
        
        if not capable_nodes:
            return False, None
            
        # Find node with lowest load
        best_node = min(capable_nodes, key=lambda n: self.node_loads.get(n, 0))
        
        return True, best_node
    
    def _distribute_capability_match(self, task: Task, node_ids: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Distribute task based on capability matching
        
        Args:
            task: The task to distribute
            node_ids: List of active node IDs
            
        Returns:
            Tuple[bool, Optional[str]]: (success, node_id)
        """
        if not node_ids:
            return False, None
            
        # Calculate capability score for each node
        node_scores = []
        
        for node_id in node_ids:
            node_info = self.nodes[node_id]
            
            # Skip nodes that can't execute this task
            if not task.can_execute_on(node_info["capabilities"]):
                continue
                
            # Base score - lower load is better
            load = self.node_loads.get(node_id, 0)
            score = 100.0 / (1.0 + load)
            
            # Capability surplus score - more capability than required is better
            capabilities = node_info["capabilities"]
            if "model_size" in capabilities and "hidden_dim" in capabilities["model_size"]:
                required_dim = task.required_capabilities.get("model_size.hidden_dim", 0)
                node_dim = capabilities["model_size"]["hidden_dim"]
                
                if required_dim > 0 and node_dim > required_dim:
                    # Node has more capacity than required
                    surplus_factor = min(2.0, node_dim / required_dim)
                    score *= (0.8 + 0.2 * surplus_factor)
            
            # Add to candidates
            node_scores.append((node_id, score))
        
        if not node_scores:
            return False, None
            
        # Select best scoring node
        best_node, _ = max(node_scores, key=lambda x: x[1])
        
        return True, best_node
    
    def _distribute_domain_specialized(self, task: Task, node_ids: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Distribute task based on domain specialization
        
        Args:
            task: The task to distribute
            node_ids: List of active node IDs
            
        Returns:
            Tuple[bool, Optional[str]]: (success, node_id)
        """
        if not node_ids or not task.preferred_domains:
            # Fall back to capability matching if no preferred domains
            return self._distribute_capability_match(task, node_ids)
            
        # Calculate domain match score for each node
        node_scores = []
        
        for node_id in node_ids:
            node_info = self.nodes[node_id]
            
            # Skip nodes that can't execute this task
            if not task.can_execute_on(node_info["capabilities"]):
                continue
                
            # Base score - lower load is better
            load = self.node_loads.get(node_id, 0)
            score = 100.0 / (1.0 + load)
            
            # Domain match score
            node_domains = node_info.get("domains", [])
            domain_matches = sum(1.0 for d in task.preferred_domains if d in node_domains)
            
            if domain_matches > 0:
                # Boost score based on domain matches
                domain_factor = 1.0 + (domain_matches / len(task.preferred_domains))
                score *= domain_factor
            
            # Add to candidates
            node_scores.append((node_id, score))
        
        if not node_scores:
            return False, None
            
        # Select best scoring node
        best_node, _ = max(node_scores, key=lambda x: x[1])
        
        return True, best_node
    
    def _distribute_performance_optimized(self, task: Task, node_ids: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Distribute task based on past performance
        
        Args:
            task: The task to distribute
            node_ids: List of active node IDs
            
        Returns:
            Tuple[bool, Optional[str]]: (success, node_id)
        """
        if not node_ids:
            return False, None
            
        # Calculate performance score for each node
        node_scores = []
        
        for node_id in node_ids:
            node_info = self.nodes[node_id]
            
            # Skip nodes that can't execute this task
            if not task.can_execute_on(node_info["capabilities"]):
                continue
                
            # Base score - lower load is better
            load = self.node_loads.get(node_id, 0)
            score = 100.0 / (1.0 + load)
            
            # Performance factor
            perf = self.node_performance.get(node_id, {})
            success_rate = perf.get("success_rate", 1.0)
            exec_time = perf.get("avg_execution_time", 1.0)
            
            # Adjust score based on performance
            performance_factor = success_rate / max(0.1, exec_time / 10.0)  # Higher success, lower time is better
            score *= performance_factor
            
            # Domain match bonus
            node_domains = node_info.get("domains", [])
            if task.preferred_domains and any(d in node_domains for d in task.preferred_domains):
                score *= 1.2  # 20% bonus for domain match
            
            # Add to candidates
            node_scores.append((node_id, score))
        
        if not node_scores:
            return False, None
            
        # Select best scoring node
        best_node, _ = max(node_scores, key=lambda x: x[1])
        
        return True, best_node
    
    def _notify_node(self, node_id: str, task: Task) -> None:
        """
        Notify a node about a task assignment
        
        Args:
            node_id: ID of the node
            task: The task
        """
        # This would send a notification to the node via the hive network
        # Actual implementation depends on the hive network API
        if not self.hive:
            return
            
        try:
            # Send task notification
            self.hive.send_notification(
                node_id,
                {
                    "type": "task_assignment",
                    "task_id": task.task_id,
                    "task": task.to_dict()
                }
            )
        except Exception as e:
            logger.error(f"Error notifying node {node_id} about task {task.task_id}: {e}")
    
    def update_task_status(self, task_id: str, status: str, 
                          result: Any = None, 
                          error: Optional[str] = None,
                          node_id: Optional[str] = None) -> bool:
        """
        Update the status of a task
        
        Args:
            task_id: ID of the task
            status: New status
            result: Optional task result
            error: Optional error message
            node_id: Optional ID of the node updating status
            
        Returns:
            bool: True if update was successful
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.warning(f"Attempted to update unknown task: {task_id}")
            return False
            
        old_status = task.status
        task.update_status(status, node_id)
        
        # Set result if provided
        if result is not None or error is not None:
            task.set_result(result, error)
        
        # Update node load if task is completed or failed
        if task.assigned_to and status in ["completed", "failed"]:
            self.node_loads[task.assigned_to] = max(0, self.node_loads.get(task.assigned_to, 0) - 1)
        
        # Update statistics
        if status == "completed" and old_status != "completed":
            self.stats["tasks_completed"] += 1
            if task.execution_time:
                self.stats["total_execution_time"] += task.execution_time
                
            # Update node performance
            if task.assigned_to:
                node_id = task.assigned_to
                
                if node_id not in self.node_performance:
                    self.node_performance[node_id] = {
                        "avg_execution_time": 0.0,
                        "success_rate": 1.0,
                        "last_results": []
                    }
                
                perf = self.node_performance[node_id]
                
                # Update execution time
                if task.execution_time:
                    if perf["avg_execution_time"] == 0:
                        perf["avg_execution_time"] = task.execution_time
                    else:
                        perf["avg_execution_time"] = 0.9 * perf["avg_execution_time"] + 0.1 * task.execution_time
                
                # Update success rate
                perf["last_results"].append(True)
                if len(perf["last_results"]) > 10:
                    perf["last_results"].pop(0)
                
                perf["success_rate"] = sum(1 for r in perf["last_results"] if r) / max(1, len(perf["last_results"]))
                
                # Update node stats
                self.nodes[node_id]["total_tasks"] += 1
                self.nodes[node_id]["successful_tasks"] += 1
        
        elif status == "failed" and old_status != "failed":
            self.stats["tasks_failed"] += 1
            
            # Update node performance
            if task.assigned_to:
                node_id = task.assigned_to
                
                if node_id not in self.node_performance:
                    self.node_performance[node_id] = {
                        "avg_execution_time": 0.0,
                        "success_rate": 1.0,
                        "last_results": []
                    }
                
                perf = self.node_performance[node_id]
                
                # Update success rate
                perf["last_results"].append(False)
                if len(perf["last_results"]) > 10:
                    perf["last_results"].pop(0)
                
                perf["success_rate"] = sum(1 for r in perf["last_results"] if r) / max(1, len(perf["last_results"]))
                
                # Update node stats
                self.nodes[node_id]["total_tasks"] += 1
                self.nodes[node_id]["failed_tasks"] += 1
                
            # Check if task should be retried
            if task.attempts < 3:  # Max 3 attempts
                logger.info(f"Requeuing failed task {task_id} for retry (attempt {task.attempts + 1})")
                
                # Reset task status
                task.status = "pending"
                task.assigned_to = None
                task.assigned_at = None
                task.started_at = None
                task.completed_at = None
                task.result = None
                
                # Add back to queue with higher priority
                score = self._calculate_task_priority_score(task)
                heapq.heappush(self.task_queue, (-score, task.created_at, task.task_id))
        
        return True
    
    def get_status(self) -> Dict:
        """
        Get distributor status
        
        Returns:
            Dict: Status information
        """
        # Calculate queue statistics
        pending_count = sum(1 for t in self.tasks.values() if t.status == "pending")
        assigned_count = sum(1 for t in self.tasks.values() if t.status == "assigned")
        in_progress_count = sum(1 for t in self.tasks.values() if t.status == "in_progress")
        
        # Calculate node statistics
        active_nodes = self._get_active_nodes()
        load_by_node = {node_id: self.node_loads.get(node_id, 0) for node_id in active_nodes}
        total_capacity = sum(1 for _ in active_nodes) * 5  # Assume each node can handle 5 tasks
        
        return {
            "queue_length": len(self.task_queue),
            "pending_tasks": pending_count,
            "assigned_tasks": assigned_count,
            "in_progress_tasks": in_progress_count,
            "completed_tasks": self.stats["tasks_completed"],
            "failed_tasks": self.stats["tasks_failed"],
            "active_nodes": len(active_nodes),
            "total_capacity": total_capacity,
            "capacity_used": sum(load_by_node.values()) / max(1, total_capacity),
            "strategy": self.current_strategy,
            "processing_active": self.processing
        }
    
    def get_detailed_stats(self) -> Dict:
        """
        Get detailed statistics
        
        Returns:
            Dict: Detailed statistics
        """
        active_nodes = self._get_active_nodes()
        
        return {
            "tasks": {
                "created": self.stats["tasks_created"],
                "completed": self.stats["tasks_completed"],
                "failed": self.stats["tasks_failed"],
                "avg_execution_time": (self.stats["total_execution_time"] / max(1, self.stats["tasks_completed"])),
                "pending": sum(1 for t in self.tasks.values() if t.status == "pending"),
                "assigned": sum(1 for t in self.tasks.values() if t.status == "assigned"),
                "in_progress": sum(1 for t in self.tasks.values() if t.status == "in_progress")
            },
            "nodes": {
                "active": len(active_nodes),
                "total": len(self.nodes),
                "loads": {node_id: self.node_loads.get(node_id, 0) for node_id in active_nodes},
                "performance": {
                    node_id: {
                        "success_rate": self.node_performance.get(node_id, {}).get("success_rate", 1.0),
                        "avg_execution_time": self.node_performance.get(node_id, {}).get("avg_execution_time", 0.0)
                    }
                    for node_id in active_nodes
                }
            },
            "distribution": {
                "strategy": self.current_strategy,
                "processing_active": self.processing
            }
        }
