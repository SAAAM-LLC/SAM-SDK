"""
Core Hive Network Components for SAM Hive Mind SDK
"""
import asyncio
import json
import logging
import threading
import time
import uuid
import torch
import zlib
import base64
from typing import Dict, List, Optional, Union, Callable, Any

from aiohttp import web
import requests

from .config import HiveConfig
from .security import HiveSecurity, ConnectionPolicy
from .monitor import NetworkStats

logger = logging.getLogger("sam_hive_sdk")

class HiveNode:
    """
    Base class representing a single SAM instance in a hive network.
    
    A HiveNode can operate independently or as part of a larger network,
    synchronizing knowledge, experiences, and thought states with other nodes.
    """
    
    def __init__(self, 
                 sam_instance, 
                 node_id: str = None, 
                 name: str = None,
                 config: Optional[HiveConfig] = None):
        """
        Initialize a HiveNode.
        
        Args:
            sam_instance: The SAM instance this node represents
            node_id: Unique identifier for this node (generated if None)
            name: Human-readable name for this node
            config: Configuration for hive behavior
        """
        self.sam = sam_instance
        self.node_id = node_id or str(uuid.uuid4())
        self.name = name or f"SAM-Node-{self.node_id[:8]}"
        self.config = config or HiveConfig()
        
        # Connection state
        self.network = None
        self.is_connected = False
        self.last_sync_time = 0
        self.sync_history = []
        
        # Performance tracking
        self.stats = NetworkStats(self.node_id)
        
        # Security
        self.security = HiveSecurity(self.config.security_policy)
        
        logger.info(f"Initialized HiveNode: {self.name} ({self.node_id})")
    
    def connect(self, network_url: str, api_key: Optional[str] = None) -> bool:
        """
        Connect this node to a hive network.
        
        Args:
            network_url: URL of the hive network server
            api_key: Optional API key for authentication
            
        Returns:
            bool: True if connection successful
        """
        if self.is_connected:
            logger.warning(f"Node {self.name} is already connected to a network")
            return False
            
        try:
            # Prepare registration payload
            payload = {
                'node_id': self.node_id,
                'name': self.name,
                'capabilities': self._get_capabilities(),
                'timestamp': time.time()
            }
            
            # Add security if needed
            if api_key:
                payload['api_key'] = api_key
                
            # Send registration request
            response = requests.post(
                f"{network_url}/register",
                json=payload,
                timeout=self.config.connection_timeout
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to connect to network: {response.text}")
                return False
                
            # Process response
            data = response.json()
            self.network = network_url
            self.is_connected = True
            
            logger.info(f"Node {self.name} connected to network at {network_url}")
            
            # Start background sync if configured
            if self.config.auto_sync:
                self._start_background_sync()
                
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to network: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect this node from its current network.
        
        Returns:
            bool: True if disconnect successful
        """
        if not self.is_connected:
            return True
            
        # Stop background sync
        self._stop_background_sync()
        
        try:
            # Notify network of disconnection
            if self.network:
                requests.post(
                    f"{self.network}/unregister",
                    json={'node_id': self.node_id},
                    timeout=self.config.connection_timeout
                )
                
            self.network = None
            self.is_connected = False
            logger.info(f"Node {self.name} disconnected from network")
            return True
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
            return False
    
    def sync(self, force: bool = False) -> Dict:
        """
        Synchronize this node with the network.
        
        Args:
            force: Force sync even if interval hasn't elapsed
            
        Returns:
            Dict: Sync results
        """
        if not self.is_connected:
            logger.warning("Cannot sync: Not connected to a network")
            return {'success': False, 'error': 'Not connected'}
            
        # Check sync interval unless forced
        if not force and time.time() - self.last_sync_time < self.config.sync_interval:
            return {'success': True, 'message': 'Skipped (interval not elapsed)'}
            
        try:
            # Prepare sync payload
            concepts = self.sam.concept_bank.get_concepts_for_sync(
                limit=self.config.sync_concept_limit
            )
            
            experiences = self.sam.experience_manager.get_experiences_for_sync(
                limit=self.config.sync_experience_limit
            )
            
            thought = self.sam.thought_state.get_shared_thought()
            
            payload = {
                'node_id': self.node_id,
                'timestamp': time.time(),
                'concepts': concepts,
                'experiences': experiences,
                'thought': thought.tolist() if thought is not None else None
            }
            
            # Compress if enabled
            if self.config.enable_compression:
                payload = self._compress_payload(payload)
                headers = {'Content-Type': 'application/octet-stream', 'Content-Encoding': 'zlib'}
            else:
                headers = {'Content-Type': 'application/json'}
            
            # Send sync request
            response = requests.post(
                f"{self.network}/sync",
                headers=headers,
                data=payload if isinstance(payload, bytes) else json.dumps(payload),
                timeout=self.config.sync_timeout
            )
            
            if response.status_code != 200:
                logger.error(f"Sync failed: {response.text}")
                return {'success': False, 'error': response.text}
                
            # Process response
            data = response.json()
            
            # Process received concepts
            if 'concepts' in data:
                concept_ids = [c.get('local_id') for c in concepts]
                self.sam.concept_bank.mark_concepts_synced(concept_ids)
                
                integrated, updated = self.sam.concept_bank.integrate_hive_concepts(
                    data['concepts'], self.network
                )
                
                logger.debug(f"Sync: Integrated {integrated} concepts, updated {updated}")
            
            # Process received experiences
            if 'experiences' in data:
                exp_ids = [e.get('experience_id') for e in experiences]
                self.sam.experience_manager.mark_experiences_synced(exp_ids)
                
                integrated_exp = self.sam.experience_manager.integrate_hive_experiences(
                    data['experiences']
                )
                
                logger.debug(f"Sync: Integrated {integrated_exp} experiences")
            
            # Process received thought
            if 'thought' in data and data['thought'] is not None:
                thought_tensor = torch.tensor(
                    data['thought'],
                    device=self.sam.config.device,
                    dtype=torch.float
                )
                
                self.sam.thought_state.set_shared_thought(
                    thought_tensor, 
                    blend_factor=self.config.thought_blend_factor
                )
            
            # Update sync time and history
            self.last_sync_time = time.time()
            self.sync_history.append({
                'timestamp': time.time(),
                'sent_concepts': len(concepts),
                'received_concepts': len(data.get('concepts', [])),
                'sent_experiences': len(experiences),
                'received_experiences': len(data.get('experiences', [])),
                'network_size': data.get('network_size', 1)
            })
            
            # Update stats
            self.stats.record_sync(
                sent_concepts=len(concepts),
                received_concepts=len(data.get('concepts', [])),
                sent_experiences=len(experiences),
                received_experiences=len(data.get('experiences', [])),
                success=True
            )
            
            return {
                'success': True,
                'sent_concepts': len(concepts),
                'received_concepts': len(data.get('concepts', [])),
                'sent_experiences': len(experiences),
                'received_experiences': len(data.get('experiences', [])),
                'network_size': data.get('network_size', 1)
            }
            
        except Exception as e:
            logger.error(f"Error during sync: {e}")
            self.stats.record_sync(success=False)
            return {'success': False, 'error': str(e)}
    
    def execute_task(self, task: Dict) -> Dict:
        """
        Execute a task assigned from the network.
        
        Args:
            task: Task description and parameters
            
        Returns:
            Dict: Task results
        """
        logger.info(f"Node {self.name} executing task: {task.get('task_id')}")
        
        try:
            # Extract task parameters
            task_id = task.get('task_id')
            task_type = task.get('type')
            parameters = task.get('parameters', {})
            
            # Execute task based on type
            result = None
            
            if task_type == 'generate':
                # Text generation task
                prompt = parameters.get('prompt', '')
                max_length = parameters.get('max_length', 256)
                temperature = parameters.get('temperature', 0.7)
                
                result = self.sam.generate(
                    input_text=prompt,
                    max_length=max_length,
                    temperature=temperature
                )
                
            elif task_type == 'learn':
                # Learning task
                content = parameters.get('content', '')
                is_private = parameters.get('private', False)
                
                # Process the content for learning
                concept_ids, segments = self.sam.process_text(
                    content, private_context=is_private
                )
                
                # Record the experience
                self.sam.experience_manager.record_experience(
                    experience_type='learning',
                    content=content,
                    metadata={'task_id': task_id},
                    private=is_private
                )
                
                result = {
                    'concepts_processed': len(concept_ids) if isinstance(concept_ids, list) else 1,
                    'segments_created': len(segments) if isinstance(segments, list) else 1
                }
                
            elif task_type == 'analyze':
                # Analysis task
                content = parameters.get('content', '')
                
                # Process the content
                concept_ids, _ = self.sam.process_text(content)
                
                # Generate analysis
                analysis = self.sam.generate(
                    input_text=f"Analyze the following content:\n\n{content}\n\nAnalysis:",
                    max_length=512,
                    temperature=0.5
                )
                
                result = {'analysis': analysis}
                
            elif task_type == 'dream':
                # Dreaming task
                duration = parameters.get('duration_minutes', 0.5)
                
                # Perform dreaming cycle
                dream_results = self.sam.dreaming.dream_cycle(duration_minutes=duration)
                
                result = dream_results
                
            else:
                logger.warning(f"Unknown task type: {task_type}")
                return {
                    'task_id': task_id,
                    'success': False,
                    'error': f"Unknown task type: {task_type}"
                }
                
            # Return results
            return {
                'task_id': task_id,
                'success': True,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            return {
                'task_id': task.get('task_id'),
                'success': False,
                'error': str(e)
            }
    
    def get_status(self) -> Dict:
        """
        Get detailed status of this node.
        
        Returns:
            Dict: Node status information
        """
        return {
            'node_id': self.node_id,
            'name': self.name,
            'connected': self.is_connected,
            'network': self.network,
            'last_sync': self.last_sync_time,
            'sync_count': len(self.sync_history),
            'stats': self.stats.get_summary(),
            'capabilities': self._get_capabilities(),
            'config': self.config.to_dict()
        }
    
    def _get_capabilities(self) -> Dict:
        """Get the capabilities of this node."""
        return {
            'model_size': {
                'hidden_dim': self.sam.config.initial_hidden_dim,
                'num_layers': len(self.sam.layers) if hasattr(self.sam, 'layers') else 0,
                'total_concepts': self.sam.concept_bank.next_concept_id,
                'parameter_count': sum(p.numel() for p in self.sam.parameters() if p.requires_grad)
            },
            'multimodal': self.sam.config.multimodal_enabled if hasattr(self.sam.config, 'multimodal_enabled') else False,
            'device': str(self.sam.config.device),
            'thought_dim': self.sam.config.thought_dim,
            'version': getattr(self.sam, 'version', '0.1.0')
        }
    
    # Background sync functionality
    def _start_background_sync(self):
        """Start background synchronization thread."""
        self._stop_sync = threading.Event()
        
        def sync_loop():
            while not self._stop_sync.is_set():
                if time.time() - self.last_sync_time >= self.config.sync_interval:
                    try:
                        self.sync()
                    except Exception as e:
                        logger.error(f"Error in background sync: {e}")
                
                # Sleep with checking for stop event
                for _ in range(min(10, max(1, int(self.config.sync_interval)))):
                    if self._stop_sync.is_set():
                        break
                    time.sleep(0.1)
        
        self._sync_thread = threading.Thread(target=sync_loop, daemon=True)
        self._sync_thread.start()
        logger.info(f"Started background sync for node {self.name}")
    
    def _stop_background_sync(self):
        """Stop background synchronization thread."""
        if hasattr(self, '_stop_sync') and not self._stop_sync.is_set():
            self._stop_sync.set()
            if hasattr(self, '_sync_thread'):
                self._sync_thread.join(timeout=5)
            logger.info(f"Stopped background sync for node {self.name}")
    
    def _compress_payload(self, payload):
        """Compress data payload."""
        json_str = json.dumps(payload).encode('utf-8')
        return zlib.compress(json_str, level=self.config.compression_level)
    
    def _decompress_payload(self, compressed_data):
        """Decompress data payload."""
        json_str = zlib.decompress(compressed_data).decode('utf-8')
        return json.loads(json_str)


class MasterNode(HiveNode):
    """
    A specialized HiveNode that can coordinate and direct other nodes.
    
    MasterNodes have the ability to:
    - Create and manage subordinate nodes
    - Assign specialized tasks to nodes
    - Coordinate knowledge sharing between nodes
    - Manage higher-level cognitive strategies
    """
    
    def __init__(self, 
                 sam_instance, 
                 node_id: str = None, 
                 name: str = None,
                 config: Optional[HiveConfig] = None):
        """Initialize a MasterNode."""
        super().__init__(sam_instance, node_id, name or "Master", config)
        self.subordinates = {}  # node_id -> connection info
        self.task_assignments = {}  # node_id -> assigned tasks
        self.specializations = {}  # domain -> [node_ids]
        
        logger.info(f"Initialized MasterNode: {self.name} ({self.node_id})")
    
    def create_subordinate(self, 
                          config_overrides: Dict = None, 
                          specialization: str = None,
                          name: str = None) -> str:
        """
        Create a new subordinate node with this node as its master.
        
        Args:
            config_overrides: Configuration overrides for the new SAM instance
            specialization: Specialized domain for this subordinate
            name: Custom name for the subordinate
            
        Returns:
            str: Node ID of the new subordinate
        """
        try:
            # Create a new SAM instance with appropriate configuration
            from sam.sam import SAM, SAMConfig  # Import here to avoid circular imports
            
            base_config = SAMConfig()
            
            # Apply master's config as base
            for key, value in self.sam.config.__dict__.items():
                if key in base_config.__dict__:
                    setattr(base_config, key, value)
            
            # Apply overrides
            if config_overrides:
                for key, value in config_overrides.items():
                    if key in base_config.__dict__:
                        setattr(base_config, key, value)
            
            # Create specialized SAM instance
            subordinate_sam = SAM(base_config)
            
            # Generate node ID and name
            sub_id = str(uuid.uuid4())
            sub_name = name or f"Sub-{specialization or 'General'}-{sub_id[:6]}"
            
            # Create subordinate node
            sub_node = HiveNode(subordinate_sam, sub_id, sub_name, self.config)
            
            # Register subordinate
            self.subordinates[sub_id] = {
                'node': sub_node,
                'created_at': time.time(),
                'specialization': specialization,
                'last_contact': time.time()
            }
            
            # Register specialization
            if specialization:
                if specialization not in self.specializations:
                    self.specializations[specialization] = []
                self.specializations[specialization].append(sub_id)
            
            logger.info(f"Created subordinate node: {sub_name} ({sub_id}) with specialization: {specialization}")
            
            # Initialize the subordinate with master's knowledge
            self._initialize_subordinate(sub_node)
            
            return sub_id
            
        except Exception as e:
            logger.error(f"Error creating subordinate: {e}")
            raise
    
    def assign_task(self, task: Dict, node_id: str = None) -> Dict:
        """
        Assign a task to a specific node or an appropriate subordinate.
        
        Args:
            task: Task description and parameters
            node_id: Specific node to assign task to (or None for auto-selection)
            
        Returns:
            Dict: Assignment results
        """
        if node_id and node_id not in self.subordinates:
            raise ValueError(f"Node {node_id} is not a registered subordinate")
            
        # Auto-select node if not specified
        if not node_id:
            node_id = self._select_best_node_for_task(task)
            
        # Get the subordinate
        subordinate = self.subordinates[node_id]['node']
        
        # Update last contact
        self.subordinates[node_id]['last_contact'] = time.time()
        
        # Track assignment
        task_id = task.get('task_id', str(uuid.uuid4()))
        if 'task_id' not in task:
            task['task_id'] = task_id
            
        self.task_assignments[task_id] = {
            'node_id': node_id,
            'task': task,
            'assigned_at': time.time(),
            'status': 'assigned'
        }
        
        # Execute the task
        result = subordinate.execute_task(task)
        
        # Update task status
        self.task_assignments[task_id]['status'] = 'completed' if result.get('success') else 'failed'
        self.task_assignments[task_id]['completed_at'] = time.time()
        self.task_assignments[task_id]['result'] = result
        
        return {
            'task_id': task_id,
            'node_id': node_id,
            'success': result.get('success', False),
            'result': result
        }
    
    def get_task_status(self, task_id: str) -> Dict:
        """
        Get the status of a specific task.
        
        Args:
            task_id: ID of the task to check
            
        Returns:
            Dict: Task status information
        """
        if task_id not in self.task_assignments:
            return {'error': 'Task not found'}
            
        return self.task_assignments[task_id]
    
    def get_subordinate_status(self, node_id: str = None) -> Union[Dict, List[Dict]]:
        """
        Get status of one or all subordinate nodes.
        
        Args:
            node_id: Specific node ID or None for all
            
        Returns:
            Dict or List[Dict]: Status information
        """
        if node_id:
            if node_id not in self.subordinates:
                return {'error': 'Node not found'}
                
            sub = self.subordinates[node_id]
            return {
                'node_id': node_id,
                'name': sub['node'].name,
                'specialization': sub['specialization'],
                'created_at': sub['created_at'],
                'last_contact': sub['last_contact'],
                'status': sub['node'].get_status()
            }
            
        # Return all subordinates
        return [
            {
                'node_id': node_id,
                'name': sub['node'].name,
                'specialization': sub['specialization'],
                'created_at': sub['created_at'],
                'last_contact': sub['last_contact'],
                'age_hours': (time.time() - sub['created_at']) / 3600
            }
            for node_id, sub in self.subordinates.items()
        ]
    
    def share_knowledge(self, target_node_id: str, domains: List[str] = None) -> Dict:
        """
        Explicitly share knowledge with a specific subordinate.
        
        Args:
            target_node_id: Node to share knowledge with
            domains: Specific knowledge domains to share (or None for all)
            
        Returns:
            Dict: Results of knowledge sharing
        """
        if target_node_id not in self.subordinates:
            return {'error': 'Target node not found'}
            
        target = self.subordinates[target_node_id]['node']
        
        # Get concepts to share
        concepts = self.sam.concept_bank.get_concepts_for_sync(limit=1000)
        
        # Filter by domain if specified
        if domains:
            # Simple domain filtering based on concept metadata
            filtered_concepts = []
            for concept in concepts:
                concept_domains = concept.get('metadata', {}).get('domains', [])
                if any(d in concept_domains for d in domains):
                    filtered_concepts.append(concept)
            concepts = filtered_concepts
        
        # Share concepts
        integrated, updated = target.sam.concept_bank.integrate_hive_concepts(
            concepts, self.node_id
        )
        
        # Share thought state
        thought = self.sam.thought_state.get_shared_thought()
        if thought is not None:
            thought_tensor = torch.tensor(
                thought.tolist(),
                device=target.sam.config.device,
                dtype=torch.float
            )
            target.sam.thought_state.set_shared_thought(
                thought_tensor, 
                blend_factor=0.5  # Strong influence from master
            )
        
        # Update last contact
        self.subordinates[target_node_id]['last_contact'] = time.time()
        
        return {
            'success': True,
            'shared_concepts': len(concepts),
            'integrated_concepts': integrated,
            'updated_concepts': updated,
            'shared_thought': thought is not None
        }
    
    def _initialize_subordinate(self, subordinate):
        """Initialize a new subordinate with core knowledge."""
        # Share critical concepts
        concepts = self.sam.concept_bank.get_concepts_for_sync(limit=500)
        subordinate.sam.concept_bank.integrate_hive_concepts(concepts, self.node_id)
        
        # Share baseline experiences
        experiences = self.sam.experience_manager.get_experiences_for_sync(limit=20)
        subordinate.sam.experience_manager.integrate_hive_experiences(experiences)
        
        # Set consciousness parameters
        if hasattr(subordinate.sam, 'consciousness') and hasattr(self.sam, 'consciousness'):
            # Copy some identity parameters while maintaining uniqueness
            with torch.no_grad():
                master_personality = self.sam.consciousness.personality_vector
                if master_personality is not None:
                    # Create a related but distinct personality vector
                    new_vector = master_personality * 0.7 + torch.randn_like(master_personality) * 0.3
                    subordinate.sam.consciousness.personality_vector = F.normalize(new_vector, dim=0)
                    subordinate.sam.consciousness.personality_initialized = True
    
    def _select_best_node_for_task(self, task):
        """Select the best subordinate for a given task."""
        task_type = task.get('type')
        task_domain = task.get('parameters', {}).get('domain')
        
        # Check for specialized nodes
        if task_domain and task_domain in self.specializations:
            candidates = self.specializations[task_domain]
            if candidates:
                # Find the least busy node
                return min(
                    candidates,
                    key=lambda node_id: sum(1 for t in self.task_assignments.values() 
                                         if t['node_id'] == node_id and t['status'] == 'assigned')
                )
        
        # Otherwise select based on load balancing
        active_counts = {
            node_id: sum(1 for t in self.task_assignments.values() 
                       if t['node_id'] == node_id and t['status'] == 'assigned')
            for node_id in self.subordinates
        }
        
        # Find least busy node
        return min(active_counts.items(), key=lambda x: x[1])[0]


class HiveNetwork:
    """
    Server implementation for a SAM Hive Mind network.
    
    Manages registration, synchronization, and coordination of multiple
    SAM instances working together as a collective intelligence.
    """
    
    def __init__(self, 
                 host_sam=None, 
                 config: Optional[HiveConfig] = None,
                 host: str = "0.0.0.0",
                 port: int = 8765):
        """
        Initialize a HiveNetwork server.
        
        Args:
            host_sam: Optional SAM instance that hosts the network
            config: Network configuration
            host: Host address to bind the server
            port: Port to bind the server
        """
        self.host_sam = host_sam
        self.config = config or HiveConfig()
        self.host = host
        self.port = port
        
        # Network state
        self.network_id = str(uuid.uuid4())
        self.nodes = {}  # node_id -> node info
        self.tasks = {}  # task_id -> task info
        
        # Server components
        self.app = None
        self.runner = None
        self.site = None
        self.server_thread = None
        self._stop_event = threading.Event()
        
        # Security
        self.security = HiveSecurity(self.config.security_policy)
        
        # Initialize server
        self._setup_server()
        
        logger.info(f"Initialized HiveNetwork server with ID {self.network_id}")
    
    def _setup_server(self):
        """Set up the web server for the network."""
        self.app = web.Application()
        
        # Define routes
        self.app.router.add_post('/register', self._handle_register)
        self.app.router.add_post('/unregister', self._handle_unregister)
        self.app.router.add_post('/sync', self._handle_sync)
        self.app.router.add_post('/task', self._handle_task)
        self.app.router.add_get('/status', self._handle_status)
        
        # Middlewares for security and logging
        self.app.middlewares.append(self._security_middleware)
        self.app.middlewares.append(self._logging_middleware)
    
    async def _security_middleware(self, app, handler):
        """Middleware for request security."""
        async def middleware(request):
            # Implement security checks
            if self.config.require_api_key:
                api_key = request.headers.get('X-API-Key')
                if not api_key or not self.security.validate_api_key(api_key):
                    return web.json_response({'error': 'Unauthorized'}, status=401)
            
            return await handler(request)
        return middleware
    
    async def _logging_middleware(self, app, handler):
        """Middleware for request logging."""
        async def middleware(request):
            start_time = time.time()
            response = await handler(request)
            duration = time.time() - start_time
            logger.debug(f"{request.method} {request.path} - {response.status} ({duration:.2f}s)")
            return response
        return middleware
    
    async def _handle_register(self, request):
        """Handle node registration requests."""
        try:
            data = await request.json()
            
            node_id = data.get('node_id')
            name = data.get('name', node_id)
            
            if not node_id:
                return web.json_response({'error': 'Missing node_id'}, status=400)
            
            # Register the node
            self.nodes[node_id] = {
                'name': name,
                'registered_at': time.time(),
                'last_seen': time.time(),
                'capabilities': data.get('capabilities', {}),
                'ip_address': request.remote,
                'sync_count': 0
            }
            
            logger.info(f"Node registered: {name} ({node_id})")
            
            return web.json_response({
                'status': 'success',
                'network_id': self.network_id,
                'network_size': len(self.nodes),
                'timestamp': time.time()
            })
        except Exception as e:
            logger.error(f"Error in registration handler: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_unregister(self, request):
        """Handle node unregistration requests."""
        try:
            data = await request.json()
            node_id = data.get('node_id')
            
            if not node_id:
                return web.json_response({'error': 'Missing node_id'}, status=400)
            
            if node_id in self.nodes:
                node_name = self.nodes[node_id]['name']
                del self.nodes[node_id]
                logger.info(f"Node unregistered: {node_name} ({node_id})")
            
            return web.json_response({
                'status': 'success',
                'network_size': len(self.nodes)
            })
        except Exception as e:
            logger.error(f"Error in unregistration handler: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_sync(self, request):
        """Handle synchronization requests."""
        try:
            # Check content type for compression
            is_compressed = request.content_type == 'application/octet-stream'
            
            if is_compressed:
                # Handle compressed data
                data_bytes = await request.read()
                json_str = zlib.decompress(data_bytes).decode('utf-8')
                data = json.loads(json_str)
            else:
                # Handle regular JSON
                data = await request.json()
            
            node_id = data.get('node_id')
            
            if not node_id or node_id not in self.nodes:
                return web.json_response({'error': 'Unknown node'}, status=401)
            
            # Update node status
            self.nodes[node_id]['last_seen'] = time.time()
            self.nodes[node_id]['sync_count'] += 1
            
            # Process incoming data with host SAM if available
            integrated_concepts = 0
            integrated_experiences = 0
            
            if self.host_sam:
                # Process concepts
                if 'concepts' in data and data['concepts']:
                    integrated_concepts, _ = self.host_sam.concept_bank.integrate_hive_concepts(
                        data['concepts'], node_id
                    )
                
                # Process experiences
                if 'experiences' in data and data['experiences']:
                    integrated_experiences = self.host_sam.experience_manager.integrate_hive_experiences(
                        data['experiences']
                    )
                
                # Process thought
                if 'thought' in data and data['thought'] is not None:
                    thought_tensor = torch.tensor(
                        data['thought'],
                        device=self.host_sam.config.device,
                        dtype=torch.float
                    )
                    self.host_sam.thought_state.set_shared_thought(thought_tensor)
            
            # Prepare response data
            response_concepts = []
            response_experiences = []
            response_thought = None
            
            # Get concepts to share from host SAM or other nodes
            if self.host_sam:
                response_concepts = self.host_sam.concept_bank.get_concepts_for_sync(
                    limit=self.config.sync_concept_limit
                )
                response_experiences = self.host_sam.experience_manager.get_experiences_for_sync(
                    limit=self.config.sync_experience_limit
                )
                shared_thought = self.host_sam.thought_state.get_shared_thought()
                response_thought = shared_thought.tolist() if shared_thought is not None else None
            else:
                # Without host SAM, share from other nodes
                # Get recently active nodes (excluding requester)
                active_nodes = [
                    n_id for n_id, info in self.nodes.items()
                    if n_id != node_id and time.time() - info['last_seen'] < 3600
                ]
                
                if active_nodes and 'concepts' in data:
                    # Randomly select concepts from recent sync data
                    # (In a real implementation, these would be stored in a database)
                    response_concepts = data['concepts'][:self.config.sync_concept_limit // 2]
            
            # Prepare the response
            response = {
                'status': 'success',
                'timestamp': time.time(),
                'network_id': self.network_id,
                'network_size': len(self.nodes),
                'concepts': response_concepts,
                'experiences': response_experiences,
                'thought': response_thought,
                'stats': {
                    'integrated_concepts': integrated_concepts,
                    'integrated_experiences': integrated_experiences
                }
            }
            
            return web.json_response(response)
            
        except Exception as e:
            logger.error(f"Error in sync handler: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_task(self, request):
        """Handle task submission and results."""
        try:
            data = await request.json()
            
            node_id = data.get('node_id')
            task_id = data.get('task_id')
            action = data.get('action', 'submit')
            
            if not node_id or node_id not in self.nodes:
                return web.json_response({'error': 'Unknown node'}, status=401)
            
            # Update node last seen
            self.nodes[node_id]['last_seen'] = time.time()
            
            if action == 'submit':
                # Node is submitting a new task
                task = data.get('task')
                if not task:
                    return web.json_response({'error': 'Missing task data'}, status=400)
                
                # Generate task ID if not provided
                if not task_id:
                    task_id = str(uuid.uuid4())
                
                # Record task
                self.tasks[task_id] = {
                    'task': task,
                    'submitter_id': node_id,
                    'submitted_at': time.time(),
                    'status': 'pending',
                    'assigned_to': None
                }
                
                # Process task (in a real implementation, this might be queued)
                if self.host_sam:
                    # Execute the task with the host SAM
                    result = self._execute_task_with_host(task_id, task)
                    self.tasks[task_id].update({
                        'status': 'completed',
                        'completed_at': time.time(),
                        'result': result
                    })
                    
                    return web.json_response({
                        'status': 'success',
                        'task_id': task_id,
                        'result': result
                    })
                else:
                    # No host SAM to execute - task remains pending
                    return web.json_response({
                        'status': 'success',
                        'task_id': task_id,
                        'message': 'Task submitted and pending assignment'
                    })
            
            elif action == 'update':
                # Node is updating an existing task
                if not task_id or task_id not in self.tasks:
                    return web.json_response({'error': 'Unknown task_id'}, status=400)
                
                task_info = self.tasks[task_id]
                
                # Check if node is authorized to update this task
                if task_info['assigned_to'] != node_id and task_info['submitter_id'] != node_id:
                    return web.json_response({'error': 'Not authorized to update this task'}, status=403)
                
                # Update task status
                status = data.get('status')
                result = data.get('result')
                
                if status:
                    task_info['status'] = status
                
                if result:
                    task_info['result'] = result
                    task_info['completed_at'] = time.time()
                
                return web.json_response({
                    'status': 'success',
                    'task_id': task_id,
                    'task_status': task_info['status']
                })
            
            elif action == 'get':
                # Node is requesting task status or result
                if not task_id or task_id not in self.tasks:
                    return web.json_response({'error': 'Unknown task_id'}, status=400)
                
                return web.json_response({
                    'status': 'success',
                    'task_id': task_id,
                    'task_info': self.tasks[task_id]
                })
            
            else:
                return web.json_response({'error': f"Unknown action: {action}"}, status=400)
                
        except Exception as e:
            logger.error(f"Error in task handler: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_status(self, request):
        """Handle network status requests."""
        try:
            include_nodes = request.query.get('nodes', 'false').lower() == 'true'
            include_tasks = request.query.get('tasks', 'false').lower() == 'true'
            
            # Basic network status
            status = {
                'network_id': self.network_id,
                'node_count': len(self.nodes),
                'active_nodes': sum(1 for info in self.nodes.values() 
                                  if time.time() - info['last_seen'] < 3600),
                'task_count': len(self.tasks),
                'pending_tasks': sum(1 for task in self.tasks.values() 
                                   if task['status'] == 'pending'),
                'uptime_seconds': time.time() - self.start_time if hasattr(self, 'start_time') else 0
            }
            
            # Include detailed node information if requested
            if include_nodes:
                status['nodes'] = [
                    {
                        'node_id': node_id,
                        'name': info['name'],
                        'registered_at': info['registered_at'],
                        'last_seen': info['last_seen'],
                        'sync_count': info['sync_count'],
                        'capabilities': info.get('capabilities', {})
                    }
                    for node_id, info in self.nodes.items()
                ]
            
            # Include task information if requested
            if include_tasks:
                status['tasks'] = [
                    {
                        'task_id': task_id,
                        'submitter_id': info['submitter_id'],
                        'status': info['status'],
                        'submitted_at': info['submitted_at'],
                        'completed_at': info.get('completed_at')
                    }
                    for task_id, info in self.tasks.items()
                ]
            
            return web.json_response(status)
            
        except Exception as e:
            logger.error(f"Error in status handler: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    def _execute_task_with_host(self, task_id, task):
        """Execute a task with the host SAM instance."""
        try:
            # Extract task parameters
            task_type = task.get('type')
            parameters = task.get('parameters', {})
            
            # Execute based on task type
            if task_type == 'generate':
                prompt = parameters.get('prompt', '')
                max_length = parameters.get('max_length', 256)
                temperature = parameters.get('temperature', 0.7)
                
                result = self.host_sam.generate(
                    input_text=prompt,
                    max_length=max_length,
                    temperature=temperature
                )
                
                return {'generated_text': result}
                
            elif task_type == 'learn':
                content = parameters.get('content', '')
                is_private = parameters.get('private', False)
                
                concept_ids, segments = self.host_sam.process_text(
                    content, private_context=is_private
                )
                
                # Record experience
                self.host_sam.experience_manager.record_experience(
                    experience_type='learning',
                    content=content,
                    metadata={'task_id': task_id},
                    private=is_private
                )
                
                return {
                    'concepts_processed': len(concept_ids) if isinstance(concept_ids, list) else 1,
                    'segments_created': len(segments) if isinstance(segments, list) else 1
                }
                
            elif task_type == 'dream':
                duration = parameters.get('duration_minutes', 0.5)
                
                result = self.host_sam.dreaming.dream_cycle(duration_minutes=duration)
                
                return result
                
            else:
                return {'error': f"Unknown task type: {task_type}"}
                
        except Exception as e:
            logger.error(f"Error executing task {task_id}: {e}")
            return {'error': str(e)}
    
    def start(self):
        """Start the network server."""
        self.start_time = time.time()
        self._stop_event.clear()
        
        async def run_app():
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()
            
            logger.info(f"HiveNetwork server running at http://{self.host}:{self.port}")
            
            # Run cleanup task
            while not self._stop_event.is_set():
                self._cleanup_stale_nodes()
                await asyncio.sleep(60)  # Check every minute
            
            # Shutdown
            await self.site.stop()
            await self.runner.cleanup()
        
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_app())
            except Exception as e:
                logger.error(f"Server error: {e}")
            finally:
                loop.close()
        
        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        return True
    
    def stop(self):
        """Stop the network server."""
        if not hasattr(self, 'server_thread') or not self.server_thread.is_alive():
            return False
            
        self._stop_event.set()
        self.server_thread.join(timeout=30)
        
        logger.info("HiveNetwork server stopped")
        return True
    
    def get_status(self):
        """Get network status information."""
        active_nodes = sum(1 for info in self.nodes.values() 
                        if time.time() - info['last_seen'] < 3600)
                        
        return {
            'network_id': self.network_id,
            'node_count': len(self.nodes),
            'active_nodes': active_nodes,
            'task_count': len(self.tasks),
            'pending_tasks': sum(1 for task in self.tasks.values() 
                               if task['status'] == 'pending'),
            'uptime_seconds': time.time() - self.start_time if hasattr(self, 'start_time') else 0,
            'server_running': hasattr(self, 'server_thread') and self.server_thread.is_alive()
        }
    
    def _cleanup_stale_nodes(self):
        """Remove nodes that haven't been seen in a long time."""
        stale_threshold = time.time() - (3600 * 24)  # 24 hours
        stale_nodes = [
            node_id for node_id, info in self.nodes.items()
            if info['last_seen'] < stale_threshold
        ]
        
        for node_id in stale_nodes:
            logger.info(f"Removing stale node: {self.nodes[node_id]['name']} ({node_id})")
            del self.nodes[node_id]
