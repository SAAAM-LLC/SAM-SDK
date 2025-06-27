"""
Core SAM functionality for the SAM-SDK
"""
import logging
import os
import json
import time
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

from .concepts import ConceptBank
from .thought import ThoughtState
from .consciousness import ConsciousnessSystem
from .experience import ExperienceManager
from .dreaming import DreamingSystem
from .neural import DynamicSegmentation, NeuroplasticLayers
from .growth import NeuralGrowth, EvolutionManager
from .multimodal import MultimodalProcessor

logger = logging.getLogger("sam_sdk")

class SAMConfig:
    """Configuration for SAM instances and components"""
    
    def __init__(self, config_dict=None, **kwargs):
        """
        Initialize SAM configuration
        
        Args:
            config_dict: Dictionary of configuration values
            **kwargs: Additional configuration overrides
        """
        # Core dimensions
        self.initial_char_dim = 256
        self.initial_hidden_dim = 1536
        self.initial_num_layers = 8
        self.max_position_embeddings = 8192
        
        # Growth parameters
        self.max_hidden_dim = 4096
        self.max_num_layers = 16
        self.growth_factor = 1.2
        self.min_layer_usage_threshold = 0.3
        
        # Memory systems
        self.concept_memory_size = 10000
        self.concept_dim = 1536
        self.thought_dim = 2048
        self.max_thought_depth = 10
        self.pattern_memory_capacity = 50000
        
        # Learning parameters
        self.learning_rate = 3e-5
        self.warmup_steps = 100
        self.adaption_rate = 0.4
        
        # Segmentation parameters
        self.max_segment_length = 16
        self.min_segment_frequency = 5
        self.concept_frequency_threshold = 10
        
        # Dreaming parameters
        self.dream_batch_size = 5
        self.dream_max_length = 256
        self.dream_cycle_minutes = 0.5
        self.dream_on_idle_only = True
        
        # Consciousness parameters
        self.stability_threshold = 0.95
        self.novelty_weight = 0.4
        
        # Paths for persistence
        self.save_dir = "./data"
        self.experiences_path = "./data/experiences.json"
        self.concepts_path = "./data/concepts.json"
        self.growth_log_path = "./data/growth_log.json"
        
        # Runtime parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Communication Style
        self.communication_style = "flexible"
        
        # Hive Mind parameters
        self.hive_enabled = False
        self.hive_sync_interval_seconds = 300
        self.hive_server_url = ""
        self.hive_identity = ""
        
        # Multimodal capabilities
        self.multimodal_enabled = False
        self.image_dim = 768
        self.audio_dim = 512
        self.multimodal_fusion_strategy = "attention"
        
        # Apply config_dict overrides
        if config_dict:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        
        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
        # Validate configuration
        self.validate()
    
    def validate(self):
        """Validate configuration parameters"""
        # Check dimension relationships
        if self.concept_dim > self.initial_hidden_dim:
            logger.warning("concept_dim should not be larger than initial_hidden_dim")
            self.concept_dim = self.initial_hidden_dim
            
        # Check growth parameters
        if self.growth_factor <= 1.0:
            logger.warning("growth_factor must be greater than 1.0, setting to default 1.2")
            self.growth_factor = 1.2
            
        # Check device configuration
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
            self.dtype = torch.float32
            
        # Validate paths
        for path_attr in ['save_dir', 'experiences_path', 'concepts_path', 'growth_log_path']:
            path = getattr(self, path_attr)
            if path and isinstance(path, str):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                
        return self
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_') and not callable(value)}
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'SAMConfig':
        """Create configuration from dictionary"""
        return cls(config_dict)
    
    @classmethod
    def from_file(cls, file_path: str) -> 'SAMConfig':
        """Load configuration from JSON file"""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def save(self, file_path: str = None) -> str:
        """Save configuration to JSON file"""
        if file_path is None:
            file_path = os.path.join(self.save_dir, "config.json")
            
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Handle non-serializable types
        config_dict = self.to_dict()
        for key, value in config_dict.items():
            if isinstance(value, torch.dtype):
                config_dict[key] = str(value)
                
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
        return file_path


class SAM(nn.Module):
    """
    Synergistic Autonomous Machine - core implementation
    
    This class provides a high-level interface to the SAM architecture,
    integrating all components into a unified cognitive system.
    """
    
    def __init__(self, config: Union[SAMConfig, Dict] = None):
        """
        Initialize a SAM instance
        
        Args:
            config: Configuration for this SAM instance
        """
        super().__init__()
        
        # Process configuration
        if isinstance(config, dict):
            self.config = SAMConfig(config)
        elif config is None:
            self.config = SAMConfig()
        else:
            self.config = config
            
        # Initialize core components
        logger.info("Initializing SAM components...")
        
        # Memory systems
        self.concept_bank = ConceptBank(
            concept_dim=self.config.concept_dim,
            initial_size=self.config.concept_memory_size,
            device=self.config.device
        )
        
        # Neural processing
        self.segmentation = DynamicSegmentation(self.config, self.concept_bank)
        self.position_embeddings = nn.Embedding(
            self.config.max_position_embeddings, 
            self.config.initial_hidden_dim
        )
        
        # Core layers
        self.layers = NeuroplasticLayers(
            hidden_dim=self.config.initial_hidden_dim,
            num_layers=self.config.initial_num_layers,
            growth_factor=self.config.growth_factor
        )
        
        self.norm = nn.LayerNorm(self.config.initial_hidden_dim)
        
        # Tie weights between concept embeddings and output layer
        self.lm_head = nn.Linear(
            self.config.initial_hidden_dim, 
            self.config.concept_memory_size, 
            bias=False
        )
        self.lm_head.weight = self.concept_bank.concept_embeddings.weight
        
        # Cognitive systems
        self.thought_state = ThoughtState(
            concept_dim=self.config.initial_hidden_dim,
            thought_dim=self.config.thought_dim,
            max_thought_depth=self.config.max_thought_depth
        )
        
        self.experience_manager = ExperienceManager(self.config)
        self.dreaming = DreamingSystem(self)
        self.consciousness = ConsciousnessSystem(self)
        
        # Multimodal components (if enabled)
        if self.config.multimodal_enabled:
            self.multimodal_processor = MultimodalProcessor(self.config)
            
        # Growth management
        self.evolution_manager = EvolutionManager(self)
        self.growth_history = []
        
        # Tracking
        self.global_step = 0
        self.current_modality = "text"
        
        # Move to specified device
        self.to(self.config.device)
        
        logger.info(f"SAM initialized with {sum(p.numel() for p in self.parameters())} parameters")
    
    def forward(self, 
               input_chars=None, 
               input_concepts=None, 
               target_concepts=None, 
               **kwargs):
        """
        Forward pass through the model
        
        Args:
            input_chars: Raw character input
            input_concepts: Pre-processed concept IDs
            target_concepts: Target concept IDs for loss calculation
            
        Returns:
            Tuple of (loss, logits, hidden_states)
        """
        # Process raw characters if provided
        if input_chars is not None:
            processed_concepts = self.segmentation(input_chars)
            
            # Flatten and pad concepts
            max_len = max(len(c) for c in processed_concepts) if processed_concepts else 0
            if max_len == 0:
                # Handle empty input
                logits = torch.empty(
                    input_chars.shape[0], 0, self.config.concept_memory_size, 
                    device=self.config.device
                )
                return None, logits, None
                
            padded_concepts = [c + [0] * (max_len - len(c)) for c in processed_concepts]
            input_concepts = torch.tensor(
                padded_concepts, dtype=torch.long, device=self.config.device
            )
        elif not isinstance(input_concepts, torch.Tensor):
            input_concepts = torch.tensor(
                input_concepts, dtype=torch.long, device=self.config.device
            )
            
        # Check concept bank size for lm_head
        if self.lm_head.out_features != self.concept_bank.concept_embeddings.num_embeddings:
            self.lm_head = nn.Linear(
                self.config.initial_hidden_dim,
                self.concept_bank.concept_embeddings.num_embeddings,
                bias=False
            ).to(self.config.device)
            self.lm_head.weight = self.concept_bank.concept_embeddings.weight
            
        # Process concepts
        batch_size, seq_len = input_concepts.shape
        concept_embeds = self.concept_bank(input_concepts)
        
        # Add position embeddings
        pos_ids = torch.arange(seq_len, device=self.config.device).unsqueeze(0)
        pos_embeds = self.position_embeddings(pos_ids)
        hidden_states = concept_embeds + pos_embeds
        
        # Integrate thought state
        thought_context = self.thought_state.update(concept_embeds)
        projected_thought = self.thought_state.project_to_concept_space(thought_context)
        if projected_thought is not None:
            hidden_states = hidden_states + projected_thought.expand_as(hidden_states)
            
        # Process through layers
        hidden_states = self.layers(hidden_states)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Project to vocabulary space
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if targets provided
        loss = None
        if target_concepts is not None:
            loss_fct = nn.CrossEntropyLoss()
            
            # Align targets and logits
            if logits.shape[1] > 1 and target_concepts.shape[1] > 1:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = target_concepts[:, 1:].contiguous()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
        return loss, logits, hidden_states
    
    def process_text(self, text, private_context=False):
        """
        Process raw text into concept IDs and segments
        
        Args:
            text: Input text to process
            private_context: Whether this is private (not shared with hive)
            
        Returns:
            Tuple of (concept_ids, segments)
        """
        if private_context:
            self.segmentation.set_private_context("private")
            
        # Convert to character tensor
        char_tensor = torch.tensor(
            [ord(c) for c in text], 
            dtype=torch.long, 
            device=self.config.device
        )
        
        # Process through segmentation
        concept_ids, segments = self.segmentation(char_tensor, return_segments=True)
        
        if private_context:
            self.segmentation.clear_private_context()
            
        return concept_ids, segments
    
    def generate(self, 
                input_text, 
                max_length=128, 
                temperature=1.0, 
                top_k=50, 
                **kwargs):
        """
        Generate text from a prompt
        
        Args:
            input_text: Input text prompt
            max_length: Maximum generation length
            temperature: Temperature for sampling
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text
        """
        self.eval()
        
        # Convert input to char IDs
        char_ids = [ord(c) for c in input_text]
        input_tensor = torch.tensor(
            char_ids, 
            dtype=torch.long, 
            device=self.config.device
        ).unsqueeze(0)
        
        # Process through segmentation
        generated_concepts = self.segmentation(input_tensor)
        
        # Ensure proper tensor format
        if not isinstance(generated_concepts, torch.Tensor):
            generated_concepts = torch.tensor(
                generated_concepts, 
                dtype=torch.long, 
                device=self.config.device
            )
        if generated_concepts.dim() == 1:
            generated_concepts = generated_concepts.unsqueeze(0)
            
        # Generate tokens
        with torch.no_grad():
            for _ in range(max_length):
                _, logits, _ = self.forward(input_concepts=generated_concepts)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                    
                # Sample from distribution
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated_concepts = torch.cat((generated_concepts, next_token), dim=1)
                
        # Convert back to text
        output_ids = generated_concepts[0].tolist()
        output_text = self._concepts_to_text(output_ids)
        
        return output_text
    
    def _concepts_to_text(self, concept_ids):
        """Convert concept IDs back to text"""
        text_parts = []
        
        for concept_id in concept_ids:
            if isinstance(concept_id, list):
                # Handle nested lists from segmentation
                text_parts.append(self._concepts_to_text(concept_id))
                continue
                
            # Look up concept in metadata
            metadata = self.concept_bank.concept_metadata.get(concept_id)
            if metadata and metadata.get("source"):
                text_parts.append(metadata["source"])
                
        return "".join(text_parts)
    
    def save(self, path=None):
        """
        Save the model state
        
        Args:
            path: Directory path to save to
            
        Returns:
            Path where model was saved
        """
        if path is None:
            path = os.path.join(self.config.save_dir, f"sam_checkpoint_{self.global_step}")
            
        os.makedirs(path, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        
        # Save configuration
        self.config.save(os.path.join(path, "config.json"))
        
        logger.info(f"Model saved to {path}")
        return path
    
    @classmethod
    def load(cls, path):
        """
        Load model from saved state
        
        Args:
            path: Directory path to load from
            
        Returns:
            Loaded SAM instance
        """
        # Load configuration
        config = SAMConfig.from_file(os.path.join(path, "config.json"))
        
        # Create model instance
        model = cls(config)
        
        # Load weights
        model.load_state_dict(
            torch.load(
                os.path.join(path, "model.pt"),
                map_location=config.device
            )
        )
        
        logger.info(f"Model loaded from {path}")
        return model
    
    def start_services(self):
        """Start background services (dreaming, hive sync)"""
        if hasattr(self, 'dreaming'):
            self.dreaming.start_background_dreaming(
                self.config.dream_cycle_minutes
            )
            
        if hasattr(self, 'hive_synchronizer') and self.hive_synchronizer:
            self.hive_synchronizer.start_sync()
            
        logger.info("SAM services started")
    
    def stop_services(self):
        """Stop background services"""
        if hasattr(self, 'dreaming'):
            self.dreaming.stop_background_dreaming()
            
        if hasattr(self, 'hive_synchronizer') and self.hive_synchronizer:
            self.hive_synchronizer.stop_sync()
            
        logger.info("SAM services stopped")
    
    def evolve(self):
        """Trigger a model evolution cycle"""
        if hasattr(self, 'evolution_manager'):
            results = self.evolution_manager.run_evolution_cycle()
            logger.info(f"Evolution cycle completed: {results}")
            return results
        return {"error": "Evolution manager not available"}
    
    def get_status(self):
        """
        Get comprehensive status of the model
        
        Returns:
            Dict of status information
        """
        status = {
            "model_size": {
                "hidden_dim": self.layers.hidden_dim,
                "num_layers": len(self.layers.layers),
                "total_concepts": self.concept_bank.next_concept_id,
                "parameter_count": sum(p.numel() for p in self.parameters() if p.requires_grad)
            },
            "training": {
                "global_step": self.global_step
            },
            "concept_stats": self.concept_bank.get_concept_stats() 
                if hasattr(self.concept_bank, "get_concept_stats") else {},
            "consciousness": self.consciousness.get_identity_summary() 
                if hasattr(self.consciousness, "get_identity_summary") else None,
        }
        
        # Add hive mind stats if available
        if hasattr(self, 'hive_synchronizer') and self.hive_synchronizer:
            status["hive_mind"] = self.hive_synchronizer.get_sync_stats()
            
        return status


class SAMFactory:
    """
    Factory for creating and configuring SAM instances
    
    Provides convenient methods for creating SAM instances with different
    configurations and capabilities.
    """
    
    @staticmethod
    def create_minimal(device="cpu"):
        """
        Create a minimal SAM instance suitable for CPU
        
        Args:
            device: Device to use
            
        Returns:
            SAM instance
        """
        config = SAMConfig({
            "initial_hidden_dim": 256,
            "initial_num_layers": 4,
            "concept_dim": 256,
            "thought_dim": 512,
            "device": device
        })
        
        return SAM(config)
    
    @staticmethod
    def create_standard(device="cuda"):
        """
        Create a standard SAM instance
        
        Args:
            device: Device to use
            
        Returns:
            SAM instance
        """
        config = SAMConfig({
            "initial_hidden_dim": 768,
            "initial_num_layers": 8,
            "concept_dim": 768,
            "thought_dim": 1536,
            "device": device
        })
        
        return SAM(config)
    
    @staticmethod
    def create_large(device="cuda"):
        """
        Create a large SAM instance
        
        Args:
            device: Device to use
            
        Returns:
            SAM instance
        """
        config = SAMConfig({
            "initial_hidden_dim": 1536,
            "initial_num_layers": 12,
            "concept_dim": 1536,
            "thought_dim": 2048,
            "device": device
        })
        
        return SAM(config)
    
    @staticmethod
    def create_multimodal(device="cuda"):
        """
        Create a SAM instance with multimodal capabilities
        
        Args:
            device: Device to use
            
        Returns:
            SAM instance with multimodal capabilities
        """
        config = SAMConfig({
            "initial_hidden_dim": 1024,
            "initial_num_layers": 10,
            "concept_dim": 1024,
            "thought_dim": 2048,
            "multimodal_enabled": True,
            "image_dim": 768,
            "audio_dim": 512,
            "device": device
        })
        
        return SAM(config)
    
    @staticmethod
    def create_networked(server_url=None, identity=None, device="cuda"):
        """
        Create a SAM instance with hive mind capabilities
        
        Args:
            server_url: URL of hive mind server
            identity: Identity for this instance
            device: Device to use
            
        Returns:
            SAM instance with hive mind capabilities
        """
        config = SAMConfig({
            "initial_hidden_dim": 768,
            "initial_num_layers": 8,
            "hive_enabled": True,
            "hive_server_url": server_url,
            "hive_identity": identity or str(uuid.uuid4()),
            "device": device
        })
        
        return SAM(config)
    
    @staticmethod
    def create_custom(config_dict):
        """
        Create a SAM instance with custom configuration
        
        Args:
            config_dict: Dictionary of configuration values
            
        Returns:
            SAM instance with custom configuration
        """
        config = SAMConfig(config_dict)
        return SAM(config)
    
    @staticmethod
    def create_from_file(config_path):
        """
        Create a SAM instance from configuration file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            SAM instance with configuration from file
        """
        config = SAMConfig.from_file(config_path)
        return SAM(config)
