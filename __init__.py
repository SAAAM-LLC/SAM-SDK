"""
SAM-SDK: Complete Development Kit for Synergistic Autonomous Machine
-------------------------------------------------------------------

A comprehensive toolkit for creating, customizing, and deploying
SAM instances with full access to all cognitive and neural systems.

Created by SAAAM LLC
"""

__version__ = "0.5.0"

# Core systems
from .core import SAM, SAMConfig, SAMFactory
from .concepts import ConceptBank, ConceptManager, PatternRecognition
from .thought import ThoughtState, ThoughtManager, RecursiveReasoning
from .consciousness import ConsciousnessSystem, IdentityManager
from .experience import ExperienceManager, ExperienceRecorder, Memory
from .dreaming import DreamingSystem, Reflection, ConceptSynthesis
from .growth import NeuralGrowth, EvolutionManager, GrowthMonitor

# Neural components
from .neural import DynamicSegmentation, NeuroplasticLayers, AdaptiveAttention

# Multimodal systems
from .multimodal import MultimodalProcessor, ModalityFusion, ImageProcessor, AudioProcessor

# Networking and distribution
from .hive import HiveNetwork, HiveNode, MasterNode

# Developer tools
from .tools import Visualizer, Debugger, PerformanceMonitor, ModelInspector
