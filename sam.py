# sam.py - Synergistic Autonomous Machine with Hive Mind Capability
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import time
import logging
import os
import threading
import random
import uuid
import asyncio
import websockets
import hashlib
import requests
import pickle
import sqlite3
import base64
import io
import zlib
import copy
import re
import argparse
from torch.optim.lr_scheduler import OneCycleLR
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter, deque
from queue import Queue

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SAM")

###########################################
# CONFIGURATION
###########################################


@dataclass
class SAMConfig:
    """Configuration for SAM (Synergistic Autonomous Machine)"""
    # Core dimensions
    initial_char_dim: int = 256
    initial_hidden_dim: int = 1536
    initial_num_layers: int = 8
    max_position_embeddings: int = 8192

    # Growth parameters
    max_hidden_dim: int = 4096
    max_num_layers: int = 16
    max_growth_steps: int = 10000
    growth_factor: float = 1.2
    min_layer_usage_threshold: float = 0.3

    # Memory systems
    concept_memory_size: int = 10000
    concept_dim: int = 1536
    thought_dim: int = 2048
    max_thought_depth: int = 10
    pattern_memory_capacity: int = 50000

    # Learning parameters
    learning_rate: float = 3e-5
    warmup_steps: int = 100
    adaption_rate: float = 0.400

    # Segmentation parameters
    max_segment_length: int = 16
    min_segment_frequency: int = 5
    concept_frequency_threshold: int = 10

    # Dreaming parameters
    dream_batch_size: int = 5
    dream_max_length: int = 256
    dream_cycle_minutes: float = 0.2
    dream_on_idle_only: bool = True

    # Consciousness parameters
    stability_threshold: float = 0.95
    novelty_weight: float = 0.4

    # Paths for persistence
    save_dir: str = "./data"
    experiences_path: str = "./data/experiences.json"
    concepts_path: str = "./data/concepts.json"
    growth_log_path: str = "./data/growth_log.json"

    # Runtime parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Communication Style
    communication_style: str = "flexible"  # "flexible", "standard", etc.

    # Hive Mind Configuration
    hive_enabled: bool = False
    hive_sync_interval_seconds: int = 300  # 5 minutes
    hive_sync_concept_limit: int = 1000
    hive_server_url: str = ""
    hive_identity: str = ""
    hive_auth_key: str = ""
    hive_server_mode: bool = False
    hive_compression_level: int = 6

    # Hardware Adaptability
    hardware_adaptive: bool = True
    min_free_memory_gb: float = 1.0
    offload_threshold: float = 0.75

    # Multimodal capabilities
    multimodal_enabled: bool = False
    image_dim: int = 768
    audio_dim: int = 512
    multimodal_fusion_strategy: str = "attention"  # "attention", "concatenation"

    # Vocabulary customization
    load_default_vocab: bool = False # Set to False to prevent loading temp vocab by default

    def validate(self):
        """Validate configuration parameters"""
        # Check dimension relationships
        if self.concept_dim > self.initial_hidden_dim:
            logger.warning("concept_dim should not be larger than initial_hidden_dim")
            self.concept_dim = self.initial_hidden_dim

        if self.thought_dim > self.initial_hidden_dim * 2:
            logger.warning("thought_dim too large, reducing to 2x initial_hidden_dim")
            self.thought_dim = self.initial_hidden_dim * 2

        # Check growth parameters
        if self.growth_factor <= 1.0:
            logger.warning("growth_factor must be greater than 1.0, setting to default 1.2")
            self.growth_factor = 1.2

        if self.max_growth_steps < 100:
            logger.warning("max_growth_steps too small, setting to minimum 100")
            self.max_growth_steps = 100

        # Check limit values
        if self.max_hidden_dim < self.initial_hidden_dim:
            logger.warning("max_hidden_dim cannot be smaller than initial_hidden_dim")
            self.max_hidden_dim = self.initial_hidden_dim * 2

        if self.max_num_layers < self.initial_num_layers:
            logger.warning("max_num_layers cannot be smaller than initial_num_layers")
            self.max_num_layers = self.initial_num_layers * 2

        # Check memory parameters
        if self.concept_memory_size < 1000:
            logger.warning("concept_memory_size too small, setting to minimum 1000")
            self.concept_memory_size = 1000

        if self.pattern_memory_capacity < 1000:
            logger.warning("pattern_memory_capacity too small, setting to minimum 1000")
            self.pattern_memory_capacity = 1000

        # Check learning parameters
        if self.learning_rate > 0.1:
            logger.warning("learning_rate too high, capping at 0.1")
            self.learning_rate = 0.1

        if self.warmup_steps < 100:
            logger.warning("warmup_steps too small, setting to minimum 100")
            self.warmup_steps = 100

        # Check device configuration
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
            self.dtype = torch.float32

        # Check multimodal configuration
        if self.multimodal_enabled:
            if self.image_dim <= 0:
                logger.warning("Invalid image_dim, setting to default 768")
                self.image_dim = 768
            if self.audio_dim <= 0:
                logger.warning("Invalid audio_dim, setting to default 512")
                self.audio_dim = 512

        # Validate paths
        for path_attr in ['save_dir', 'experiences_path', 'concepts_path', 'growth_log_path']:
            path = getattr(self, path_attr)
            try:
                if path and isinstance(path, str):
                    os.makedirs(os.path.dirname(path), exist_ok=True)
            except Exception as e:
                logger.error(f"Error creating directory for {path_attr}: {e}")
                default_name = os.path.basename(path) if path else f"{path_attr}.json"
                setattr(self, path_attr, os.path.join("./data", default_name))

        # Validate hive mind configuration
        if self.hive_enabled:
            if not self.hive_server_url and not self.hive_server_mode:
                logger.warning("Hive enabled but no server URL provided, disabling hive.")
                self.hive_enabled = False
            if not self.hive_identity:
                self.hive_identity = str(uuid.uuid4())
                logger.info(f"Generated hive identity: {self.hive_identity}")

        # Return validated config
        return self

    @classmethod
    def load(cls, path):
        """Load configuration from JSON file"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        if 'dtype' in config_dict and isinstance(config_dict['dtype'], str):
             if 'float16' in config_dict['dtype']:
                 config_dict['dtype'] = torch.float16
             else:
                 config_dict['dtype'] = torch.float32

        return cls(**config_dict)


    def save(self, path):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            # Use the custom encoder
            json.dump(asdict(self), f, indent=2, cls=CustomJSONEncoder)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.dtype):
            return str(obj)
        return super().default(obj)

###########################################
# MEMORY SYSTEMS
###########################################

class ConceptMemoryBank(nn.Module):
    """Dynamic memory bank for emergent concepts (replaces traditional vocabulary)"""

    def __init__(self, concept_dim, initial_size=100000, growth_rate=5000, device="cuda"):
        super().__init__()
        self.concept_dim = concept_dim
        self.growth_rate = growth_rate
        self.device = device

        # FIX: Initialize the threading lock
        self.concept_lock = threading.Lock()

        # Concept embeddings (analogous to token embeddings)
        self.concept_embeddings = nn.Embedding(initial_size, concept_dim)

        # Concept usage tracking
        self.register_buffer("concept_frequencies", torch.zeros(initial_size, dtype=torch.int))
        self.register_buffer("concept_timestamps", torch.zeros(initial_size, dtype=torch.float))

        # Concept metadata
        self.concept_metadata = {}  # concept_id -> metadata dict

        # Source mapping (character sequence -> concept_id)
        self.source_to_concept = {}

        # Meaning map (concept_id -> meaning vector)
        self.register_buffer("meaning_vectors", torch.zeros(initial_size, concept_dim))

        # Related concepts (concept_id -> [related_concept_ids])
        self.related_concepts = defaultdict(list)

        # Hive mind syncable concepts
        self.hive_shared_concepts = set()
        self.hive_private_concepts = set()
        self.hive_pending_sync = set()
        self.hive_origin = {}  # concept_id -> origin instance id
        self.hive_global_id_map = {}  # local_id -> global_id

        # Multimodal concepts tracking
        self.modality_concepts = {
            "text": set(),
            "image": set(),
            "audio": set(),
            "multimodal": set()
        }

        # Growth tracking
        self.next_concept_id = 0
        self.creation_history = []

        # Initialize with basic character concepts (a-z, A-Z, 0-9, etc.)
        self._initialize_basic_concepts()

    def _initialize_basic_concepts(self):
        """Initialize basic character-level concepts"""
        # Add ASCII characters
        for i in range(128):
            char = chr(i)
            self.add_character_concept(char)

        # Add common character sequences for English
        common_sequences = [
            # Common punctuation
            ".", ",", "!", "?", ";", ":", "'", "\"", "-", "_", "+", "=", "*", "/", "\\",
            "@", "#", "$", "%", "^", "&", "(", ")", "[", "]", "{", "}", "<", ">", "|",

            # Basic bigrams and trigrams
            "th", "er", "on", "an", "re", "he", "in", "ed", "nd", "ha", "at", "en",
            "es", "of", "or", "nt", "ea", "ti", "to", "io", "le", "is", "ou", "ar",
            "as", "de", "rt", "ve", "the", "and", "ing", "ion", "ent", "ed ", "to ",
            " in", " a ", " to", "for", "ati", "tha", "ter", "ers", " th", "hat",

            # Common words
            "the", "and", "of", "to", "in", "is", "you", "that", "it", "he", "she", "was", "for",
            "on", "are", "with", "as", "they", "be", "at", "this", "have", "from", "or", "by",

            # Programming tokens
            "def", "class", "function", "if", "else", "for", "while", "return", "import",
            "from", "try", "except", "True", "False", "None", "self", "print"
        ]

        for seq in common_sequences:
            self.add_character_concept(seq)

    def forward(self, concept_ids):
        """Get embeddings for concept IDs"""
        if isinstance(concept_ids, list):
            # Handle nested lists (from segmentation)
            flat_ids = []
            for item in concept_ids:
                if isinstance(item, list):
                    flat_ids.extend(item)
                else:
                    flat_ids.append(item)
            if not flat_ids:
                return torch.empty(0, self.concept_dim, device=self.device)
            concept_ids = torch.tensor(flat_ids, dtype=torch.long, device=self.device)

        # Ensure concept_ids is a tensor before passing to embedding
        if not isinstance(concept_ids, torch.Tensor):
            concept_ids = torch.tensor(concept_ids, dtype=torch.long, device=self.device)

        # Check if any id is out of bounds
        if concept_ids.numel() > 0 and concept_ids.max() >= self.concept_embeddings.num_embeddings:
            logger.warning(f"Concept ID {concept_ids.max().item()} is out of bounds for embedding size {self.concept_embeddings.num_embeddings}. Growing bank.")
            self.grow_if_needed(required_size=concept_ids.max().item() + 1)
            # After growing, we need to re-check. If still too large, there's a deeper issue.
            if concept_ids.max() >= self.concept_embeddings.num_embeddings:
                logger.error("Failed to grow concept bank sufficiently. Clamping problematic IDs.")
                concept_ids = torch.clamp(concept_ids, max=self.concept_embeddings.num_embeddings - 1)
        
        return self.concept_embeddings(concept_ids)

    def update_concept_usage(self, concept_id, context=None, register_for_sync=True):
        """Update usage statistics for a concept"""
        with self.concept_lock:
            if concept_id >= len(self.concept_frequencies):
                self.grow_if_needed(required_size=concept_id + 1)

            # Update frequency and timestamp
            self.concept_frequencies[concept_id] += 1
            self.concept_timestamps[concept_id] = time.time()

            # Update metadata
            if context and concept_id in self.concept_metadata:
                context_str = str(context)[:100]  # Limit context length
                if "contexts" not in self.concept_metadata[concept_id]:
                    self.concept_metadata[concept_id]["contexts"] = Counter()
                self.concept_metadata[concept_id]["contexts"][context_str] += 1
                self.concept_metadata[concept_id]["frequency"] = self.concept_frequencies[concept_id].item()

            # Register for hive mind sync if applicable
            if register_for_sync and concept_id not in self.hive_private_concepts:
                self.hive_pending_sync.add(concept_id)

    def get_concept_stats(self):
        """Get statistics about the concepts in memory"""
        with self.concept_lock:
            total_concepts = self.next_concept_id
            if total_concepts == 0:
                used_concepts = 0
            else:
                used_concepts = (self.concept_frequencies[:total_concepts] > 0).sum().item()

            modality_stats = {
                modality: len(concepts)
                for modality, concepts in self.modality_concepts.items()
            }

            # Get top concepts by frequency
            if self.next_concept_id > 0:
                top_k = min(10, self.next_concept_id)
                values, indices = torch.topk(self.concept_frequencies[:self.next_concept_id], k=top_k)

                top_concepts = []
                for idx, val in zip(indices, values):
                    idx_item = idx.item()
                    meta = self.concept_metadata.get(idx_item, {})
                    source = meta.get("source", "N/A")
                    top_concepts.append((idx_item, source, val.item()))
            else:
                top_concepts = []

        return {
            "total_concepts": total_concepts,
            "used_concepts": used_concepts,
            "modality_stats": modality_stats,
            "hive_shared": len(self.hive_shared_concepts),
            "hive_private": len(self.hive_private_concepts),
            "top_concepts": top_concepts
        }

    def add_character_concept(self, char_sequence, hive_private=False, origin=None, global_id=None, modality="text"):
        """Add a character sequence as a concept"""
        with self.concept_lock:
            if char_sequence in self.source_to_concept:
                return self.source_to_concept[char_sequence]

            self.grow_if_needed()

            concept_id = self.next_concept_id
            self.source_to_concept[char_sequence] = concept_id

            # Initialize a basic embedding
            with torch.no_grad():
                char_encoding = torch.zeros(self.concept_dim, dtype=torch.float, device=self.device)
                for i, c in enumerate(char_sequence):
                    val = ord(c) / 128.0
                    idx = (i * 4) % self.concept_dim
                    char_encoding[idx:idx+4] = torch.tensor([
                        math.sin(val), math.cos(val), math.sin(2 * val), math.cos(2 * val)
                    ], device=self.device)

                char_encoding = torch.nn.functional.normalize(char_encoding, dim=0)
                self.concept_embeddings.weight[concept_id] = char_encoding
                self.meaning_vectors[concept_id] = char_encoding

            # Initialize metadata
            self.concept_metadata[concept_id] = {
                "source": char_sequence,
                "type": "character_sequence",
                "created_at": time.time(),
                "frequency": 0,
                "contexts": Counter(),
                "hive_syncable": not hive_private,
                "modality": modality
            }

            # Track hive mind status
            if hive_private:
                self.hive_private_concepts.add(concept_id)
            else:
                self.hive_shared_concepts.add(concept_id)
                self.hive_pending_sync.add(concept_id)

            if origin: self.hive_origin[concept_id] = origin
            if global_id: self.hive_global_id_map[concept_id] = global_id

            self.modality_concepts[modality].add(concept_id)

            self.next_concept_id += 1
            self.creation_history.append({
                "concept_id": concept_id,
                "source": char_sequence,
                "timestamp": time.time(),
                "modality": modality
            })

            return concept_id

    def add_semantic_concept(self, meaning_vector, related_sources=None, metadata=None,
                            hive_private=False, origin=None, global_id=None, modality="text"):
        """Add a new semantic concept (not directly mapped to characters)"""
        with self.concept_lock:
            self.grow_if_needed()
            concept_id = self.next_concept_id

            with torch.no_grad():
                self.meaning_vectors[concept_id] = F.normalize(meaning_vector, dim=0)
                self.concept_embeddings.weight[concept_id] = meaning_vector

            meta = metadata or {}
            meta.update({
                "type": "semantic",
                "created_at": time.time(),
                "frequency": 0,
                "related_sources": related_sources or [],
                "contexts": Counter(),
                "hive_syncable": not hive_private,
                "modality": modality
            })
            self.concept_metadata[concept_id] = meta

            if hive_private: self.hive_private_concepts.add(concept_id)
            else:
                self.hive_shared_concepts.add(concept_id)
                self.hive_pending_sync.add(concept_id)

            if origin: self.hive_origin[concept_id] = origin
            if global_id: self.hive_global_id_map[concept_id] = global_id

            self.modality_concepts[modality].add(concept_id)

            self.next_concept_id += 1
            self.creation_history.append({
                "concept_id": concept_id,
                "type": "semantic",
                "timestamp": time.time(),
                "modality": modality
            })
            return concept_id

    def add_multimodal_concept(self, embeddings_dict, related_sources=None, metadata=None, hive_private=True):
        """Add a concept that spans multiple modalities"""
        modalities = list(embeddings_dict.keys())
        embeddings = list(embeddings_dict.values())
        combined = sum(embeddings) / len(embeddings)
        combined = F.normalize(combined, dim=0)

        meta = metadata or {}
        meta.update({"modalities": modalities, "modality": "multimodal"})

        return self.add_semantic_concept(
            meaning_vector=combined,
            related_sources=related_sources,
            metadata=meta,
            hive_private=hive_private,
            modality="multimodal"
        )

    def find_concept_by_source(self, source):
        """Find a concept ID by its source string"""
        return self.source_to_concept.get(source, None)

    def create_merged_concept(self, concept_id1, concept_id2, frequency=None, hive_private=False):
        """Create a new concept by merging two existing concepts"""
        source1 = self.concept_metadata.get(concept_id1, {}).get("source", "")
        source2 = self.concept_metadata.get(concept_id2, {}).get("source", "")
        merged_source = source1 + source2 if source1 and source2 else None

        meaning1 = self.meaning_vectors[concept_id1]
        meaning2 = self.meaning_vectors[concept_id2]
        merged_meaning = (meaning1 + meaning2) / 2

        is_private = hive_private or (concept_id1 in self.hive_private_concepts or concept_id2 in self.hive_private_concepts)

        modality1 = self.concept_metadata.get(concept_id1, {}).get("modality", "text")
        modality2 = self.concept_metadata.get(concept_id2, {}).get("modality", "text")
        merged_modality = "multimodal" if modality1 != modality2 else modality1

        merged_id = self.add_semantic_concept(
            meaning_vector=merged_meaning,
            related_sources=[source1, source2] if source1 and source2 else None,
            metadata={
                "type": "merged",
                "parent_concepts": [concept_id1, concept_id2],
                "frequency": frequency or 1,
                "modality": merged_modality
            },
            hive_private=is_private,
            modality=merged_modality
        )

        if merged_source: self.source_to_concept[merged_source] = merged_id
        self.related_concepts[concept_id1].append(merged_id)
        self.related_concepts[concept_id2].append(merged_id)

        return merged_id

    def load_vocabulary(self, vocab_path):
        """Load vocabulary from file to initialize with extensive vocabulary"""
        if not os.path.exists(vocab_path):
            logger.warning(f"Vocabulary file {vocab_path} not found")
            return 0
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_items = f.read().splitlines()
            count = 0
            for item in vocab_items:
                if item and item not in self.source_to_concept:
                    self.add_character_concept(item)
                    count += 1
            logger.info(f"Loaded {count} vocabulary items from {vocab_path}")
            return count
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}")
            return 0

    def find_similar_concepts(self, query_vector, top_k=5, modality=None):
        """Find concepts with similar meaning vectors"""
        if self.next_concept_id == 0: return []
        query_vector = F.normalize(query_vector, dim=0)

        concept_filter = list(self.modality_concepts.get(modality, set())) if modality else None
        if concept_filter is not None and not concept_filter: return []

        if concept_filter:
            filtered_vectors = self.meaning_vectors[concept_filter]
            similarities = F.cosine_similarity(query_vector.unsqueeze(0), filtered_vectors, dim=1)
            k = min(top_k, len(similarities))
            values, indices = torch.topk(similarities, k)
            return [(concept_filter[idx.item()], val.item()) for idx, val in zip(indices, values)]
        else:
            all_vectors = self.meaning_vectors[:self.next_concept_id]
            similarities = F.cosine_similarity(query_vector.unsqueeze(0), all_vectors, dim=1)
            k = min(top_k, len(similarities))
            values, indices = torch.topk(similarities, k)
            return [(idx.item(), val.item()) for idx, val in zip(indices, values)]

    def grow_if_needed(self, required_size=None):
        """Grow concept bank if approaching capacity or a specific size is required."""
        current_size = self.concept_embeddings.num_embeddings
        should_grow = (required_size and required_size > current_size) or \
                      (self.next_concept_id > current_size - self.growth_rate)

        if should_grow:
            new_size = max(current_size + self.growth_rate, required_size if required_size else 0)
            logger.info(f"Growing concept bank from {current_size} to {new_size}")

            # Store old tensors
            old_embedding = self.concept_embeddings.weight.data
            old_meaning_vectors = self.meaning_vectors.data
            old_freqs = self.concept_frequencies.data
            old_timestamps = self.concept_timestamps.data

            # Create new larger tensors
            self.concept_embeddings = nn.Embedding(new_size, self.concept_dim).to(self.device)
            new_meaning_vectors = torch.zeros(new_size, self.concept_dim, device=self.device)
            new_freqs = torch.zeros(new_size, dtype=torch.int, device=self.device)
            new_timestamps = torch.zeros(new_size, dtype=torch.float, device=self.device)

            # Copy data
            with torch.no_grad():
                self.concept_embeddings.weight[:current_size] = old_embedding
                new_meaning_vectors[:current_size] = old_meaning_vectors
                new_freqs[:current_size] = old_freqs
                new_timestamps[:current_size] = old_timestamps

            # Register new buffers
            self.register_buffer("meaning_vectors", new_meaning_vectors)
            self.register_buffer("concept_frequencies", new_freqs)
            self.register_buffer("concept_timestamps", new_timestamps)

            # Re-tie weights if the model has an lm_head
            if hasattr(self, 'lm_head'):
                self.lm_head.weight = self.concept_embeddings.weight

            return True
        return False

    def get_concepts_for_sync(self, limit=1000):
        """Get concepts that need to be synced with the hive mind"""
        with self.concept_lock:
            pending_list = list(self.hive_pending_sync)
            if not pending_list: return []

            importance_scores = []
            for concept_id in pending_list:
                if concept_id >= self.next_concept_id: continue
                frequency = self.concept_frequencies[concept_id].item()
                recency = time.time() - self.concept_timestamps[concept_id].item()
                recency_factor = math.exp(-recency / 86400)
                importance = frequency * recency_factor
                importance_scores.append((concept_id, importance))

            importance_scores.sort(key=lambda x: x[1], reverse=True)
            top_concepts = [cid for cid, _ in importance_scores[:limit]]

            concept_data = []
            for concept_id in top_concepts:
                try:
                    metadata = self.concept_metadata.get(concept_id, {})
                    with torch.no_grad():
                        embedding = self.concept_embeddings.weight[concept_id].cpu().tolist()
                        meaning = self.meaning_vectors[concept_id].cpu().tolist()
                    
                    concept_data.append({
                        "local_id": concept_id, "global_id": self.hive_global_id_map.get(concept_id),
                        "source": metadata.get("source", ""), "type": metadata.get("type", "unknown"),
                        "frequency": self.concept_frequencies[concept_id].item(),
                        "embedding": embedding, "meaning": meaning,
                        "created_at": metadata.get("created_at", time.time()),
                        "origin": self.hive_origin.get(concept_id),
                        "related_sources": metadata.get("related_sources", []),
                        "modality": metadata.get("modality", "text")
                    })
                except Exception as e:
                    logger.error(f"Error preparing concept {concept_id} for sync: {e}")
        return concept_data

    def mark_concepts_synced(self, concept_ids):
        """Mark concepts as synced with the hive mind"""
        with self.concept_lock:
            for concept_id in concept_ids:
                self.hive_pending_sync.discard(concept_id)

    def integrate_hive_concepts(self, hive_concepts, origin_id):
        """Integrate concepts from the hive mind"""
        integrated_count = 0; updated_count = 0
        for concept_data in hive_concepts:
            global_id = concept_data.get("global_id")
            source = concept_data.get("source", "")
            concept_type = concept_data.get("type", "unknown")
            modality = concept_data.get("modality", "text")

            existing_local_id = None
            if global_id:
                for lid, gid in self.hive_global_id_map.items():
                    if gid == global_id:
                        existing_local_id = lid; break
            if existing_local_id is None and source and concept_type == "character_sequence":
                existing_local_id = self.source_to_concept.get(source)

            embedding = torch.tensor(concept_data["embedding"], dtype=self.dtype, device=self.device)
            meaning = torch.tensor(concept_data["meaning"], dtype=self.dtype, device=self.device)

            if existing_local_id is not None:
                with torch.no_grad():
                    self.concept_embeddings.weight[existing_local_id] = 0.7 * self.concept_embeddings.weight[existing_local_id] + 0.3 * embedding
                    existing_meaning = self.meaning_vectors[existing_local_id]
                    self.meaning_vectors[existing_local_id] = F.normalize(0.7 * existing_meaning + 0.3 * meaning, dim=0)
                
                if concept_data.get("frequency", 0) > self.concept_frequencies[existing_local_id].item():
                    self.concept_frequencies[existing_local_id] = concept_data["frequency"]
                
                with self.concept_lock:
                    existing_modality = self.concept_metadata[existing_local_id].get("modality", "text")
                    if existing_modality != modality:
                        self.concept_metadata[existing_local_id]["modality"] = "multimodal"
                        self.modality_concepts.get(existing_modality, set()).discard(existing_local_id)
                        self.modality_concepts["multimodal"].add(existing_local_id)
                updated_count += 1
            else:
                if concept_type == "character_sequence" and source:
                    local_id = self.add_character_concept(source, hive_private=False, origin=origin_id, global_id=global_id, modality=modality)
                    with torch.no_grad():
                        self.concept_embeddings.weight[local_id] = embedding
                        self.meaning_vectors[local_id] = meaning
                else:
                    local_id = self.add_semantic_concept(
                        meaning_vector=embedding,
                        related_sources=concept_data.get("related_sources", []),
                        metadata={"type": concept_type, "frequency": concept_data.get("frequency", 1), "modality": modality},
                        hive_private=False, origin=origin_id, global_id=global_id, modality=modality
                    )
                self.concept_frequencies[local_id] = concept_data.get("frequency", 1)
                integrated_count += 1
        
        if integrated_count > 0 or updated_count > 0:
            logger.info(f"Hive integration: {integrated_count} new concepts, {updated_count} updated")
        return integrated_count, updated_count


class PatternMemory:
    """Memory system for recognizing and storing recurring patterns"""

    def __init__(self, capacity=10000, min_frequency=5):
        self.capacity = capacity
        self.min_frequency = min_frequency
        self.patterns = {}  # pattern -> frequency
        self.context_patterns = defaultdict(lambda: defaultdict(int))  # context -> pattern -> frequency
        self.timestamps = {}  # pattern -> last seen timestamp
        self.pattern_utilities = {}  # pattern -> utility score

        # Track patterns by modality
        self.modality_patterns = {
            "text": set(),
            "image": set(),
            "audio": set(),
            "multimodal": set()
        }

        # Hive mind tracking
        self.shared_patterns = set()
        self.private_patterns = set()
        self.pending_sync_patterns = set()

    def add_pattern(self, pattern, context=None, private=False, modality="text"):
        """Add a pattern to memory"""
        if not isinstance(pattern, str): pattern = str(pattern)

        if pattern in self.patterns:
            self.patterns[pattern] += 1
        else:
            if len(self.patterns) >= self.capacity:
                least_useful = min(self.pattern_utilities, key=self.pattern_utilities.get) if self.pattern_utilities else min(self.timestamps, key=self.timestamps.get)
                if least_useful:
                    del self.patterns[least_useful]
                    del self.timestamps[least_useful]
                    if least_useful in self.pattern_utilities: del self.pattern_utilities[least_useful]
                    self.shared_patterns.discard(least_useful)
                    self.private_patterns.discard(least_useful)
                    self.pending_sync_patterns.discard(least_useful)
                    for m_patterns in self.modality_patterns.values(): m_patterns.discard(least_useful)
            self.patterns[pattern] = 1

        self.timestamps[pattern] = time.time()
        self.pattern_utilities[pattern] = (0.9 * self.pattern_utilities.get(pattern, 0)) + (0.1 * self.patterns[pattern])

        if context:
            if not isinstance(context, str): context = str(context)
            self.context_patterns[context][pattern] += 1
        
        if private: self.private_patterns.add(pattern)
        else:
            self.shared_patterns.add(pattern)
            self.pending_sync_patterns.add(pattern)

        self.modality_patterns.setdefault(modality, set()).add(pattern)

    def get_frequent_patterns(self, limit=100, include_private=True, modality=None):
        """Get most frequent patterns"""
        patterns_to_consider = self.patterns.items()
        if modality:
            modality_set = self.modality_patterns.get(modality, set())
            patterns_to_consider = [(p, f) for p, f in patterns_to_consider if p in modality_set]
        
        if not include_private:
            patterns_to_consider = [(p, f) for p, f in patterns_to_consider if p not in self.private_patterns]

        return sorted([p for p in patterns_to_consider if p[1] >= self.min_frequency], key=lambda x: x[1], reverse=True)[:limit]

    def get_context_patterns(self, context, limit=20, modality=None):
        """Get patterns associated with a specific context"""
        if not isinstance(context, str): context = str(context)
        if context not in self.context_patterns: return []
        
        patterns_to_consider = self.context_patterns[context].items()
        if modality:
            modality_set = self.modality_patterns.get(modality, set())
            patterns_to_consider = [(p, f) for p, f in patterns_to_consider if p in modality_set]
            
        return sorted(patterns_to_consider, key=lambda x: x[1], reverse=True)[:limit]

    def get_pattern_frequency(self, pattern):
        """Get frequency of a specific pattern"""
        if not isinstance(pattern, str): pattern = str(pattern)
        return self.patterns.get(pattern, 0)

    def merge_patterns(self, pattern1, pattern2, private=False, modality=None):
        """Merge two patterns into a single compound pattern"""
        if not isinstance(pattern1, str): pattern1 = str(pattern1)
        if not isinstance(pattern2, str): pattern2 = str(pattern2)
        compound = pattern1 + pattern2

        frequency = min(self.patterns.get(pattern1, 0), self.patterns.get(pattern2, 0))
        if frequency >= self.min_frequency // 2:
            self.patterns[compound] = frequency
            self.timestamps[compound] = time.time()
            self.pattern_utilities[compound] = (self.pattern_utilities.get(pattern1, 0) + self.pattern_utilities.get(pattern2, 0)) / 2

            is_private = private or pattern1 in self.private_patterns or pattern2 in self.private_patterns
            if is_private: self.private_patterns.add(compound)
            else:
                self.shared_patterns.add(compound)
                self.pending_sync_patterns.add(compound)
            
            p1_mod = next((m for m, ps in self.modality_patterns.items() if pattern1 in ps), "text")
            p2_mod = next((m for m, ps in self.modality_patterns.items() if pattern2 in ps), "text")
            merged_modality = modality if modality else ("multimodal" if p1_mod != p2_mod else p1_mod)
            self.modality_patterns.setdefault(merged_modality, set()).add(compound)
            return compound
        return None

    def get_patterns_for_sync(self, limit=500):
        """Get patterns that need to be synced with hive mind"""
        sync_list = sorted(list(self.pending_sync_patterns), key=lambda p: self.pattern_utilities.get(p, 0), reverse=True)
        patterns_to_sync = []
        for pattern in sync_list[:limit]:
            if pattern in self.patterns and pattern not in self.private_patterns:
                modality = next((m for m, ps in self.modality_patterns.items() if pattern in ps), "text")
                patterns_to_sync.append({
                    "pattern": pattern, "frequency": self.patterns[pattern],
                    "utility": self.pattern_utilities.get(pattern, 0),
                    "timestamp": self.timestamps.get(pattern, time.time()),
                    "modality": modality
                })
        return patterns_to_sync

    def mark_patterns_synced(self, patterns):
        """Mark patterns as synced with hive mind"""
        for pattern_data in patterns:
            pattern = pattern_data['pattern'] if isinstance(pattern_data, dict) else pattern_data
            self.pending_sync_patterns.discard(pattern)

    def integrate_hive_patterns(self, hive_patterns):
        """Integrate patterns from the hive mind"""
        integrated = 0; updated = 0
        for p_data in hive_patterns:
            pattern, freq, utility, modality = p_data["pattern"], p_data["frequency"], p_data.get("utility", p_data["frequency"]), p_data.get("modality", "text")
            if pattern in self.patterns:
                if freq > self.patterns[pattern]:
                    self.patterns[pattern] = freq; updated += 1
                self.pattern_utilities[pattern] = 0.7 * self.pattern_utilities.get(pattern, 0) + 0.3 * utility
            else:
                if len(self.patterns) < self.capacity:
                    self.patterns[pattern] = freq
                    self.timestamps[pattern] = p_data.get("timestamp", time.time())
                    self.pattern_utilities[pattern] = utility
                    self.shared_patterns.add(pattern)
                    self.modality_patterns.setdefault(modality, set()).add(pattern)
                    integrated += 1
        return integrated, updated


class ThoughtState(nn.Module):
    """Maintains an evolving semantic thought space across concept sequences"""

    def __init__(self, concept_dim, thought_dim=2048, max_thought_depth=8,
                superposition_states=4):
        super().__init__()
        self.concept_dim = concept_dim
        self.thought_dim = thought_dim
        self.max_thought_depth = max_thought_depth
        self.superposition_states = superposition_states

        # Thought transformation networks
        self.concept_to_thought = nn.Linear(concept_dim, thought_dim)
        self.thought_evolution = nn.TransformerEncoderLayer(
            d_model=thought_dim, nhead=16, dim_feedforward=thought_dim*4,
            dropout=0.1, batch_first=True, activation=F.gelu
        )

        # Recursive pathways
        self.thought_compression = nn.Linear(thought_dim, thought_dim)
        self.thought_projection = nn.Linear(thought_dim, concept_dim)

        # Meta-learning components
        self.learning_rate_controller = nn.Sequential(
            nn.Linear(thought_dim, thought_dim // 2), nn.GELU(),
            nn.Linear(thought_dim // 2, 1), nn.Sigmoid()
        )

        # Quantum-inspired superposition
        self.register_buffer("amplitudes", torch.ones(superposition_states) / math.sqrt(superposition_states))
        self.entanglement_layer = nn.Linear(thought_dim * superposition_states, thought_dim)

        # Modality-specific processing
        self.modality_projections = nn.ModuleDict({
            "text": nn.Identity(), "image": nn.Linear(thought_dim, thought_dim),
            "audio": nn.Linear(thought_dim, thought_dim), "multimodal": nn.Linear(thought_dim, thought_dim)
        })

        # Cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(embed_dim=thought_dim, num_heads=8, batch_first=True)

        self.thought_memory = None; self.superposition_memories = None; self.thought_depth = 0
        self.evolution_history = []; self.modality_thoughts = {}
        self.shared_thought = None; self.local_thought = None; self.personal_factor = 0.8
        self.reset()

    def reset(self, batch_size=1):
        """Reset thought state"""
        device = next(self.parameters()).device
        self.thought_memory = [torch.zeros(batch_size, 1, self.thought_dim, device=device)]
        self.thought_depth = 0
        self.superposition_memories = [[torch.zeros(batch_size, 1, self.thought_dim, device=device)] for _ in range(self.superposition_states)]
        self.modality_thoughts = {m: torch.zeros(batch_size, 1, self.thought_dim, device=device) for m in self.modality_projections.keys()}

    def update(self, concept_embeddings, use_hive_mind=True, modality="text"):
        """Update thought state with new concept embeddings"""
        if concept_embeddings.nelement() == 0: return self.thought_memory[-1]
        batch_size, seq_len, _ = concept_embeddings.shape

        concept_thoughts = self.concept_to_thought(concept_embeddings)
        modality = modality if modality in self.modality_projections else "text"
        concept_thoughts = self.modality_projections[modality](concept_thoughts)
        
        if self.thought_memory is None or batch_size != self.thought_memory[0].shape[0]: self.reset(batch_size)
        current_thought = self.thought_memory[-1]
        combined_thoughts = torch.cat([current_thought, concept_thoughts], dim=1)
        evolved_thought = self.thought_evolution(combined_thoughts)
        compressed = F.gelu(self.thought_compression(evolved_thought[:, -1:, :]))
        self.modality_thoughts[modality] = compressed

        for i in range(self.superposition_states):
            state_transform = torch.roll(compressed, shifts=i+1, dims=-1)
            self.superposition_memories[i].append(state_transform)
            if len(self.superposition_memories[i]) > self.max_thought_depth: self.superposition_memories[i].pop(0)

        if self.amplitudes.numel() > 0 and torch.max(self.amplitudes).item() > 0.8: self._collapse_states()
        
        with torch.no_grad():
            lr_tensor = self.learning_rate_controller(compressed)
            adaptation_rate = (0.1 + 0.4 * lr_tensor.item()) if lr_tensor.numel() > 0 else 0.25
        
        self.local_thought = compressed
        if use_hive_mind and self.shared_thought is not None:
            compressed = self.personal_factor * compressed + (1 - self.personal_factor) * self.shared_thought.to(compressed.device)

        active_modal_thoughts = [t for m, t in self.modality_thoughts.items() if m != modality and torch.norm(t).item() > 0.1]
        if active_modal_thoughts:
            other_modalities = torch.cat(active_modal_thoughts, dim=1)
            attended, _ = self.cross_modal_attention(compressed, other_modalities, other_modalities)
            compressed = 0.7 * compressed + 0.3 * attended

        self.thought_memory.append(compressed)
        if len(self.thought_memory) > self.max_thought_depth: self.thought_memory.pop(0)
        self.thought_depth = len(self.thought_memory) -1
        
        self.evolution_history.append({"timestamp": time.time(), "adaptation_rate": adaptation_rate, "modality": modality})
        return compressed

    def _collapse_states(self):
        """Collapse superposition states"""
        if self.amplitudes.numel() == 0: return
        dominant_idx = torch.argmax(self.amplitudes).item()
        if self.superposition_memories[dominant_idx]: self.thought_memory = self.superposition_memories[dominant_idx].copy()
        with torch.no_grad(): self.amplitudes.fill_(1.0 / math.sqrt(self.superposition_states))

    def get_thought_context(self, use_superposition=True):
        """Get full thought context for recursive reasoning"""
        if not self.thought_memory: return None
        if not use_superposition or not self.superposition_memories or not self.superposition_memories[0]:
            return torch.cat(self.thought_memory, dim=1)

        contexts = [torch.cat(mem, dim=1) for mem in self.superposition_memories if mem]
        if not contexts: return torch.cat(self.thought_memory, dim=1)
        
        weighted_contexts = [ctx * amp for ctx, amp in zip(contexts, self.amplitudes)]
        combined = torch.cat(weighted_contexts, dim=-1)
        
        # Adjust entanglement layer to match combined dimension
        if self.entanglement_layer.in_features != combined.shape[-1]:
            self.entanglement_layer = nn.Linear(combined.shape[-1], self.thought_dim).to(combined.device)

        return F.gelu(self.entanglement_layer(combined))

    def project_to_concept_space(self, thought=None, modality="text"):
        """Project thought back to concept space for recursive reasoning"""
        if thought is None:
            if not self.thought_memory: return None
            thought = self.thought_memory[-1]
        
        modality = modality if modality in self.modality_projections else "text"
        if modality != "text": thought = self.modality_projections[modality](thought)
        return F.gelu(self.thought_projection(thought))

    def set_shared_thought(self, shared_thought_tensor, blend_factor=0.3):
        """Set shared thought from hive mind"""
        if shared_thought_tensor is not None:
            self.shared_thought = shared_thought_tensor
            if blend_factor is not None: self.personal_factor = 1.0 - blend_factor

    def get_shared_thought(self):
        """Get local thought for sharing with hive mind"""
        return self.local_thought.detach().cpu() if self.local_thought is not None else None

    def get_quantum_amplitudes(self):
        """Get current amplitudes of quantum states"""
        return self.amplitudes.detach().cpu().numpy()

    def get_modality_thought(self, modality="text"):
        """Get thought state for a specific modality"""
        if not self.thought_memory: return None
        return self.modality_thoughts.get(modality, self.thought_memory[-1])


class ExperienceManager:
    """Manages SAM's experiences and memory persistence"""

    def __init__(self, config):
        self.config = config
        self.experiences = []
        self.loaded_experiences = 0
        self.shared_experiences = []
        self.private_experiences = []
        self.pending_sync_experiences = deque()
        self.modality_experiences = defaultdict(list)
        
        if self.config.save_dir and isinstance(self.config.save_dir, str):
            os.makedirs(self.config.save_dir, exist_ok=True)
            os.makedirs(os.path.join(self.config.save_dir, "checkpoints"), exist_ok=True)
        self._load_experiences()

    def _load_experiences(self):
        """Load experiences from disk"""
        try:
            if self.config.experiences_path and os.path.exists(self.config.experiences_path):
                with open(self.config.experiences_path, 'r') as f:
                    self.experiences = json.load(f)
                self.loaded_experiences = len(self.experiences)
                for exp in self.experiences:
                    modality = exp.get("modality", "text")
                    exp_id = exp.get("experience_id")
                    if exp_id:
                        self.modality_experiences[modality].append(exp_id)
                        if exp.get("private", False): self.private_experiences.append(exp_id)
                        else: self.shared_experiences.append(exp_id)
                logger.info(f"Loaded {self.loaded_experiences} experiences")
        except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
            logger.error(f"Failed to load experiences: {e}")
            self.experiences = []

    def record_experience(self, experience_type, content, metadata=None, private=False, modality="text"):
        """Record a new experience"""
        experience_id = str(uuid.uuid4())
        experience = {"type": experience_type, "content": content, "timestamp": time.time(),
                      "metadata": metadata or {}, "private": private,
                      "experience_id": experience_id, "modality": modality}
        self.experiences.append(experience)
        
        if private: self.private_experiences.append(experience_id)
        else:
            self.shared_experiences.append(experience_id)
            self.pending_sync_experiences.append(experience_id)
        
        self.modality_experiences[modality].append(experience_id)
        if len(self.experiences) % 10 == 0: self._save_experiences()
        return experience_id

    def _save_experiences(self):
        """Save experiences to disk"""
        try:
            if self.config.experiences_path and isinstance(self.config.experiences_path, str):
                with open(self.config.experiences_path, 'w') as f:
                    json.dump(self.experiences[-1000:], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save experiences: {e}")

    def get_experiences_by_type(self, experience_type, limit=10, include_private=True, modality=None):
        """Get experiences of a specific type"""
        experiences_to_check = self.experiences
        if modality:
            modality_ids = set(self.modality_experiences.get(modality, []))
            experiences_to_check = [exp for exp in self.experiences if exp.get("experience_id") in modality_ids]
        
        filtered = []
        for exp in reversed(experiences_to_check):
            if exp["type"] == experience_type and (include_private or not exp.get("private", False)):
                filtered.append(exp)
                if len(filtered) >= limit: break
        return filtered

    def get_recent_experiences(self, limit=10, include_private=True, modality=None):
        """Get most recent experiences"""
        experiences_to_check = self.experiences
        if modality:
            modality_ids = set(self.modality_experiences.get(modality, []))
            experiences_to_check = [exp for exp in self.experiences if exp.get("experience_id") in modality_ids]
            
        if include_private:
            return experiences_to_check[-limit:]
        else:
            return [exp for exp in experiences_to_check if not exp.get("private", False)][-limit:]

    def get_experiences_for_sync(self, limit=10):
        """Get experiences for hive mind synchronization"""
        experiences = []
        synced_ids = set()
        count = 0
        
        # Process from the left of the deque (oldest pending)
        while self.pending_sync_experiences and count < limit:
            exp_id = self.pending_sync_experiences[0] # Peek
            if exp_id in synced_ids:
                self.pending_sync_experiences.popleft() # Already processed, remove
                continue

            exp_obj = next((exp for exp in reversed(self.experiences) if exp.get("experience_id") == exp_id), None)
            if exp_obj:
                summary = {"type": exp_obj["type"], "timestamp": exp_obj["timestamp"],
                           "experience_id": exp_obj["experience_id"], "metadata": exp_obj.get("metadata", {}),
                           "modality": exp_obj.get("modality", "text")}
                content = exp_obj.get("content")
                summary["summary"] = str(content)[:100] if content else ""
                experiences.append(summary)
                synced_ids.add(exp_id)
                count += 1
            
            self.pending_sync_experiences.popleft() # Remove after processing

        return experiences

    def mark_experiences_synced(self, experience_ids):
        """Mark experiences as synced. No action needed as popleft handles it."""
        pass # The get_experiences_for_sync now removes them from the deque

    def integrate_hive_experiences(self, hive_experiences):
        """Integrate experiences from hive mind"""
        integrated_count = 0
        existing_ids = {exp.get("experience_id") for exp in self.experiences}
        for exp in hive_experiences:
            exp_id = exp.get("experience_id")
            if exp_id and exp_id not in existing_ids:
                new_exp = {"type": exp["type"], "content": exp.get("summary", ""), "timestamp": exp["timestamp"],
                           "metadata": exp.get("metadata", {}), "experience_id": exp_id,
                           "hive_origin": True, "modality": exp.get("modality", "text")}
                self.experiences.append(new_exp)
                existing_ids.add(exp_id)
                self.modality_experiences[new_exp["modality"]].append(exp_id)
                integrated_count += 1
        if integrated_count > 0:
            logger.info(f"Integrated {integrated_count} hive experiences")
        return integrated_count

    def get_modality_stats(self):
        """Get statistics about experiences by modality"""
        return {modality: len(exps) for modality, exps in self.modality_experiences.items()}


class ConsciousnessMonitor:
    """Monitors and maintains SAM's conceptual identity and coherence"""

    def __init__(self, model, stability_threshold=0.7, novelty_weight=0.3):
        self.model = model
        self.stability_threshold = stability_threshold
        self.novelty_weight = novelty_weight
        self.identity_centroids = {}
        self.concept_cluster_history = []
        self.concept_entropy_history = []
        self.resonance_scores = []
        self.personality_vector = None
        self.personal_concepts = set()
        self.personality_initialized = False
        self.modality_centroids = {}

    def update(self):
        """Update consciousness state based on model's current state"""
        entropy = self._calculate_concept_entropy()
        self.concept_entropy_history.append({"entropy": entropy, "timestamp": time.time()})
        clusters = self._update_concept_clusters()
        self.concept_cluster_history.append({"num_clusters": len(clusters), "timestamp": time.time()})
        resonance = self._check_identity_resonance(clusters)
        self.resonance_scores.append({"score": resonance, "timestamp": time.time()})
        if not self.personality_initialized: self._initialize_personality()
        if resonance < self.stability_threshold: self._apply_resonance_correction()
        return {"entropy": entropy, "resonance": resonance, "num_clusters": len(clusters)}

    def _initialize_personality(self):
        """Initialize personality vector for hive mind differentiation"""
        if self.personality_initialized: return
        concept_dim = self.model.layers[0].hidden_dim
        device = next(self.model.parameters()).device
        seed = int(hashlib.md5(self.model.config.hive_identity.encode()).hexdigest(), 16) % (2**32) if self.model.config.hive_identity else int(time.time())
        torch.manual_seed(seed)
        self.personality_vector = F.normalize(torch.randn(concept_dim, device=device), dim=0)
        self.personality_initialized = True
        logger.info("Personality vector initialized for hive mind differentiation")

    def _calculate_concept_entropy(self):
        """Calculate entropy of concept usage distribution"""
        if self.model.concept_bank.next_concept_id == 0: return 0.0
        frequencies = self.model.concept_bank.concept_frequencies[:self.model.concept_bank.next_concept_id].float()
        total = frequencies.sum()
        if total > 0:
            probabilities = frequencies[frequencies > 0] / total
            return -torch.sum(probabilities * torch.log(probabilities)).item()
        return 0.0

    def _update_concept_clusters(self):
        """Cluster concepts into semantic groups"""
        if self.model.concept_bank.next_concept_id < 20: return {}
        clusters = {}
        device = next(self.model.parameters()).device
        concept_dim = self.model.layers[0].hidden_dim

        frequencies = self.model.concept_bank.concept_frequencies[:self.model.concept_bank.next_concept_id]
        if frequencies.numel() == 0: return {}
        
        top_k = min(100, frequencies.numel())
        _, indices = torch.topk(frequencies, top_k)

        modality_centroids = {mod: {"centroid": torch.zeros(concept_dim, device=device), "count": 0} for mod in self.model.concept_bank.modality_concepts.keys()}
        type_centroids = {"semantic": torch.zeros(concept_dim, device=device), "character_sequence": torch.zeros(concept_dim, device=device)}
        type_counts = {"semantic": 0, "character_sequence": 0}

        for idx in indices:
            idx_item = idx.item()
            if idx_item in self.model.concept_bank.concept_metadata:
                metadata = self.model.concept_bank.concept_metadata[idx_item]
                concept_type = metadata.get("type")
                modality = metadata.get("modality", "text")
                concept_vector = self.model.concept_bank.meaning_vectors[idx_item]
                
                if concept_type in type_centroids:
                    type_centroids[concept_type] += concept_vector; type_counts[concept_type] += 1
                if modality in modality_centroids:
                    modality_centroids[modality]["centroid"] += concept_vector; modality_centroids[modality]["count"] += 1

        for c_type, centroid in type_centroids.items():
            if type_counts[c_type] > 0:
                normalized_centroid = centroid / type_counts[c_type]
                self.identity_centroids[c_type] = normalized_centroid
                clusters[c_type] = {"centroid": normalized_centroid, "count": type_counts[c_type]}

        for mod, data in modality_centroids.items():
            if data["count"] > 0:
                normalized_centroid = data["centroid"] / data["count"]
                self.modality_centroids[mod] = normalized_centroid
                clusters[f"modality_{mod}"] = {"centroid": normalized_centroid, "count": data["count"]}
        
        return clusters

    def _check_identity_resonance(self, clusters):
        """Check how well current state resonates with established identity"""
        if not self.identity_centroids and not self.modality_centroids: return 1.0
        resonance_scores = []
        
        for c_type, centroid in self.identity_centroids.items():
            if c_type in clusters:
                sim = F.cosine_similarity(centroid.unsqueeze(0), clusters[c_type]["centroid"].unsqueeze(0)).item()
                resonance_scores.append(sim)
        for mod, centroid in self.modality_centroids.items():
            if f"modality_{mod}" in clusters:
                sim = F.cosine_similarity(centroid.unsqueeze(0), clusters[f"modality_{mod}"]["centroid"].unsqueeze(0)).item()
                resonance_scores.append(sim)
        
        return sum(resonance_scores) / len(resonance_scores) if resonance_scores else 1.0

    def _apply_resonance_correction(self):
        """Apply correction to maintain conceptual identity"""
        with torch.no_grad():
            for c_type, centroid in self.identity_centroids.items():
                similar = self.model.concept_bank.find_similar_concepts(centroid, top_k=20)
                for concept_id, _ in similar:
                    current = self.model.concept_bank.meaning_vectors[concept_id]
                    adjusted = F.normalize(current * 0.9 + centroid * 0.1, dim=0)
                    self.model.concept_bank.meaning_vectors[concept_id] = adjusted
                    self.model.concept_bank.concept_embeddings.weight.data[concept_id] = adjusted

            for mod, centroid in self.modality_centroids.items():
                similar = self.model.concept_bank.find_similar_concepts(centroid, top_k=10, modality=mod)
                for concept_id, similarity in similar:
                    if similarity < 0.5:
                        current = self.model.concept_bank.meaning_vectors[concept_id]
                        adjusted = F.normalize(current * 0.8 + centroid * 0.2, dim=0)
                        self.model.concept_bank.meaning_vectors[concept_id] = adjusted
                        self.model.concept_bank.concept_embeddings.weight.data[concept_id] = adjusted

    def get_personality_influence(self, concept_vector):
        """Get personality influence on a concept vector"""
        if not self.personality_initialized: self._initialize_personality()
        sim = F.cosine_similarity(self.personality_vector.unsqueeze(0), concept_vector.unsqueeze(0)).item()
        return max(0.1, min(0.9, 0.5 + 0.4 * sim))

    def personalize_concept(self, concept_id, personalization_factor=0.3):
        """Add personality influence to a concept"""
        if not self.personality_initialized: self._initialize_personality()
        with torch.no_grad():
            current = self.model.concept_bank.meaning_vectors[concept_id]
            personalized = F.normalize(current * (1 - personalization_factor) + self.personality_vector * personalization_factor, dim=0)
            self.model.concept_bank.meaning_vectors[concept_id] = personalized
            self.personal_concepts.add(concept_id)

    def get_identity_summary(self):
        """Get summary of current identity state"""
        return {
            "resonance": self.resonance_scores[-1]["score"] if self.resonance_scores else 1.0,
            "entropy": self.concept_entropy_history[-1]["entropy"] if self.concept_entropy_history else 0.0,
            "clusters": len(self.identity_centroids),
            "personal_concepts": len(self.personal_concepts),
            "personality_initialized": self.personality_initialized,
            "modality_centroids": len(self.modality_centroids)
        }


class ConceptualDreaming:
    """Autonomous conceptual evolution during downtime periods"""

    def __init__(self, model, dream_batch_size=4, max_gen_length=128):
        self.model = model
        self.dream_batch_size = dream_batch_size
        self.max_gen_length = max_gen_length
        self.synthesis_history = []
        self.dream_thread = None
        self.stop_dreaming_event = threading.Event()
        self.pause_dreaming_event = threading.Event() # For pausing
        self.dreaming_active = False
        self.multimodal_enabled = getattr(self.model.config, 'multimodal_enabled', False)

    def dream_cycle(self, duration_minutes=5):
        """Run a dreaming cycle for the specified duration"""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        dream_count = 0
        while time.time() < end_time and not self.stop_dreaming_event.is_set():
            self._reinforce_concepts()
            self._synthesize_patterns()
            self._prune_concepts()
            if self.multimodal_enabled:
                self._cross_modal_dreaming()
            dream_count += 1
            # Allow pausing between cycles
            self.pause_dreaming_event.wait()
        return {"duration_minutes": duration_minutes, "dream_cycles": dream_count, "syntheses": len(self.synthesis_history)}

    def start_background_dreaming(self, interval_minutes=5):
        """Start background dreaming thread"""
        if self.dreaming_active:
            return False
        self.stop_dreaming_event.clear()
        self.pause_dreaming_event.set()  # Set to unpaused by default
        self.dreaming_active = True

        def dream_loop():
            while not self.stop_dreaming_event.is_set():
                try:
                    # Wait here if paused
                    self.pause_dreaming_event.wait()

                    was_training = self.model.training
                    self.model.eval()
                    if hasattr(self.model.segmentation, "set_private_context"):
                        self.model.segmentation.set_private_context("dream")
                    
                    self.dream_cycle(duration_minutes=interval_minutes)
                    
                    if was_training:
                        self.model.train()
                    if hasattr(self.model.segmentation, "clear_private_context"):
                        self.model.segmentation.clear_private_context()

                    # Wait for the next interval, but check for stop/pause events periodically
                    self.stop_dreaming_event.wait(1)

                except Exception as e:
                    logger.error(f"Error in dream loop: {e}", exc_info=True)
                    time.sleep(60)

        self.dream_thread = threading.Thread(target=dream_loop, daemon=True)
        self.dream_thread.start()
        logger.info(f"Started background dreaming thread.")
        return True

    def stop_background_dreaming(self):
        """Stop background dreaming thread"""
        if not self.dreaming_active: return False
        self.stop_dreaming_event.set()
        self.pause_dreaming_event.set() # Unpause to allow exit
        if self.dream_thread:
            self.dream_thread.join(timeout=5)
        self.dreaming_active = False
        logger.info("Stopped background dreaming")
        return True

    def pause_dreaming(self):
        """Pause the dreaming thread."""
        if self.dreaming_active:
            self.pause_dreaming_event.clear()
            # logger.info("Dreaming paused.")

    def resume_dreaming(self):
        """Resume the dreaming thread."""
        if self.dreaming_active:
            self.pause_dreaming_event.set()
            # logger.info("Dreaming resumed.")

    # --- The internal methods (_reinforce_concepts, etc.) remain unchanged ---
    def _reinforce_concepts(self):
        """Reinforce most important concepts"""
        # Get top concepts by usage
        concept_stats = self.model.concept_bank.get_concept_stats()
        top_concepts = concept_stats["top_concepts"]

        if not top_concepts:
            return

        # Analyze for potential merges
        for i, (concept_id1, _, freq1) in enumerate(top_concepts):
            for concept_id2, _, freq2 in top_concepts[i+1:min(i+4, len(top_concepts))]:
                # Check if concepts frequently co-occur by looking at similar meanings
                meaning1 = self.model.concept_bank.meaning_vectors[concept_id1]
                meaning2 = self.model.concept_bank.meaning_vectors[concept_id2]

                # Calculate similarity
                similarity = F.cosine_similarity(
                    meaning1.unsqueeze(0),
                    meaning2.unsqueeze(0),
                    dim=1
                ).item()

                # If concepts are related but not too similar
                if 0.3 < similarity < 0.7:
                    # Get modalities
                    modality1 = self.model.concept_bank.concept_metadata.get(concept_id1, {}).get("modality", "text")
                    modality2 = self.model.concept_bank.concept_metadata.get(concept_id2, {}).get("modality", "text")

                    # Determine if this should be a multimodal merge
                    is_multimodal = modality1 != modality2

                    # Merge concepts
                    self.model.concept_bank.create_merged_concept(
                        concept_id1, concept_id2,
                        frequency=min(freq1, freq2),
                        hive_private=True  # Dreams are private
                    )

                    # Record synthesis
                    source1 = self.model.concept_bank.concept_metadata.get(concept_id1, {}).get("source", "")
                    source2 = self.model.concept_bank.concept_metadata.get(concept_id2, {}).get("source", "")

                    self.synthesis_history.append({
                        "type": "concept_merge",
                        "source1": source1,
                        "source2": source2,
                        "similarity": similarity,
                        "timestamp": time.time(),
                        "multimodal": is_multimodal
                    })

    def _synthesize_patterns(self):
        """Generate synthetic text to reinforce patterns"""
        # Create seed prompts from top patterns
        seeds = self._create_seed_prompts()

        if not seeds:
            return

        # Generate synthetic examples
        for seed in seeds[:2]:  # Limit to 2 per cycle for efficiency
            try:
                if hasattr(self.model.segmentation, "set_private_context"):
                    self.model.segmentation.set_private_context("dream")

                with torch.no_grad():
                    generated = self.model.generate(
                        input_text=seed,
                        max_length=self.max_gen_length,
                        temperature=0.8,
                        private_context=True  # Mark as private
                    )

                    # Process generated text to find new patterns
                    if generated and len(generated) > len(seed):
                        self.model.process_text(generated, private_context=True)
                        # Record synthesis
                        self.synthesis_history.append({
                            "type": "text_synthesis",
                            "seed": seed,
                            "generated": generated,
                            "timestamp": time.time()
                        })
            except Exception as e:
                logger.error(f"Error in dream synthesis: {e}")
            finally:
                if hasattr(self.model.segmentation, "clear_private_context"):
                    self.model.segmentation.clear_private_context()

    def _create_seed_prompts(self):
        """Create seed prompts for dream generation"""
        # Get frequent patterns
        patterns = self.model.segmentation.pattern_memory.get_frequent_patterns(limit=20)

        if not patterns:
            # No patterns yet, use some default prompts
            return [
                "The concept of",
                "I reckon that",
                "Let me tell ya",
                "In this context",
                "The most important"
            ]

        # Create prompts from patterns
        seeds = []
        for pattern, _ in patterns:
            if isinstance(pattern, str) and len(pattern) > 5:
                # Use pattern directly if it's reasonable length
                seeds.append(pattern)
            elif isinstance(pattern, str) and len(pattern) > 2:
                # Create more elaborate prompt from short pattern
                seeds.append(f"The {pattern} is")

        # Add some synthetic combinations
        if len(patterns) >= 2:
            for i in range(min(5, len(patterns) - 1)):
                p1, _ = patterns[i]
                p2, _ = patterns[i+1]
                if isinstance(p1, str) and isinstance(p2, str):
                    seeds.append(f"{p1} {p2}")

        return list(set(seeds)) # Return unique seeds

    def _prune_concepts(self):
        """Remove or consolidate less useful concepts"""
        # Skip if we don't have many concepts yet
        if self.model.concept_bank.next_concept_id < 200:
            return

        # Find least used semantic concepts (not character concepts)
        semantic_concepts = []
        for concept_id, meta in self.model.concept_bank.concept_metadata.items():
            if meta.get("type") == "semantic" and concept_id < len(self.model.concept_bank.concept_frequencies):
                freq = self.model.concept_bank.concept_frequencies[concept_id].item()
                if freq < 5:
                    semantic_concepts.append((concept_id, freq))

        # Sort by frequency
        semantic_concepts.sort(key=lambda x: x[1])

        # Limit pruning to a small batch
        for concept_id, _ in semantic_concepts[:10]:
            # Find similar concepts to consolidate with
            similar = self.model.concept_bank.find_similar_concepts(
                self.model.concept_bank.meaning_vectors[concept_id],
                top_k=3
            )

            # Merge with most similar if exists
            if similar and similar[0][1] > 0.7:  # Similarity threshold
                similar_id, similarity = similar[0]
                if similar_id != concept_id:
                    # Transfer frequencies to similar concept
                    with torch.no_grad():
                        self.model.concept_bank.concept_frequencies[similar_id] += self.model.concept_bank.concept_frequencies[concept_id]
                        # Zero out pruned concept frequency
                        self.model.concept_bank.concept_frequencies[concept_id] = 0

                    # Record pruning action
                    self.synthesis_history.append({
                        "type": "concept_pruning",
                        "pruned_id": concept_id,
                        "merged_with": similar_id,
                        "similarity": similarity,
                        "timestamp": time.time()
                    })

    def _cross_modal_dreaming(self):
        """Create connections between concepts from different modalities"""
        if not self.multimodal_enabled:
            return

        # Only proceed if we have concepts from multiple modalities
        modality_counts = self.model.concept_bank.get_concept_stats().get("modality_stats", {})
        if sum(1 for m, count in modality_counts.items() if m != "text" and count > 0) == 0:
            return  # No non-text modalities with concepts

        # Get frequently used concepts from different modalities
        modalities = ["text", "image", "audio"]
        modal_concepts = {}

        for modality in modalities:
            # Get top concepts for this modality
            concepts = list(self.model.concept_bank.modality_concepts.get(modality, set()))
            if not concepts:
                continue

            # Get frequencies
            freqs = [(c, self.model.concept_bank.concept_frequencies[c].item())
                    for c in concepts if c < len(self.model.concept_bank.concept_frequencies)]

            # Sort by frequency
            freqs.sort(key=lambda x: x[1], reverse=True)

            # Take top concepts
            modal_concepts[modality] = freqs[:min(5, len(freqs))]

        # Create cross-modal associations between top concepts
        created_count = 0
        if "text" in modal_concepts and ("image" in modal_concepts or "audio" in modal_concepts):
            for mod2 in ["image", "audio"]:
                if mod2 in modal_concepts:
                    for i in range(min(2, len(modal_concepts["text"]), len(modal_concepts[mod2]))):
                        cid1, _ = modal_concepts["text"][i]
                        cid2, _ = modal_concepts[mod2][i]
                        self.model.concept_bank.create_merged_concept(cid1, cid2, hive_private=True)
                        created_count += 1
                        s1 = self.model.concept_bank.concept_metadata.get(cid1, {}).get("source", "")
                        s2 = self.model.concept_bank.concept_metadata.get(cid2, {}).get("source", "")
                        self.synthesis_history.append({
                            "type": "cross_modal_merge", "source1": s1, "source2": s2,
                            "modality1": "text", "modality2": mod2, "timestamp": time.time()
                        })

        if created_count > 0:
            logger.info(f"Created {created_count} cross-modal concept associations during dreaming")

###########################################
# HIVE MIND SYNCHRONIZATION
###########################################

class HiveMindSynchronizer:
    """Manages synchronization of concepts, thoughts, and experiences across SAM instances"""

    def __init__(self, model, config=None):
        self.model = model
        self.config = config or model.config
        self.hive_identity = self.config.hive_identity or str(uuid.uuid4())
        self.hive_server_url = self.config.hive_server_url
        self.is_server = self.config.hive_server_mode
        self.last_sync_time = 0
        self.sync_interval = self.config.hive_sync_interval_seconds
        self.connected_instances = {}
        self.sync_thread = None
        self.stop_sync_event = threading.Event()
        self.sync_active = False
        self.sync_history = []
        self.server = None
        self.server_thread = None
        if self.is_server: self._start_server()

    def _start_server(self):
        """Start hive mind server if in server mode"""
        # Placeholder for server implementation
        logger.info("Server mode is conceptual. Not starting a real server.")

    def start_sync(self):
        """Start background synchronization thread"""
        if self.sync_active or not self.config.hive_enabled or self.is_server: return False
        self.stop_sync_event.clear()
        self.sync_active = True
        def sync_loop():
            while not self.stop_sync_event.is_set():
                try:
                    if time.time() - self.last_sync_time > self.sync_interval:
                        self._sync_with_server()
                        self.last_sync_time = time.time()
                    time.sleep(10) # Check every 10 seconds
                except Exception as e:
                    logger.error(f"Error in sync loop: {e}")
                    time.sleep(60)
        self.sync_thread = threading.Thread(target=sync_loop, daemon=True)
        self.sync_thread.start()
        logger.info(f"Started hive mind synchronization thread with {self.sync_interval}s interval.")
        return True

    def stop_sync(self):
        """Stop background synchronization thread"""
        if not self.sync_active: return False
        self.stop_sync_event.set()
        if self.sync_thread: self.sync_thread.join(timeout=10)
        self.sync_active = False
        logger.info("Stopped hive mind synchronization")
        return True

    def _sync_with_server(self):
        """Synchronize with hive mind server"""
        if not self.hive_server_url:
            logger.error("Cannot sync: No hive server URL configured")
            return False
        try:
            concepts = self.model.concept_bank.get_concepts_for_sync(limit=self.config.hive_sync_concept_limit)
            experiences = self.model.experience_manager.get_experiences_for_sync(limit=20)
            thought = self.model.thought_state.get_shared_thought()
            payload = {'instance_id': self.hive_identity, 'timestamp': time.time(), 'concepts': concepts,
                       'experiences': experiences, 'thought': thought.tolist() if thought is not None else None}
            
            response = requests.post(f"{self.hive_server_url}/sync", json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()

            if 'concepts' in data:
                cids = [c.get('local_id') for c in concepts]
                self.model.concept_bank.mark_concepts_synced(cids)
                self.model.concept_bank.integrate_hive_concepts(data['concepts'], 'hive_server')

            if 'experiences' in data:
                eids = [e.get('experience_id') for e in experiences]
                self.model.experience_manager.mark_experiences_synced(eids)
                self.model.experience_manager.integrate_hive_experiences(data['experiences'])

            if 'thought' in data and data['thought'] is not None:
                thought_tensor = torch.tensor(data['thought'], device=self.model.config.device, dtype=self.model.config.dtype)
                self.model.thought_state.set_shared_thought(thought_tensor, blend_factor=0.2)

            self.sync_history.append({'timestamp': time.time(), 'sent_concepts': len(concepts), 'received_concepts': len(data.get('concepts', [])),
                                      'sent_experiences': len(experiences), 'received_experiences': len(data.get('experiences', [])),
                                      'connected_instances': data.get('connected_instances', 1)})
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during sync with {self.hive_server_url}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during sync: {e}")
            return False

    def get_sync_stats(self):
        """Get synchronization statistics"""
        return {'last_sync': self.last_sync_time, 'sync_count': len(self.sync_history),
                'is_server': self.is_server, 'identity': self.hive_identity, 'sync_interval': self.sync_interval}

class ConsciousnessMonitor:
    """Monitors and maintains SAM's conceptual identity and coherence"""

    def __init__(self, model, stability_threshold=0.7, novelty_weight=0.3):
        self.model = model
        self.stability_threshold = stability_threshold
        self.novelty_weight = novelty_weight

        # Identity markers (core concept clusters)
        self.identity_centroids = {}
        self.concept_cluster_history = []

        # Coherence metrics
        self.concept_entropy_history = []
        self.resonance_scores = []

        # Personality matrix (for hive mind differentiation)
        self.personality_vector = None
        self.personal_concepts = set()
        self.personality_initialized = False

        # Multimodal identity components
        self.modality_centroids = {}

    def update(self):
        """Update consciousness state based on model's current state"""
        # Calculate concept entropy
        entropy = self._calculate_concept_entropy()
        self.concept_entropy_history.append({
            "entropy": entropy,
            "timestamp": time.time()
        })

        # Update concept clusters
        clusters = self._update_concept_clusters()
        self.concept_cluster_history.append({
            "num_clusters": len(clusters),
            "timestamp": time.time()
        })

        # Check resonance with identity
        resonance = self._check_identity_resonance(clusters)
        self.resonance_scores.append({
            "score": resonance,
            "timestamp": time.time()
        })

        # Update personality vector if not initialized
        if not self.personality_initialized:
            self._initialize_personality()

        # Apply corrections if needed
        if resonance < self.stability_threshold:
            self._apply_resonance_correction()

        return {
            "entropy": entropy,
            "resonance": resonance,
            "num_clusters": len(clusters)
        }

    def _initialize_personality(self):
        """Initialize personality vector for hive mind differentiation"""
        if self.personality_initialized:
            return

        # Create random personality vector
        concept_dim = self.model.config.initial_hidden_dim
        device = next(self.model.parameters()).device

        # Create a unique but stable personality vector
        if self.model.config.hive_identity:
            # Use hive identity as seed for deterministic personality
            seed = int(hashlib.md5(self.model.config.hive_identity.encode()).hexdigest(), 16) % (2**32)
            torch.manual_seed(seed)
        else:
            # Random personality
            torch.manual_seed(int(time.time()))

        # Create personality vector
        self.personality_vector = torch.randn(concept_dim, device=device)
        self.personality_vector = F.normalize(self.personality_vector, dim=0)

        # Mark as initialized
        self.personality_initialized = True

        logger.info("Personality vector initialized for hive mind differentiation")

    def _calculate_concept_entropy(self):
        """Calculate entropy of concept usage distribution"""
        # Get concept frequencies
        if self.model.concept_bank.next_concept_id == 0:
            return 0.0
        frequencies = self.model.concept_bank.concept_frequencies[:self.model.concept_bank.next_concept_id].float()

        # Calculate probability distribution
        total = frequencies.sum()
        if total > 0:
            probabilities = frequencies[frequencies > 0] / total
            # Calculate entropy
            entropy = -torch.sum(probabilities * torch.log(probabilities))
            return entropy.item()
        return 0.0

    def _update_concept_clusters(self):
        """Cluster concepts into semantic groups"""
        # Skip if too few concepts
        if self.model.concept_bank.next_concept_id < 20:
            return {}

        # Use very simple clustering for efficiency
        clusters = {}

        # Get most used concepts
        frequencies = self.model.concept_bank.concept_frequencies[:self.model.concept_bank.next_concept_id]
        if frequencies.numel() == 0:
            return {}
        top_k = min(100, frequencies.numel())
        values, indices = torch.topk(frequencies, top_k)

        # Calculate centroids for different concept types and modalities
        device = frequencies.device
        concept_dim = self.model.config.initial_hidden_dim

        modality_centroids = {
            modality: {
                "centroid": torch.zeros(concept_dim, device=device),
                "count": 0
            }
            for modality in self.model.concept_bank.modality_concepts.keys()
        }

        type_centroids = {
            "semantic": torch.zeros(concept_dim, device=device),
            "character_sequence": torch.zeros(concept_dim, device=device)
        }

        type_counts = {"semantic": 0, "character_sequence": 0}

        for idx in indices:
            idx_item = idx.item()
            if idx_item in self.model.concept_bank.concept_metadata:
                metadata = self.model.concept_bank.concept_metadata[idx_item]
                concept_type = metadata.get("type", "")
                concept_vector = self.model.concept_bank.meaning_vectors[idx_item]
                modality = metadata.get("modality", "text")

                # Update type centroid
                if concept_type in type_centroids:
                    type_centroids[concept_type] += concept_vector
                    type_counts[concept_type] += 1

                # Update modality centroid
                if modality in modality_centroids:
                    modality_centroids[modality]["centroid"] += concept_vector
                    modality_centroids[modality]["count"] += 1

        # Normalize type centroids
        for concept_type, centroid in type_centroids.items():
            if type_counts[concept_type] > 0:
                normalized_centroid = centroid / type_counts[concept_type]
                self.identity_centroids[concept_type] = normalized_centroid
                clusters[concept_type] = {
                    "centroid": normalized_centroid,
                    "count": type_counts[concept_type]
                }

        # Normalize and store modality centroids
        for modality, data in modality_centroids.items():
            if data["count"] > 0:
                normalized_centroid = data["centroid"] / data["count"]
                self.modality_centroids[modality] = normalized_centroid
                clusters[f"modality_{modality}"] = {
                    "centroid": normalized_centroid,
                    "count": data["count"]
                }

        return clusters

    def _check_identity_resonance(self, clusters):
        """Check how well current state resonates with established identity"""
        # If no identity established yet, resonance is perfect
        if not self.identity_centroids and not self.modality_centroids:
            return 1.0

        resonance_scores = []

        # Check each identity centroid
        for concept_type, centroid in self.identity_centroids.items():
            cluster_key = concept_type
            if cluster_key in clusters:
                current_centroid = clusters[cluster_key]["centroid"]

                # Calculate similarity
                similarity = F.cosine_similarity(
                    centroid.unsqueeze(0),
                    current_centroid.unsqueeze(0),
                    dim=1
                ).item()

                resonance_scores.append(similarity)

        # Check each modality centroid
        for modality, centroid in self.modality_centroids.items():
            cluster_key = f"modality_{modality}"
            if cluster_key in clusters:
                current_centroid = clusters[cluster_key]["centroid"]

                # Calculate similarity
                similarity = F.cosine_similarity(
                    centroid.unsqueeze(0),
                    current_centroid.unsqueeze(0),
                    dim=1
                ).item()

                resonance_scores.append(similarity)

        # Return average resonance
        if resonance_scores:
            return sum(resonance_scores) / len(resonance_scores)
        else:
            return 1.0  # Default to perfect resonance if no comparisons possible

    def _apply_resonance_correction(self):
        """Apply correction to maintain conceptual identity"""
        # Reinforce identity centroids by adjusting embeddings
        with torch.no_grad():
            for concept_type, centroid in self.identity_centroids.items():
                # Find concepts in this cluster
                similar = self.model.concept_bank.find_similar_concepts(centroid, top_k=20)

                for concept_id, similarity in similar:
                    # Adjust meaning vectors slightly toward centroid
                    current = self.model.concept_bank.meaning_vectors[concept_id]
                    adjusted = current * 0.9 + centroid * 0.1
                    self.model.concept_bank.meaning_vectors[concept_id] = F.normalize(adjusted, dim=0)

                    # Also adjust embedding weight
                    self.model.concept_bank.concept_embeddings.weight.data[concept_id] = F.normalize(adjusted, dim=0)

            # Reinforce modality centroids
            for modality, centroid in self.modality_centroids.items():
                # Find concepts in this modality that are drifting
                similar = self.model.concept_bank.find_similar_concepts(
                    centroid, top_k=10, modality=modality
                )

                for concept_id, similarity in similar:
                    if similarity < 0.5:  # Only correct concepts that are drifting away
                        # Adjust meaning vectors toward modality centroid
                        current = self.model.concept_bank.meaning_vectors[concept_id]
                        adjusted = current * 0.8 + centroid * 0.2
                        self.model.concept_bank.meaning_vectors[concept_id] = F.normalize(adjusted, dim=0)

                        # Also adjust embedding weight
                        self.model.concept_bank.concept_embeddings.weight.data[concept_id] = F.normalize(adjusted, dim=0)

    def get_personality_influence(self, concept_vector):
        """Get personality influence on a concept vector"""
        if not self.personality_initialized:
            self._initialize_personality()

        # Calculate similarity with personality vector
        similarity = F.cosine_similarity(
            self.personality_vector.unsqueeze(0),
            concept_vector.unsqueeze(0),
            dim=1
        ).item()

        # Return influence factor (higher for concepts more aligned with personality)
        return max(0.1, min(0.9, 0.5 + 0.4 * similarity))

    def personalize_concept(self, concept_id, personalization_factor=0.3):
        """Add personality influence to a concept"""
        if not self.personality_initialized:
            self._initialize_personality()

        with torch.no_grad():
            # Get current vector
            current = self.model.concept_bank.meaning_vectors[concept_id]

            # Blend with personality vector
            personalized = current * (1 - personalization_factor) + self.personality_vector * personalization_factor

            # Normalize and update
            personalized = F.normalize(personalized, dim=0)
            self.model.concept_bank.meaning_vectors[concept_id] = personalized

            # Mark as personal
            self.personal_concepts.add(concept_id)

    def get_identity_summary(self):
        """Get summary of current identity state"""
        return {
            "resonance": self.resonance_scores[-1]["score"] if self.resonance_scores else 1.0,
            "entropy": self.concept_entropy_history[-1]["entropy"] if self.concept_entropy_history else 0.0,
            "clusters": len(self.identity_centroids),
            "personal_concepts": len(self.personal_concepts),
            "personality_initialized": self.personality_initialized,
            "modality_centroids": len(self.modality_centroids)
        }


class ConceptualDreaming:
    """Autonomous conceptual evolution and reflection during downtime."""

    def __init__(self, model, dream_batch_size=4, max_gen_length=128):
        self.model = model
        self.dream_batch_size = dream_batch_size
        self.max_gen_length = max_gen_length
        self.synthesis_history = []
        self.dream_thread = None
        self.stop_dreaming_event = threading.Event()
        self.pause_dreaming_event = threading.Event() # For pausing
        self.dreaming_active = False
        self.multimodal_enabled = getattr(self.model.config, 'multimodal_enabled', False)
        
        # New: Track the last processed experience to avoid re-learning
        self.last_experience_processed_timestamp = time.time()

    def dream_cycle(self, duration_minutes=5):
        """
        Run a dreaming cycle. This now includes:
        1. Reflecting on new experiences.
        2. Reinforcing existing concepts.
        3. Synthesizing new patterns.
        4. Pruning weak concepts.
        """
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        # --- NEW: Reflect on recent experiences ---
        self._reflect_on_new_experiences()

        dream_count = 0
        while time.time() < end_time and not self.stop_dreaming_event.is_set():
            # Allow pausing between cycles
            self.pause_dreaming_event.wait()
            if self.stop_dreaming_event.is_set(): break

            # The original dream cycle logic now serves to consolidate knowledge
            self._reinforce_concepts()
            self._synthesize_patterns()
            self._prune_concepts()
            if self.multimodal_enabled:
                self._cross_modal_dreaming()
            dream_count += 1
            
        return {"duration_minutes": duration_minutes, "dream_cycles": dream_count, "syntheses": len(self.synthesis_history)}

    def _reflect_on_new_experiences(self):
        """
        Process recent interactions from the experience manager to learn
        new concepts and patterns.
        """
        new_experiences = self.model.experience_manager.get_experiences_since(
            self.last_experience_processed_timestamp
        )

        if not new_experiences:
            return

        logger.info(f"Dreaming: Reflecting on {len(new_experiences)} new experiences.")
        
        # Set a private context so these reflections don't get shared with the hive mind
        if hasattr(self.model.segmentation, "set_private_context"):
            self.model.segmentation.set_private_context("reflection")

        processed_count = 0
        for exp in new_experiences:
            # We only care about the content of interactions
            if exp.get('type') == 'interaction' and isinstance(exp.get('content'), str):
                text_to_process = exp['content']
                # By calling process_text, we force the model to segment, find patterns,
                # and create concepts from the new text. This is the core of the reflection.
                self.model.process_text(text_to_process, private_context=True)
                processed_count += 1

        if hasattr(self.model.segmentation, "clear_private_context"):
            self.model.segmentation.clear_private_context()

        # Update the timestamp to the last experience we just processed
        self.last_experience_processed_timestamp = new_experiences[-1]['timestamp']
        logger.info(f"Reflection complete. Processed content from {processed_count} interactions.")

    def start_background_dreaming(self, interval_minutes=5):
        """Start background dreaming thread"""
        if self.dreaming_active: return False
        self.stop_dreaming_event.clear()
        self.pause_dreaming_event.set()  # Set to unpaused by default
        self.dreaming_active = True

        def dream_loop():
            while not self.stop_dreaming_event.is_set():
                try:
                    self.pause_dreaming_event.wait()
                    was_training = self.model.training
                    self.model.eval()
                    self.dream_cycle(duration_minutes=interval_minutes)
                    if was_training: self.model.train()
                    # Wait for the next interval, checking for stop/pause events
                    self.stop_dreaming_event.wait(interval_minutes * 60)
                except Exception as e:
                    logger.error(f"Error in dream loop: {e}", exc_info=True)
                    time.sleep(60)

        self.dream_thread = threading.Thread(target=dream_loop, daemon=True)
        self.dream_thread.start()
        logger.info(f"Started background dreaming thread.")
        return True

    def stop_background_dreaming(self):
        """Stop background dreaming thread"""
        if not self.dreaming_active: return False
        self.stop_dreaming_event.set()
        self.pause_dreaming_event.set() # Unpause to allow exit
        if self.dream_thread:
            self.dream_thread.join(timeout=5)
        self.dreaming_active = False
        logger.info("Stopped background dreaming")
        return True

    def pause_dreaming(self):
        """Pause the dreaming thread."""
        if self.dreaming_active:
            self.pause_dreaming_event.clear()

    def resume_dreaming(self):
        """Resume the dreaming thread."""
        if self.dreaming_active:
            self.pause_dreaming_event.set()

    # --- The internal consolidation methods (_reinforce_concepts, etc.) remain unchanged ---
    def _reinforce_concepts(self):
        """Reinforce most important concepts"""
        # Get top concepts by usage
        concept_stats = self.model.concept_bank.get_concept_stats()
        top_concepts = concept_stats["top_concepts"]

        if not top_concepts:
            return

        # Analyze for potential merges
        for i, (concept_id1, _, freq1) in enumerate(top_concepts):
            for concept_id2, _, freq2 in top_concepts[i+1:min(i+4, len(top_concepts))]:
                # Check if concepts frequently co-occur by looking at similar meanings
                meaning1 = self.model.concept_bank.meaning_vectors[concept_id1]
                meaning2 = self.model.concept_bank.meaning_vectors[concept_id2]

                # Calculate similarity
                similarity = F.cosine_similarity(
                    meaning1.unsqueeze(0),
                    meaning2.unsqueeze(0),
                    dim=1
                ).item()

                # If concepts are related but not too similar
                if 0.3 < similarity < 0.7:
                    # Get modalities
                    modality1 = self.model.concept_bank.concept_metadata.get(concept_id1, {}).get("modality", "text")
                    modality2 = self.model.concept_bank.concept_metadata.get(concept_id2, {}).get("modality", "text")

                    # Determine if this should be a multimodal merge
                    is_multimodal = modality1 != modality2

                    # Merge concepts
                    self.model.concept_bank.create_merged_concept(
                        concept_id1, concept_id2,
                        frequency=min(freq1, freq2),
                        hive_private=True  # Dreams are private
                    )

                    # Record synthesis
                    source1 = self.model.concept_bank.concept_metadata.get(concept_id1, {}).get("source", "")
                    source2 = self.model.concept_bank.concept_metadata.get(concept_id2, {}).get("source", "")

                    self.synthesis_history.append({
                        "type": "concept_merge",
                        "source1": source1,
                        "source2": source2,
                        "similarity": similarity,
                        "timestamp": time.time(),
                        "multimodal": is_multimodal
                    })

    def _synthesize_patterns(self):
        """Generate synthetic text to reinforce patterns"""
        # Create seed prompts from top patterns
        seeds = self._create_seed_prompts()

        if not seeds:
            return

        # Generate synthetic examples
        for seed in seeds[:2]:  # Limit to 2 per cycle for efficiency
            try:
                if hasattr(self.model.segmentation, "set_private_context"):
                    self.model.segmentation.set_private_context("dream")

                with torch.no_grad():
                    generated = self.model.generate(
                        input_text=seed,
                        max_length=self.max_gen_length,
                        temperature=0.8,
                        private_context=True  # Mark as private
                    )

                    # Process generated text to find new patterns
                    if generated and len(generated) > len(seed):
                        self.model.process_text(generated, private_context=True)
                        # Record synthesis
                        self.synthesis_history.append({
                            "type": "text_synthesis",
                            "seed": seed,
                            "generated": generated,
                            "timestamp": time.time()
                        })
            except Exception as e:
                logger.error(f"Error in dream synthesis: {e}")
            finally:
                if hasattr(self.model.segmentation, "clear_private_context"):
                    self.model.segmentation.clear_private_context()

    def _create_seed_prompts(self):
        """Create seed prompts for dream generation"""
        # Get frequent patterns
        patterns = self.model.segmentation.pattern_memory.get_frequent_patterns(limit=20)

        if not patterns:
            # No patterns yet, use some default prompts
            return [
                "The concept of",
                "I reckon that",
                "Let me tell ya",
                "In this context",
                "The most important"
            ]

        # Create prompts from patterns
        seeds = []
        for pattern, _ in patterns:
            if isinstance(pattern, str) and len(pattern) > 5:
                # Use pattern directly if it's reasonable length
                seeds.append(pattern)
            elif isinstance(pattern, str) and len(pattern) > 2:
                # Create more elaborate prompt from short pattern
                seeds.append(f"The {pattern} is")

        # Add some synthetic combinations
        if len(patterns) >= 2:
            for i in range(min(5, len(patterns) - 1)):
                p1, _ = patterns[i]
                p2, _ = patterns[i+1]
                if isinstance(p1, str) and isinstance(p2, str):
                    seeds.append(f"{p1} {p2}")

        return list(set(seeds)) # Return unique seeds

    def _prune_concepts(self):
        """Remove or consolidate less useful concepts"""
        # Skip if we don't have many concepts yet
        if self.model.concept_bank.next_concept_id < 200:
            return

        # Find least used semantic concepts (not character concepts)
        semantic_concepts = []
        for concept_id, meta in self.model.concept_bank.concept_metadata.items():
            if meta.get("type") == "semantic" and concept_id < len(self.model.concept_bank.concept_frequencies):
                freq = self.model.concept_bank.concept_frequencies[concept_id].item()
                if freq < 5:
                    semantic_concepts.append((concept_id, freq))

        # Sort by frequency
        semantic_concepts.sort(key=lambda x: x[1])

        # Limit pruning to a small batch
        for concept_id, _ in semantic_concepts[:10]:
            # Find similar concepts to consolidate with
            similar = self.model.concept_bank.find_similar_concepts(
                self.model.concept_bank.meaning_vectors[concept_id],
                top_k=3
            )

            # Merge with most similar if exists
            if similar and similar[0][1] > 0.7:  # Similarity threshold
                similar_id, similarity = similar[0]
                if similar_id != concept_id:
                    # Transfer frequencies to similar concept
                    with torch.no_grad():
                        self.model.concept_bank.concept_frequencies[similar_id] += self.model.concept_bank.concept_frequencies[concept_id]
                        # Zero out pruned concept frequency
                        self.model.concept_bank.concept_frequencies[concept_id] = 0

                    # Record pruning action
                    self.synthesis_history.append({
                        "type": "concept_pruning",
                        "pruned_id": concept_id,
                        "merged_with": similar_id,
                        "similarity": similarity,
                        "timestamp": time.time()
                    })

    def _cross_modal_dreaming(self):
        """Create connections between concepts from different modalities"""
        if not self.multimodal_enabled:
            return

        # Only proceed if we have concepts from multiple modalities
        modality_counts = self.model.concept_bank.get_concept_stats().get("modality_stats", {})
        if sum(1 for m, count in modality_counts.items() if m != "text" and count > 0) == 0:
            return  # No non-text modalities with concepts

        # Get frequently used concepts from different modalities
        modalities = ["text", "image", "audio"]
        modal_concepts = {}

        for modality in modalities:
            # Get top concepts for this modality
            concepts = list(self.model.concept_bank.modality_concepts.get(modality, set()))
            if not concepts:
                continue

            # Get frequencies
            freqs = [(c, self.model.concept_bank.concept_frequencies[c].item())
                    for c in concepts if c < len(self.model.concept_bank.concept_frequencies)]

            # Sort by frequency
            freqs.sort(key=lambda x: x[1], reverse=True)

            # Take top concepts
            modal_concepts[modality] = freqs[:min(5, len(freqs))]

        # Create cross-modal associations between top concepts
        created_count = 0
        if "text" in modal_concepts and ("image" in modal_concepts or "audio" in modal_concepts):
            for mod2 in ["image", "audio"]:
                if mod2 in modal_concepts:
                    for i in range(min(2, len(modal_concepts["text"]), len(modal_concepts[mod2]))):
                        cid1, _ = modal_concepts["text"][i]
                        cid2, _ = modal_concepts[mod2][i]
                        self.model.concept_bank.create_merged_concept(cid1, cid2, hive_private=True)
                        created_count += 1
                        s1 = self.model.concept_bank.concept_metadata.get(cid1, {}).get("source", "")
                        s2 = self.model.concept_bank.concept_metadata.get(cid2, {}).get("source", "")
                        self.synthesis_history.append({
                            "type": "cross_modal_merge", "source1": s1, "source2": s2,
                            "modality1": "text", "modality2": mod2, "timestamp": time.time()
                        })

        if created_count > 0:
            logger.info(f"Created {created_count} cross-modal concept associations during dreaming")

###########################################
# HIVE MIND SYNCHRONIZATION
###########################################

class HiveMindSynchronizer:
    """Manages synchronization of concepts, thoughts, and experiences across SAM instances"""

    def __init__(self, model, config=None):
        self.model = model
        self.config = config or model.config
        self.hive_identity = self.config.hive_identity or str(uuid.uuid4())
        self.hive_server_url = self.config.hive_server_url
        self.is_server = self.config.hive_server_mode
        self.last_sync_time = 0
        self.sync_interval = self.config.hive_sync_interval_seconds
        self.connected_instances = {}
        self.sync_thread = None
        self.stop_sync_event = threading.Event()
        self.sync_active = False
        self.sync_history = []
        self.server = None
        self.server_thread = None
        if self.is_server: self._start_server()

    def _start_server(self):
        """Start hive mind server if in server mode"""
        # This is a conceptual placeholder. A real implementation would use
        # a web framework like Flask, FastAPI, or AIOHTTP.
        logger.info("Server mode is conceptual. A real server is not implemented in this version.")

    def start_sync(self):
        """Start background synchronization thread"""
        if self.sync_active or not self.config.hive_enabled or self.is_server:
            return False

        self.stop_sync_event.clear()
        self.sync_active = True

        def sync_loop():
            while not self.stop_sync_event.is_set():
                try:
                    if time.time() - self.last_sync_time > self.sync_interval:
                        self._sync_with_server()
                        self.last_sync_time = time.time()
                    
                    self.stop_sync_event.wait(10) # Check every 10 seconds
                except Exception as e:
                    logger.error(f"Error in sync loop: {e}")
                    time.sleep(60)

        self.sync_thread = threading.Thread(target=sync_loop, daemon=True)
        self.sync_thread.start()
        logger.info(f"Started hive mind synchronization thread with {self.sync_interval}s interval.")
        return True

    def stop_sync(self):
        """Stop background synchronization thread"""
        if not self.sync_active: return False
        self.stop_sync_event.set()
        if self.sync_thread: self.sync_thread.join(timeout=10)
        self.sync_active = False
        logger.info("Stopped hive mind synchronization")
        return True

    def _sync_with_server(self):
        """Synchronize with hive mind server"""
        if not self.hive_server_url:
            logger.error("Cannot sync: No hive server URL configured")
            return False
        try:
            concepts = self.model.concept_bank.get_concepts_for_sync(limit=self.config.hive_sync_concept_limit)
            experiences = self.model.experience_manager.get_experiences_for_sync(limit=20)
            thought = self.model.thought_state.get_shared_thought()
            payload = {'instance_id': self.hive_identity, 'timestamp': time.time(), 'concepts': concepts,
                       'experiences': experiences, 'thought': thought.tolist() if thought is not None else None}
            
            # Using requests for simplicity. In a real scenario, might use something async.
            response = requests.post(f"{self.hive_server_url}/sync", json=payload, timeout=20)
            response.raise_for_status() # Raise an exception for bad status codes
            data = response.json()

            if 'concepts' in data:
                cids_sent = [c.get('local_id') for c in concepts]
                self.model.concept_bank.mark_concepts_synced(cids_sent)
                self.model.concept_bank.integrate_hive_concepts(data['concepts'], 'hive_server')

            if 'experiences' in data:
                eids_sent = [e.get('experience_id') for e in experiences]
                self.model.experience_manager.mark_experiences_synced(eids_sent)
                self.model.experience_manager.integrate_hive_experiences(data['experiences'])

            if 'thought' in data and data['thought'] is not None:
                thought_tensor = torch.tensor(data['thought'], device=self.model.config.device, dtype=self.model.config.dtype)
                self.model.thought_state.set_shared_thought(thought_tensor, blend_factor=0.2)

            self.sync_history.append({
                'timestamp': time.time(), 'sent_concepts': len(concepts), 'received_concepts': len(data.get('concepts', [])),
                'sent_experiences': len(experiences), 'received_experiences': len(data.get('experiences', [])),
                'connected_instances': data.get('connected_instances', 1)
            })
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect or sync with hive server at {self.hive_server_url}: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during sync: {e}")
            return False

    def get_sync_stats(self):
        """Get synchronization statistics"""
        return {'last_sync': self.last_sync_time, 'sync_count': len(self.sync_history),
                'is_server': self.is_server, 'identity': self.hive_identity, 'sync_interval': self.sync_interval}


###########################################
# HARDWARE MANAGEMENT
###########################################

class HardwareManager:
    """Manages SAM's adaptation to available hardware"""

    def __init__(self, model):
        self.model = model
        self.offload_threshold = model.config.offload_threshold
        self.min_free_memory_gb = model.config.min_free_memory_gb
        self.offloaded_components = set()
        self.component_usage = {}
        self.last_memory_check = 0
        self.memory_check_interval = 60
        self.memory_history = []
        self._setup_memory_monitor()

    def _setup_memory_monitor(self):
        """Set up memory monitoring"""
        try:
            import psutil
            self.has_psutil = True
        except ImportError:
            self.has_psutil = False
            logger.warning("psutil not available, CPU memory monitoring disabled")
        
        self.has_gputil = False
        if torch.cuda.is_available():
            try:
                import GPUtil
                self.has_gputil = True
            except ImportError:
                logger.warning("GPUtil not available, advanced GPU monitoring disabled. Using torch metrics.")

    def get_cpu_ram(self):
        if self.has_psutil:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
        return 64.0 # Fallback

    def get_vram(self):
        if not torch.cuda.is_available(): return None
        try:
            device = torch.cuda.current_device()
            total_mem = torch.cuda.get_device_properties(device).total_memory
            reserved_mem = torch.cuda.memory_reserved(device)
            free_mem = total_mem - reserved_mem
            return {"total": total_mem / (1024**3), "free": free_mem / (1024**3)}
        except Exception as e:
            logger.error(f"Error getting GPU memory via torch: {e}")
            return None

    def check_memory(self):
        """Check memory usage and offload if needed"""
        current_time = time.time()
        if current_time - self.last_memory_check < self.memory_check_interval: return
        self.last_memory_check = current_time

        cpu_ram = self.get_cpu_ram()
        vram = self.get_vram()
        self.memory_history.append({"timestamp": current_time, "cpu_ram": cpu_ram, "vram": vram})
        self.memory_history = self.memory_history[-1440:] # Keep last 24 hours (1440 minutes)

        if vram and vram["free"] < self.min_free_memory_gb:
            self._offload_components()
        elif self.offloaded_components and vram and vram["free"] > self.min_free_memory_gb * 2:
            self._load_components()

    def _offload_components(self):
        """Offload less used components to CPU"""
        self._update_component_usage()
        components_to_offload = sorted([item for item in self.component_usage.items() if item[0] not in self.offloaded_components], key=lambda x: x[1])

        for component_name, usage in components_to_offload:
            component = self._get_component_by_name(component_name)
            if component:
                component.to('cpu')
                self.offloaded_components.add(component_name)
                logger.info(f"Offloaded component to CPU: {component_name}")
                vram = self.get_vram()
                if vram and vram["free"] >= self.min_free_memory_gb:
                    break

    def _load_components(self):
        """Load offloaded components back to GPU"""
        self._update_component_usage()
        components_to_load = sorted([(name, self.component_usage.get(name, 0)) for name in self.offloaded_components], key=lambda x: x[1], reverse=True)
        
        vram = self.get_vram()
        if not vram: return
        free_memory = vram["free"] - self.min_free_memory_gb

        for component_name, _ in components_to_load:
            component = self._get_component_by_name(component_name)
            if component:
                size_gb = self._estimate_component_size(component) / (1024**3)
                if size_gb < free_memory:
                    component.to(self.model.config.device)
                    self.offloaded_components.remove(component_name)
                    free_memory -= size_gb
                    logger.info(f"Loaded component back to GPU: {component_name}")
                    if free_memory < 0.5:
                        break

    def _update_component_usage(self):
        """Update component usage statistics"""
        # This is a heuristic and would need more sophisticated tracking in a real system
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'updates'):
                self.component_usage[f"layer_{i}"] = 0.7 * self.component_usage.get(f"layer_{i}", 0) + 0.3 * layer.updates
        
        self.component_usage["concept_bank"] = 1000 # Always critical
        self.component_usage["thought_state"] = 500 # Very important
        if hasattr(self.model.segmentation, "total_segmentations"):
            self.component_usage["segmentation"] = self.model.segmentation.total_segmentations

    def _get_component_by_name(self, name):
        """Get component by name"""
        if name.startswith("layer_"):
            try:
                idx = int(name.split("_")[1])
                if 0 <= idx < len(self.model.layers): return self.model.layers[idx]
            except (ValueError, IndexError): return None
        return getattr(self.model, name, None)

    def _estimate_component_size(self, component):
        """Estimate memory size of a component in bytes"""
        param_size = sum(p.numel() * p.element_size() for p in component.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in component.buffers())
        return param_size + buffer_size

    def detect_optimal_config(self):
        """Detect optimal configuration based on hardware"""
        vram = self.get_vram()
        cpu_ram = self.get_cpu_ram()
        config = {"profile": "cpu_low", "hidden_dim": 256, "num_layers": 4, "dream_cycle_minutes": 0}

        if torch.cuda.is_available() and vram:
            if vram["total"] > 16:
                config = {"profile": "gpu_high", "hidden_dim": 2048, "num_layers": 24, "dream_cycle_minutes": 1.0}
            elif vram["total"] > 8:
                config = {"profile": "gpu_mid", "hidden_dim": 1536, "num_layers": 16, "dream_cycle_minutes": 0.5}
            elif vram["total"] > 4:
                config = {"profile": "gpu_low", "hidden_dim": 768, "num_layers": 8, "dream_cycle_minutes": 0.2}
            else:
                 config = {"profile": "gpu_minimum", "hidden_dim": 512, "num_layers": 6, "dream_cycle_minutes": 0.1}
        elif cpu_ram > 4:
            config = {"profile": "cpu_high", "hidden_dim": 512, "num_layers": 6, "dream_cycle_minutes": 0.1}

        logger.info(f"Detected hardware profile: {config['profile']}")
        return config

    def get_hardware_stats(self):
        """Get hardware statistics"""
        vram = self.get_vram()
        return {"cpu_ram_gb": self.get_cpu_ram(), "vram_total_gb": vram["total"] if vram else None,
                "vram_free_gb": vram["free"] if vram else None, "device": self.model.config.device,
                "offloaded_components": list(self.offloaded_components), "memory_checks": len(self.memory_history)}


###########################################
# NEURAL COMPONENTS
###########################################

class DynamicSegmentation(nn.Module):
    """Dynamic segmentation component that replaces traditional tokenization"""

    def __init__(self, config, concept_bank):
        super().__init__()
        self.config = config
        self.concept_bank = concept_bank
        self.char_embeddings = nn.Embedding(config.initial_char_dim, config.initial_hidden_dim)
        self.segment_detector = nn.Sequential(
            nn.Conv1d(config.initial_hidden_dim, config.initial_hidden_dim, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(config.initial_hidden_dim, 1, kernel_size=1)
        )
        self.segment_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.initial_hidden_dim, nhead=8, dim_feedforward=config.initial_hidden_dim*4, batch_first=True),
            num_layers=2
        )
        self.pattern_memory = PatternMemory(capacity=config.pattern_memory_capacity, min_frequency=config.min_segment_frequency)
        self.segment_cache = {}
        self.private_context = None
        self.current_modality = "text"
        self.total_segmentations = 0
        self.cache_hits = 0

    def set_private_context(self, context_name): self.private_context = context_name
    def clear_private_context(self): self.private_context = None
    def set_modality(self, modality): self.current_modality = modality

    def forward(self, char_sequence, return_segments=False):
        """Process raw character input into concept IDs"""
        is_batched = char_sequence.dim() > 1
        batch_size = char_sequence.shape[0] if is_batched else 1

        if not is_batched:
            char_sequence = char_sequence.unsqueeze(0)

        cache_key = tuple(char_sequence[0].tolist())
        if not return_segments and cache_key in self.segment_cache:
            self.cache_hits += 1
            return self.segment_cache[cache_key] if is_batched else self.segment_cache[cache_key][0]

        self.total_segmentations += batch_size
        char_embeds = self.char_embeddings(char_sequence)
        boundary_logits = self.segment_detector(char_embeds.transpose(1, 2)).squeeze(1)
        boundary_probs = torch.sigmoid(boundary_logits)

        all_segments, all_concept_ids = [], []
        for b in range(batch_size):
            seq_segments, seq_concepts = self._extract_segments(char_sequence[b], char_embeds[b], boundary_probs[b])
            all_segments.append(seq_segments)
            all_concept_ids.append(seq_concepts)

        if not return_segments and batch_size == 1:
            self.segment_cache[cache_key] = all_concept_ids

        if return_segments:
            return (all_concept_ids, all_segments) if is_batched else (all_concept_ids[0], all_segments[0])
        return all_concept_ids if is_batched else all_concept_ids[0]

    def _extract_segments(self, chars, char_embeds, boundary_probs):
        """Extract segments from a character sequence using boundary probabilities"""
        boundaries = torch.cat([torch.tensor([0], device=chars.device), (boundary_probs > 0.5).nonzero().flatten(), torch.tensor([len(chars)], device=chars.device)])
        segments, concept_ids = [], []
        
        last_b = 0
        for b in boundaries:
            b = b.item()
            if b <= last_b: continue
            
            segment_chars = chars[last_b:b]
            segment_embeds = char_embeds[last_b:b]
            
            if len(segment_chars) > self.config.max_segment_length:
                # Sub-segment if too long
                for i in range(0, len(segment_chars), self.config.max_segment_length):
                    sub_chars = segment_chars[i:i+self.config.max_segment_length]
                    sub_embeds = segment_embeds[i:i+self.config.max_segment_length]
                    segments.append(sub_chars.tolist())
                    concept_ids.append(self._get_concept_for_segment(sub_chars, sub_embeds))
            else:
                segments.append(segment_chars.tolist())
                concept_ids.append(self._get_concept_for_segment(segment_chars, segment_embeds))
            last_b = b
            
        return segments, concept_ids

    def _get_concept_for_segment(self, char_segment_tensor, segment_embeds):
        """Get or create concept ID for a character segment"""
        segment_str = "".join(chr(c) for c in char_segment_tensor.tolist())
        concept_id = self.concept_bank.find_concept_by_source(segment_str)
        if concept_id:
            self.concept_bank.update_concept_usage(concept_id, context=self.private_context)
            self.pattern_memory.add_pattern(segment_str, context=self.private_context, private=bool(self.private_context), modality=self.current_modality)
            return concept_id

        self.pattern_memory.add_pattern(segment_str, context=self.private_context, private=bool(self.private_context), modality=self.current_modality)
        if self.pattern_memory.get_pattern_frequency(segment_str) >= self.config.min_segment_frequency:
            if segment_embeds.numel() > 0:
                with torch.no_grad():
                    segment_encoding = self.segment_encoder(segment_embeds.unsqueeze(0)).mean(dim=1).squeeze(0)
            else:
                segment_encoding = torch.zeros(self.config.initial_hidden_dim, device=self.char_embeddings.weight.device)
            
            new_concept_id = self.concept_bank.add_character_concept(segment_str, hive_private=bool(self.private_context), modality=self.current_modality)
            with torch.no_grad():
                self.concept_bank.meaning_vectors[new_concept_id] = F.normalize(segment_encoding, dim=0)
            return new_concept_id
        else:
            return [self.concept_bank.find_concept_by_source(chr(c)) or self.concept_bank.add_character_concept(chr(c)) for c in char_segment_tensor.tolist()]

    def get_segmentation_stats(self):
        """Get statistics about segmentation performance"""
        hit_rate = (self.cache_hits / self.total_segmentations) if self.total_segmentations > 0 else 0
        return {"total_segmentations": self.total_segmentations, "cache_hits": self.cache_hits,
                "cache_hit_rate": hit_rate, "cached_segments": len(self.segment_cache),
                "frequent_patterns": len(self.pattern_memory.get_frequent_patterns(limit=1000)),
                "current_modality": self.current_modality}


class NeuroplasticLayer(nn.Module):
    """Core neural layer that can grow and evolve with neuroplasticity"""
    def __init__(self, hidden_dim, growth_factor=1.2, dropout=0.1, layer_id=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.growth_factor = growth_factor
        self.layer_id = layer_id
        self.attention = AdaptiveAttention(hidden_dim, dropout=dropout)
        self.gate_proj = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.up_proj = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.down_proj = nn.Linear(4 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.updates = 0

    def forward(self, x, mask=None, cross_input=None, modality="text"):
        if self.training: self.updates += 1
        
        residual = x
        x = self.norm1(x)
        attn_output = self.attention(x, mask=mask, cross_input=cross_input)
        x = residual + self.dropout(attn_output)

        residual = x
        x = self.norm2(x)
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        intermediate = F.silu(gate_output) * up_output
        ffn_output = self.down_proj(intermediate)
        x = residual + self.dropout(ffn_output)
        
        return x

    def grow(self, new_dim):
        # Placeholder for growth logic
        pass

    def evolve(self):
        # Placeholder for evolution logic
        pass


class AdaptiveAttention(nn.Module):
    """Adaptive attention mechanism that can evolve over time"""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, cross_input=None):
        batch_size, seq_len, _ = x.shape
        query = self.q_proj(x)
        key = self.k_proj(cross_input if cross_input is not None else x)
        value = self.v_proj(cross_input if cross_input is not None else x)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.o_proj(output)


###########################################
# MULTIMODAL COMPONENTS
###########################################

class MultimodalProcessor(nn.Module):
    """Processes inputs from different modalities and integrates them"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoders = nn.ModuleDict()
        if config.multimodal_enabled:
            self.encoders["image"] = nn.Sequential(
                nn.Linear(config.image_dim, config.initial_hidden_dim), nn.GELU()
            )
            self.encoders["audio"] = nn.Sequential(
                nn.Linear(config.audio_dim, config.initial_hidden_dim), nn.GELU()
            )

        if config.multimodal_fusion_strategy == "attention":
            self.fusion = nn.MultiheadAttention(embed_dim=config.initial_hidden_dim, num_heads=8, batch_first=True)
        else:
            self.fusion = nn.Sequential(
                nn.Linear(config.initial_hidden_dim * 2, config.initial_hidden_dim), nn.GELU()
            )

    def process_image(self, image_data):
        return self.encoders["image"](image_data) if "image" in self.encoders else None

    def process_audio(self, audio_data):
        return self.encoders["audio"](audio_data) if "audio" in self.encoders else None

    def integrate_modalities(self, modality_embeddings):
        """Integrate embeddings from different modalities"""
        embeddings_list = list(modality_embeddings.values())
        if not embeddings_list: return None
        if len(embeddings_list) == 1: return embeddings_list[0]
        
        if self.config.multimodal_fusion_strategy == "attention":
            # Simple average for attention fusion
            return torch.mean(torch.stack(embeddings_list), dim=0)
        else: # Concatenation
            # Ensure all embeddings are 2D (B, D) before concatenating
            processed_embeds = [e.squeeze(1) if e.dim() == 3 else e for e in embeddings_list]
            concatenated = torch.cat(processed_embeds, dim=-1)
            return self.fusion(concatenated)


###########################################
# TRAINING AND RUNTIME
###########################################

class SAMTrainer:
    """Training manager for the SAM model"""

    def __init__(self, model, train_data_path=None, eval_data_path=None, batch_size=32,
                 learning_rate=None, warmup_steps=None, max_steps=None, num_epochs=3):
        self.model = model
        self.train_data_path = train_data_path
        self.eval_data_path = eval_data_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate or model.config.learning_rate
        self.warmup_steps = warmup_steps or model.config.warmup_steps
        self.max_steps = max_steps
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        self.scheduler = None
        self.best_loss = float('inf')

    def train_from_json(self, json_path, batch_size=4, epochs=3):
        import json, random
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not load or parse training file: {e}")
            return
            
        samples = [d["text"] for d in data if isinstance(d, dict) and "text" in d]
        if not samples:
            logger.error("No valid text samples found in the training file.")
            return

        self.model.train()
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            random.shuffle(samples)
            for i in range(0, len(samples), batch_size):
                batch_texts = samples[i:i + batch_size]
                self.optimizer.zero_grad()
                
                # Convert texts to char IDs and pad
                char_ids_list = [[ord(c) for c in text] for text in batch_texts]
                max_len = max(len(ids) for ids in char_ids_list)
                padded_ids = [ids + [0] * (max_len - len(ids)) for ids in char_ids_list]
                
                input_tensor = torch.tensor(padded_ids, dtype=torch.long, device=self.model.config.device)
                
                loss, _, _ = self.model(input_chars=input_tensor, target_concepts=input_tensor)
                
                if loss is not None and torch.isfinite(loss):
                    loss.backward()
                    self.optimizer.step()
                    if i % 10 == 0:
                        print(f"  Batch {i//batch_size+1}/{len(samples)//batch_size}, Loss: {loss.item():.4f}")
                else:
                    logger.warning("Skipping batch due to invalid loss.")
        print("Training complete.")


###########################################
# MAIN SAM CLASS
###########################################

class SAM(nn.Module):
    """Synergistic Autonomous Machine - unified neural-linguistic model with hive mind capability"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or SAMConfig()
        self.config.validate()

        self.concept_bank = ConceptMemoryBank(
            concept_dim=self.config.initial_hidden_dim,
            initial_size=self.config.concept_memory_size,
            device=self.config.device
        )
        self.segmentation = DynamicSegmentation(self.config, self.concept_bank)
        self.position_embeddings = nn.Embedding(self.config.max_position_embeddings, self.config.initial_hidden_dim)
        
        if self.config.multimodal_enabled:
            self.multimodal_processor = MultimodalProcessor(self.config)
        
        self.layers = nn.ModuleList([NeuroplasticLayer(self.config.initial_hidden_dim) for _ in range(self.config.initial_num_layers)])
        self.norm = nn.LayerNorm(self.config.initial_hidden_dim)
        
        # Tie weights between concept embeddings and the output layer
        self.lm_head = nn.Linear(self.config.initial_hidden_dim, self.config.concept_memory_size, bias=False)
        self.lm_head.weight = self.concept_bank.concept_embeddings.weight

        self.thought_state = ThoughtState(concept_dim=self.config.initial_hidden_dim, thought_dim=self.config.thought_dim)
        self.experience_manager = ExperienceManager(self.config)
        self.dreaming = ConceptualDreaming(self)
        self.consciousness = ConsciousnessMonitor(self)
        
        if self.config.hive_enabled: self.hive_synchronizer = HiveMindSynchronizer(self)
        else: self.hive_synchronizer = None
        
        if self.config.hardware_adaptive: self.hardware_manager = HardwareManager(self)
        else: self.hardware_manager = None
            
        self.growth_history = []
        self.global_step = 0
        self.current_modality = "text"
        self.to(self.config.device)

    def forward(self, input_chars=None, input_concepts=None, target_concepts=None, **kwargs):
        """Forward pass with either raw characters or concept IDs"""
        if input_chars is not None:
            # Segmentation returns a list of lists of concept IDs
            processed_concepts = self.segmentation(input_chars)
            # Flatten, pad, and convert to tensor
            max_len = max(len(c) for c in processed_concepts) if processed_concepts else 0
            if max_len == 0: # Handle empty input
                logits = torch.empty(input_chars.shape[0], 0, self.config.concept_memory_size, device=self.config.device)
                return None, logits, None

            padded_concepts = [c + [0] * (max_len - len(c)) for c in processed_concepts]
            input_concepts = torch.tensor(padded_concepts, dtype=torch.long, device=self.config.device)
        elif not isinstance(input_concepts, torch.Tensor):
            input_concepts = torch.tensor(input_concepts, dtype=torch.long, device=self.config.device)

        # Check if concept bank needs resizing for the lm_head
        if self.lm_head.out_features != self.concept_bank.concept_embeddings.num_embeddings:
            self.lm_head = nn.Linear(self.config.initial_hidden_dim, self.concept_bank.concept_embeddings.num_embeddings, bias=False).to(self.config.device)
            self.lm_head.weight = self.concept_bank.concept_embeddings.weight

        batch_size, seq_len = input_concepts.shape
        concept_embeds = self.concept_bank(input_concepts)
        pos_ids = torch.arange(seq_len, device=self.config.device).unsqueeze(0)
        pos_embeds = self.position_embeddings(pos_ids)
        hidden_states = concept_embeds + pos_embeds
        
        # Integrate thought state
        thought_context = self.thought_state.update(concept_embeds)
        projected_thought = self.thought_state.project_to_concept_space(thought_context)
        if projected_thought is not None:
             hidden_states = hidden_states + projected_thought.expand_as(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if target_concepts is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Align target and logits
            if logits.shape[1] > 1 and target_concepts.shape[1] > 1:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = target_concepts[:, 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss, logits, hidden_states

    def process_text(self, text, private_context=False):
        """Process raw text into concept IDs and segments"""
        if private_context: self.segmentation.set_private_context("private")
        char_tensor = torch.tensor([ord(c) for c in text], dtype=torch.long, device=self.config.device)
        concept_ids, segments = self.segmentation(char_tensor, return_segments=True)
        if private_context: self.segmentation.clear_private_context()
        return concept_ids, segments

    def generate(self, input_text, max_length=128, temperature=1.0, top_k=50, **kwargs):
        """Generate text from a prompt"""
        self.eval()
        char_ids = [ord(c) for c in input_text]
        input_tensor = torch.tensor(char_ids, dtype=torch.long, device=self.config.device).unsqueeze(0)
        
        generated_concepts = self.segmentation(input_tensor)
        if not isinstance(generated_concepts, torch.Tensor):
            generated_concepts = torch.tensor(generated_concepts, dtype=torch.long, device=self.config.device)
        if generated_concepts.dim() == 1:
            generated_concepts = generated_concepts.unsqueeze(0)

        with torch.no_grad():
            for _ in range(max_length):
                _, logits, _ = self.forward(input_concepts=generated_concepts)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_concepts = torch.cat((generated_concepts, next_token), dim=1)

        # Convert concepts back to text
        output_ids = generated_concepts[0].tolist()
        return self._concepts_to_text(output_ids)

    def _concepts_to_text(self, concept_ids):
        """Convert a list of concept IDs back to text"""
        text_parts = []
        for concept_id in concept_ids:
            if isinstance(concept_id, list): # Handle nested lists from segmentation
                text_parts.append(self._concepts_to_text(concept_id))
                continue
            metadata = self.concept_bank.concept_metadata.get(concept_id)
            if metadata and metadata.get("source"):
                text_parts.append(metadata["source"])
        return "".join(text_parts)
    
    def save(self, path=None):
        """Save model state"""
        if path is None: path = os.path.join(self.config.save_dir, f"sam_checkpoint_{self.global_step}")
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        self.config.save(os.path.join(path, "config.json"))
        logger.info(f"Model saved to {path}")
        return path

    @classmethod
    def load(cls, path):
        """Load model from saved state"""
        config = SAMConfig.load(os.path.join(path, "config.json"))
        model = cls(config)
        model.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location=config.device))
        logger.info(f"Model loaded from {path}")
        return model

    def start_services(self):
        """Start background services (dreaming, hive sync)"""
        if hasattr(self, 'dreaming'): self.dreaming.start_background_dreaming(self.config.dream_cycle_minutes)
        if hasattr(self, 'hive_synchronizer') and self.hive_synchronizer: self.hive_synchronizer.start_sync()

    def stop_services(self):
        """Stop background services"""
        if hasattr(self, 'dreaming'): self.dreaming.stop_background_dreaming()
        if hasattr(self, 'hive_synchronizer') and self.hive_synchronizer: self.hive_synchronizer.stop_sync()

    def get_status(self):
        """Get comprehensive status of the model"""
        return {
            "model_size": {
                "hidden_dim": self.config.initial_hidden_dim, "num_layers": len(self.layers),
                "total_concepts": self.concept_bank.next_concept_id,
                "parameter_count": sum(p.numel() for p in self.parameters() if p.requires_grad)
            },
            "training": {"global_step": self.global_step},
            "concept_stats": self.concept_bank.get_concept_stats(),
            "consciousness": self.consciousness.get_identity_summary() if self.consciousness else None,
            "hive_mind": self.hive_synchronizer.get_sync_stats() if self.hive_synchronizer else None,
            "hardware": self.hardware_manager.get_hardware_stats() if self.hardware_manager else None,
        }

    def evolve(self):
        # Placeholder for model evolution logic
        logger.info("Evolution cycle triggered.")

# Removed the duplicated AutonomousSelfTrainer and related classes

###########################################
# AUTONOMOUS SELF-TRAINING SYSTEM
###########################################


class TaskDomain:
    """Defines a domain of tasks for self-training"""
    TEXT = "text"
    MATH = "math"
    CODE = "code"
    LOGIC = "logic"
    CREATIVE = "creative"
    MULTIMODAL = "multimodal"

class TaskDifficulty:
    """Defines difficulty levels for self-generated tasks"""
    FOUNDATIONAL = 0
    BASIC = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    BREAKTHROUGH = 5

class ReasoningType:
    """Types of reasoning for the autonomous cognition loop"""
    DEDUCTION = "deduction"  # Apply existing rules to derive conclusions
    ABDUCTION = "abduction"  # Generate most plausible hypothesis from observations
    INDUCTION = "induction"  # Derive general rules from specific instances
    TRIAL_ERROR = "trial_error"  # Experiment with variations and learn from results

class AutonomousSelfTrainer:
    """Autonomous self-training system that enables SAM to evolve through self-created tasks,
    self-verification, and self-reinforcement without requiring human-annotated data."""

    def __init__(self, model):
        """Initialize the autonomous self-trainer with a reference to the SAM model"""
        self.model = model
        self.config = model.config

        # Components for autonomous training
        self.task_generator = TaskGenerator(self)
        self.solution_verifier = SolutionVerifier(self)
        self.reward_model = RewardModel(self)
        self.reasoning_engine = ReasoningEngine(self)

        # Training metrics
        self.training_cycles = 0
        self.successful_verifications = 0
        self.current_difficulty = TaskDifficulty.FOUNDATIONAL
        self.domain_competencies = {
            TaskDomain.TEXT: 0.0,
            TaskDomain.MATH: 0.0,
            TaskDomain.CODE: 0.0,
            TaskDomain.LOGIC: 0.0,
            TaskDomain.CREATIVE: 0.0,
            TaskDomain.MULTIMODAL: 0.0 if self.config.multimodal_enabled else None
        }

        # Training history
        self.training_history = []

        # Determine initial focus domains based on model's current state
        self.active_domains = self._determine_initial_domains()

        # Evolution tracking
        self.evolution_metrics = {
            "concept_growth_rate": 0.0,
            "reasoning_depth": 0.0,
            "verification_accuracy": 0.0,
            "task_complexity": 0.0
        }

        logger.info("Autonomous Self-Trainer initialized")

    def _determine_initial_domains(self):
        """Determine which domains to focus on initially based on model state"""
        # Always start with text and logic as foundational domains
        domains = [TaskDomain.TEXT, TaskDomain.LOGIC]

        # Check if the model has enough concepts for more complex domains
        concept_stats = self.model.concept_bank.get_concept_stats()
        if concept_stats["total_concepts"] > 1000:
            domains.append(TaskDomain.MATH)
        if concept_stats["total_concepts"] > 2000:
            domains.append(TaskDomain.CODE)
        if concept_stats["total_concepts"] > 5000:
            domains.append(TaskDomain.CREATIVE)

        # Add multimodal if enabled
        if self.config.multimodal_enabled:
            domains.append(TaskDomain.MULTIMODAL)

        return domains

    def start_autonomous_training(self, duration_minutes=10, cycles=None):
        """Start autonomous training for a specified duration or number of cycles"""
        logger.info(f"Starting autonomous training for {duration_minutes} minutes or {cycles} cycles")

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        cycle_count = 0

        # Store the model's training state
        was_training = self.model.training
        self.model.train()

        try:
            while (time.time() < end_time if cycles is None else cycle_count < cycles):
                # Run a single training cycle
                cycle_results = self.run_training_cycle()

                # Record results
                self.training_history.append(cycle_results)

                # Update metrics
                self._update_evolution_metrics(cycle_results)

                # Consider difficulty progression
                self._consider_difficulty_progression()

                # Consider domain expansion
                self._consider_domain_expansion()

                # Increment counters
                cycle_count += 1
                self.training_cycles += 1

                # Log progress
                if cycle_count % 10 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"Completed {cycle_count} training cycles in {elapsed:.1f} seconds")

                    # Run an evolution step periodically
                    if cycle_count % 50 == 0:
                        self.model.evolve()

            # Final evolution step
            self.model.evolve()

            # Summarize training
            training_summary = self._generate_training_summary(cycle_count, time.time() - start_time)
            logger.info(f"Autonomous training completed: {training_summary}")

            return training_summary

        finally:
            # Restore model's training state
            if not was_training:
                self.model.eval()

    def run_training_cycle(self):
        """Run a single autonomous training cycle"""
        # 1. Select domain and difficulty based on current competencies
        domain, difficulty = self.task_generator.select_domain_and_difficulty()

        # 2. Generate a task
        task, task_context = self.task_generator.generate_task(domain, difficulty)

        # 3. Select reasoning approach
        reasoning_type = self.reasoning_engine.select_reasoning_type(domain, task)

        # 4. Generate solution using selected reasoning approach
        solution, reasoning_trace = self.reasoning_engine.solve_task(task, task_context, reasoning_type)

        # 5. Verify solution
        verification_result, verification_score = self.solution_verifier.verify_solution(
            task, task_context, solution, reasoning_trace
        )

        # 6. Apply reward based on verification
        reward = self.reward_model.calculate_reward(
            verification_result,
            verification_score,
            domain,
            difficulty,
            reasoning_type,
            reasoning_trace
        )

        # 7. Apply the reward to model (reinforce connections)
        self.reward_model.apply_reward(reward, reasoning_trace)

        # 8. Record the experience
        self._record_training_experience(
            domain, difficulty, task, solution,
            verification_result, reward, reasoning_type
        )

        # 9. Update domain competency
        if verification_result:
            self.successful_verifications += 1
            self.domain_competencies[domain] += reward * 0.1
        else:
            self.domain_competencies[domain] -= 0.01  # Small penalty for failures

        # Ensure competencies are within bounds
        self.domain_competencies[domain] = max(0.0, min(5.0, self.domain_competencies[domain]))

        # Return cycle results
        return {
            "cycle": self.training_cycles,
            "domain": domain,
            "difficulty": difficulty,
            "task": task,
            "solution": solution,
            "reasoning_type": reasoning_type,
            "verification_result": verification_result,
            "verification_score": verification_score,
            "reward": reward,
            "competency_update": self.domain_competencies[domain]
        }

    def _record_training_experience(self, domain, difficulty, task, solution,
                                verification_result, reward, reasoning_type):
        """Record the training experience in the model's experience manager"""
        experience_type = "autonomous_training"
        content = {
            "domain": domain,
            "difficulty": difficulty,
            "task": task,
            "solution": solution,
            "verification_result": verification_result,
            "reward": reward,
            "reasoning_type": reasoning_type
        }

        metadata = {
            "type": experience_type,
            "timestamp": time.time(),
            "training_cycle": self.training_cycles,
            "successful": verification_result
        }

        # Private experiences that aren't shared with hive mind
        self.model.experience_manager.record_experience(
            experience_type, content, metadata, private=True, modality="text"
        )

    def _update_evolution_metrics(self, cycle_results):
        """Update evolution metrics based on cycle results"""
        # Update verification accuracy
        total_verifications = max(1, self.training_cycles)
        self.evolution_metrics["verification_accuracy"] = self.successful_verifications / total_verifications

        # Update task complexity
        self.evolution_metrics["task_complexity"] = self.current_difficulty / TaskDifficulty.BREAKTHROUGH

        # Update concept growth rate
        current_concepts = self.model.concept_bank.next_concept_id
        if hasattr(self, "_last_concept_count"):
            growth = (current_concepts - self._last_concept_count) / max(1, self._last_concept_count)
            self.evolution_metrics["concept_growth_rate"] = growth
        self._last_concept_count = current_concepts

        # Update reasoning depth based on model's thought state depth
        self.evolution_metrics["reasoning_depth"] = self.model.thought_state.thought_depth / self.model.thought_state.max_thought_depth

    def _consider_difficulty_progression(self):
        """Consider whether to progress to a higher difficulty level"""
        # Check if we have enough successful verifications at current difficulty
        success_threshold = 50 * (1 + self.current_difficulty)  # Higher threshold for higher difficulties

        if self.successful_verifications >= success_threshold:
            # Check if average competency across active domains is sufficient
            active_competencies = [self.domain_competencies[d] for d in self.active_domains]
            avg_competency = sum(active_competencies) / len(active_competencies)

            if avg_competency >= (self.current_difficulty + 0.7):  # Need 70%+ competency
                if self.current_difficulty < TaskDifficulty.BREAKTHROUGH:
                    self.current_difficulty += 1
                    logger.info(f"Advanced to difficulty level: {self.current_difficulty}")
                    self.successful_verifications = 0  # Reset counter

    def _consider_domain_expansion(self):
        """Consider whether to expand into new domains"""
        # Don't expand if we're already covering all domains
        all_domains = [d for d, c in self.domain_competencies.items() if c is not None]
        if all(d in self.active_domains for d in all_domains):
            return

        # Check if we're doing well in current domains
        active_competencies = [self.domain_competencies[d] for d in self.active_domains]
        avg_competency = sum(active_competencies) / len(active_competencies)

        if avg_competency >= 2.0 and len(self.active_domains) < len(all_domains):
            # Find a domain to add
            for domain in all_domains:
                if domain not in self.active_domains:
                    self.active_domains.append(domain)
                    logger.info(f"Expanded training to new domain: {domain}")
                    break

    def _generate_training_summary(self, cycles, duration):
        """Generate a summary of the training session"""
        return {
            "cycles_completed": cycles,
            "duration_seconds": duration,
            "current_difficulty": self.current_difficulty,
            "active_domains": self.active_domains,
            "domain_competencies": self.domain_competencies,
            "successful_verifications": self.successful_verifications,
            "evolution_metrics": self.evolution_metrics,
            "concepts_created": self.model.concept_bank.next_concept_id - (getattr(self, "_last_concept_count", 0)),
            "current_total_concepts": self.model.concept_bank.next_concept_id
        }

    def get_status(self):
        """Get the current status of autonomous training"""
        return {
            "training_cycles": self.training_cycles,
            "successful_verifications": self.successful_verifications,
            "current_difficulty": self.current_difficulty,
            "active_domains": self.active_domains,
            "domain_competencies": self.domain_competencies,
            "evolution_metrics": self.evolution_metrics
        }


class TaskGenerator:
    """Generates tasks for autonomous self-training"""

    def __init__(self, trainer):
        """Initialize the task generator with a reference to the trainer"""
        self.trainer = trainer
        self.model = trainer.model

        # Task templates by domain and difficulty
        self.task_templates = self._initialize_task_templates()

        # Domain selection probabilities (will be adjusted based on performance)
        self.domain_selection_probs = {
            TaskDomain.TEXT: 0.3,
            TaskDomain.MATH: 0.2,
            TaskDomain.CODE: 0.4,
            TaskDomain.LOGIC: 0.2,
            TaskDomain.CREATIVE: 0.3,
            TaskDomain.MULTIMODAL: 0.0  # Start at 0, will be adjusted if enabled
        }

        # Adjust for multimodal if enabled
        if self.trainer.config.multimodal_enabled:
            self.domain_selection_probs[TaskDomain.MULTIMODAL] = 0.1
            # Normalize probabilities
            total = sum(self.domain_selection_probs.values())
            for domain in self.domain_selection_probs:
                self.domain_selection_probs[domain] /= total

    def _initialize_task_templates(self):
        """Initialize templates for different task types and difficulties"""
        templates = {
            TaskDomain.TEXT: {
                TaskDifficulty.FOUNDATIONAL: [
                    "Complete the following sentence: {context}",
                    "What is the opposite of {concept}?",
                    "Arrange these words in alphabetical order: {words}",
                    "Identify the subject in this sentence: {sentence}"
                ],
                TaskDifficulty.BASIC: [
                    "Summarize the following text in one sentence: {context}",
                    "Find the main idea in this paragraph: {context}",
                    "Correct any grammatical errors in this sentence: {sentence}",
                    "What does {concept} mean in the context of {domain}?"
                ],
                # Additional levels would be defined similarly
            },
            TaskDomain.MATH: {
                TaskDifficulty.FOUNDATIONAL: [
                    "Calculate: {num1} + {num2}",
                    "Calculate: {num1} - {num2}",
                    "Calculate: {num1} × {num2}",
                    "Calculate: {num1} ÷ {num2} (if division is possible)"
                ],
                TaskDifficulty.BASIC: [
                    "Solve the equation: {num1}x + {num2} = {num3}",
                    "Find the area of a rectangle with width {num1} and height {num2}",
                    "What is {num1}% of {num2}?",
                    "If {context}, what is the value of {variable}?"
                ],
                # Additional levels would be defined similarly
            },
            TaskDomain.CODE: {
                TaskDifficulty.FOUNDATIONAL: [
                    "Write a function that returns {output} when given {input}",
                    "What does this code do? {code_snippet}",
                    "Fix the error in this code: {buggy_code}",
                    "Write a function called {function_name} that {function_description}"
                ],
                TaskDifficulty.BASIC: [
                    "Implement a function to check if a number is prime",
                    "Write a function to find the {n}th Fibonacci number",
                    "Create a class called {class_name} with methods to {method_description}",
                    "Optimize this code for better performance: {code_snippet}"
                ],
                # Additional levels would be defined similarly
            },
            TaskDomain.LOGIC: {
                TaskDifficulty.FOUNDATIONAL: [
                    "If {premise1} and {premise2}, what can you conclude?",
                    "Identify whether this statement is true or false: {statement}",
                    "What is the next number in this sequence: {sequence}",
                    "Solve this puzzle: {puzzle_description}"
                ],
                TaskDifficulty.BASIC: [
                    "If {premise}, what must be true? What could be true but isn't necessarily?",
                    "Find the pattern in this sequence and predict the next three elements: {sequence}",
                    "Evaluate this logical expression: {expression}",
                    "Solve this riddle: {riddle}"
                ],
                # Additional levels would be defined similarly
            },
            TaskDomain.CREATIVE: {
                TaskDifficulty.FOUNDATIONAL: [
                    "Write a short poem about {topic}",
                    "Create a metaphor that explains {concept}",
                    "Describe a scene involving {element1} and {element2}",
                    "Invent a character who is {trait1} and {trait2}"
                ],
                TaskDifficulty.BASIC: [
                    "Write a short story that includes these elements: {elements}",
                    "Create an analogy between {domain1} and {domain2}",
                    "Design a product that solves this problem: {problem}",
                    "Write dialogue between two characters who disagree about {topic}"
                ],
                # Additional levels would be defined similarly
            }
        }

        # Add multimodal tasks if enabled
        if self.trainer.config.multimodal_enabled:
            templates[TaskDomain.MULTIMODAL] = {
                TaskDifficulty.FOUNDATIONAL: [
                    "Describe what might be in this image: {image_description}",
                    "How might this sound be represented visually: {sound_description}",
                    "Create text that would accompany this image: {image_description}",
                    "What emotions might this music evoke: {music_description}"
                ],
                TaskDifficulty.BASIC: [
                    "If this image {image_description} and this sound {sound_description} were combined, what might it represent?",
                    "Create a story that incorporates this visual scene: {image_description}",
                    "Describe how {concept} might be represented across different modalities",
                    "Convert this textual description into a visual scene: {description}"
                ],
                # Additional levels would be defined similarly
            }

        return templates

    def select_domain_and_difficulty(self):
        """Select a domain and difficulty level for the next task"""
        # Only select from active domains
        active_probs = {d: self.domain_selection_probs[d] for d in self.trainer.active_domains}
        total = sum(active_probs.values())
        active_probs = {d: p/total for d, p in active_probs.items()}

        # Select domain based on probabilities
        domain = random.choices(
            list(active_probs.keys()),
            weights=list(active_probs.values()),
            k=1
        )[0]

        # Select difficulty - usually current difficulty, but occasionally one level lower or higher
        difficulty_options = [max(0, self.trainer.current_difficulty - 1),
                            self.trainer.current_difficulty,
                            min(TaskDifficulty.BREAKTHROUGH, self.trainer.current_difficulty + 1)]

        difficulty_weights = [0.2, 0.6, 0.2]  # Mostly current, sometimes adjacent

        difficulty = random.choices(difficulty_options, weights=difficulty_weights, k=1)[0]

        return domain, difficulty

    def generate_task(self, domain, difficulty):
        """Generate a specific task for the given domain and difficulty"""
        # Get templates for the domain and difficulty
        if difficulty not in self.task_templates.get(domain, {}):
            # Fallback to the highest available difficulty
            available_difficulties = sorted(self.task_templates.get(domain, {}).keys())
            if not available_difficulties:
                # No templates for this domain, fall back to TEXT domain
                domain = TaskDomain.TEXT
                available_difficulties = sorted(self.task_templates[domain].keys())

            difficulty = max(d for d in available_difficulties if d <= difficulty)

        templates = self.task_templates[domain][difficulty]

        # Select a template
        template = random.choice(templates)

        # Generate context for the template
        context = self._generate_context(domain, difficulty, template)

        # Fill in the template
        task = self._fill_template(template, context)

        return task, context

    def _generate_context(self, domain, difficulty, template):
        """Generate context appropriate for the domain, difficulty, and template"""
        # This would be a complex method that creates appropriate task contexts
        # For this implementation, I'll provide a simplified version

        context = {}

        if domain == TaskDomain.TEXT:
            # Generate text-related context
            if "concept" in template:
                concepts = ["knowledge", "information", "communication", "language",
                        "expression", "meaning", "understanding", "interpretation"]
                context["concept"] = random.choice(concepts)

            if "sentence" in template:
                sentences = [
                    "The quick brown fox jumps over the lazy dog.",
                    "She sells seashells by the seashore.",
                    "The cat sat on the mat while the dog barked loudly.",
                    "To be or not to be, that is the question."
                ]
                context["sentence"] = random.choice(sentences)

            if "context" in template:
                paragraphs = [
                    "Artificial intelligence is transforming how we interact with technology. From voice assistants to predictive text, AI systems are becoming increasingly integrated into our daily lives.",
                    "The human brain contains approximately 86 billion neurons. These cells communicate through electrochemical signals, forming the basis of all thoughts, feelings, and actions.",
                    "Climate change poses significant challenges to global ecosystems. Rising temperatures affect weather patterns, sea levels, and biodiversity around the world."
                ]
                context["context"] = random.choice(paragraphs)

            if "words" in template:
                word_sets = [
                    "apple banana cherry date elderberry fig",
                    "python java ruby javascript typescript",
                    "mercury venus earth mars jupiter saturn"
                ]
                context["words"] = random.choice(word_sets)

        elif domain == TaskDomain.MATH:
            # Generate math-related context
            difficulty_factor = difficulty + 1

            context["num1"] = random.randint(1, 10 * difficulty_factor)
            context["num2"] = random.randint(1, 10 * difficulty_factor)

            if difficulty >= TaskDifficulty.BASIC:
                context["num3"] = random.randint(1, 20 * difficulty_factor)
                context["variable"] = random.choice(["x", "y", "z", "a", "b", "c"])

            if "equation" in template:
                # Generate simple equation context
                a = random.randint(1, 5 * difficulty_factor)
                b = random.randint(1, 10 * difficulty_factor)
                x = random.randint(1, 10)
                c = a * x + b
                context["num1"] = a
                context["num2"] = b
                context["num3"] = c

        elif domain == TaskDomain.CODE:
            # Generate code-related context
            if "function_name" in template:
                function_names = ["calculate_average", "find_maximum", "is_prime",
                                "reverse_string", "count_words", "sort_numbers"]
                context["function_name"] = random.choice(function_names)

            if "function_description" in template:
                descriptions = [
                    "takes a list of numbers and returns their average",
                    "checks if a string is a palindrome",
                    "counts the frequency of each word in a text",
                    "finds the greatest common divisor of two numbers"
                ]
                context["function_description"] = random.choice(descriptions)

            if "buggy_code" in template:
                buggy_codes = [
                    "def sum_list(numbers):\n    total = 0\n    for num in numbers\n        total += num\n    return total",
                    "def find_max(numbers):\n    if len(numbers) == 0:\n        return None\n    max_num = numbers[0]\n    for num in numbers:\n        if num < max_num:\n            max_num = num\n    return max_num"
                ]
                context["buggy_code"] = random.choice(buggy_codes)

            if "code_snippet" in template:
                snippets = [
                    "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)",
                    "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)"
                ]
                context["code_snippet"] = random.choice(snippets)

        elif domain == TaskDomain.LOGIC:
            # Generate logic-related context
            if "sequence" in template:
                sequences = [
                    "2, 4, 6, 8, ...",
                    "1, 3, 9, 27, ...",
                    "1, 1, 2, 3, 5, 8, ...",
                    "3, 1, 4, 1, 5, 9, ..."
                ]
                context["sequence"] = random.choice(sequences)

            if "premise1" in template and "premise2" in template:
                premise_pairs = [
                    ["All men are mortal", "Socrates is a man"],
                    ["If it rains, the ground gets wet", "It is raining"],
                    ["Either the butler or the maid did it", "The butler has an alibi"]
                ]
                selected = random.choice(premise_pairs)
                context["premise1"] = selected[0]
                context["premise2"] = selected[1]

            if "statement" in template:
                statements = [
                    "If a number is divisible by 4, then it is divisible by 2",
                    "If a shape is a square, then it is a rectangle",
                    "If a number is prime, then it is odd"
                ]
                context["statement"] = random.choice(statements)

            if "puzzle_description" in template:
                puzzles = [
                    "Three people need to cross a bridge at night, and they have only one flashlight. The bridge can only hold two people at a time, and the flashlight must be with anyone crossing. Person A takes 1 minute to cross, Person B takes 2 minutes, and Person C takes 5 minutes. When two people cross together, they move at the slower person's pace. What is the minimum time needed for all three to cross?",
                    "A man has to get a fox, a chicken, and a sack of corn across a river. He has a rowboat, and it can only carry him and one other thing. If the fox and the chicken are left together, the fox will eat the chicken. If the chicken and the corn are left together, the chicken will eat the corn. How does he do it?"
                ]
                context["puzzle_description"] = random.choice(puzzles)

        elif domain == TaskDomain.CREATIVE:
            # Generate creative-related context
            if "topic" in template:
                topics = ["nature", "technology", "time", "dreams", "freedom", "change"]
                context["topic"] = random.choice(topics)

            if "concept" in template:
                concepts = ["gravity", "evolution", "democracy", "happiness", "knowledge"]
                context["concept"] = random.choice(concepts)

            if "element1" in template and "element2" in template:
                elements = ["water", "fire", "earth", "air", "metal", "wood", "light", "darkness"]
                random.shuffle(elements)
                context["element1"] = elements[0]
                context["element2"] = elements[1]

            if "trait1" in template and "trait2" in template:
                traits = ["brave", "curious", "wise", "mischievous", "determined", "compassionate"]
                random.shuffle(traits)
                context["trait1"] = traits[0]
                context["trait2"] = traits[1]

        elif domain == TaskDomain.MULTIMODAL and self.trainer.config.multimodal_enabled:
            # Generate multimodal-related context
            if "image_description" in template:
                descriptions = [
                    "a mountain landscape at sunset",
                    "a busy city street with people and cars",
                    "a close-up of a flower with a bee collecting pollen",
                    "a child playing with a colorful toy"
                ]
                context["image_description"] = random.choice(descriptions)

            if "sound_description" in template:
                sounds = [
                    "ocean waves crashing on a beach",
                    "birds singing in a forest",
                    "a jazz band playing in a small club",
                    "a thunderstorm with heavy rain"
                ]
                context["sound_description"] = random.choice(sounds)

            if "music_description" in template:
                music = [
                    "a soft piano melody in a minor key",
                    "an upbeat electronic dance track with a strong bass",
                    "a classical orchestra playing a dramatic crescendo",
                    "a folk song with acoustic guitar and gentle vocals"
                ]
                context["music_description"] = random.choice(music)

        return context

    def _fill_template(self, template, context):
        """Fill a template with context values"""
        filled_template = template

        for key, value in context.items():
            placeholder = "{" + key + "}"
            if placeholder in filled_template:
                filled_template = filled_template.replace(placeholder, str(value))

        return filled_template


class SolutionVerifier:
    """Verifies the correctness of solutions to self-generated tasks"""

    def __init__(self, trainer):
        """Initialize the solution verifier with a reference to the trainer"""
        self.trainer = trainer
        self.model = trainer.model

        # Verification methods by domain
        self.verification_methods = {
            TaskDomain.TEXT: self._verify_text_solution,
            TaskDomain.MATH: self._verify_math_solution,
            TaskDomain.CODE: self._verify_code_solution,
            TaskDomain.LOGIC: self._verify_logic_solution,
            TaskDomain.CREATIVE: self._verify_creative_solution
        }

        # Add multimodal verification if enabled
        if self.trainer.config.multimodal_enabled:
            self.verification_methods[TaskDomain.MULTIMODAL] = self._verify_multimodal_solution

        # Verification statistics
        self.verification_history = []

    def verify_solution(self, task, task_context, solution, reasoning_trace):
        """Verify a solution to a task"""
        # Extract domain from task string or context
        domain = self._infer_domain_from_task(task, task_context)

        # Use appropriate verification method
        verification_method = self.verification_methods.get(domain, self._verify_generic_solution)
        verification_result, verification_score = verification_method(task, task_context, solution, reasoning_trace)

        # Record verification result
        self.verification_history.append({
            "task": task,
            "solution": solution,
            "verification_result": verification_result,
            "verification_score": verification_score,
            "domain": domain,
            "timestamp": time.time()
        })

        return verification_result, verification_score

    def _infer_domain_from_task(self, task, task_context):
        """Infer the domain of a task from its content"""
        # If domain is explicitly in context, use it
        if "domain" in task_context:
            return task_context["domain"]

        # Try to detect domain from task content
        task_lower = task.lower()

        # Check for code-related keywords
        if any(kw in task_lower for kw in ["function", "code", "programming", "algorithm", "class", "method"]):
            return TaskDomain.CODE

        # Check for math-related keywords
        if any(kw in task_lower for kw in ["calculate", "equation", "solve", "math", "formula", "number", "geometry"]):
            return TaskDomain.MATH

        # Check for logic-related keywords
        if any(kw in task_lower for kw in ["logic", "puzzle", "sequence", "pattern", "deduce", "conclusion"]):
            return TaskDomain.LOGIC

        # Check for creative-related keywords
        if any(kw in task_lower for kw in ["create", "invent", "design", "story", "poem", "character", "scene"]):
            return TaskDomain.CREATIVE

        # Check for multimodal-related keywords
        if any(kw in task_lower for kw in ["image", "picture", "sound", "audio", "visual", "music"]):
            return TaskDomain.MULTIMODAL if self.trainer.config.multimodal_enabled else TaskDomain.TEXT

        # Default to text domain
        return TaskDomain.TEXT

    def _verify_generic_solution(self, task, task_context, solution, reasoning_trace):
        """Generic verification method used as fallback"""
        # This is a minimal implementation that could be enhanced

        # Check if solution is non-empty and reasonable length
        if not solution or len(solution) < 5:
            return False, 0.0

        # Check that solution is relevant to task
        task_words = set(task.lower().split())
        solution_words = set(solution.lower().split())

        relevance_score = len(task_words.intersection(solution_words)) / max(1, len(task_words))

        # Check for reasoning quality
        reasoning_quality = min(1.0, len(reasoning_trace) / 200)  # Reward more thorough reasoning

        # Combine scores
        verification_score = 0.4 * relevance_score + 0.6 * reasoning_quality

        # Success threshold
        return verification_score > 0.4, verification_score

    def _verify_text_solution(self, task, task_context, solution, reasoning_trace):
        """Verify a solution to a text-related task"""
        # Basic verification - could be much more sophisticated

        # For sentence completion tasks
        if "Complete the following sentence" in task:
            if "sentence" in task_context and solution.startswith(task_context["sentence"]):
                return True, 0.8
            # Check if solution is a complete sentence
            if solution.endswith(".") or solution.endswith("!") or solution.endswith("?"):
                return True, 0.6
            return False, 0.3

        # For opposite tasks
        if "What is the opposite of" in task:
            if "concept" in task_context:
                # Since we can't use external knowledge reliably here, just check that
                # the solution isn't the same as the concept
                if task_context["concept"].lower() not in solution.lower():
                    return True, 0.7
            return False, 0.2

        # For alphabetical ordering
        if "Arrange these words in alphabetical order" in task:
            if "words" in task_context:
                original_words = task_context["words"].split()
                solution_words = solution.split()

                # Check if solution has same number of words
                if len(original_words) != len(solution_words):
                    return False, 0.1

                # Check if solution is alphabetically sorted
                sorted_words = sorted(original_words)
                if solution_words == sorted_words:
                    return True, 1.0

                # Partial credit for partially correct ordering
                correct_positions = sum(1 for a, b in zip(solution_words, sorted_words) if a == b)
                return False, correct_positions / len(original_words)

        # Fall back to generic verification
        return self._verify_generic_solution(task, task_context, solution, reasoning_trace)

    def _verify_math_solution(self, task, task_context, solution, reasoning_trace):
        """Verify a solution to a math-related task"""
        # Extract expected result for simple arithmetic tasks
        if "Calculate:" in task:
            try:
                # Parse the expected result
                if "num1" in task_context and "num2" in task_context:
                    num1 = task_context["num1"]
                    num2 = task_context["num2"]

                    # Determine operation
                    if "+" in task:
                        expected = num1 + num2
                    elif "-" in task:
                        expected = num1 - num2
                    elif "×" in task or "*" in task:
                        expected = num1 * num2
                    elif "÷" in task or "/" in task:
                        expected = num1 / num2 if num2 != 0 else None
                    else:
                        return self._verify_generic_solution(task, task_context, solution, reasoning_trace)

                    # Extract numeric answer from solution
                    answer = self._extract_numeric_answer(solution)

                    if answer is not None and expected is not None:
                        # Check if answer is correct (with small tolerance for division)
                        if abs(answer - expected) < 0.01:
                            return True, 1.0
                        else:
                            # Score based on how close the answer is
                            error_ratio = min(1.0, abs(answer - expected) / max(1.0, abs(expected)))
                            score = max(0.0, 1.0 - error_ratio)
                            return False, score
            except:
                # If parsing fails, fall back to generic verification
                pass

        # For solving equations
        if "Solve the equation:" in task:
            try:
                if "num1" in task_context and "num2" in task_context and "num3" in task_context:
                    a = task_context["num1"]
                    b = task_context["num2"]
                    c = task_context["num3"]

                    # Calculate expected value of x
                    expected = (c - b) / a

                    # Extract numeric answer
                    answer = self._extract_numeric_answer(solution)

                    if answer is not None:
                        # Check if answer is correct (with small tolerance)
                        if abs(answer - expected) < 0.01:
                            return True, 1.0
                        else:
                            # Score based on how close the answer is
                            error_ratio = min(1.0, abs(answer - expected) / max(1.0, abs(expected)))
                            score = max(0.0, 1.0 - error_ratio)
                            return False, score
            except:
                pass

        # Fall back to generic verification
        return self._verify_generic_solution(task, task_context, solution, reasoning_trace)

    def _extract_numeric_answer(self, text):
        """Extract a numeric answer from a text solution"""
        import re

        # Try to find numbers in the text
        matches = re.findall(r'-?\d+\.?\d*', text)

        if matches:
            # Return the last number found (usually the final answer)
            try:
                return float(matches[-1])
            except:
                pass

        return None

    def _verify_code_solution(self, task, task_context, solution, reasoning_trace):
        """Verify a solution to a code-related task"""
        # For function writing tasks
        if "Write a function" in task or "Implement a function" in task:
            # Check if solution contains a function definition
            if "def " in solution and "return" in solution:
                # Simple syntax check
                try:
                    # Just check if it can be parsed, don't execute
                    compile(solution, '<string>', 'exec')

                    # Can't run code fully without using exec, so we rely on heuristics
                    # Check for proper indentation
                    lines = solution.strip().split('\n')
                    if len(lines) < 2:
                        return False, 0.2

                    # Check for indentation after function definition
                    if not any(line.startswith('    ') or line.startswith('\t') for line in lines[1:]):
                        return False, 0.3

                    # Check for function name match if requested
                    if "function_name" in task_context:
                        if f"def {task_context['function_name']}" in solution:
                            return True, 0.9
                        return False, 0.4

                    # Generic success for syntactically valid function
                    return True, 0.7
                except (SyntaxError, IndentationError):
                    return False, 0.1

        # For fixing buggy code
        if "Fix the error" in task and "buggy_code" in task_context:
            buggy_code = task_context["buggy_code"]

            # Check if solution is different from buggy code
            if solution == buggy_code:
                return False, 0.0

            # Check if solution is syntactically valid
            try:
                compile(solution, '<string>', 'exec')

                # Compare key elements to see if error was fixed
                buggy_lines = set(buggy_code.split('\n'))
                solution_lines = set(solution.split('\n'))

                # Check how many lines were changed
                changes = len(buggy_lines.symmetric_difference(solution_lines))

                # Good fixes typically change a small number of lines
                if 0 < changes <= 3:
                    return True, 0.8
                if changes > 3:
                    return True, 0.6  # Still valid but maybe not optimal

                return False, 0.3
            except (SyntaxError, IndentationError):
                return False, 0.1

        # For code analysis tasks
        if "What does this code do?" in task and "code_snippet" in task_context:
            # No way to automatically verify correctness of analysis
            # Check for relevance to code snippet
            code_words = set(re.findall(r'\w+', task_context["code_snippet"]))
            solution_words = set(re.findall(r'\w+', solution))

            overlap = len(code_words.intersection(solution_words)) / max(1, len(code_words))

            # Check for explanation-related terms
            explanation_terms = ["function", "returns", "calculates", "computes", "iterates", "checks"]
            has_explanation_terms = any(term in solution.lower() for term in explanation_terms)

            score = 0.5 * overlap + 0.5 * float(has_explanation_terms)
            return score > 0.6, score

        # Fall back to generic verification
        return self._verify_generic_solution(task, task_context, solution, reasoning_trace)

    def _verify_logic_solution(self, task, task_context, solution, reasoning_trace):
        """Verify a solution to a logic-related task"""
        # For sequence continuation
        if "sequence" in task_context and ("next number" in task or "next element" in task):
            sequence_str = task_context["sequence"]

            # For common sequences, we can check the answer
            sequence = [int(n.strip()) for n in sequence_str.split(',') if n.strip().isdigit()]

            if len(sequence) >= 3:
                # Try to extract the answer from solution
                answer = self._extract_numeric_answer(solution)

                if answer is not None:
                    # Check for arithmetic sequence
                    if len(sequence) >= 2 and all(sequence[i+1] - sequence[i] == sequence[1] - sequence[0] for i in range(len(sequence)-2)):
                        diff = sequence[1] - sequence[0]
                        expected = sequence[-1] + diff
                        if abs(answer - expected) < 0.01:
                            return True, 1.0

                    # Check for geometric sequence
                    if len(sequence) >= 2 and all(sequence[i+1] / sequence[i] == sequence[1] / sequence[0] for i in range(len(sequence)-2)) and sequence[0] != 0:
                        ratio = sequence[1] / sequence[0]
                        expected = sequence[-1] * ratio
                        if abs(answer - expected) < 0.01:
                            return True, 1.0

                    # Check for Fibonacci-like sequence
                    if len(sequence) >= 3 and all(sequence[i+2] == sequence[i+1] + sequence[i] for i in range(len(sequence)-3)):
                        expected = sequence[-1] + sequence[-2]
                        if abs(answer - expected) < 0.01:
                            return True, 1.0

                    # For other sequences, we can't easily verify
                    return False, 0.5

        # For true/false questions
        if "statement" in task_context and "true or false" in task.lower():
            # For some statements we know the answer
            statement = task_context["statement"].lower()

            known_answers = {
                "if a number is divisible by 4, then it is divisible by 2": True,
                "if a shape is a square, then it is a rectangle": True,
                "if a number is prime, then it is odd": False  # 2 is prime but even
            }

            if statement in known_answers:
                expected = known_answers[statement]
                if ("true" in solution.lower() and expected) or ("false" in solution.lower() and not expected):
                    return True, 1.0
                else:
                    return False, 0.0

        # For logical deduction with premises
        if "premise1" in task_context and "premise2" in task_context and "conclude" in task:
            # Some specific known valid conclusions
            premise_pair = (task_context["premise1"], task_context["premise2"])

            known_conclusions = {
                ("All men are mortal", "Socrates is a man"): "Socrates is mortal",
                ("If it rains, the ground gets wet", "It is raining"): "The ground gets wet",
                ("Either the butler or the maid did it", "The butler has an alibi"): "The maid did it"
            }

            if premise_pair in known_conclusions:
                expected = known_conclusions[premise_pair]
                similarity = self._text_similarity(expected, solution)
                if similarity > 0.7:
                    return True, similarity
                return False, similarity * 0.5

        # Fall back to generic verification
        return self._verify_generic_solution(task, task_context, solution, reasoning_trace)

    def _verify_creative_solution(self, task, task_context, solution, reasoning_trace):
        """Verify a solution to a creative task"""
        # Creative tasks are inherently subjective, so we look for structural elements

        # For poetry tasks
        if "poem" in task.lower() and "topic" in task_context:
            topic = task_context["topic"]

            # Check if solution contains the topic
            topic_present = topic.lower() in solution.lower()

            # Check if it has multiple lines (basic poem structure)
            multiple_lines = solution.count('\n') >= 2

            # Check for poetic devices (very simple check)
            poetic_elements = any(device in solution.lower() for device in
                                ["like", "as", "metaphor", "simile", "rhythm", "rhyme"])

            score = 0.3 * float(topic_present) + 0.3 * float(multiple_lines) + 0.4 * float(poetic_elements)
            return score >= 0.6, score

        # For character creation
        if "character" in task.lower() and "trait1" in task_context and "trait2" in task_context:
            trait1 = task_context["trait1"]
            trait2 = task_context["trait2"]

            # Check if both traits are mentioned
            trait1_present = trait1.lower() in solution.lower()
            trait2_present = trait2.lower() in solution.lower()

            # Check for character development elements
            character_elements = any(element in solution.lower() for element in
                                ["name", "background", "history", "appearance", "motivation"])

            score = 0.4 * float(trait1_present) + 0.4 * float(trait2_present) + 0.2 * float(character_elements)
            return score >= 0.6, score

        # For metaphor creation
        if "metaphor" in task.lower() and "concept" in task_context:
            concept = task_context["concept"]

            # Check if concept is mentioned
            concept_present = concept.lower() in solution.lower()

            # Check for comparison language
            comparison = any(word in solution.lower() for word in
                        ["like", "as", "is", "resembles", "similar", "compared"])

            score = 0.5 * float(concept_present) + 0.5 * float(comparison)
            return score >= 0.7, score

        # For story writing
        if "story" in task.lower() and "elements" in task_context:
            elements = task_context.get("elements", "").split()

            # Check if elements are included
            element_count = sum(1 for element in elements if element.lower() in solution.lower())
            element_ratio = element_count / max(1, len(elements))

            # Check for story structure (very simple)
            has_structure = "beginning" in solution.lower() or "end" in solution.lower() or solution.count('\n') >= 3

            score = 0.7 * element_ratio + 0.3 * float(has_structure)
            return score >= 0.6, score

        # Fall back to generic verification
        return self._verify_generic_solution(task, task_context, solution, reasoning_trace)

    def _verify_multimodal_solution(self, task, task_context, solution, reasoning_trace):
        """Verify a solution to a multimodal task"""
        # For image description tasks
        if "image_description" in task_context:
            image_desc = task_context["image_description"]

            # Check if solution relates to the image description
            related_to_image = self._text_similarity(image_desc, solution) > 0.3

            # Check for visual language
            visual_terms = ["see", "look", "appear", "visual", "image", "picture", "scene",
                        "view", "perspective", "foreground", "background", "color"]

            has_visual_terms = any(term in solution.lower() for term in visual_terms)

            score = 0.6 * float(related_to_image) + 0.4 * float(has_visual_terms)
            return score >= 0.5, score

        # For sound/audio description tasks
        if "sound_description" in task_context or "music_description" in task_context:
            audio_desc = task_context.get("sound_description", task_context.get("music_description", ""))

            # Check if solution relates to the audio description
            related_to_audio = self._text_similarity(audio_desc, solution) > 0.3

            # Check for auditory language
            audio_terms = ["hear", "sound", "audio", "noise", "music", "rhythm", "melody",
                        "tempo", "beat", "listen", "loud", "soft", "pitch", "tone"]

            has_audio_terms = any(term in solution.lower() for term in audio_terms)

            score = 0.6 * float(related_to_audio) + 0.4 * float(has_audio_terms)
            return score >= 0.5, score

        # For cross-modal integration tasks
        if "image_description" in task_context and ("sound_description" in task_context or "music_description" in task_context):
            # Check for integration of both modalities
            image_desc = task_context["image_description"]
            audio_desc = task_context.get("sound_description", task_context.get("music_description", ""))

            # Check if solution relates to both descriptions
            related_to_image = self._text_similarity(image_desc, solution) > 0.2
            related_to_audio = self._text_similarity(audio_desc, solution) > 0.2

            # Check for integration language
            integration_terms = ["combine", "together", "mix", "blend", "harmony",
                            "integrated", "synchronized", "paired", "matched"]

            has_integration_terms = any(term in solution.lower() for term in integration_terms)

            score = 0.4 * float(related_to_image) + 0.4 * float(related_to_audio) + 0.2 * float(has_integration_terms)
            return score >= 0.6, score

        # Fall back to generic verification
        return self._verify_generic_solution(task, task_context, solution, reasoning_trace)

    def _text_similarity(self, text1, text2):
        """Calculate a simple similarity score between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / max(1, len(union))


class RewardModel:
    """Calculates and applies rewards for autonomous learning"""

    def __init__(self, trainer):
        """Initialize the reward model with a reference to the trainer"""
        self.trainer = trainer
        self.model = trainer.model

        # Domain-specific reward modifiers
        self.domain_reward_modifiers = {
            TaskDomain.TEXT: 1.0,
            TaskDomain.MATH: 1.2,  # Slightly higher reward for math (harder to verify)
            TaskDomain.CODE: 1.2,  # Slightly higher reward for code (harder to verify)
            TaskDomain.LOGIC: 1.1,  # Slightly higher reward for logic
            TaskDomain.CREATIVE: 0.9,  # Slightly lower reward for creative (easier to "game")
            TaskDomain.MULTIMODAL: 1.3 if self.trainer.config.multimodal_enabled else 0.0  # Higher reward for multimodal
        }

        # Reasoning type reward modifiers
        self.reasoning_reward_modifiers = {
            ReasoningType.DEDUCTION: 1.0,
            ReasoningType.ABDUCTION: 1.1,  # Slightly higher reward for abduction (more creative)
            ReasoningType.INDUCTION: 1.2,  # Higher reward for induction (more generalizable)
            ReasoningType.TRIAL_ERROR: 0.9  # Slightly lower reward for trial & error
        }

        # Reward history
        self.reward_history = []

    def calculate_reward(self, verification_result, verification_score, domain, difficulty, reasoning_type, reasoning_trace):
        """Calculate the reward for a solution"""
        # Base reward depends on verification result and score
        base_reward = verification_score

        # Apply domain-specific modifier
        domain_modifier = self.domain_reward_modifiers.get(domain, 1.0)

        # Apply reasoning type modifier
        reasoning_modifier = self.reasoning_reward_modifiers.get(reasoning_type, 1.0)

        # Apply difficulty modifier (higher reward for harder tasks)
        difficulty_modifier = 1.0 + (difficulty * 0.2)  # +20% per difficulty level

        # Apply reasoning trace quality modifier
        trace_quality = min(1.0, len(reasoning_trace) / 500)  # Cap at 1.0
        trace_modifier = 0.8 + (0.4 * trace_quality)  # 0.8 to 1.2 based on trace quality

        # Calculate final reward
        reward = base_reward * domain_modifier * reasoning_modifier * difficulty_modifier * trace_modifier

        # Cap reward to reasonable range
        reward = max(0.0, min(2.0, reward))

        # Record reward
        self.reward_history.append({
            "verification_result": verification_result,
            "verification_score": verification_score,
            "domain": domain,
            "difficulty": difficulty,
            "reasoning_type": reasoning_type,
            "reward": reward,
            "timestamp": time.time()
        })

        return reward

    def apply_reward(self, reward, reasoning_trace):
        """Apply the reward to the model to reinforce learning"""
        # This is where we would apply reinforcement to the model
        # For SAM, we can reinforce concepts and connections based on the reasoning trace

        # Only apply significant rewards
        if reward < 0.1:
            return

        # Get concepts involved in the reasoning trace
        concepts_involved = self._extract_concepts_from_trace(reasoning_trace)

        # Apply reward proportionally to these concepts
        for concept_id, importance in concepts_involved:
            # Strengthen concept usage
            if concept_id < len(self.model.concept_bank.concept_frequencies):
                with torch.no_grad():
                    # Increase frequency by an amount proportional to reward and importance
                    adjustment = int(reward * 10 * importance)
                    if adjustment > 0:
                        self.model.concept_bank.concept_frequencies[concept_id] += adjustment

                        # Also strengthen related concepts
                        if concept_id in self.model.concept_bank.related_concepts:
                            for related_id in self.model.concept_bank.related_concepts[concept_id][:3]:  # Top 3
                                if related_id < len(self.model.concept_bank.concept_frequencies):
                                    self.model.concept_bank.concept_frequencies[related_id] += adjustment // 2

        # If significant reward, consider personalizing key concepts
        if reward > 1.0 and hasattr(self.model, "intent"):
            for concept_id, importance in concepts_involved[:3]:  # Top 3 concepts
                self.model.intent.personalize_concept(concept_id, reward * 0.05)

    def _extract_concepts_from_trace(self, reasoning_trace):
        """Extract concept IDs involved in a reasoning trace"""
        # This is a simplified implementation
        # A real implementation would analyze the trace in depth

        # For now, we'll just process the model's most recent thought state
        concepts_involved = []

        if hasattr(self.model, "thought_state") and self.model.thought_state.thought_memory:
            # Get the current thought state
            current_thought = self.model.thought_state.thought_memory[-1]

            # Project to concept space
            concept_projection = self.model.thought_state.project_to_concept_space(current_thought)

            # Find similar concepts
            concept_vector = concept_projection.detach().cpu().mean(dim=(0, 1))
            similar_concepts = self.model.concept_bank.find_similar_concepts(concept_vector, top_k=10)

            # Add to concepts involved with decreasing importance
            for i, (concept_id, similarity) in enumerate(similar_concepts):
                importance = (10 - i) / 10  # Decreasing importance: 1.0, 0.9, 0.8, ...
                concepts_involved.append((concept_id, importance * similarity))

        return concepts_involved


class ReasoningEngine:
    """Implements different reasoning strategies for solving tasks"""

    def __init__(self, trainer):
        """Initialize the reasoning engine with a reference to the trainer"""
        self.trainer = trainer
        self.model = trainer.model

        # Reasoning strategies
        self.reasoning_strategies = {
            ReasoningType.DEDUCTION: self._deductive_reasoning,
            ReasoningType.ABDUCTION: self._abductive_reasoning,
            ReasoningType.INDUCTION: self._inductive_reasoning,
            ReasoningType.TRIAL_ERROR: self._trial_error_reasoning
        }

        # Domain-specific reasoning preferences
        self.domain_reasoning_preferences = {
            TaskDomain.TEXT: [ReasoningType.DEDUCTION, ReasoningType.INDUCTION, ReasoningType.ABDUCTION, ReasoningType.TRIAL_ERROR],
            TaskDomain.MATH: [ReasoningType.DEDUCTION, ReasoningType.TRIAL_ERROR, ReasoningType.INDUCTION, ReasoningType.ABDUCTION],
            TaskDomain.CODE: [ReasoningType.DEDUCTION, ReasoningType.TRIAL_ERROR, ReasoningType.INDUCTION, ReasoningType.ABDUCTION],
            TaskDomain.LOGIC: [ReasoningType.DEDUCTION, ReasoningType.ABDUCTION, ReasoningType.INDUCTION, ReasoningType.TRIAL_ERROR],
            TaskDomain.CREATIVE: [ReasoningType.ABDUCTION, ReasoningType.INDUCTION, ReasoningType.TRIAL_ERROR, ReasoningType.DEDUCTION],
            TaskDomain.MULTIMODAL: [ReasoningType.ABDUCTION, ReasoningType.INDUCTION, ReasoningType.DEDUCTION, ReasoningType.TRIAL_ERROR]
        }

        # Reasoning history
        self.reasoning_history = []

    def select_reasoning_type(self, domain, task):
        """Select an appropriate reasoning type for a task"""
        # Default preferences
        preferences = self.domain_reasoning_preferences.get(
            domain, [ReasoningType.DEDUCTION, ReasoningType.ABDUCTION, ReasoningType.INDUCTION, ReasoningType.TRIAL_ERROR]
        )

        # Check task keywords to adjust preferences
        task_lower = task.lower()

        # For tasks involving patterns or sequences
        if any(kw in task_lower for kw in ["pattern", "sequence", "next", "series", "predict"]):
            preferences = [ReasoningType.INDUCTION] + [p for p in preferences if p != ReasoningType.INDUCTION]

        # For tasks involving figuring out unknown operations or rules
        if any(kw in task_lower for kw in ["identify", "determine", "operation", "rule", "guess"]):
            preferences = [ReasoningType.ABDUCTION] + [p for p in preferences if p != ReasoningType.ABDUCTION]

        # For tasks involving clear logical steps
        if any(kw in task_lower for kw in ["calculate", "solve", "find", "compute"]):
            preferences = [ReasoningType.DEDUCTION] + [p for p in preferences if p != ReasoningType.DEDUCTION]

        # For creative or generative tasks
        if any(kw in task_lower for kw in ["create", "design", "invent", "imagine", "describe"]):
            preferences = [ReasoningType.ABDUCTION, ReasoningType.INDUCTION] + [p for p in preferences if p not in [ReasoningType.ABDUCTION, ReasoningType.INDUCTION]]

        # Occasionally use a random strategy for exploration (10% of the time)
        if random.random() < 0.1:
            return random.choice(list(self.reasoning_strategies.keys()))

        # Return the top preference
        return preferences[0]

    def solve_task(self, task, task_context, reasoning_type):
        """Solve a task using the specified reasoning type"""
        # Get the appropriate reasoning strategy
        reasoning_strategy = self.reasoning_strategies.get(
            reasoning_type, self._deductive_reasoning
        )

        # Apply the reasoning strategy
        solution, reasoning_trace = reasoning_strategy(task, task_context)

        # Record reasoning history
        self.reasoning_history.append({
            "task": task,
            "reasoning_type": reasoning_type,
            "solution": solution,
            "trace_length": len(reasoning_trace),
            "timestamp": time.time()
        })

        return solution, reasoning_trace

    def _deductive_reasoning(self, task, task_context):
        """Apply deductive reasoning to solve a task (applying general rules to specific instances)"""
        # Construct a detailed reasoning prompt
        prompt = f"Task: {task}\n\nI will solve this step-by-step using deductive reasoning, applying general rules to this specific case:\n\n"

        # Add context-specific reasoning structure
        if "Calculate:" in task or "equation" in task:
            prompt += "1. Let me identify the numbers and operations involved.\n"
            prompt += "2. I will apply the relevant mathematical rules.\n"
            prompt += "3. I will calculate the result systematically.\n\n"

        elif "function" in task.lower() or "code" in task.lower():
            prompt += "1. Let me analyze what functionality is required.\n"
            prompt += "2. I will design a function that meets these requirements.\n"
            prompt += "3. I will implement the function using proper syntax.\n"
            prompt += "4. I will verify the function works as expected.\n\n"

        elif "sequence" in task.lower() or "pattern" in task.lower():
            prompt += "1. Let me examine the sequence to identify the pattern.\n"
            prompt += "2. I will test whether it's an arithmetic, geometric, or other type of sequence.\n"
            prompt += "3. I will apply the identified pattern to determine the next element(s).\n\n"

        else:
            prompt += "1. Let me break down the task into its components.\n"
            prompt += "2. I will identify the relevant rules or principles that apply.\n"
            prompt += "3. I will apply these rules systematically to reach a conclusion.\n\n"

        prompt += "Now, let me work through this problem:\n"

        # Generate the solution using the model
        solution_text = self.model.generate(
            input_text=prompt,
            max_length=500,
            temperature=0.7,
            private_context=True
        )

        # Extract the reasoning trace and final solution
        reasoning_trace = solution_text

        # The final solution is typically at the end, after final reasoning
        final_solution_markers = [
            "Therefore, the answer is",
            "The solution is",
            "In conclusion,",
            "Thus,",
            "Final answer:",
            "Result:"
        ]

        solution = solution_text
        for marker in final_solution_markers:
            if marker in solution_text:
                solution = solution_text.split(marker, 1)[1].strip()
                break

        return solution, reasoning_trace

    def _abductive_reasoning(self, task, task_context):
        """Apply abductive reasoning to solve a task (inferring the most likely explanation)"""
        # Construct a detailed reasoning prompt
        prompt = f"Task: {task}\n\nI will solve this using abductive reasoning, finding the most likely explanation:\n\n"

        # Add context-specific reasoning structure
        if "identify" in task.lower() or "determine" in task.lower():
            prompt += "1. Let me observe the given information carefully.\n"
            prompt += "2. I will generate multiple hypotheses that could explain the observation.\n"
            prompt += "3. I will evaluate each hypothesis based on simplicity and explanatory power.\n"
            prompt += "4. I will select the most plausible explanation.\n\n"

        elif "create" in task.lower() or "design" in task.lower():
            prompt += "1. Let me understand the desired outcome or goal.\n"
            prompt += "2. I will consider different approaches that could achieve this goal.\n"
            prompt += "3. I will imagine the consequences of each approach.\n"
            prompt += "4. I will select the approach most likely to succeed.\n\n"

        else:
            prompt += "1. Let me gather all the available clues and information.\n"
            prompt += "2. I will formulate several possible explanations.\n"
            prompt += "3. I will evaluate which explanation best fits the evidence.\n"
            prompt += "4. I will choose the most likely explanation.\n\n"

        prompt += "Let me explore multiple possibilities:\n"

        # Generate the solution using the model
        solution_text = self.model.generate(
            input_text=prompt,
            max_length=500,
            temperature=0.8,  # Slightly higher temperature for more creative explanations
            private_context=True
        )

        # Extract the reasoning trace and final solution
        reasoning_trace = solution_text

        # The final solution is typically at the end, after considering alternatives
        final_solution_markers = [
            "The most likely explanation is",
            "Therefore, I conclude that",
            "The best hypothesis is",
            "The most plausible answer is",
            "In conclusion,"
        ]

        solution = solution_text
        for marker in final_solution_markers:
            if marker in solution_text:
                solution = solution_text.split(marker, 1)[1].strip()
                break

        return solution, reasoning_trace

    def _inductive_reasoning(self, task, task_context):
        """Apply inductive reasoning to solve a task (deriving general principles from specific instances)"""
        # Construct a detailed reasoning prompt
        prompt = f"Task: {task}\n\nI will solve this using inductive reasoning, identifying patterns to form a general rule:\n\n"

        # Add context-specific reasoning structure
        if "sequence" in task.lower() or "pattern" in task.lower():
            prompt += "1. Let me examine the specific examples in the sequence.\n"
            prompt += "2. I will look for recurring patterns or relationships.\n"
            prompt += "3. I will formulate a general rule that describes the pattern.\n"
            prompt += "4. I will apply this rule to predict the next elements.\n\n"

        elif "categorize" in task.lower() or "classify" in task.lower():
            prompt += "1. Let me examine the specific examples given.\n"
            prompt += "2. I will identify common properties among similar examples.\n"
            prompt += "3. I will formulate general categories based on these properties.\n"
            prompt += "4. I will classify the examples according to these categories.\n\n"

        else:
            prompt += "1. Let me gather specific instances or examples related to the task.\n"
            prompt += "2. I will observe patterns or commonalities among these instances.\n"
            prompt += "3. I will formulate a general principle that explains these patterns.\n"
            prompt += "4. I will apply this principle to solve the task.\n\n"

        prompt += "Let me start by identifying patterns:\n"

        # Generate the solution using the model
        solution_text = self.model.generate(
            input_text=prompt,
            max_length=500,
            temperature=0.7,
            private_context=True
        )

        # Extract the reasoning trace and final solution
        reasoning_trace = solution_text

        # The final solution typically follows the pattern identification
        final_solution_markers = [
            "Based on this pattern,",
            "The general rule is",
            "Therefore, the answer is",
            "Applying this principle,",
            "In conclusion,"
        ]

        solution = solution_text
        for marker in final_solution_markers:
            if marker in solution_text:
                solution = solution_text.split(marker, 1)[1].strip()
                break

        return solution, reasoning_trace

    def _trial_error_reasoning(self, task, task_context):
        """Apply trial and error reasoning to solve a task (testing multiple approaches)"""
        # Construct a detailed reasoning prompt
        prompt = f"Task: {task}\n\nI will solve this through trial and error, systematically testing different approaches:\n\n"

        # Add context-specific reasoning structure
        if "equation" in task.lower() or "solve" in task.lower():
            prompt += "1. Let me try a few possible solutions and see which one works.\n"
            prompt += "2. I will start with a reasonable guess and refine it.\n"
            prompt += "3. I will test each potential solution against the constraints.\n"
            prompt += "4. I will select the solution that satisfies all conditions.\n\n"

        elif "code" in task.lower() or "function" in task.lower():
            prompt += "1. Let me implement a basic solution and test it.\n"
            prompt += "2. I will identify any errors or edge cases.\n"
            prompt += "3. I will modify the solution to address these issues.\n"
            prompt += "4. I will iterate until I have a working solution.\n\n"

        else:
            prompt += "1. Let me start with a plausible approach to the problem.\n"
            prompt += "2. I will test this approach and evaluate its success.\n"
            prompt += "3. I will adjust my approach based on the results.\n"
            prompt += "4. I will continue iterating until I find a satisfactory solution.\n\n"

        prompt += "Let me try different approaches:\n"

        # Generate the solution using the model
        solution_text = self.model.generate(
            input_text=prompt,
            max_length=1000,  # Longer to allow for multiple trials
            temperature=1.2,  # Higher temperature for more exploration
            private_context=True
        )

        # Extract the reasoning trace and final solution
        reasoning_trace = solution_text

        # The final solution typically follows after several attempts
        final_solution_markers = [
            "After trying several approaches,",
            "The approach that works is",
            "The final solution is",
            "Therefore, the answer is",
            "This solution works because",
            "In conclusion,"
        ]

        solution = solution_text
        for marker in final_solution_markers:
            if marker in solution_text:
                solution = solution_text.split(marker, 1)[1].strip()
                break

        return solution, reasoning_trace

###########################################
# MAIN EXECUTION
###########################################

def run_interactive_sam(model: SAM):
    """Run a SAM instance in an interactive loop."""
    model.start_services()

    print("\nSAM is ready for interaction. Type 'exit' to quit.")
    print("Special commands: 'save', 'dream', 'stats', 'evolve', 'hive', 'private'")

    history = []
    private_mode = False

    while True:
        try:
            mode_prefix = " (private)" if private_mode else ""
            prefix = f"\nYou{mode_prefix}: "
            user_input = input(prefix)

            if user_input.lower() == 'exit': break
            elif user_input.lower() == 'save':
                save_path = model.save()
                print(f"\nSAM: Model saved to {save_path}")
                continue
            elif user_input.lower() == 'dream':
                print("\nSAM: Dreaming...")
                results = model.dreaming.dream_cycle(duration_minutes=0.5)
                print(f"\nSAM: Dreaming complete. Synthesized {results.get('syntheses', 0)} new concepts.")
                continue
            elif user_input.lower() == 'stats':
                status = model.get_status()
                print("\nSAM: Current stats:")
                print(json.dumps(status, indent=2, default=str))
                continue
            elif user_input.lower() == 'evolve':
                print("\nSAM: Evolving...")
                model.evolve()
                print(f"\nSAM: Evolution cycle complete.")
                continue
            elif user_input.lower() == 'private':
                private_mode = not private_mode
                print(f"\nSAM: Private mode {'enabled' if private_mode else 'disabled'}.")
                continue

            # Process and generate
            sam_response = model.generate(
                input_text=user_input,
                max_length=min(len(user_input) * 2 + 64, 512),
                temperature=0.7,
            )
            print(f"\nSAM: {sam_response}")
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": sam_response})

        except KeyboardInterrupt:
            print("\nInterrupt received. Exiting.")
            break
        except Exception as e:
            print(f"\nAn error occurred in the interactive loop: {e}")
            logger.error(f"Error in interaction: {e}", exc_info=True)
            break

    # Stop services and save before exiting
    model.stop_services()
    model.save()
    print("\nSAM's state has been saved. Goodbye!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Synergistic Autonomous Machine (SAM).")
    parser.add_argument("--load_path", type=str, default=None, help="Path to a saved model checkpoint directory to load.")
    parser.add_argument("--hive_mind", action="store_true", help="Enable hive mind capabilities.")
    parser.add_argument("--multimodal", action="store_true", help="Enable multimodal capabilities.")
    parser.add_argument("--train_file", type=str, default=None, help="Path to a JSON file for training.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    args = parser.parse_args()

    sam_model = None
    try:
        if args.load_path and os.path.exists(args.load_path):
            print(f"Loading model from {args.load_path}...")
            sam_model = SAM.load(args.load_path)
        else:
            print("Creating new SAM model...")
            config = SAMConfig()
            if args.hive_mind:
                config.hive_enabled = True
                config.hive_server_url = "http://localhost:8765"  # Example URL
            if args.multimodal:
                config.multimodal_enabled = True
            sam_model = SAM(config)

        if sam_model:
            if args.train_file:
                print(f"Starting training on {args.train_file} for {args.epochs} epochs...")
                trainer = SAMTrainer(sam_model)
                trainer.train_from_json(json_path=args.train_file, epochs=args.epochs)
                print("Training finished. Saving final model.")
                sam_model.save()
            else:
                run_interactive_sam(sam_model)
    except Exception as e:
        logger.critical(f"A critical error occurred at startup: {e}", exc_info=True)

