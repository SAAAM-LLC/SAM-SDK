"""
Advanced synchronization mechanisms for SAM Hive Mind SDK
"""
import json
import logging
import threading
import time
import zlib
import base64
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any

import torch
import numpy as np

logger = logging.getLogger("sam_hive_sdk")

class SyncStrategy(Enum):
    """Strategies for hive mind synchronization"""
    FULL = "full"               # Sync everything
    SELECTIVE = "selective"     # Sync only important concepts/experiences
    DIFFERENTIAL = "differential"  # Sync only changes since last sync
    FEDERATED = "federated"     # Sync in a federated learning style
    DOMAIN_SPECIFIC = "domain_specific"  # Sync only specific knowledge domains


class SyncDomain(Enum):
    """Knowledge domains for selective synchronization"""
    CORE = "core"               # Core operational concepts
    LANGUAGE = "language"       # Language understanding
    REASONING = "reasoning"     # Reasoning capabilities
    DOMAIN_KNOWLEDGE = "domain_knowledge"  # Specific subject matter
    CREATIVE = "creative"       # Creative capabilities
    MULTIMODAL = "multimodal"   # Multimodal understanding
    PERSONALITY = "personality"  # Personality traits
    SKILLS = "skills"           # Operational skills


class ConceptRank:
    """
    Ranks concepts for synchronization importance
    
    Determines which concepts should be prioritized during selective
    synchronization based on various metrics.
    """
    
    def __init__(self):
        """Initialize concept ranker"""
        self.importance_cache = {}  # concept_id -> importance score
        self.cache_timestamp = {}   # concept_id -> last calculation time
        self.cache_expiry = 3600    # 1 hour
    
    def rank_concepts(self, 
                     concepts: List[Dict], 
                     limit: int = 1000,
                     strategy: SyncStrategy = SyncStrategy.SELECTIVE,
                     domains: List[SyncDomain] = None) -> List[Dict]:
        """
        Rank concepts by importance for synchronization
        
        Args:
            concepts: List of concept dictionaries
            limit: Maximum number of concepts to return
            strategy: Synchronization strategy
            domains: Specific domains to include
            
        Returns:
            List[Dict]: Ranked concepts
        """
        # Apply appropriate ranking based on strategy
        if strategy == SyncStrategy.FULL:
            # No ranking needed, just respect the limit
            return concepts[:limit]
            
        elif strategy == SyncStrategy.DIFFERENTIAL:
            # Rank by recency
            return sorted(
                concepts, 
                key=lambda c: c.get('metadata', {}).get('last_modified', 0),
                reverse=True
            )[:limit]
            
        elif strategy == SyncStrategy.DOMAIN_SPECIFIC and domains:
            # Filter by domains
            domain_concepts = [
                c for c in concepts
                if any(d.value in c.get('metadata', {}).get('domains', []) for d in domains)
            ]
            # Then rank by importance
            return self._rank_by_importance(domain_concepts, limit)
            
        else:  # Default to selective ranking
            return self._rank_by_importance(concepts, limit)
    
    def _rank_by_importance(self, concepts: List[Dict], limit: int) -> List[Dict]:
        """
        Rank concepts by calculated importance
        
        Args:
            concepts: List of concept dictionaries
            limit: Maximum number of concepts to return
            
        Returns:
            List[Dict]: Ranked concepts
        """
        # Calculate or retrieve importance for each concept
        ranked_concepts = []
        
        for concept in concepts:
            concept_id = concept.get('local_id')
            
            # Get cached importance if available and fresh
            if (concept_id in self.importance_cache and
                time.time() - self.cache_timestamp.get(concept_id, 0) < self.cache_expiry):
                importance = self.importance_cache[concept_id]
            else:
                # Calculate importance
                importance = self._calculate_concept_importance(concept)
                
                # Cache it
                self.importance_cache[concept_id] = importance
                self.cache_timestamp[concept_id] = time.time()
            
            ranked_concepts.append((concept, importance))
        
        # Sort by importance (descending)
        ranked_concepts.sort(key=lambda x: x[1], reverse=True)
        
        # Return concepts up to limit
        return [c for c, _ in ranked_concepts[:limit]]
    
    def _calculate_concept_importance(self, concept: Dict) -> float:
        """
        Calculate importance score for a concept
        
        Args:
            concept: Concept dictionary
            
        Returns:
            float: Importance score (0-1)
        """
        # Start with base score
        score = 0.0
        
        # Get metadata
        metadata = concept.get('metadata', {})
        
        # Factor 1: Usage frequency (0-0.3)
        frequency = concept.get('frequency', 0)
        freq_score = min(0.3, frequency / 1000 * 0.3)
        score += freq_score
        
        # Factor 2: Recency (0-0.2)
        last_used = metadata.get('last_used', 0)
        if last_used > 0:
            recency = min(1.0, (time.time() - last_used) / (86400 * 30))  # 30 days max
            recency_score = 0.2 * (1 - recency)
            score += recency_score
        
        # Factor 3: Connectivity (0-0.2)
        related_concepts = metadata.get('related_concepts', [])
        connectivity_score = min(0.2, len(related_concepts) / 50 * 0.2)
        score += connectivity_score
        
        # Factor 4: Type importance (0-0.15)
        concept_type = metadata.get('type', '')
        type_score = 0.0
        if concept_type == 'character_sequence':
            type_score = 0.05
        elif concept_type == 'semantic':
            type_score = 0.10
        elif concept_type == 'merged':
            type_score = 0.15
        score += type_score
        
        # Factor 5: Domain importance (0-0.15)
        domains = metadata.get('domains', [])
        domain_score = 0.0
        if SyncDomain.CORE.value in domains:
            domain_score = 0.15
        elif SyncDomain.REASONING.value in domains:
            domain_score = 0.12
        elif SyncDomain.LANGUAGE.value in domains:
            domain_score = 0.10
        elif domains:  # Any other domain
            domain_score = 0.08
        score += domain_score
        
        return score


class SyncManager:
    """
    Manages synchronization between SAM instances
    
    Implements different synchronization strategies and handles
    the details of synchronizing concepts, experiences, and thoughts.
    """
    
    def __init__(self, sam_instance, config=None):
        """
        Initialize sync manager
        
        Args:
            sam_instance: SAM instance to manage
            config: Configuration options
        """
        self.sam = sam_instance
        self.config = config or {}
        
        # Synchronization state
        self.last_sync_time = 0
        self.sync_history = []
        self.sync_errors = []
        
        # Tracking for differential sync
        self.last_sync_hashes = {}  # node_id -> {concept_id: hash}
        
        # Sync components
        self.concept_ranker = ConceptRank()
        
        # Sync monitoring
        self.sync_stats = {
            'total_syncs': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'concepts_sent': 0,
            'concepts_received': 0,
            'experiences_sent': 0,
            'experiences_received': 0
        }
        
        logger.info("SyncManager initialized")
    
    def sync_with_node(self, 
                      target_node, 
                      strategy: SyncStrategy = SyncStrategy.SELECTIVE,
                      domains: List[SyncDomain] = None,
                      force: bool = False) -> Dict:
        """
        Synchronize with another node
        
        Args:
            target_node: Target node for synchronization
            strategy: Synchronization strategy to use
            domains: Specific domains to synchronize
            force: Force sync even if interval hasn't elapsed
            
        Returns:
            Dict: Synchronization results
        """
        # Check sync interval
        if not force and time.time() - self.last_sync_time < self.config.get('sync_interval', 60):
            return {
                'success': True,
                'message': 'Skipped (interval not elapsed)',
                'target_node': target_node.node_id
            }
            
        try:
            start_time = time.time()
            
            # Get concepts to sync
            concepts = self._prepare_concepts_for_sync(target_node.node_id, strategy, domains)
            
            # Get experiences to sync
            experiences = self._prepare_experiences_for_sync(target_node.node_id, strategy, domains)
            
            # Get thought state to sync
            thought = self._prepare_thought_for_sync(target_node.node_id, strategy)
            
            # Prepare sync payload
            payload = {
                'node_id': self.sam.hive_identity,
                'timestamp': time.time(),
                'strategy': strategy.value,
                'concepts': concepts,
                'experiences': experiences,
                'thought': thought.tolist() if thought is not None else None,
                'domains': [d.value for d in domains] if domains else None
            }
            
            # Perform the sync (implementation depends on target node type)
            sync_result = self._perform_sync(target_node, payload)
            
            # Process received data
            if sync_result.get('success', False):
                self._process_sync_response(target_node.node_id, sync_result, strategy, domains)
            
            # Update sync time and history
            self.last_sync_time = time.time()
            duration = time.time() - start_time
            
            # Record sync history
            sync_record = {
                'timestamp': time.time(),
                'target_node': target_node.node_id,
                'strategy': strategy.value,
                'domains': [d.value for d in domains] if domains else None,
                'sent_concepts': len(concepts),
                'received_concepts': len(sync_result.get('concepts', [])),
                'sent_experiences': len(experiences),
                'received_experiences': len(sync_result.get('experiences', [])),
                'duration': duration,
                'success': sync_result.get('success', False)
            }
            self.sync_history.append(sync_record)
            
            # Update stats
            self.sync_stats['total_syncs'] += 1
            if sync_result.get('success', False):
                self.sync_stats['successful_syncs'] += 1
                self.sync_stats['concepts_sent'] += len(concepts)
                self.sync_stats['concepts_received'] += len(sync_result.get('concepts', []))
                self.sync_stats['experiences_sent'] += len(experiences)
                self.sync_stats['experiences_received'] += len(sync_result.get('experiences', []))
            else:
                self.sync_stats['failed_syncs'] += 1
                
            # Return results
            return {
                'success': sync_result.get('success', False),
                'sent_concepts': len(concepts),
                'received_concepts': len(sync_result.get('concepts', [])),
                'sent_experiences': len(experiences),
                'received_experiences': len(sync_result.get('experiences', [])),
                'duration': duration,
                'target_node': target_node.node_id
            }
            
        except Exception as e:
            logger.error(f"Error during sync with {target_node.node_id}: {e}")
            
            # Record error
            self.sync_errors.append({
                'timestamp': time.time(),
                'target_node': target_node.node_id,
                'error': str(e)
            })
            
            # Update stats
            self.sync_stats['total_syncs'] += 1
            self.sync_stats['failed_syncs'] += 1
            
            return {
                'success': False,
                'error': str(e),
                'target_node': target_node.node_id
            }
    
    def sync_with_network(self, 
                         network,
                         strategy: SyncStrategy = SyncStrategy.SELECTIVE,
                         domains: List[SyncDomain] = None,
                         node_limit: int = None) -> Dict:
        """
        Synchronize with multiple nodes in a network
        
        Args:
            network: Network containing nodes
            strategy: Synchronization strategy to use
            domains: Specific domains to synchronize
            node_limit: Maximum number of nodes to sync with
            
        Returns:
            Dict: Aggregated synchronization results
        """
        # Get nodes to sync with
        if hasattr(network, 'get_active_nodes'):
            nodes = network.get_active_nodes()
        else:
            nodes = list(network.nodes.values())
            
        # Limit number of nodes if specified
        if node_limit and len(nodes) > node_limit:
            nodes = nodes[:node_limit]
            
        # Sync with each node
        results = []
        for node in nodes:
            result = self.sync_with_node(node, strategy, domains)
            results.append(result)
            
        # Aggregate results
        success_count = sum(1 for r in results if r.get('success', False))
        sent_concepts = sum(r.get('sent_concepts', 0) for r in results)
        received_concepts = sum(r.get('received_concepts', 0) for r in results)
        sent_experiences = sum(r.get('sent_experiences', 0) for r in results)
        received_experiences = sum(r.get('received_experiences', 0) for r in results)
        
        return {
            'success': success_count == len(results),
            'node_count': len(nodes),
            'success_count': success_count,
            'sent_concepts': sent_concepts,
            'received_concepts': received_concepts,
            'sent_experiences': sent_experiences,
            'received_experiences': received_experiences,
            'detailed_results': results
        }
    
    def _prepare_concepts_for_sync(self, 
                                 target_node_id: str, 
                                 strategy: SyncStrategy,
                                 domains: List[SyncDomain] = None) -> List[Dict]:
        """
        Prepare concepts for synchronization
        
        Args:
            target_node_id: ID of target node
            strategy: Synchronization strategy
            domains: Specific domains to include
            
        Returns:
            List[Dict]: Concepts to synchronize
        """
        # Get raw concepts from SAM
        raw_concepts = self.sam.concept_bank.get_concepts_for_sync(
            limit=self.config.get('sync_concept_limit', 1000)
        )
        
        # Add domain metadata if missing
        for concept in raw_concepts:
            metadata = concept.get('metadata', {})
            if 'domains' not in metadata:
                # Infer domains based on concept type and content
                inferred_domains = self._infer_concept_domains(concept)
                metadata['domains'] = inferred_domains
                concept['metadata'] = metadata
        
        # Apply strategy-specific filtering
        if strategy == SyncStrategy.DIFFERENTIAL:
            # Only include concepts that have changed since last sync
            filtered_concepts = []
            for concept in raw_concepts:
                concept_id = concept.get('local_id')
                # Generate hash of concept for comparison
                concept_hash = self._hash_concept(concept)
                
                # Check if changed since last sync
                last_hash = self.last_sync_hashes.get(target_node_id, {}).get(concept_id)
                if last_hash != concept_hash:
                    filtered_concepts.append(concept)
                    
                    # Update hash
                    if target_node_id not in self.last_sync_hashes:
                        self.last_sync_hashes[target_node_id] = {}
                    self.last_sync_hashes[target_node_id][concept_id] = concept_hash
                    
            raw_concepts = filtered_concepts
            
        # Use ConceptRank to prioritize concepts
        return self.concept_ranker.rank_concepts(
            raw_concepts,
            limit=self.config.get('sync_concept_limit', 1000),
            strategy=strategy,
            domains=domains
        )
    
    def _prepare_experiences_for_sync(self, 
                                    target_node_id: str, 
                                    strategy: SyncStrategy,
                                    domains: List[SyncDomain] = None) -> List[Dict]:
        """
        Prepare experiences for synchronization
        
        Args:
            target_node_id: ID of target node
            strategy: Synchronization strategy
            domains: Specific domains to include
            
        Returns:
            List[Dict]: Experiences to synchronize
        """
        # Get raw experiences from SAM
        raw_experiences = self.sam.experience_manager.get_experiences_for_sync(
            limit=self.config.get('sync_experience_limit', 20)
        )
        
        # Apply domain filtering if needed
        if strategy == SyncStrategy.DOMAIN_SPECIFIC and domains:
            filtered_experiences = []
            for exp in raw_experiences:
                exp_domains = exp.get('metadata', {}).get('domains', [])
                # Include if any specified domain matches
                if any(d.value in exp_domains for d in domains):
                    filtered_experiences.append(exp)
            raw_experiences = filtered_experiences
            
        # Apply selective filtering if needed
        if strategy == SyncStrategy.SELECTIVE:
            # Prioritize experiences by recency and importance
            sorted_experiences = sorted(
                raw_experiences,
                key=lambda e: (
                    e.get('metadata', {}).get('importance', 0.5),
                    e.get('timestamp', 0)
                ),
                reverse=True
            )
            raw_experiences = sorted_experiences[:self.config.get('sync_experience_limit', 20)]
            
        return raw_experiences
    
    def _prepare_thought_for_sync(self, 
                                target_node_id: str, 
                                strategy: SyncStrategy) -> Optional[torch.Tensor]:
        """
        Prepare thought state for synchronization
        
        Args:
            target_node_id: ID of target node
            strategy: Synchronization strategy
            
        Returns:
            Optional[torch.Tensor]: Thought tensor or None
        """
        # Get thought state from SAM
        thought = self.sam.thought_state.get_shared_thought()
        
        # Apply strategy-specific processing
        if strategy == SyncStrategy.SELECTIVE or strategy == SyncStrategy.FEDERATED:
            # Apply dimensionality reduction for more efficient sync
            if thought is not None and self.config.get('compress_thoughts', False):
                # Simple dimensionality reduction (in real implementation would be more sophisticated)
                reduced_dim = min(thought.shape[-1], 512)  # Reduce to at most 512 dimensions
                if reduced_dim < thought.shape[-1]:
                    # This is a placeholder - real implementation would use proper dim reduction
                    thought = thought[:, :, :reduced_dim]
        
        return thought
    
    def _perform_sync(self, target_node, payload: Dict) -> Dict:
        """
        Perform the actual synchronization with a node
        
        Args:
            target_node: Target node
            payload: Sync payload
            
        Returns:
            Dict: Sync response
        """
        # Implementation depends on node type
        if hasattr(target_node, 'sync'):
            # Direct sync with a node object
            return target_node.sync(payload)
        elif hasattr(target_node, 'node_id'):
            # Sync via network request to a remote node
            # This is a placeholder - real implementation would use proper network request
            logger.debug(f"Would sync with remote node {target_node.node_id}")
            return {'success': True, 'concepts': [], 'experiences': [], 'thought': None}
        else:
            raise ValueError(f"Unknown node type: {type(target_node)}")
    
    def _process_sync_response(self, 
                             node_id: str, 
                             response: Dict,
                             strategy: SyncStrategy,
                             domains: List[SyncDomain] = None) -> None:
        """
        Process synchronization response from a node
        
        Args:
            node_id: ID of the node
            response: Sync response
            strategy: Synchronization strategy used
            domains: Domains that were synchronized
        """
        # Process received concepts
        received_concepts = response.get('concepts', [])
        if received_concepts:
            if strategy == SyncStrategy.FEDERATED:
                # In federated mode, we merge concepts rather than replacing
                self._integrate_federated_concepts(received_concepts, node_id)
            else:
                # Standard integration
                self.sam.concept_bank.integrate_hive_concepts(received_concepts, node_id)
        
        # Process received experiences
        received_experiences = response.get('experiences', [])
        if received_experiences:
            self.sam.experience_manager.integrate_hive_experiences(received_experiences)
        
        # Process received thought
        received_thought = response.get('thought')
        if received_thought is not None:
            # Convert to tensor
            thought_tensor = torch.tensor(
                received_thought,
                device=self.sam.config.device,
                dtype=torch.float
            )
            
            # Determine blend factor based on strategy
            blend_factor = 0.2  # Default
            if strategy == SyncStrategy.FULL:
                blend_factor = 0.3  # Stronger influence
            elif strategy == SyncStrategy.SELECTIVE:
                blend_factor = 0.2  # Moderate influence
            elif strategy == SyncStrategy.DIFFERENTIAL:
                blend_factor = 0.1  # Lighter influence
                
            # Set shared thought
            self.sam.thought_state.set_shared_thought(thought_tensor, blend_factor=blend_factor)
    
    def _integrate_federated_concepts(self, concepts: List[Dict], origin_node: str) -> None:
        """
        Integrate concepts using federated learning approach
        
        Args:
            concepts: Concepts to integrate
            origin_node: Node that sent the concepts
        """
        # For each concept, merge rather than replace
        for concept in concepts:
            concept_id = concept.get('global_id')
            if not concept_id:
                continue
                
            # Find matching local concept
            local_id = None
            for lid, gid in self.sam.concept_bank.hive_global_id_map.items():
                if gid == concept_id:
                    local_id = lid
                    break
                    
            if local_id is None:
                # No matching concept, just integrate normally
                self.sam.concept_bank.integrate_hive_concepts([concept], origin_node)
                continue
                
            # We have a matching concept, perform federated merge
            try:
                # Get vectors
                remote_embedding = torch.tensor(concept['embedding'], device=self.sam.config.device)
                remote_meaning = torch.tensor(concept['meaning'], device=self.sam.config.device)
                
                local_embedding = self.sam.concept_bank.concept_embeddings.weight[local_id]
                local_meaning = self.sam.concept_bank.meaning_vectors[local_id]
                
                # Calculate merge weights
                local_freq = self.sam.concept_bank.concept_frequencies[local_id].item()
                remote_freq = concept.get('frequency', 1)
                
                total = local_freq + remote_freq
                local_weight = local_freq / total
                remote_weight = remote_freq / total
                
                # Ensure minimum influence
                remote_weight = max(remote_weight, 0.1)
                local_weight = 1.0 - remote_weight
                
                # Merge embeddings
                with torch.no_grad():
                    merged_embedding = local_weight * local_embedding + remote_weight * remote_embedding
                    merged_meaning = local_weight * local_meaning + remote_weight * remote_meaning
                    
                    # Normalize
                    merged_meaning = torch.nn.functional.normalize(merged_meaning, dim=0)
                    
                    # Update vectors
                    self.sam.concept_bank.concept_embeddings.weight[local_id] = merged_embedding
                    self.sam.concept_bank.meaning_vectors[local_id] = merged_meaning
                
                # Update metadata
                if local_id in self.sam.concept_bank.concept_metadata:
                    local_meta = self.sam.concept_bank.concept_metadata[local_id]
                    remote_meta = concept.get('metadata', {})
                    
                    # Merge domains
                    local_domains = local_meta.get('domains', [])
                    remote_domains = remote_meta.get('domains', [])
                    merged_domains = list(set(local_domains + remote_domains))
                    
                    # Update metadata
                    local_meta['domains'] = merged_domains
                    local_meta['last_modified'] = time.time()
                    local_meta['federated_update'] = True
                    
                    # Record federated origin
                    if 'federated_origins' not in local_meta:
                        local_meta['federated_origins'] = []
                    if origin_node not in local_meta['federated_origins']:
                        local_meta['federated_origins'].append(origin_node)
                
            except Exception as e:
                logger.error(f"Error during federated merge for concept {concept_id}: {e}")
    
    def _infer_concept_domains(self, concept: Dict) -> List[str]:
        """
        Infer domains for a concept based on its content
        
        Args:
            concept: Concept dictionary
            
        Returns:
            List[str]: Inferred domain values
        """
        domains = []
        
        # Get metadata
        metadata = concept.get('metadata', {})
        concept_type = metadata.get('type', '')
        source = metadata.get('source', '')
        
        # Core concepts get the CORE domain
        if concept_type == 'character_sequence' and len(source) <= 3:
            domains.append(SyncDomain.CORE.value)
            
        # Check source for domain clues
        if source:
            source_lower = source.lower()
            
            # Check for language-related patterns
            language_patterns = ['grammar', 'sentence', 'verb', 'noun', 'word', 'pronoun', 'adjective']
            if any(pattern in source_lower for pattern in language_patterns):
                domains.append(SyncDomain.LANGUAGE.value)
                
            # Check for reasoning-related patterns
            reasoning_patterns = ['logic', 'reason', 'inference', 'deduction', 'conclusion', 'therefore', 'because']
            if any(pattern in source_lower for pattern in reasoning_patterns):
                domains.append(SyncDomain.REASONING.value)
                
            # Check for creative-related patterns
            creative_patterns = ['create', 'imagine', 'story', 'poem', 'art', 'music', 'design']
            if any(pattern in source_lower for pattern in creative_patterns):
                domains.append(SyncDomain.CREATIVE.value)
                
            # Check for multimodal-related patterns
            multimodal_patterns = ['image', 'picture', 'audio', 'sound', 'video', 'visual', 'hear']
            if any(pattern in source_lower for pattern in multimodal_patterns):
                domains.append(SyncDomain.MULTIMODAL.value)
        
        # Add DOMAIN_KNOWLEDGE as fallback
        if not domains:
            domains.append(SyncDomain.DOMAIN_KNOWLEDGE.value)
            
        return domains
    
    def _hash_concept(self, concept: Dict) -> str:
        """
        Generate a hash of a concept for change detection
        
        Args:
            concept: Concept dictionary
            
        Returns:
            str: Hash string
        """
        # Create a reduced representation for hashing
        hash_dict = {
            'embedding': concept.get('embedding')[:10] if concept.get('embedding') else None,  # First 10 values
            'meaning': concept.get('meaning')[:10] if concept.get('meaning') else None,  # First 10 values
            'frequency': concept.get('frequency'),
            'metadata': {
                'type': concept.get('metadata', {}).get('type'),
                'domains': concept.get('metadata', {}).get('domains'),
                'last_modified': concept.get('metadata', {}).get('last_modified')
            }
        }
        
        # Convert to string and hash
        hash_str = json.dumps(hash_dict, sort_keys=True)
        return hashlib.md5(hash_str.encode()).hexdigest()

    def get_sync_stats(self) -> Dict:
        """
        Get synchronization statistics
        
        Returns:
            Dict: Sync statistics
        """
        return {
            'total_syncs': self.sync_stats['total_syncs'],
            'successful_syncs': self.sync_stats['successful_syncs'],
            'failed_syncs': self.sync_stats['failed_syncs'],
            'success_rate': self.sync_stats['successful_syncs'] / max(1, self.sync_stats['total_syncs']),
            'concepts_sent': self.sync_stats['concepts_sent'],
            'concepts_received': self.sync_stats['concepts_received'],
            'experiences_sent': self.sync_stats['experiences_sent'],
            'experiences_received': self.sync_stats['experiences_received'],
            'last_sync_time': self.last_sync_time,
            'recent_syncs': self.sync_history[-10:] if self.sync_history else [],
            'recent_errors': self.sync_errors[-5:] if self.sync_errors else []
        }
