"""
Knowledge Domain Specialization System for SAM-SDK

Enables SAM nodes to develop specialized expertise in particular domains
while maintaining their general capabilities. Implements domain-based
routing, learning, and intelligence distribution.
"""

import logging
import time
import uuid
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Set, Tuple, Optional, Union, Any

logger = logging.getLogger("sam_sdk.domains")

class KnowledgeDomain:
    """
    Represents a specialized knowledge domain that a SAM node can focus on.
    
    Knowledge domains enable:
    - Targeted concept development
    - Specialized task handling
    - Efficient knowledge distribution across the hive
    """
    
    def __init__(self, 
                 name: str,
                 description: str = "",
                 priority: float = 1.0,
                 centroid_dim: int = 1536):
        """
        Initialize a knowledge domain
        
        Args:
            name: Domain identifier (e.g., "medical", "finance", "creative")
            description: Human-readable description of the domain
            priority: Relative importance of this domain (affects resource allocation)
            centroid_dim: Dimension of the domain centroid vector
        """
        self.name = name
        self.description = description
        self.priority = priority
        self.created_at = time.time()
        
        # Domain identification
        self.id = f"domain_{name}_{uuid.uuid4().hex[:8]}"
        
        # Domain semantic representation
        self.centroid = torch.zeros(centroid_dim)
        self.concept_ids = set()  # Concepts associated with this domain
        self.pattern_ids = set()  # Patterns associated with this domain
        
        # Domain metrics
        self.task_count = 0
        self.success_rate = 0.0
        self.confidence_score = 0.0
        self.expertise_level = 0.0  # 0.0-5.0 scale
        self.last_updated = time.time()
        
        # Semantic keywords
        self.keywords = set()
        
        # Related domains
        self.related_domains = {}  # domain_id -> similarity score
        
        logger.info(f"Created knowledge domain: {name} ({self.id})")
    
    def update_centroid(self, concept_vectors: torch.Tensor, concept_ids: List[int]) -> None:
        """
        Update the domain centroid based on new concept vectors
        
        Args:
            concept_vectors: Tensor of concept vectors
            concept_ids: IDs of the concepts
        """
        if len(concept_vectors) == 0:
            return
            
        # Update centroid as weighted average
        if torch.sum(self.centroid) == 0:
            # First update - initialize centroid
            self.centroid = torch.mean(concept_vectors, dim=0)
        else:
            # Weighted update
            weight = min(0.2, 10.0 / (len(self.concept_ids) + 10))  # Diminishing influence
            self.centroid = (1 - weight) * self.centroid + weight * torch.mean(concept_vectors, dim=0)
        
        # Normalize centroid
        self.centroid = F.normalize(self.centroid, dim=0)
        
        # Update concept IDs
        self.concept_ids.update(concept_ids)
        self.last_updated = time.time()
    
    def calculate_relevance(self, query_vector: torch.Tensor) -> float:
        """
        Calculate relevance of this domain to a query
        
        Args:
            query_vector: Query vector to compare against domain
            
        Returns:
            float: Relevance score (0-1)
        """
        if torch.sum(self.centroid) == 0:
            return 0.0
            
        # Calculate cosine similarity
        query_norm = F.normalize(query_vector, dim=0)
        similarity = F.cosine_similarity(query_norm.unsqueeze(0), self.centroid.unsqueeze(0)).item()
        
        # Weight by domain priority and expertise
        weighted_score = similarity * (0.5 + 0.5 * self.priority) * (0.5 + 0.5 * self.expertise_level / 5.0)
        
        return max(0.0, min(1.0, weighted_score))
    
    def record_task_result(self, success: bool, confidence: float = None) -> None:
        """
        Record the result of a task in this domain
        
        Args:
            success: Whether the task was successful
            confidence: Confidence score for the task (0-1)
        """
        self.task_count += 1
        
        # Update success rate
        if self.task_count == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            self.success_rate = ((self.task_count - 1) * self.success_rate + (1.0 if success else 0.0)) / self.task_count
        
        # Update confidence score
        if confidence is not None:
            if self.task_count == 1:
                self.confidence_score = confidence
            else:
                self.confidence_score = ((self.task_count - 1) * self.confidence_score + confidence) / self.task_count
        
        # Update expertise level based on tasks and success rate
        experience_factor = min(1.0, self.task_count / 100)  # Scales with tasks up to 100
        self.expertise_level = 5.0 * experience_factor * (0.5 + 0.5 * self.success_rate)
        
        self.last_updated = time.time()
    
    def add_keywords(self, keywords: List[str]) -> None:
        """
        Add semantic keywords to the domain
        
        Args:
            keywords: List of keyword strings
        """
        self.keywords.update([k.lower() for k in keywords])
    
    def to_dict(self) -> Dict:
        """Convert domain to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
            "created_at": self.created_at,
            "concept_count": len(self.concept_ids),
            "pattern_count": len(self.pattern_ids),
            "task_count": self.task_count,
            "success_rate": self.success_rate,
            "confidence_score": self.confidence_score,
            "expertise_level": self.expertise_level,
            "last_updated": self.last_updated,
            "keywords": list(self.keywords),
            "related_domains": self.related_domains
        }


class DomainManager:
    """
    Manages specialized knowledge domains across SAM nodes
    
    The DomainManager enables:
    - Dynamic creation and evolution of specialized domains
    - Intelligent routing of tasks to appropriate domains
    - Cross-domain knowledge transfer
    - Domain-specific learning and growth
    """
    
    def __init__(self, sam_instance):
        """
        Initialize the domain manager
        
        Args:
            sam_instance: Reference to the SAM instance
        """
        self.sam = sam_instance
        self.domains = {}  # domain_id -> KnowledgeDomain
        self.default_domain = self._create_default_domain()
        
        # Domain assignment
        self.concept_domain_map = {}  # concept_id -> domain_ids
        self.pattern_domain_map = {}  # pattern_id -> domain_ids
        
        # Domain analysis
        self.domain_similarity_matrix = {}  # (domain_id1, domain_id2) -> similarity
        
        logger.info("Initialized DomainManager")
    
    def _create_default_domain(self) -> KnowledgeDomain:
        """Create the default general domain"""
        domain = KnowledgeDomain(
            name="general",
            description="General knowledge domain",
            priority=1.0
        )
        self.domains[domain.id] = domain
        return domain
    
    def create_domain(self, name: str, description: str = "", priority: float = 1.0) -> str:
        """
        Create a new knowledge domain
        
        Args:
            name: Domain identifier
            description: Human-readable description
            priority: Relative importance
            
        Returns:
            str: Domain ID
        """
        domain = KnowledgeDomain(
            name=name,
            description=description,
            priority=priority,
            centroid_dim=self.sam.config.concept_dim if hasattr(self.sam.config, 'concept_dim') else 1536
        )
        self.domains[domain.id] = domain
        
        # Initialize domain with relevant existing concepts
        self._initialize_domain_from_existing_concepts(domain)
        
        # Update domain similarities
        self._update_domain_similarities(domain)
        
        return domain.id
    
    def _initialize_domain_from_existing_concepts(self, domain: KnowledgeDomain) -> None:
        """
        Initialize a new domain using existing relevant concepts
        
        Args:
            domain: The domain to initialize
        """
        # Skip if no concept bank or few concepts
        if not hasattr(self.sam, 'concept_bank') or self.sam.concept_bank.next_concept_id < 100:
            return
            
        # Find concepts related to domain keywords
        keywords = domain.name.split('_') + domain.description.split()
        keywords = [k.lower() for k in keywords if len(k) > 3]
        
        if not keywords:
            return
            
        relevant_concepts = []
        relevant_ids = []
        
        # Search concept metadata for keyword matches
        for concept_id, metadata in self.sam.concept_bank.concept_metadata.items():
            if concept_id >= self.sam.concept_bank.concept_embeddings.num_embeddings:
                continue
                
            source = metadata.get('source', '')
            if not source:
                continue
                
            if any(keyword in source.lower() for keyword in keywords):
                # Check if concept exists in embedding space
                if concept_id < len(self.sam.concept_bank.meaning_vectors):
                    vec = self.sam.concept_bank.meaning_vectors[concept_id]
                    if torch.sum(vec) > 0:  # Ensure non-zero vector
                        relevant_concepts.append(vec)
                        relevant_ids.append(concept_id)
        
        # Update domain with found concepts
        if relevant_concepts:
            concepts_tensor = torch.stack(relevant_concepts)
            domain.update_centroid(concepts_tensor, relevant_ids)
            
            # Add found keywords to domain
            found_keywords = set()
            for concept_id in relevant_ids:
                meta = self.sam.concept_bank.concept_metadata.get(concept_id, {})
                source = meta.get('source', '')
                if source:
                    found_keywords.update(source.lower().split())
            
            # Filter keywords by length
            domain.add_keywords([k for k in found_keywords if len(k) > 3])
            
            logger.info(f"Initialized domain {domain.name} with {len(relevant_ids)} existing concepts")
    
    def _update_domain_similarities(self, domain: KnowledgeDomain) -> None:
        """
        Update similarity measures between domains
        
        Args:
            domain: The domain to compare with others
        """
        for other_id, other_domain in self.domains.items():
            if other_id == domain.id:
                continue
                
            # Skip if either centroid is zero
            if torch.sum(domain.centroid) == 0 or torch.sum(other_domain.centroid) == 0:
                continue
                
            # Calculate similarity
            similarity = F.cosine_similarity(
                domain.centroid.unsqueeze(0),
                other_domain.centroid.unsqueeze(0)
            ).item()
            
            # Update similarity matrix
            self.domain_similarity_matrix[(domain.id, other_id)] = similarity
            self.domain_similarity_matrix[(other_id, domain.id)] = similarity
            
            # Update related domains if similarity is significant
            if similarity > 0.3:
                domain.related_domains[other_id] = similarity
                other_domain.related_domains[domain.id] = similarity
    
    def get_domain(self, domain_id: str) -> Optional[KnowledgeDomain]:
        """
        Get a domain by ID
        
        Args:
            domain_id: ID of the domain
            
        Returns:
            Optional[KnowledgeDomain]: The domain if found
        """
        return self.domains.get(domain_id)
    
    def get_domain_by_name(self, name: str) -> Optional[KnowledgeDomain]:
        """
        Get a domain by name
        
        Args:
            name: Name of the domain
            
        Returns:
            Optional[KnowledgeDomain]: The domain if found
        """
        for domain in self.domains.values():
            if domain.name == name:
                return domain
        return None
    
    def select_domains_for_task(self, task_description: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Select the most relevant domains for a task
        
        Args:
            task_description: Description of the task
            top_k: Number of domains to return
            
        Returns:
            List[Tuple[str, float]]: List of (domain_id, relevance) pairs
        """
        # Skip if no concept processing available
        if not hasattr(self.sam, 'process_text'):
            return [(self.default_domain.id, 1.0)]
            
        # Process the task description to get concept vectors
        try:
            concept_ids, _ = self.sam.process_text(task_description)
            
            if not concept_ids:
                return [(self.default_domain.id, 1.0)]
                
            # Flatten concept IDs if nested
            if isinstance(concept_ids[0], list):
                flat_ids = []
                for sublist in concept_ids:
                    if isinstance(sublist, list):
                        flat_ids.extend(sublist)
                    else:
                        flat_ids.append(sublist)
                concept_ids = flat_ids
            
            # Get vectors for concepts
            vectors = []
            for cid in concept_ids:
                if cid < len(self.sam.concept_bank.meaning_vectors):
                    vec = self.sam.concept_bank.meaning_vectors[cid]
                    if torch.sum(vec) > 0:  # Ensure non-zero vector
                        vectors.append(vec)
            
            if not vectors:
                return [(self.default_domain.id, 1.0)]
                
            # Calculate task vector as mean of concept vectors
            task_vector = torch.mean(torch.stack(vectors), dim=0)
            
            # Calculate relevance for each domain
            relevance_scores = []
            for domain_id, domain in self.domains.items():
                relevance = domain.calculate_relevance(task_vector)
                relevance_scores.append((domain_id, relevance))
            
            # Sort by relevance
            relevance_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k
            return relevance_scores[:top_k]
            
        except Exception as e:
            logger.error(f"Error selecting domains for task: {e}")
            return [(self.default_domain.id, 1.0)]
    
    def assign_concept_to_domains(self, concept_id: int, 
                                concept_vector: torch.Tensor,
                                domains: Optional[List[str]] = None) -> List[str]:
        """
        Assign a concept to appropriate domains
        
        Args:
            concept_id: ID of the concept
            concept_vector: Vector representation of the concept
            domains: Optional list of domain IDs to consider (None for all)
            
        Returns:
            List[str]: Domain IDs the concept was assigned to
        """
        assigned_domains = []
        
        # If domains provided, only consider those
        domain_list = [self.domains[d] for d in domains if d in self.domains] if domains else self.domains.values()
        
        # Always consider default domain
        if self.default_domain.id not in [d.id for d in domain_list]:
            domain_list = list(domain_list) + [self.default_domain]
        
        # Calculate relevance to each domain
        for domain in domain_list:
            relevance = domain.calculate_relevance(concept_vector)
            
            # Assign if relevance is significant
            if relevance > 0.4:
                domain.concept_ids.add(concept_id)
                assigned_domains.append(domain.id)
                
                # Update domain centroid
                domain.update_centroid(concept_vector.unsqueeze(0), [concept_id])
        
        # If no relevant domains found, assign to default
        if not assigned_domains:
            self.default_domain.concept_ids.add(concept_id)
            self.default_domain.update_centroid(concept_vector.unsqueeze(0), [concept_id])
            assigned_domains.append(self.default_domain.id)
        
        # Update concept domain map
        self.concept_domain_map[concept_id] = assigned_domains
        
        return assigned_domains
    
    def get_concept_domains(self, concept_id: int) -> List[str]:
        """
        Get domains a concept belongs to
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            List[str]: Domain IDs
        """
        return self.concept_domain_map.get(concept_id, [self.default_domain.id])
    
    def get_domain_concepts(self, domain_id: str, limit: int = 100) -> List[int]:
        """
        Get concepts belonging to a domain
        
        Args:
            domain_id: ID of the domain
            limit: Maximum number of concepts to return
            
        Returns:
            List[int]: Concept IDs
        """
        domain = self.domains.get(domain_id)
        if not domain:
            return []
            
        return list(domain.concept_ids)[:limit]
    
    def record_domain_task_result(self, domain_id: str, success: bool, confidence: float = None) -> None:
        """
        Record a task result for a domain
        
        Args:
            domain_id: ID of the domain
            success: Whether the task was successful
            confidence: Confidence score (0-1)
        """
        domain = self.domains.get(domain_id)
        if domain:
            domain.record_task_result(success, confidence)
    
    def get_domain_status(self) -> Dict:
        """
        Get status information about all domains
        
        Returns:
            Dict: Domain status information
        """
        return {
            "total_domains": len(self.domains),
            "domains": [domain.to_dict() for domain in self.domains.values()],
            "cross_domain_connections": sum(1 for s in self.domain_similarity_matrix.values() if s > 0.3)
        }
    
    def get_domain_metrics(self) -> Dict:
        """
        Get performance metrics for domains
        
        Returns:
            Dict: Domain performance metrics
        """
        domains_by_expertise = sorted(
            self.domains.values(),
            key=lambda d: d.expertise_level,
            reverse=True
        )
        
        return {
            "top_domains_by_expertise": [
                {
                    "id": d.id,
                    "name": d.name,
                    "expertise_level": d.expertise_level,
                    "task_count": d.task_count,
                    "success_rate": d.success_rate
                }
                for d in domains_by_expertise[:5]
            ],
            "domain_concept_distribution": {
                d.name: len(d.concept_ids) for d in self.domains.values()
            },
            "cross_domain_relationships": [
                {
                    "domain1": d1,
                    "domain2": d2,
                    "similarity": sim
                }
                for (d1, d2), sim in self.domain_similarity_matrix.items()
                if sim > 0.5 and d1 < d2  # Only include one direction and significant similarities
            ]
        }
