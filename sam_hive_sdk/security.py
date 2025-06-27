"""
Security components for the SAM Hive Mind SDK
"""
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

class HiveSecurity:
    """
    Security management for HiveNetwork and HiveNode
    
    Handles authentication, authorization, and other security concerns.
    """
    
    def __init__(self, policy: str = "standard"):
        """
        Initialize security with specified policy
        
        Args:
            policy: Security policy (standard, strict, open)
        """
        self.policy = policy
        self.api_keys = []
        self.tokens = {}  # token -> expiration time
        self.token_secret = secrets.token_hex(32)
    
    def generate_api_key(self) -> str:
        """Generate a new API key"""
        api_key = secrets.token_hex(16)
        self.api_keys.append(api_key)
        return api_key
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key"""
        return api_key in self.api_keys
    
    def generate_token(self, node_id: str, expires_in: int = 3600) -> str:
        """
        Generate an authentication token
        
        Args:
            node_id: ID of the node
            expires_in: Seconds until expiration
            
        Returns:
            str: Authentication token
        """
        expiration = int(time.time()) + expires_in
        payload = {
            'node_id': node_id,
            'exp': expiration,
            'iat': int(time.time())
        }
        
        # Create signature
        payload_str = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            self.token_secret.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Combine payload and signature
        token = f"{payload_str}.{signature}"
        
        # Store token
        self.tokens[token] = expiration
        
        return token
    
    def validate_token(self, token: str) -> Optional[str]:
        """
        Validate a token and return node_id if valid
        
        Args:
            token: Authentication token
            
        Returns:
            Optional[str]: node_id if valid, None otherwise
        """
        try:
            # Check if token exists and isn't expired
            if token not in self.tokens:
                return None
                
            if time.time() > self.tokens[token]:
                # Token expired
                del self.tokens[token]
                return None
            
            # Split token
            payload_str, signature = token.split('.')
            
            # Verify signature
            expected_signature = hmac.new(
                self.token_secret.encode(),
                payload_str.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if signature != expected_signature:
                return None
                
            # Extract payload
            payload = json.loads(payload_str)
            
            # Check expiration
            if time.time() > payload['exp']:
                return None
                
            return payload['node_id']
            
        except Exception:
            return None
    
    def generate_request_signature(self, payload: Dict, timestamp: int) -> str:
        """
        Generate a signature for a request payload
        
        Args:
            payload: Request payload
            timestamp: Request timestamp
            
        Returns:
            str: Request signature
        """
        # Convert payload to string
        payload_str = json.dumps(payload, sort_keys=True)
        
        # Create message
        message = f"{payload_str}.{timestamp}"
        
        # Create signature
        signature = hmac.new(
            self.token_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def validate_request_signature(self, payload: Dict, timestamp: int, signature: str) -> bool:
        """
        Validate a request signature
        
        Args:
            payload: Request payload
            timestamp: Request timestamp
            signature: Request signature
            
        Returns:
            bool: True if signature is valid
        """
        # Check timestamp freshness (5 minute window)
        if abs(time.time() - timestamp) > 300:
            return False
            
        # Generate expected signature
        expected = self.generate_request_signature(payload, timestamp)
        
        # Compare signatures
        return hmac.compare_digest(signature, expected)


class ConnectionPolicy:
    """
    Defines connection policy for hive networks
    
    Controls which nodes can connect and what data can be shared.
    """
    
    def __init__(self, policy_type: str = "open"):
        """
        Initialize connection policy
        
        Args:
            policy_type: Type of policy (open, restricted, whitelist)
        """
        self.policy_type = policy_type
        self.whitelist = []  # Whitelisted node IDs
        self.blacklist = []  # Blacklisted node IDs
        self.domain_restrictions = {}  # domain -> allowed node IDs
    
    def can_connect(self, node_id: str, node_info: Dict) -> bool:
        """
        Check if a node can connect to the network
        
        Args:
            node_id: ID of the node
            node_info: Information about the node
            
        Returns:
            bool: True if connection is allowed
        """
        # Check blacklist
        if node_id in self.blacklist:
            return False
            
        # Check policy type
        if self.policy_type == "open":
            return True
        elif self.policy_type == "whitelist":
            return node_id in self.whitelist
        elif self.policy_type == "restricted":
            # Check capabilities
            capabilities = node_info.get('capabilities', {})
            model_size = capabilities.get('model_size', {})
            
            # Example: require at least a certain size model
            param_count = model_size.get('parameter_count', 0)
            return param_count >= 10000000  # 10M parameters minimum
            
        return False
    
    def can_share_concept(self, concept: Dict, node_id: str) -> bool:
        """
        Check if a concept can be shared with a node
        
        Args:
            concept: Concept data
            node_id: ID of the target node
            
        Returns:
            bool: True if sharing is allowed
        """
        # If domain restrictions are defined for this concept
        concept_domain = concept.get('metadata', {}).get('domain')
        if concept_domain and concept_domain in self.domain_restrictions:
            allowed_nodes = self.domain_restrictions[concept_domain]
            return node_id in allowed_nodes
            
        # By default, allow sharing
        return True
