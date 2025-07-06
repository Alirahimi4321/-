"""
Project AWARENESS - Security Agent
Enforces permissions based on trust scores and autonomy levels with sandboxing.
"""

import asyncio
import time
import hashlib
import jwt
import os
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from agents.base_agent import BaseAgent, AgentMessage
from core.config import AwarenessConfig


class AutonomyLevel(Enum):
    """Autonomy levels for agent permissions."""
    RESTRICTED = 0   # No autonomous actions
    LIMITED = 1      # Basic actions only
    MODERATE = 2     # Standard operations
    ELEVATED = 3     # Advanced operations
    FULL = 4         # All operations
    ADMIN = 5        # System administration


@dataclass
class SecurityPolicy:
    """Security policy definition."""
    name: str
    description: str
    min_trust_score: float
    min_autonomy_level: AutonomyLevel
    allowed_actions: Set[str]
    denied_actions: Set[str] = field(default_factory=set)
    

@dataclass
class SecurityEvent:
    """Security event for logging."""
    timestamp: float
    event_type: str
    agent_name: str
    action: str
    allowed: bool
    trust_score: float
    autonomy_level: int
    details: Dict[str, Any] = field(default_factory=dict)
    

class SecurityAgent(BaseAgent):
    """
    Security Agent implementing trust-based permission enforcement.
    Provides sandboxing and security policy management.
    """
    
    def __init__(self, name: str, config: AwarenessConfig, agent_config: Dict[str, Any]):
        super().__init__(name, config, agent_config)
        
        # Security configuration
        self.trust_threshold = self.config.security.trust_threshold
        self.max_autonomy_level = self.config.security.max_autonomy_level
        self.enable_sandboxing = self.config.security.enable_sandboxing
        
        # Security policies
        self.policies: Dict[str, SecurityPolicy] = {}
        self._initialize_default_policies()
        
        # Agent trust scores and autonomy levels
        self.agent_trust_scores: Dict[str, float] = {}
        self.agent_autonomy_levels: Dict[str, AutonomyLevel] = {}
        
        # Security events log
        self.security_events: List[SecurityEvent] = []
        self.max_events = 1000
        
        # JWT tokens for secure communication
        self.jwt_secret = self._load_jwt_secret()
        
        # Register handlers
        self._register_security_handlers()
        
    async def _initialize(self):
        """Initialize the security agent."""
        self.logger.info("SecurityAgent initializing...")
        
        # Initialize security components
        await self._initialize_security_components()
        
        self.logger.info("SecurityAgent initialized successfully")
        
    async def _shutdown(self):
        """Shutdown the security agent."""
        self.logger.info("SecurityAgent shutting down...")
        
        # Save security state
        await self._save_security_state()
        
        self.logger.info("SecurityAgent shutdown complete")
        
    def _register_security_handlers(self):
        """Register security-specific message handlers."""
        self.register_handler("security_check", self._handle_security_check)
        self.register_handler("trust_update", self._handle_trust_update)
        self.register_handler("autonomy_request", self._handle_autonomy_request)
        self.register_handler("security_events", self._handle_security_events)
        self.register_handler("create_token", self._handle_create_token)
        self.register_handler("verify_token", self._handle_verify_token)
        
    def _get_background_tasks(self) -> List:
        """Get security-specific background tasks."""
        return [
            self._security_monitoring_loop(),
            self._trust_decay_loop(),
        ]
        
    async def _handle_security_check(self, message: AgentMessage):
        """Handle security check requests."""
        try:
            agent_name = message.payload.get("agent_name")
            action = message.payload.get("action")
            resource = message.payload.get("resource")
            
            if not agent_name or not action:
                await self._send_error_response(message, "Missing agent_name or action")
                return
                
            # Perform security check
            allowed, reason = await self.check_permission(agent_name, action, resource)
            
            await self._send_response(message, {
                "status": "success",
                "allowed": allowed,
                "reason": reason,
                "agent_name": agent_name,
                "action": action
            })
            
        except Exception as e:
            self.logger.error(f"Error handling security check: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_trust_update(self, message: AgentMessage):
        """Handle trust score update requests."""
        try:
            agent_name = message.payload.get("agent_name")
            trust_score = message.payload.get("trust_score")
            reason = message.payload.get("reason", "Manual update")
            
            if not agent_name or trust_score is None:
                await self._send_error_response(message, "Missing agent_name or trust_score")
                return
                
            # Update trust score
            await self.update_trust_score(agent_name, trust_score, reason)
            
            await self._send_response(message, {
                "status": "success",
                "agent_name": agent_name,
                "trust_score": trust_score
            })
            
        except Exception as e:
            self.logger.error(f"Error handling trust update: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_autonomy_request(self, message: AgentMessage):
        """Handle autonomy level requests."""
        try:
            agent_name = message.payload.get("agent_name")
            requested_level = message.payload.get("autonomy_level")
            
            if not agent_name or requested_level is None:
                await self._send_error_response(message, "Missing agent_name or autonomy_level")
                return
                
            # Process autonomy request
            granted_level = await self.request_autonomy_level(agent_name, requested_level)
            
            await self._send_response(message, {
                "status": "success",
                "agent_name": agent_name,
                "requested_level": requested_level,
                "granted_level": granted_level.value
            })
            
        except Exception as e:
            self.logger.error(f"Error handling autonomy request: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_security_events(self, message: AgentMessage):
        """Handle security events query requests."""
        try:
            limit = message.payload.get("limit", 100)
            event_type = message.payload.get("event_type")
            
            # Get security events
            events = await self.get_security_events(limit, event_type)
            
            await self._send_response(message, {
                "status": "success",
                "events": events,
                "count": len(events)
            })
            
        except Exception as e:
            self.logger.error(f"Error handling security events: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_create_token(self, message: AgentMessage):
        """Handle JWT token creation requests."""
        try:
            agent_name = message.payload.get("agent_name")
            permissions = message.payload.get("permissions", [])
            expires_in = message.payload.get("expires_in", 3600)  # 1 hour default
            
            if not agent_name:
                await self._send_error_response(message, "Missing agent_name")
                return
                
            # Create JWT token
            token = await self.create_token(agent_name, permissions, expires_in)
            
            await self._send_response(message, {
                "status": "success",
                "token": token,
                "agent_name": agent_name,
                "expires_in": expires_in
            })
            
        except Exception as e:
            self.logger.error(f"Error handling create token: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_verify_token(self, message: AgentMessage):
        """Handle JWT token verification requests."""
        try:
            token = message.payload.get("token")
            
            if not token:
                await self._send_error_response(message, "Missing token")
                return
                
            # Verify JWT token
            valid, payload = await self.verify_token(token)
            
            await self._send_response(message, {
                "status": "success",
                "valid": valid,
                "payload": payload if valid else None
            })
            
        except Exception as e:
            self.logger.error(f"Error handling verify token: {e}")
            await self._send_error_response(message, str(e))
            
    async def check_permission(self, agent_name: str, action: str, 
                              resource: Optional[str] = None) -> tuple[bool, str]:
        """
        Check if an agent has permission to perform an action.
        
        Args:
            agent_name: Name of the agent
            action: Action to check
            resource: Optional resource being accessed
            
        Returns:
            Tuple of (allowed, reason)
        """
        # Get agent trust score and autonomy level
        trust_score = self.agent_trust_scores.get(agent_name, 0.0)
        autonomy_level = self.agent_autonomy_levels.get(agent_name, AutonomyLevel.RESTRICTED)
        
        # Find applicable policy
        policy = self._find_applicable_policy(action, resource)
        
        if not policy:
            # No specific policy found, use default restrictions
            allowed = trust_score >= self.trust_threshold
            reason = f"Default policy: trust_score={trust_score:.2f}"
        else:
            # Check against policy requirements
            allowed = (trust_score >= policy.min_trust_score and
                      autonomy_level.value >= policy.min_autonomy_level.value and
                      action in policy.allowed_actions and
                      action not in policy.denied_actions)
            
            if not allowed:
                if trust_score < policy.min_trust_score:
                    reason = f"Insufficient trust score: {trust_score:.2f} < {policy.min_trust_score:.2f}"
                elif autonomy_level.value < policy.min_autonomy_level.value:
                    reason = f"Insufficient autonomy level: {autonomy_level.value} < {policy.min_autonomy_level.value}"
                elif action not in policy.allowed_actions:
                    reason = f"Action not in allowed list: {action}"
                elif action in policy.denied_actions:
                    reason = f"Action explicitly denied: {action}"
                else:
                    reason = "Unknown policy violation"
            else:
                reason = f"Policy '{policy.name}' allows action"
                
        # Log security event
        await self._log_security_event(
            event_type="permission_check",
            agent_name=agent_name,
            action=action,
            allowed=allowed,
            trust_score=trust_score,
            autonomy_level=autonomy_level.value,
            details={"resource": resource, "reason": reason}
        )
        
        return allowed, reason
        
    async def update_trust_score(self, agent_name: str, trust_score: float, reason: str):
        """
        Update an agent's trust score.
        
        Args:
            agent_name: Name of the agent
            trust_score: New trust score (0.0 to 1.0)
            reason: Reason for the update
        """
        # Validate trust score
        trust_score = max(0.0, min(1.0, trust_score))
        
        old_score = self.agent_trust_scores.get(agent_name, 0.0)
        self.agent_trust_scores[agent_name] = trust_score
        
        # Log trust update event
        await self._log_security_event(
            event_type="trust_update",
            agent_name=agent_name,
            action="trust_score_update",
            allowed=True,
            trust_score=trust_score,
            autonomy_level=self.agent_autonomy_levels.get(agent_name, AutonomyLevel.RESTRICTED).value,
            details={"old_score": old_score, "new_score": trust_score, "reason": reason}
        )
        
        self.logger.info(f"Updated trust score for {agent_name}: {old_score:.2f} -> {trust_score:.2f} ({reason})")
        
    async def request_autonomy_level(self, agent_name: str, requested_level: int) -> AutonomyLevel:
        """
        Request an autonomy level for an agent.
        
        Args:
            agent_name: Name of the agent
            requested_level: Requested autonomy level
            
        Returns:
            Granted autonomy level
        """
        # Validate requested level
        requested_level = max(0, min(self.max_autonomy_level, requested_level))
        requested_autonomy = AutonomyLevel(requested_level)
        
        # Get current trust score
        trust_score = self.agent_trust_scores.get(agent_name, 0.0)
        
        # Determine granted level based on trust score
        if trust_score >= 0.9:
            max_allowed = AutonomyLevel.FULL
        elif trust_score >= 0.8:
            max_allowed = AutonomyLevel.ELEVATED
        elif trust_score >= 0.6:
            max_allowed = AutonomyLevel.MODERATE
        elif trust_score >= 0.4:
            max_allowed = AutonomyLevel.LIMITED
        else:
            max_allowed = AutonomyLevel.RESTRICTED
            
        # Grant the lower of requested and allowed level
        granted_level = min(requested_autonomy, max_allowed, key=lambda x: x.value)
        
        old_level = self.agent_autonomy_levels.get(agent_name, AutonomyLevel.RESTRICTED)
        self.agent_autonomy_levels[agent_name] = granted_level
        
        # Log autonomy update event
        await self._log_security_event(
            event_type="autonomy_update",
            agent_name=agent_name,
            action="autonomy_level_update",
            allowed=True,
            trust_score=trust_score,
            autonomy_level=granted_level.value,
            details={
                "old_level": old_level.value,
                "requested_level": requested_level,
                "granted_level": granted_level.value,
                "max_allowed": max_allowed.value
            }
        )
        
        self.logger.info(f"Updated autonomy level for {agent_name}: {old_level.value} -> {granted_level.value}")
        
        return granted_level
        
    async def create_token(self, agent_name: str, permissions: List[str], expires_in: int) -> str:
        """
        Create a JWT token for an agent.
        
        Args:
            agent_name: Name of the agent
            permissions: List of permissions
            expires_in: Token expiration time in seconds
            
        Returns:
            JWT token string
        """
        payload = {
            "agent_name": agent_name,
            "permissions": permissions,
            "trust_score": self.agent_trust_scores.get(agent_name, 0.0),
            "autonomy_level": self.agent_autonomy_levels.get(agent_name, AutonomyLevel.RESTRICTED).value,
            "iat": time.time(),
            "exp": time.time() + expires_in
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        
        # Log token creation
        await self._log_security_event(
            event_type="token_created",
            agent_name=agent_name,
            action="create_token",
            allowed=True,
            trust_score=payload["trust_score"],
            autonomy_level=payload["autonomy_level"],
            details={"permissions": permissions, "expires_in": expires_in}
        )
        
        return token
        
    async def verify_token(self, token: str) -> tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verify a JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Tuple of (valid, payload)
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check expiration
            if payload.get("exp", 0) < time.time():
                return False, None
                
            return True, payload
            
        except jwt.InvalidTokenError:
            return False, None
            
    async def get_security_events(self, limit: int = 100, 
                                 event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get security events.
        
        Args:
            limit: Maximum number of events to return
            event_type: Filter by event type
            
        Returns:
            List of security events
        """
        events = self.security_events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
            
        # Sort by timestamp (newest first) and limit
        events = sorted(events, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        # Convert to dict format
        return [
            {
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "agent_name": event.agent_name,
                "action": event.action,
                "allowed": event.allowed,
                "trust_score": event.trust_score,
                "autonomy_level": event.autonomy_level,
                "details": event.details
            }
            for event in events
        ]
        
    def _initialize_default_policies(self):
        """Initialize default security policies."""
        # Memory operations policy
        self.policies["memory_operations"] = SecurityPolicy(
            name="memory_operations",
            description="Memory read/write operations",
            min_trust_score=0.3,
            min_autonomy_level=AutonomyLevel.LIMITED,
            allowed_actions={"memory_store", "memory_retrieve", "memory_delete"}
        )
        
        # System operations policy
        self.policies["system_operations"] = SecurityPolicy(
            name="system_operations",
            description="System-level operations",
            min_trust_score=0.7,
            min_autonomy_level=AutonomyLevel.ELEVATED,
            allowed_actions={"system_restart", "system_config", "agent_spawn"}
        )
        
        # External access policy
        self.policies["external_access"] = SecurityPolicy(
            name="external_access",
            description="External network/file access",
            min_trust_score=0.5,
            min_autonomy_level=AutonomyLevel.MODERATE,
            allowed_actions={"network_request", "file_read", "file_write"}
        )
        
    def _find_applicable_policy(self, action: str, resource: Optional[str] = None) -> Optional[SecurityPolicy]:
        """Find the applicable security policy for an action."""
        for policy in self.policies.values():
            if action in policy.allowed_actions or action in policy.denied_actions:
                return policy
        return None
        
    async def _log_security_event(self, event_type: str, agent_name: str, action: str,
                                 allowed: bool, trust_score: float, autonomy_level: int,
                                 details: Dict[str, Any]):
        """Log a security event."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            agent_name=agent_name,
            action=action,
            allowed=allowed,
            trust_score=trust_score,
            autonomy_level=autonomy_level,
            details=details
        )
        
        self.security_events.append(event)
        
        # Keep only the most recent events
        if len(self.security_events) > self.max_events:
            self.security_events = self.security_events[-self.max_events:]
            
    def _load_jwt_secret(self) -> str:
        """Load or generate JWT secret."""
        secret_file = Path(self.config.security.jwt_secret_file)
        
        if secret_file.exists():
            try:
                return secret_file.read_text().strip()
            except Exception:
                pass
                
        # Generate new secret
        secret = hashlib.sha256(os.urandom(32)).hexdigest()
        
        try:
            secret_file.parent.mkdir(parents=True, exist_ok=True)
            secret_file.write_text(secret)
            secret_file.chmod(0o600)  # Restrict permissions
        except Exception as e:
            self.logger.warning(f"Could not save JWT secret: {e}")
            
        return secret
        
    async def _security_monitoring_loop(self):
        """Background security monitoring loop."""
        while self.is_running:
            try:
                # Monitor for security threats
                await self._monitor_security_threats()
                
                # Check for anomalous behavior
                await self._check_anomalous_behavior()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in security monitoring loop: {e}")
                await asyncio.sleep(300)
                
    async def _trust_decay_loop(self):
        """Background trust score decay loop."""
        while self.is_running:
            try:
                # Decay trust scores over time
                decay_rate = self.config.agents.trust_decay_rate
                current_time = time.time()
                
                for agent_name in list(self.agent_trust_scores.keys()):
                    current_score = self.agent_trust_scores[agent_name]
                    # Decay trust score slightly over time
                    new_score = max(0.0, current_score - decay_rate)
                    self.agent_trust_scores[agent_name] = new_score
                    
                await asyncio.sleep(3600)  # Decay every hour
                
            except Exception as e:
                self.logger.error(f"Error in trust decay loop: {e}")
                await asyncio.sleep(7200)
                
    async def _monitor_security_threats(self):
        """Monitor for security threats."""
        # Check for failed permission attempts
        recent_events = [e for e in self.security_events 
                        if e.timestamp > time.time() - 300 and not e.allowed]
        
        if len(recent_events) > 10:  # More than 10 failed attempts in 5 minutes
            self.logger.warning("High number of failed permission attempts detected")
            
    async def _check_anomalous_behavior(self):
        """Check for anomalous agent behavior."""
        # Check for agents with rapidly declining trust scores
        # Implementation would go here
        pass
        
    async def _initialize_security_components(self):
        """Initialize security components."""
        # Initialize default trust scores for known agents
        known_agents = ["router", "memory_controller", "context", "learning", "watchdog"]
        for agent_name in known_agents:
            if agent_name not in self.agent_trust_scores:
                self.agent_trust_scores[agent_name] = 0.8  # Start with good trust
            if agent_name not in self.agent_autonomy_levels:
                self.agent_autonomy_levels[agent_name] = AutonomyLevel.MODERATE
                
    async def _save_security_state(self):
        """Save security state."""
        # Implementation for saving security state would go here
        pass
        
    async def _send_response(self, original_message: AgentMessage, response: Dict[str, Any]):
        """Send response to a message."""
        # Implementation depends on message bus integration
        pass
        
    async def _send_error_response(self, original_message: AgentMessage, error: str):
        """Send error response to a message."""
        await self._send_response(original_message, {
            "status": "error",
            "error": error
        })