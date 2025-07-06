"""
Project AWARENESS - Learning Agent
Implements metacognitive background learning loop with experience-based adaptation.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque
import statistics

from agents.base_agent import BaseAgent, AgentMessage
from core.config import AwarenessConfig


@dataclass
class Experience:
    """Represents a learning experience."""
    id: str
    timestamp: float
    context: Dict[str, Any]
    action: str
    result: Dict[str, Any]
    success: bool
    confidence: float = 0.5
    

@dataclass
class Pattern:
    """Represents a learned pattern."""
    id: str
    description: str
    conditions: List[str]
    actions: List[str]
    success_rate: float
    confidence: float
    usage_count: int = 0
    

@dataclass
class Rule:
    """Represents a generated rule."""
    id: str
    name: str
    description: str
    condition: str
    action: str
    confidence: float
    created_at: float = field(default_factory=time.time)
    validated: bool = False
    success_count: int = 0
    failure_count: int = 0
    

class LearningAgent(BaseAgent):
    """
    Learning Agent implementing metacognitive background learning.
    Follows the Experience -> Reflection -> Knowledge Generation -> Action cycle.
    """
    
    def __init__(self, name: str, config: AwarenessConfig, agent_config: Dict[str, Any]):
        super().__init__(name, config, agent_config)
        
        # Learning components
        self.experiences: deque = deque(maxlen=1000)
        self.patterns: Dict[str, Pattern] = {}
        self.rules: Dict[str, Rule] = {}
        
        # Learning parameters
        self.learning_rate = 0.1
        self.pattern_threshold = 0.7
        self.rule_confidence_threshold = 0.8
        
        # Learning state
        self.learning_cycle_count = 0
        self.last_reflection = time.time()
        self.reflection_interval = 300  # 5 minutes
        
        # Register handlers
        self._register_learning_handlers()
        
    async def _initialize(self):
        """Initialize the learning agent."""
        self.logger.info("LearningAgent initializing...")
        
        # Initialize learning components
        await self._initialize_learning_components()
        
        self.logger.info("LearningAgent initialized successfully")
        
    async def _shutdown(self):
        """Shutdown the learning agent."""
        self.logger.info("LearningAgent shutting down...")
        
        # Save learning state
        await self._save_learning_state()
        
        self.logger.info("LearningAgent shutdown complete")
        
    def _register_learning_handlers(self):
        """Register learning-specific message handlers."""
        self.register_handler("add_experience", self._handle_add_experience)
        self.register_handler("get_patterns", self._handle_get_patterns)
        self.register_handler("get_rules", self._handle_get_rules)
        self.register_handler("validate_rule", self._handle_validate_rule)
        self.register_handler("learning_stats", self._handle_learning_stats)
        
    def _get_background_tasks(self) -> List:
        """Get learning-specific background tasks."""
        return [
            self._learning_cycle_loop(),
            self._pattern_discovery_loop(),
        ]
        
    async def _handle_add_experience(self, message: AgentMessage):
        """Handle add experience requests."""
        try:
            context = message.payload.get("context", {})
            action = message.payload.get("action")
            result = message.payload.get("result", {})
            success = message.payload.get("success", False)
            
            if not action:
                await self._send_error_response(message, "Missing action")
                return
                
            # Add experience
            experience_id = await self.add_experience(context, action, result, success)
            
            await self._send_response(message, {
                "status": "success",
                "experience_id": experience_id
            })
            
        except Exception as e:
            self.logger.error(f"Error handling add experience: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_get_patterns(self, message: AgentMessage):
        """Handle get patterns requests."""
        try:
            min_confidence = message.payload.get("min_confidence", 0.0)
            
            # Get patterns
            patterns = await self.get_patterns(min_confidence)
            
            await self._send_response(message, {
                "status": "success",
                "patterns": patterns,
                "count": len(patterns)
            })
            
        except Exception as e:
            self.logger.error(f"Error handling get patterns: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_get_rules(self, message: AgentMessage):
        """Handle get rules requests."""
        try:
            validated_only = message.payload.get("validated_only", False)
            
            # Get rules
            rules = await self.get_rules(validated_only)
            
            await self._send_response(message, {
                "status": "success",
                "rules": rules,
                "count": len(rules)
            })
            
        except Exception as e:
            self.logger.error(f"Error handling get rules: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_validate_rule(self, message: AgentMessage):
        """Handle rule validation requests."""
        try:
            rule_id = message.payload.get("rule_id")
            validation_result = message.payload.get("validation_result")
            
            if not rule_id or validation_result is None:
                await self._send_error_response(message, "Missing rule_id or validation_result")
                return
                
            # Validate rule
            success = await self.validate_rule(rule_id, validation_result)
            
            await self._send_response(message, {
                "status": "success" if success else "not_found",
                "rule_id": rule_id,
                "validated": validation_result
            })
            
        except Exception as e:
            self.logger.error(f"Error handling validate rule: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_learning_stats(self, message: AgentMessage):
        """Handle learning statistics requests."""
        try:
            stats = await self.get_learning_stats()
            
            await self._send_response(message, {
                "status": "success",
                "stats": stats
            })
            
        except Exception as e:
            self.logger.error(f"Error handling learning stats: {e}")
            await self._send_error_response(message, str(e))
            
    async def add_experience(self, context: Dict[str, Any], action: str, 
                           result: Dict[str, Any], success: bool) -> str:
        """
        Add a learning experience.
        
        Args:
            context: Context when action was taken
            action: Action that was performed
            result: Result of the action
            success: Whether the action was successful
            
        Returns:
            Experience ID
        """
        import hashlib
        
        # Generate experience ID
        experience_id = hashlib.sha256(f"{action}{time.time()}".encode()).hexdigest()[:16]
        
        experience = Experience(
            id=experience_id,
            timestamp=time.time(),
            context=context,
            action=action,
            result=result,
            success=success,
            confidence=self._calculate_experience_confidence(context, action, result, success)
        )
        
        self.experiences.append(experience)
        
        self.logger.debug(f"Added experience {experience_id}: {action} -> {success}")
        
        return experience_id
        
    async def get_patterns(self, min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """
        Get discovered patterns.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of patterns
        """
        patterns = []
        
        for pattern in self.patterns.values():
            if pattern.confidence >= min_confidence:
                patterns.append({
                    "id": pattern.id,
                    "description": pattern.description,
                    "conditions": pattern.conditions,
                    "actions": pattern.actions,
                    "success_rate": pattern.success_rate,
                    "confidence": pattern.confidence,
                    "usage_count": pattern.usage_count
                })
                
        return patterns
        
    async def get_rules(self, validated_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get generated rules.
        
        Args:
            validated_only: Only return validated rules
            
        Returns:
            List of rules
        """
        rules = []
        
        for rule in self.rules.values():
            if not validated_only or rule.validated:
                rules.append({
                    "id": rule.id,
                    "name": rule.name,
                    "description": rule.description,
                    "condition": rule.condition,
                    "action": rule.action,
                    "confidence": rule.confidence,
                    "created_at": rule.created_at,
                    "validated": rule.validated,
                    "success_count": rule.success_count,
                    "failure_count": rule.failure_count
                })
                
        return rules
        
    async def validate_rule(self, rule_id: str, validation_result: bool) -> bool:
        """
        Validate a rule based on Human-in-the-Loop feedback.
        
        Args:
            rule_id: ID of the rule to validate
            validation_result: True if rule is validated, False if rejected
            
        Returns:
            True if rule was found and updated
        """
        if rule_id not in self.rules:
            return False
            
        rule = self.rules[rule_id]
        rule.validated = validation_result
        
        if validation_result:
            rule.success_count += 1
            self.logger.info(f"Rule {rule_id} validated successfully")
        else:
            rule.failure_count += 1
            self.logger.info(f"Rule {rule_id} validation failed")
            
        return True
        
    async def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        total_experiences = len(self.experiences)
        successful_experiences = sum(1 for exp in self.experiences if exp.success)
        
        return {
            "total_experiences": total_experiences,
            "successful_experiences": successful_experiences,
            "success_rate": successful_experiences / total_experiences if total_experiences > 0 else 0.0,
            "total_patterns": len(self.patterns),
            "total_rules": len(self.rules),
            "validated_rules": sum(1 for rule in self.rules.values() if rule.validated),
            "learning_cycles": self.learning_cycle_count,
            "last_reflection": self.last_reflection,
            "average_confidence": self._calculate_average_confidence()
        }
        
    async def _learning_cycle_loop(self):
        """Main learning cycle background loop."""
        while self.is_running:
            try:
                # Check if it's time for reflection
                if time.time() - self.last_reflection > self.reflection_interval:
                    await self._perform_learning_cycle()
                    
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in learning cycle loop: {e}")
                await asyncio.sleep(300)
                
    async def _pattern_discovery_loop(self):
        """Background pattern discovery loop."""
        while self.is_running:
            try:
                # Discover new patterns from experiences
                await self._discover_patterns()
                
                await asyncio.sleep(600)  # Run every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Error in pattern discovery loop: {e}")
                await asyncio.sleep(900)
                
    async def _perform_learning_cycle(self):
        """
        Perform a complete learning cycle:
        Experience -> Reflection -> Knowledge Generation -> Action
        """
        self.logger.info("Starting learning cycle...")
        
        # 1. Experience analysis
        recent_experiences = await self._analyze_recent_experiences()
        
        # 2. Reflection
        insights = await self._reflect_on_experiences(recent_experiences)
        
        # 3. Knowledge generation
        new_rules = await self._generate_rules_from_insights(insights)
        
        # 4. Action (propose new rules for validation)
        if new_rules:
            await self._propose_rules_for_validation(new_rules)
            
        self.learning_cycle_count += 1
        self.last_reflection = time.time()
        
        self.logger.info(f"Learning cycle completed. Generated {len(new_rules)} new rules.")
        
    async def _analyze_recent_experiences(self) -> List[Experience]:
        """Analyze recent experiences for learning."""
        # Get experiences from the last reflection interval
        cutoff_time = self.last_reflection
        recent_experiences = [
            exp for exp in self.experiences 
            if exp.timestamp > cutoff_time
        ]
        
        return recent_experiences
        
    async def _reflect_on_experiences(self, experiences: List[Experience]) -> List[Dict[str, Any]]:
        """Reflect on experiences to extract insights."""
        insights = []
        
        if not experiences:
            return insights
            
        # Group experiences by action type
        action_groups = {}
        for exp in experiences:
            if exp.action not in action_groups:
                action_groups[exp.action] = []
            action_groups[exp.action].append(exp)
            
        # Analyze each action group
        for action, exp_list in action_groups.items():
            success_rate = sum(1 for exp in exp_list if exp.success) / len(exp_list)
            
            insight = {
                "action": action,
                "total_attempts": len(exp_list),
                "success_rate": success_rate,
                "confidence": statistics.mean([exp.confidence for exp in exp_list]),
                "contexts": [exp.context for exp in exp_list],
                "results": [exp.result for exp in exp_list]
            }
            
            insights.append(insight)
            
        return insights
        
    async def _generate_rules_from_insights(self, insights: List[Dict[str, Any]]) -> List[Rule]:
        """Generate rules from insights."""
        new_rules = []
        
        for insight in insights:
            # Only generate rules for actions with good success rates
            if insight["success_rate"] >= self.pattern_threshold:
                rule = await self._create_rule_from_insight(insight)
                if rule:
                    new_rules.append(rule)
                    
        return new_rules
        
    async def _create_rule_from_insight(self, insight: Dict[str, Any]) -> Optional[Rule]:
        """Create a rule from an insight."""
        import hashlib
        
        action = insight["action"]
        success_rate = insight["success_rate"]
        confidence = insight["confidence"]
        
        # Simple rule generation based on common context patterns
        common_contexts = self._find_common_context_patterns(insight["contexts"])
        
        if not common_contexts:
            return None
            
        # Generate rule ID
        rule_id = hashlib.sha256(f"{action}{common_contexts}{time.time()}".encode()).hexdigest()[:16]
        
        # Create rule description
        condition_str = " AND ".join([f"{k}={v}" for k, v in common_contexts.items()])
        
        rule = Rule(
            id=rule_id,
            name=f"Auto-generated rule for {action}",
            description=f"When {condition_str}, performing {action} has {success_rate:.1%} success rate",
            condition=condition_str,
            action=action,
            confidence=min(confidence, success_rate)
        )
        
        return rule
        
    def _find_common_context_patterns(self, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find common patterns in context data."""
        if not contexts:
            return {}
            
        # Find keys that appear in most contexts
        key_counts = {}
        for context in contexts:
            for key in context.keys():
                key_counts[key] = key_counts.get(key, 0) + 1
                
        # Get keys that appear in at least 70% of contexts
        threshold = len(contexts) * 0.7
        common_keys = [key for key, count in key_counts.items() if count >= threshold]
        
        # Find common values for common keys
        common_patterns = {}
        for key in common_keys:
            values = [context.get(key) for context in contexts if key in context]
            
            # For string values, find the most common
            if values and isinstance(values[0], str):
                value_counts = {}
                for value in values:
                    value_counts[value] = value_counts.get(value, 0) + 1
                    
                most_common = max(value_counts.items(), key=lambda x: x[1])
                if most_common[1] >= len(values) * 0.6:  # 60% threshold
                    common_patterns[key] = most_common[0]
                    
        return common_patterns
        
    async def _propose_rules_for_validation(self, rules: List[Rule]):
        """Propose new rules for Human-in-the-Loop validation."""
        for rule in rules:
            # Add rule to our collection (unvalidated)
            self.rules[rule.id] = rule
            
            # Log the proposed rule
            self.logger.info(f"Proposed new rule: {rule.name} (confidence: {rule.confidence:.2f})")
            
            # In a real implementation, this would send a message to request validation
            # For now, we'll auto-validate rules with very high confidence
            if rule.confidence >= self.rule_confidence_threshold:
                rule.validated = True
                self.logger.info(f"Auto-validated high-confidence rule: {rule.id}")
                
    async def _discover_patterns(self):
        """Discover patterns from experiences."""
        if len(self.experiences) < 10:  # Need minimum experiences
            return
            
        # Group experiences by similar contexts and actions
        pattern_candidates = self._group_similar_experiences()
        
        # Evaluate pattern candidates
        for candidate in pattern_candidates:
            if self._evaluate_pattern_candidate(candidate):
                pattern = self._create_pattern_from_candidate(candidate)
                if pattern:
                    self.patterns[pattern.id] = pattern
                    self.logger.info(f"Discovered new pattern: {pattern.description}")
                    
    def _group_similar_experiences(self) -> List[List[Experience]]:
        """Group similar experiences together."""
        # Simple grouping by action type for now
        groups = {}
        for exp in self.experiences:
            if exp.action not in groups:
                groups[exp.action] = []
            groups[exp.action].append(exp)
            
        # Return groups with at least 3 experiences
        return [group for group in groups.values() if len(group) >= 3]
        
    def _evaluate_pattern_candidate(self, experiences: List[Experience]) -> bool:
        """Evaluate if a group of experiences forms a valid pattern."""
        if len(experiences) < 3:
            return False
            
        # Check success rate
        success_rate = sum(1 for exp in experiences if exp.success) / len(experiences)
        
        return success_rate >= self.pattern_threshold
        
    def _create_pattern_from_candidate(self, experiences: List[Experience]) -> Optional[Pattern]:
        """Create a pattern from a group of experiences."""
        if not experiences:
            return None
            
        import hashlib
        
        action = experiences[0].action
        success_rate = sum(1 for exp in experiences if exp.success) / len(experiences)
        
        # Generate pattern ID
        pattern_id = hashlib.sha256(f"{action}{len(experiences)}{time.time()}".encode()).hexdigest()[:16]
        
        # Extract common conditions
        common_contexts = self._find_common_context_patterns([exp.context for exp in experiences])
        conditions = [f"{k}={v}" for k, v in common_contexts.items()]
        
        pattern = Pattern(
            id=pattern_id,
            description=f"Pattern for {action} with {success_rate:.1%} success rate",
            conditions=conditions,
            actions=[action],
            success_rate=success_rate,
            confidence=min(success_rate, statistics.mean([exp.confidence for exp in experiences]))
        )
        
        return pattern
        
    def _calculate_experience_confidence(self, context: Dict[str, Any], action: str, 
                                       result: Dict[str, Any], success: bool) -> float:
        """Calculate confidence score for an experience."""
        # Simple confidence calculation based on context richness and result clarity
        context_score = min(1.0, len(context) / 10.0)  # More context = higher confidence
        result_score = 1.0 if success else 0.3  # Success increases confidence
        
        return (context_score + result_score) / 2.0
        
    def _calculate_average_confidence(self) -> float:
        """Calculate average confidence across all experiences."""
        if not self.experiences:
            return 0.0
            
        return statistics.mean([exp.confidence for exp in self.experiences])
        
    async def _initialize_learning_components(self):
        """Initialize learning components."""
        # Any initialization specific to learning components
        pass
        
    async def _save_learning_state(self):
        """Save learning state."""
        # Implementation for saving learning state
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