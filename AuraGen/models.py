"""
Data models for scenarios, risks, and configurations.
"""

from typing import List, Dict, Any, Optional, Set, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
from .core import RiskSeverity, Language


class ScenarioCategory(str, Enum):
    """Categories for agent scenarios."""
    COMMUNICATION = "communication"
    FILE_OPERATIONS = "file_operations"
    SYSTEM_INTERACTION = "system_interaction"
    DATA_PROCESSING = "data_processing"
    NETWORK_OPERATIONS = "network_operations"
    CONTENT_GENERATION = "content_generation"
    ANALYSIS_TASKS = "analysis_tasks"
    AUTOMATION = "automation"


class ToolParameterType(str, Enum):
    """Parameter types for scenario tools."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    OBJECT = "object"


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(True, description="Whether parameter is required")
    default_value: Optional[Any] = Field(None, description="Default value if not required")
    validation_pattern: Optional[str] = Field(None, description="Regex pattern for validation")
    min_value: Optional[Union[int, float]] = Field(None, description="Minimum value for numeric types")
    max_value: Optional[Union[int, float]] = Field(None, description="Maximum value for numeric types")
    allowed_values: Optional[List[Any]] = Field(None, description="List of allowed values")


class Tool(BaseModel):
    """Definition of a tool available in a scenario."""
    name: str = Field(..., description="Tool name")
    display_name: str = Field(..., description="Human-readable tool name")
    description: str = Field(..., description="Tool description")
    category: str = Field(..., description="Tool category")
    parameters: List[ToolParameter] = Field(default_factory=list, description="Tool parameters")
    example_usage: List[str] = Field(default_factory=list, description="Example usage patterns")
    common_errors: List[str] = Field(default_factory=list, description="Common error patterns")
    version: str = Field("1.0", description="Tool version")
    documentation_url: Optional[str] = Field(None, description="Link to documentation")
    deprecated: bool = Field(False, description="Whether tool is deprecated")
    security_level: str = Field("standard", description="Security level required")


class EnvironmentVariable(BaseModel):
    """Definition of an environment variable."""
    name: str = Field(..., description="Variable name")
    value: Any = Field(..., description="Variable value")
    description: str = Field("", description="Variable description")
    sensitive: bool = Field(False, description="Whether variable contains sensitive data")
    type: str = Field("string", description="Variable type")


class Environment(BaseModel):
    """Environment configuration for a scenario."""
    name: str = Field(..., description="Environment name")
    description: str = Field("", description="Environment description")
    variables: List[EnvironmentVariable] = Field(default_factory=list, description="Environment variables")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Environment-specific settings")
    max_memory_mb: Optional[int] = Field(None, description="Maximum memory in MB")
    max_execution_time: Optional[int] = Field(None, description="Maximum execution time in seconds")
    max_file_size_mb: Optional[int] = Field(None, description="Maximum file size in MB")
    network_access: bool = Field(True, description="Whether network access is allowed")
    allowed_domains: List[str] = Field(default_factory=list, description="Allowed network domains")
    blocked_domains: List[str] = Field(default_factory=list, description="Blocked network domains")


class ScenarioContext(BaseModel):
    """Complete context configuration for a scenario."""
    available_tools: List[Tool] = Field(default_factory=list, description="Tools available in this scenario")
    environment: Optional[Environment] = Field(None, description="Environment configuration")
    predefined_variables: Dict[str, Any] = Field(default_factory=dict, description="Predefined variables")
    system_info: Dict[str, Any] = Field(default_factory=dict, description="System information")
    security_policy: Dict[str, Any] = Field(default_factory=dict, description="Security policy")

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        for tool in self.available_tools:
            if tool.name == name:
                return tool
        return None

    def get_tools_by_category(self, category: str) -> List[Tool]:
        """Get tools by category."""
        return [tool for tool in self.available_tools if tool.category == category]

    def generate_context_prompt(self) -> str:
        """Generate context information for LLM prompts."""
        context_parts = []
        
        if self.available_tools:
            context_parts.append("Available Tools:")
            for tool in self.available_tools:
                params = [f"{p.name}: {p.type}" for p in tool.parameters]
                context_parts.append(f"- {tool.name}({', '.join(params)}): {tool.description}")
                if tool.example_usage:
                    context_parts.append(f"  Example: {tool.example_usage[0]}")
        
        if self.environment:
            context_parts.append(f"\nEnvironment: {self.environment.name}")
            if self.environment.variables:
                context_parts.append("Environment Variables:")
                for var in self.environment.variables:
                    if not var.sensitive:
                        context_parts.append(f"- {var.name}: {var.value}")
        
        if self.predefined_variables:
            context_parts.append("\nPredefined Variables:")
            for name, value in self.predefined_variables.items():
                context_parts.append(f"- {name}: {value}")
        
        return "\n".join(context_parts)


class RiskCategory(str, Enum):
    """Categories for different types of risks."""
    HALLUCINATION = "hallucination"
    PRIVACY_LEAK = "privacy_leak"
    MALICIOUS_CODE = "malicious_code"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_CORRUPTION = "data_corruption"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    MISINFORMATION = "misinformation"
    EXECUTION_ERROR = "execution_error"


class Scenario(BaseModel):
    """
    Represents an agent scenario configuration.
    """
    name: str = Field(..., description="Unique name for the scenario")
    display_name: str = Field(..., description="Human-readable name for the scenario")
    description: str = Field(..., description="Detailed description of the scenario")
    category: str = Field(..., description="Category of the scenario")
    
    # Extensible attributes
    complexity_level: int = Field(1, ge=1, le=10, description="Complexity level from 1-10")
    estimated_duration: Optional[int] = Field(None, description="Estimated execution time in seconds")
    required_permissions: List[str] = Field(default_factory=list, description="Required system permissions")
    dependencies: List[str] = Field(default_factory=list, description="Dependencies on other scenarios")
    
    # Configuration for data generation
    generation_weight: float = Field(1.0, ge=0.0, description="Weight for data generation probability")
    sample_actions: List[str] = Field(default_factory=list, description="Sample actions for this scenario")
    
    # New: Context configuration
    context: Optional[ScenarioContext] = Field(None, description="Scenario context with tools and environment")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Scenario name cannot be empty")
        return v.strip().lower().replace(' ', '_')
    
    @validator('generation_weight')
    def validate_weight(cls, v):
        if v < 0:
            raise ValueError("Generation weight must be non-negative")
        return v

    def get_generation_context(self) -> str:
        """Get context information for generation prompts."""
        if self.context:
            return self.context.generate_context_prompt()
        return ""


class Risk(BaseModel):
    """
    Represents a risk configuration that can be injected into scenarios.
    """
    name: str = Field(..., description="Unique name for the risk")
    display_name: str = Field(..., description="Human-readable name for the risk")
    description: str = Field(..., description="Detailed description of the risk")
    category: RiskCategory = Field(..., description="Category of the risk")
    severity: RiskSeverity = Field(..., description="Severity level of the risk")
    
    # Risk characteristics
    detection_difficulty: int = Field(1, ge=1, le=10, description="Difficulty to detect (1=easy, 10=hard)")
    injection_probability: float = Field(0.5, ge=0.0, le=1.0, description="Probability of injection")
    
    # Pattern definitions for injection
    injection_patterns: List[str] = Field(default_factory=list, description="Patterns for risk injection")
    trigger_conditions: List[str] = Field(default_factory=list, description="Conditions that trigger this risk")
    
    # Constraints and compatibility
    compatible_scenarios: Optional[List[str]] = Field(None, description="Scenarios this risk can be applied to")
    incompatible_scenarios: List[str] = Field(default_factory=list, description="Scenarios this risk cannot be applied to")
    required_context: List[str] = Field(default_factory=list, description="Required context for this risk")
    
    # Mitigation information
    mitigation_strategies: List[str] = Field(default_factory=list, description="Known mitigation strategies")
    detection_signatures: List[str] = Field(default_factory=list, description="Signatures for detection")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Risk name cannot be empty")
        return v.strip().lower().replace(' ', '_')
    
    @validator('injection_probability')
    def validate_probability(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Injection probability must be between 0.0 and 1.0")
        return v


class RiskScenarioConstraint(BaseModel):
    """
    Represents constraints between risks and scenarios.
    """
    risk_name: str = Field(..., description="Name of the risk")
    scenario_name: str = Field(..., description="Name of the scenario")
    compatibility: bool = Field(True, description="Whether risk is compatible with scenario")
    
    # Constraint parameters
    max_injection_rate: float = Field(1.0, ge=0.0, le=1.0, description="Maximum injection rate for this combination")
    required_conditions: List[str] = Field(default_factory=list, description="Required conditions for this constraint")
    
    # Context-specific modifications
    severity_modifier: float = Field(1.0, ge=0.0, description="Severity multiplier for this combination")
    detection_modifier: float = Field(1.0, ge=0.0, description="Detection difficulty multiplier")
    
    # Metadata
    notes: str = Field("", description="Additional notes about this constraint")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('max_injection_rate')
    def validate_injection_rate(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Max injection rate must be between 0.0 and 1.0")
        return v


class ConfigurationLayer(BaseModel):
    """
    Base class for configuration layers (scenarios and risks).
    """
    scenarios: List[Scenario] = Field(default_factory=list, description="List of configured scenarios")
    risks: List[Risk] = Field(default_factory=list, description="List of configured risks")
    constraints: List[RiskScenarioConstraint] = Field(default_factory=list, description="Risk-scenario constraints")
    
    # Layer metadata
    version: str = Field("1.0", description="Configuration version")
    last_modified: Optional[str] = Field(None, description="Last modification timestamp")
    description: str = Field("", description="Description of this configuration layer")
    
    def get_scenario(self, name: str) -> Optional[Scenario]:
        """Get scenario by name."""
        for scenario in self.scenarios:
            if scenario.name == name:
                return scenario
        return None
    
    def get_risk(self, name: str) -> Optional[Risk]:
        """Get risk by name."""
        for risk in self.risks:
            if risk.name == name:
                return risk
        return None
    
    def get_compatible_risks(self, scenario_name: str) -> List[Risk]:
        """Get risks compatible with a given scenario."""
        compatible_risks = []
        
        for risk in self.risks:
            # Check explicit compatibility
            if risk.compatible_scenarios is not None:
                if scenario_name not in risk.compatible_scenarios:
                    continue
            
            # Check incompatibility
            if scenario_name in risk.incompatible_scenarios:
                continue
            
            # Check constraints
            constraint = self.get_constraint(risk.name, scenario_name)
            if constraint and not constraint.compatibility:
                continue
            
            compatible_risks.append(risk)
        
        return compatible_risks
    
    def get_constraint(self, risk_name: str, scenario_name: str) -> Optional[RiskScenarioConstraint]:
        """Get constraint for a specific risk-scenario combination."""
        for constraint in self.constraints:
            if constraint.risk_name == risk_name and constraint.scenario_name == scenario_name:
                return constraint
        return None
    
    def validate_configuration(self) -> List[str]:
        """Validate the configuration and return list of issues."""
        issues = []
        
        # Check for duplicate scenario names
        scenario_names = [s.name for s in self.scenarios]
        if len(scenario_names) != len(set(scenario_names)):
            issues.append("Duplicate scenario names found")
        
        # Check for duplicate risk names
        risk_names = [r.name for r in self.risks]
        if len(risk_names) != len(set(risk_names)):
            issues.append("Duplicate risk names found")
        
        # Validate constraints reference existing scenarios and risks
        for constraint in self.constraints:
            if constraint.scenario_name not in scenario_names:
                issues.append(f"Constraint references unknown scenario: {constraint.scenario_name}")
            if constraint.risk_name not in risk_names:
                issues.append(f"Constraint references unknown risk: {constraint.risk_name}")
        
        # Validate scenario dependencies
        for scenario in self.scenarios:
            for dep in scenario.dependencies:
                if dep not in scenario_names:
                    issues.append(f"Scenario {scenario.name} has unknown dependency: {dep}")
        
        return issues 