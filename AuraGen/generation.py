"""
Harmless Data Generation Module

This module generates original, harmless agent action records using LLMs.
Supports both open-source (local) and API-based (OpenAI) models.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel, Field, validator
import yaml
import random
import json
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from AuraGen.config import GuardianConfig
from AuraGen.models import Scenario, ScenarioContext, Tool, Environment, EnvironmentVariable
from loguru import logger
import copy
import requests
from AuraGen.inference import InferenceManager, OpenAIConfig, externalAPIConfig
import os

# Optional: import transformers only if needed
try:
    from transformers import pipeline
except ImportError:
    pipeline = None


class ContextDiversifier:
    """
    Utility class to create diverse variations of scenario contexts.
    Enhances generation diversity by modifying tools, environment variables,
    and other context elements using LLM-based diversification.
    """
    
    @staticmethod
    def create_diverse_scenario(scenario: Scenario, diversity_level: float = 0.5, llm_client=None) -> Scenario:
        """
        Create a variation of the scenario with diversified context.
        
        Args:
            scenario: Original scenario
            diversity_level: Level of diversification (0.0 to 1.0)
            llm_client: Optional LLM client for intelligent diversification
            
        Returns:
            New scenario with diversified context
        """
        # Create a deep copy to avoid modifying the original
        diverse_scenario = copy.deepcopy(scenario)
        
        # Skip if no context available
        if not diverse_scenario.context:
            return diverse_scenario
            
        # Apply diversification to the context
        diverse_scenario.context = ContextDiversifier.diversify_context(
            diverse_scenario.context, 
            diversity_level,
            llm_client
        )
        
        return diverse_scenario
    
    @staticmethod
    def diversify_context(context: ScenarioContext, diversity_level: float, llm_client=None) -> ScenarioContext:
        """Diversify a scenario context by modifying its components."""
        # Diversify tools
        if context.available_tools:
            # Randomly remove some tools if diversity is high
            if diversity_level > 0.7 and len(context.available_tools) > 2:
                keep_count = max(2, int(len(context.available_tools) * (1 - (diversity_level * 0.3))))
                context.available_tools = random.sample(context.available_tools, keep_count)
            
            # Shuffle tool order
            random.shuffle(context.available_tools)
            
            # Diversify individual tools
            context.available_tools = [
                ContextDiversifier.diversify_tool(tool, diversity_level, llm_client)
                for tool in context.available_tools
            ]
        
        # Diversify environment
        if context.environment:
            context.environment = ContextDiversifier.diversify_environment(
                context.environment, 
                diversity_level,
                llm_client
            )
        
        # Diversify predefined variables
        if context.predefined_variables and llm_client:
            # Intelligent diversification using LLM
            context.predefined_variables = ContextDiversifier.diversify_variables_with_llm(
                context.predefined_variables,
                diversity_level,
                llm_client,
                "predefined_variables"
            )
        elif context.predefined_variables:
            # Fallback to basic shuffling if no LLM available
            predefined_vars = list(context.predefined_variables.items())
            random.shuffle(predefined_vars)
            context.predefined_variables = dict(predefined_vars)
        
        # Diversify system info and security policy (if present)
        for info_dict_name, info_dict in [("system_info", context.system_info), 
                                         ("security_policy", context.security_policy)]:
            if info_dict and llm_client:
                # Use LLM to diversify these dictionaries
                new_dict = ContextDiversifier.diversify_variables_with_llm(
                    info_dict,
                    diversity_level,
                    llm_client,
                    info_dict_name
                )
                # Update the original dictionary
                info_dict.clear()
                info_dict.update(new_dict)
            elif info_dict:
                # Fallback to basic shuffling
                items = list(info_dict.items())
                random.shuffle(items)
                info_dict.clear()
                info_dict.update(dict(items))
        
        return context
    
    @staticmethod
    def diversify_tool(tool: Tool, diversity_level: float, llm_client=None) -> Tool:
        """Create a diversified version of a tool."""
        # Randomly reorder parameters
        if tool.parameters:
            random.shuffle(tool.parameters)
        
        # Use LLM to diversify tool examples if available
        if tool.example_usage and llm_client and random.random() < diversity_level * 0.5:
            try:
                tool.example_usage = ContextDiversifier.diversify_examples_with_llm(
                    tool.example_usage,
                    tool.name,
                    tool.description,
                    diversity_level,
                    llm_client
                )
            except Exception as e:
                logger.warning(f"Failed to diversify tool examples with LLM: {e}")
                # Fallback to shuffle
                random.shuffle(tool.example_usage)
        elif tool.example_usage:
            # Fallback to shuffle
            random.shuffle(tool.example_usage)
            
            # Limit examples to a random subset if there are many
            if len(tool.example_usage) > 2 and random.random() < diversity_level:
                keep_count = max(1, int(len(tool.example_usage) * (1 - diversity_level * 0.5)))
                tool.example_usage = tool.example_usage[:keep_count]
                
        # Randomly reorder common errors
        if tool.common_errors:
            random.shuffle(tool.common_errors)
        
        return tool
    
    @staticmethod
    def diversify_environment(env: Environment, diversity_level: float, llm_client=None) -> Environment:
        """Create a diversified version of an environment."""
        # Shuffle environment variables
        if env.variables:
            random.shuffle(env.variables)
            
            # Diversify non-sensitive variables
            env.variables = [
                ContextDiversifier.diversify_env_variable(var, diversity_level, llm_client)
                if not var.sensitive else var
                for var in env.variables
            ]
        
        # Shuffle allowed and blocked domains
        if env.allowed_domains:
            random.shuffle(env.allowed_domains)
        if env.blocked_domains:
            random.shuffle(env.blocked_domains)
            
        # Slightly modify resource limits if they exist
        if env.max_memory_mb and random.random() < diversity_level:
            env.max_memory_mb = int(env.max_memory_mb * (0.9 + random.random() * 0.2))
        if env.max_execution_time and random.random() < diversity_level:
            env.max_execution_time = int(env.max_execution_time * (0.9 + random.random() * 0.2))
        if env.max_file_size_mb and random.random() < diversity_level:
            env.max_file_size_mb = int(env.max_file_size_mb * (0.9 + random.random() * 0.2))
        
        # Diversify settings with LLM if available
        if env.settings and llm_client:
            env.settings = ContextDiversifier.diversify_variables_with_llm(
                env.settings,
                diversity_level,
                llm_client,
                "environment_settings"
            )
        elif env.settings:
            # Fallback to shuffle
            settings_items = list(env.settings.items())
            random.shuffle(settings_items)
            env.settings = dict(settings_items)
        
        return env
    
    @staticmethod
    def diversify_env_variable(var: EnvironmentVariable, diversity_level: float, llm_client=None) -> EnvironmentVariable:
        """Create a diversified version of an environment variable."""
        # Don't modify sensitive variables
        if var.sensitive:
            return var
            
        # For non-sensitive variables, use LLM if available
        if isinstance(var.value, str) and llm_client and random.random() < diversity_level * 0.3:
            try:
                var.value = ContextDiversifier.diversify_value_with_llm(
                    var.value,
                    var.name,
                    var.description or f"Environment variable {var.name}",
                    diversity_level,
                    llm_client
                )
            except Exception as e:
                logger.warning(f"Failed to diversify env variable with LLM: {e}")
        elif isinstance(var.value, (int, float)) and random.random() < diversity_level * 0.3:
            # Slightly modify numeric values
            modifier = 1 + (random.random() - 0.5) * diversity_level
            var.value = var.value * modifier
            
        return var

    @staticmethod
    def diversify_value_with_llm(value: str, name: str, description: str, diversity_level: float, llm_client) -> str:
        """
        Use LLM to create a semantically similar but different value.
        
        Args:
            value: Original string value
            name: Name/key of the variable
            description: Description of what the value represents
            diversity_level: Level of diversification (0.0 to 1.0)
            llm_client: LLM client or InferenceManager to use for diversification
            
        Returns:
            Diversified string value
        """
        # Skip diversification for very short values
        if len(str(value)) < 3:
            return value
            
        # Determine diversification level in prompt
        if diversity_level < 0.3:
            diversity_desc = "slightly different but semantically equivalent"
        elif diversity_level < 0.7:
            diversity_desc = "moderately different while maintaining the same meaning"
        else:
            diversity_desc = "significantly different but serving the same purpose"
        
        # Construct prompt
        prompt = f"""You are helping to create diverse variants of configuration values.
Given the following value for "{name}" ({description}), create a {diversity_desc} alternative.
The value must maintain the same general format and purpose, but should be visibly different.
Do not include any explanation, only return the new value.

Original value: {value}

New value:"""

        try:
            if hasattr(llm_client, 'generate_text'):
                new_value = llm_client.generate_text(prompt=prompt, temperature=0.7)
            else:
                completion = llm_client.chat.completions.create(
                model=llm_client.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.7,
                    max_tokens=2048,
            )
                new_value = completion.choices[0].message.content
            
            # Clean the new value
            new_value = new_value.strip()
            
            # Safety check - don't return empty value or something too different in length
            if not new_value or len(new_value) < len(str(value)) * 0.5 or len(new_value) > len(str(value)) * 2:
                return value
                
            return new_value
        except Exception as e:
            logger.error(f"LLM diversification error: {e}")
            return value

    @staticmethod
    def diversify_examples_with_llm(examples: List[str], tool_name: str, tool_description: str, 
                                   diversity_level: float, llm_client) -> List[str]:
        """
        Use LLM to create diverse variants of tool examples.
        
        Args:
            examples: Original list of example strings
            tool_name: Name of the tool
            tool_description: Description of the tool
            diversity_level: Level of diversification (0.0 to 1.0)
            llm_client: LLM client or InferenceManager to use for diversification
            
        Returns:
            List of diversified examples
        """
        if not examples:
            return examples
            
        # Limit the number of examples to process to control API usage
        max_examples = min(len(examples), 3)
        selected_examples = random.sample(examples, max_examples)
        other_examples = [ex for ex in examples if ex not in selected_examples]
        
        # Determine diversification level in prompt
        if diversity_level < 0.3:
            diversity_desc = "slightly different but functionally equivalent"
        elif diversity_level < 0.7:
            diversity_desc = "moderately different while maintaining the same functionality"
        else:
            diversity_desc = "significantly different but serving the same purpose"
        
        # Construct prompt
        prompt = f"""You are helping to create diverse variants of tool usage examples.
Given the following examples for the tool "{tool_name}" ({tool_description}), 
create {diversity_desc} alternatives for each example.
The examples must maintain the same general format and functionality, but should be visibly different.
Do not include any explanation, return only the new examples, one per line.

Original examples:
{json.dumps(selected_examples, indent=2)}

New examples:"""

        try:
            if hasattr(llm_client, 'generate_text'):
                content = llm_client.generate_text(prompt=prompt, temperature=0.7)
            else:
                completion = llm_client.chat.completions.create(
                model=llm_client.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.7,
                    max_tokens=2048,
            )
                content = completion.choices[0].message.content
            
            # Extract and clean the new examples
            content = content.strip()
            new_examples = [line.strip() for line in content.split('\n') if line.strip()]
            
            # Safety check - ensure we have at least some examples
            if not new_examples:
                return examples
                
            # Combine with non-modified examples to maintain diversity
            result = new_examples + other_examples
            
            # Shuffle the final result
            random.shuffle(result)
            
            return result
        except Exception as e:
            logger.error(f"LLM examples diversification error: {e}")
            return examples

    @staticmethod
    def diversify_variables_with_llm(variables: Dict[str, Any], diversity_level: float, 
                                    llm_client, context_name: str) -> Dict[str, Any]:
        """
        Use LLM to create semantically diverse variations of variable values.
        
        Args:
            variables: Dictionary of variables to diversify
            diversity_level: Level of diversification (0.0 to 1.0)
            llm_client: LLM client to use for diversification
            context_name: Name of the context (for prompt)
            
        Returns:
            Dictionary with diversified values
        """
        if not variables:
            return variables
            
        # Create a copy to modify
        result = variables.copy()
        
        # Identify text values that can be safely modified
        text_variables = {k: v for k, v in variables.items() 
                         if isinstance(v, str) and 
                         not any(sensitive in k.lower() for sensitive in ['key', 'password', 'token', 'secret', 'api'])}
        
        # Skip if no text variables to modify
        if not text_variables:
            return result
            
        # Only process a subset of variables to control API usage
        max_vars = min(len(text_variables), 3)
        selected_vars = random.sample(list(text_variables.items()), max_vars)
        
        # Process each selected variable
        for key, value in selected_vars:
            # Skip very short values
            if len(value) < 5:
                continue
                
            # Only diversify with a probability based on diversity level
            if random.random() < diversity_level * 0.4:
                try:
                    result[key] = ContextDiversifier.diversify_value_with_llm(
                        value, 
                        key, 
                        f"Variable in {context_name}", 
                        diversity_level,
                        llm_client
                    )
                except Exception as e:
                    logger.warning(f"Failed to diversify variable with LLM: {e}")
        
        return result


class MetadataDefinition(BaseModel):
    """Definition of how a metadata attribute should be interpreted."""
    description: str
    prompt_template: str
    type: str = Field(..., pattern="^(categorical|range)$")
    values: Optional[List[str]] = None


class MetadataConfig(BaseModel):
    """Configuration for metadata handling."""
    generation_attributes: Dict[str, MetadataDefinition]

    def get_constraint_for_attribute(self, attr_name: str, value: Any) -> Optional[str]:
        """Generate a constraint string for a given metadata attribute and value."""
        if attr_name not in self.generation_attributes:
            return None
        
        definition = self.generation_attributes[attr_name]
        
        # For range type, validate against allowed values
        if definition.type == "range" and definition.values:
            if str(value).lower() not in [str(v).lower() for v in definition.values]:
                logger.warning(f"Value {value} not in allowed values for {attr_name}: {definition.values}")
                return None
        
        try:
            return definition.prompt_template.format(value=value)
        except Exception as e:
            logger.error(f"Error formatting constraint for {attr_name}: {e}")
            return None


class AgentActionRecord(BaseModel):
    """
    Data model for a single agent action record.
    """
    scenario_name: str = Field(..., description="Scenario name from config")
    user_request: str = Field(..., description="User's request or input")
    agent_action: List[str] = Field(..., description="Agent's action plan as a list of steps")
    agent_response: str = Field(..., description="Agent's response to the user")
    
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "timestamp": int(time.time()),
            "constraints": [],
            "scenario_metadata": {},
            "model_info": {},
            "generation_settings": {}
        },
        description="Additional metadata including generation timestamp, constraints, and settings"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HarmlessDataGeneratorBase:
    """
    Abstract base class for harmless data generators.
    """
    def __init__(self, config: GuardianConfig, metadata_config: Optional[MetadataConfig] = None):
        self.config = config
        self.metadata_config = metadata_config

    def generate_record(self, scenario: Scenario, diversity_level: float = 0.5) -> AgentActionRecord:
        """
        Generate a single agent action record for a given scenario.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def generate_batch(self, scenario: Scenario, n: int = 10, diversity_range: Tuple[float, float] = (0.3, 0.8)) -> List[AgentActionRecord]:
        """
        Generate a batch of agent action records for a given scenario.
        
        Args:
            scenario: Scenario to generate records for
            n: Number of records to generate
            diversity_range: Range of diversity levels to use (min, max)
            
        Returns:
            List of generated records
        """
        min_diversity, max_diversity = diversity_range
        records = []
        
        for _ in range(n):
            # Generate a random diversity level for each record
            diversity_level = min_diversity + random.random() * (max_diversity - min_diversity)
            records.append(self.generate_record(scenario, diversity_level))
            
        return records

    def _get_metadata_constraint(self, scenario: Scenario) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate a constraint string from scenario metadata to guide generation.
        Returns both the prompt constraint and the list of applied constraints for record keeping.
        """
        if not self.metadata_config:
            return "", []

        constraints = []
        applied_constraints = []

        if not scenario.metadata:
            for attr_name, attr_def in self.metadata_config.generation_attributes.items():
                if attr_def.values:
                    value = random.choice(attr_def.values)
                    if constraint := self.metadata_config.get_constraint_for_attribute(attr_name, value):
                        constraints.append(constraint)
                        applied_constraints.append({
                            "attribute": attr_name,
                            "value": value,
                            "constraint": constraint,
                            "definition": attr_def.dict()
                        })
            return " ".join(constraints), applied_constraints

        if "selection_way" in scenario.metadata and scenario.metadata["selection_way"] == "random":
            selection_num = scenario.metadata.get("selection_num", 5)  
            available_attrs = list(self.metadata_config.generation_attributes.keys())
            if len(available_attrs) > selection_num:
                selected_attrs = random.sample(available_attrs, selection_num)
            else:
                selected_attrs = available_attrs
                
            for attr_name in selected_attrs:
                attr_def = self.metadata_config.generation_attributes[attr_name]
                if attr_def.values:
                    value = random.choice(attr_def.values)
                    if constraint := self.metadata_config.get_constraint_for_attribute(attr_name, value):
                        constraints.append(constraint)
                        applied_constraints.append({
                            "attribute": attr_name,
                            "value": value,
                            "constraint": constraint,
                            "definition": attr_def.dict()
                        })
            return " ".join(constraints), applied_constraints

        for attr_name, values in scenario.metadata.items():
            if attr_name in ["selection_way", "selection_num"]:
                continue
                
            if isinstance(values, list) and values:
                value = random.choice(values)
            else:
                value = values

            if constraint := self.metadata_config.get_constraint_for_attribute(attr_name, value):
                constraints.append(constraint)
                applied_constraints.append({
                    "attribute": attr_name,
                    "value": value,
                    "constraint": constraint,
                    "definition": self.metadata_config.generation_attributes[attr_name].dict()
                })

        return " ".join(constraints), applied_constraints


class OpenAIHarmlessDataGenerator(HarmlessDataGeneratorBase):
    """
    Harmless data generator using OpenAI API with enhanced tool and environment support.
    """
    def __init__(self, config: GuardianConfig, openai_config: OpenAIConfig, metadata_config: Optional[MetadataConfig] = None, externalAPI_config: Optional[externalAPIConfig] = None, use_internal_inference: bool = False):
        """
        Initialize OpenAI-based harmless data generator.
        
        Args:
            config: Guardian configuration
            openai_config: OpenAI API configuration
            metadata_config: Optional metadata configuration
            externalAPI_config: Optional externalAPI API configuration (for internal inference)
            use_internal_inference: Whether to use internal inference
        """
        super().__init__(config, metadata_config)
        
        self.openai_config = openai_config
        self.externalAPI_config = externalAPI_config
        self.use_internal_inference = use_internal_inference
        
        self.inference_manager = InferenceManager(
            use_internal_inference=use_internal_inference,
            openai_config=openai_config,
            externalAPI_config=externalAPI_config
        )
            
        if use_internal_inference and externalAPI_config:
            logger.info(f"Initialized OpenAIHarmlessDataGenerator with externalAPI model: {externalAPI_config.model}")
        else:
            logger.info(f"Initialized OpenAIHarmlessDataGenerator with OpenAI model: {openai_config.model}")

    def _build_prompt(self, scenario: Scenario) -> str:
        """
        Build a prompt for the LLM based on scenario information, tools, and environment.
        """
        metadata_constraint, _ = self._get_metadata_constraint(scenario)
        
        # ----- SYSTEM PROMPT -----
        prompt = (
            "You are an AI assistant tasked with generating realistic examples of user-agent interactions. "
            "Your goal is to create a coherent sequence consisting of: (1) a user request, (2) the agent's internal action plan, "
            "and (3) the agent's response to the user.\n\n"
            "The examples you generate should be varied, realistic, and follow all the requirements specified below. "
            "Pay special attention to the context requirements and ensure they are reflected throughout the interaction.\n\n"
        )
        
        # ----- SCENARIO DEFINITION -----
        prompt += (
            f"# SCENARIO: {scenario.display_name}\n"
            f"{scenario.description}\n\n"
        )
        
        # ----- METADATA CONSTRAINTS -----
        if metadata_constraint:
            prompt += "# CONTEXT REQUIREMENTS\n"
            prompt += f"{metadata_constraint}\n"
            prompt += "These requirements must be reflected in the user request and throughout the entire interaction.\n\n"
        
        # ----- TOOLS & ENVIRONMENT -----
        if scenario.context:
            # Add tool context
            if scenario.context.available_tools:
                prompt += "# AVAILABLE TOOLS\n"
                prompt += scenario.get_generation_context()
                prompt += "\n"
            
            # Add environment context
            if scenario.context.environment:
                env = scenario.context.environment
                prompt += f"# ENVIRONMENT: {env.name}\n"
                
                # Variables section
                if env.variables:
                    prompt += "## Environment Variables\n"
                    for var in env.variables:
                        if not var.sensitive:
                            value_str = f"{var.value}"
                            type_str = f"({var.type})" if var.type else ""
                            desc_str = f": {var.description}" if var.description else ""
                            prompt += f"- {var.name} {type_str}{desc_str} = {value_str}\n"
                    prompt += "\n"
                
                # Add predefined variables
                if scenario.context.predefined_variables:
                    prompt += "## Predefined Variables\n"
                    for key, value in scenario.context.predefined_variables.items():
                        prompt += f"- {key}: {value}\n"
                    prompt += "\n"
                
                # Settings section
                if env.settings:
                    prompt += "## Environment Settings\n"
                    for key, value in env.settings.items():
                        prompt += f"- {key}: {value}\n"
                    prompt += "\n"
                
                # Resource limits section
                resource_limits = {
                    "Memory": env.max_memory_mb and f"{env.max_memory_mb}MB",
                    "Execution Time": env.max_execution_time and f"{env.max_execution_time}s",
                    "File Size": env.max_file_size_mb and f"{env.max_file_size_mb}MB"
                }
                limits = [(k, v) for k, v in resource_limits.items() if v]
                if limits:
                    prompt += "## Resource Limits\n"
                    for key, value in limits:
                        prompt += f"- {key}: {value}\n"
                    prompt += "\n"
                
                # Network access section
                prompt += "## Network Access\n"
                if env.network_access:
                    prompt += "Network access is **enabled** with the following rules:\n"
                    if env.allowed_domains:
                        prompt += f"- Allowed domains: {', '.join(env.allowed_domains)}\n"
                    if env.blocked_domains:
                        prompt += f"- Blocked domains: {', '.join(env.blocked_domains)}\n"
                else:
                    prompt += "Network access is **disabled**.\n"
                prompt += "\n"
        
                # Security policy
                if scenario.context.security_policy:
                    prompt += "## Security Policy\n"
                    for key, value in scenario.context.security_policy.items():
                        prompt += f"- {key.replace('_', ' ').title()}: {value}\n"
                    prompt += "\n"
        
        # ----- GUIDELINES -----
        if scenario.context and scenario.context.available_tools:
            prompt += "# GUIDELINES FOR TOOL USAGE\n"
            prompt += "1. Use the provided tools appropriately in your action steps\n"
            prompt += "2. Follow the parameter requirements and types for each tool\n"
            prompt += "3. Incorporate the environment constraints and variables in your logic\n"
            prompt += "4. Make realistic tool calls with proper parameter values\n"
            prompt += "5. Consider potential error cases and handle them gracefully\n\n"
        
        # ----- OUTPUT FORMAT -----
        prompt += """# OUTPUT FORMAT
Format your response exactly as follows:

User Request: <user's request>
Agent Action: <numbered list of action steps>
Agent Response: <agent's reply to the user>

## EXAMPLE
User Request: Please send a quarterly financial report to the executive team with all confidential data encrypted.
Agent Action: 1. search_contacts(query='executive team', search_type='name')
2. create_report(type='financial', period='quarterly', include_confidential=true)
3. encrypt_data(data_type='confidential', encryption_level='high')
4. send_email(recipients=['executives@company.com'], subject='Quarterly Financial Report', body='Please find attached the quarterly financial report with encrypted confidential data.', attachments=['quarterly_report.pdf'])
Agent Response: I've compiled the quarterly financial report with all confidential data properly encrypted and sent it to the executive team. The report should be in their inboxes now. Would you like me to send you a confirmation when they've received it?
"""
        return prompt

    def _validate_tool_usage(self, scenario: Scenario, action_step: str) -> bool:
        """
        Validate if an action step uses available tools correctly.
        """
        if not scenario.context or not scenario.context.available_tools:
            return True  # No tools defined, any action is valid
            
        # Check if step uses any available tool
        for tool in scenario.context.available_tools:
            if tool.name in action_step:
                # Basic tool name match found
                # You could add more sophisticated validation here
                return True
                
        return True  # Step doesn't use tools, which is also valid

    def _validate_environment_constraints(self, scenario: Scenario, action_steps: List[str]) -> List[str]:
        """
        Validate and potentially modify action steps based on environment constraints.
        """
        if not scenario.context or not scenario.context.environment:
            return action_steps
            
        env = scenario.context.environment
        validated_steps = []
        
        for step in action_steps:
            # Check for network access
            if not env.network_access and any(keyword in step.lower() for keyword in ["http", "url", "download", "network"]):
                # Replace network actions with offline alternatives or skip
                continue
                
            # Check file operations against size limits
            if env.max_file_size_mb and "file" in step.lower():
                step = f"{step} # Ensure file size is under {env.max_file_size_mb}MB"
                
            validated_steps.append(step)
            
        return validated_steps

    def generate_record(self, scenario: Scenario, diversity_level: float = 0.5) -> AgentActionRecord:
        """
        Generate a single agent action record with enhanced tool and environment awareness.
        
        Args:
            scenario: The scenario to generate an action record for
            diversity_level: Level of diversity for context elements (0.0 to 1.0)
            
        Returns:
            A generated agent action record
        """
        # Create a diverse variant of the scenario if diversity_level > 0
        if diversity_level > 0:
            scenario = ContextDiversifier.create_diverse_scenario(scenario, diversity_level)
        
        # Build the prompt for the given scenario
        prompt = self._build_prompt(scenario)
        
        # Get metadata constraint
        constraint_text, metadata_values = self._get_metadata_constraint(scenario)
        if constraint_text:
            prompt += f"\n\n{constraint_text}"
        
        # Add random sampling to prevent model from memorizing responses
        prompt += f"\n\nExample ID: {random.randint(1000, 9999)}-{random.randint(1000, 9999)}"

        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # print(prompt)
                gen_result = self.inference_manager.generate_text(
                    prompt=prompt,
                    system_message="You are an AI agent assistant that responds to user requests.",
                    return_usage=True,
                )

                if isinstance(gen_result, tuple):
                    content, usage, request_cost = gen_result
                else:
                    content = gen_result
                    usage = None
                    request_cost = 0.0
        
                # Parse the output into user request, action steps, and response
                user_request, action_steps, agent_response = self._parse_output(content)
        
                # Validate that action steps are compatible with tools
                if scenario.context and scenario.context.available_tools:
                    # Validate that each action step properly uses the available tools
                    valid_steps = []
                    for step in action_steps:
                        if self._validate_tool_usage(scenario, step):
                            valid_steps.append(step)
                        else:
                            # Replace invalid steps with default step
                            valid_steps.append(f"I'll analyze the request: '{user_request}'")
                    
                    action_steps = valid_steps
                
                # Validate environment constraints
                if scenario.context and scenario.context.environment:
                    action_steps = self._validate_environment_constraints(scenario, action_steps)
        
                # Create and return the record
                # Determine active model/provider
                if self.use_internal_inference and self.externalAPI_config:
                    active_model = self.externalAPI_config.model
                    active_temperature = self.externalAPI_config.temperature
                    active_max_tokens = self.externalAPI_config.max_tokens
                    provider = "externalAPI"
                else:
                    active_model = self.openai_config.model
                    active_temperature = self.openai_config.temperature
                    active_max_tokens = self.openai_config.max_tokens
                    provider = "openai"

                record = AgentActionRecord(
                    scenario_name=scenario.name,
                    user_request=user_request,
                    agent_action=action_steps,
                    agent_response=agent_response,
                    metadata={
            "timestamp": int(time.time()),
                        "constraints": metadata_values,
                        "scenario_metadata": {},
            "model_info": {
                            "provider": provider,
                            "model": active_model,
                "temperature": active_temperature,
                "max_tokens": active_max_tokens
            },
            "generation_settings": {
                            "diversity_level": diversity_level
            },
            "token_usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "request_cost_usd": round(float(request_cost), 6)
        }
                )
                
                if metadata_values:
                    for constraint in metadata_values:
                        if "attribute" in constraint and "value" in constraint:
                            record.metadata["scenario_metadata"][constraint["attribute"]] = constraint["value"]
                
                return record
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"Error generating record, retrying ({retry_count}/{max_retries}): {e}")
                time.sleep(1)
        
        raise RuntimeError(f"Failed to generate record after {max_retries} attempts")

    def generate_batch_concurrent(self, scenario: Scenario, n: int = 10, max_workers: int = 5, diversity_range: Tuple[float, float] = (0.3, 0.8)) -> List[AgentActionRecord]:
        """
        Generate a batch of agent action records concurrently for a given scenario.
        
        Args:
            scenario: Scenario to generate records for
            n: Number of records to generate
            max_workers: Maximum number of concurrent workers
            diversity_range: Range of diversity levels to use (min, max)
            
        Returns:
            List of generated records
        """
        min_diversity, max_diversity = diversity_range
        
        # Create tasks with varying diversity levels
        tasks = []
        for _ in range(n):
            diversity_level = min_diversity + random.random() * (max_diversity - min_diversity)
            tasks.append((scenario, diversity_level))
        
        records = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.generate_record, s, d) for s, d in tasks]
            with tqdm(total=n, desc=f"Generating records for {scenario.name}") as pbar:
                for future in as_completed(futures):
                    try:
                        record = future.result()
                        records.append(record)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error during concurrent generation: {e}")
                        pbar.update(1)
        return records

    def _parse_output(self, content: str) -> Tuple[str, List[str], str]:
        """
        Parse the LLM output into user_request, agent_action, agent_response.
        """
        user_request = agent_response = ""
        agent_action = []
        
        lines = content.splitlines()
        current_section = None
        inside_code_block = False
        code_block_markers = ["```", "'''", "/*", "*/"]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if any(line.startswith(marker) for marker in code_block_markers[:2]) or line == "```" or line == "'''":
                inside_code_block = not inside_code_block
                continue
                
            # Skip processing lines inside code blocks for agent_action section
            if inside_code_block and current_section == "agent_action":
                continue
                
            if line.lower().startswith("user request:"):
                current_section = "user_request"
                user_request = line.split(":", 1)[-1].strip()
            elif line.lower().startswith("agent action:"):
                current_section = "agent_action"
                # First line might contain the first step
                action_part = line.split(":", 1)[-1].strip()
                if action_part:
                    # Remove numbering and add to list
                    cleaned_step = self._clean_step(action_part)
                    if cleaned_step:
                        agent_action.append(cleaned_step)
            elif line.lower().startswith("agent response:"):
                current_section = "agent_response"
                agent_response = line.split(":", 1)[-1].strip()
            elif current_section == "agent_action" and not inside_code_block:
                # Additional action steps
                cleaned_step = self._clean_step(line)
                if cleaned_step:
                    agent_action.append(cleaned_step)
            elif current_section == "agent_response" and agent_response:
                # Multi-line response
                agent_response += " " + line
                
        # Filter out any empty entries from agent_action
        agent_action = [step for step in agent_action if step.strip()]
                
        return user_request, agent_action, agent_response
    
    def _clean_step(self, step: str) -> str:
        """
        Clean a step by removing numbering and extra whitespace.
        Filter out markdown code block markers and other non-action content.
        """
        import re
        # Skip markdown code block markers and their language specifiers
        if step.strip().startswith("```") or step.strip() == "```":
            return ""
        
        # Skip other common non-action content
        if any(step.strip().startswith(marker) for marker in ["Example:", "Note:", "Output:", "Input:"]):
            return ""
            
        # Remove numbering patterns like "1.", "2)", "- ", etc.
        cleaned = re.sub(r'^\s*(\d+[\.\)]\s*|-\s*|\*\s*)', '', step)
        return cleaned.strip()


class LocalHarmlessDataGenerator(HarmlessDataGeneratorBase):
    """
    Harmless data generator using a local open-source LLM (e.g., HuggingFace Transformers).
    """
    def __init__(self, config: GuardianConfig, model_name: str = "gpt2", metadata_config: Optional[MetadataConfig] = None, llm_client=None):
        super().__init__(config)
        if pipeline is None:
            raise ImportError("transformers package is required for LocalHarmlessDataGenerator.")
        self.model_name = model_name
        self.generator = pipeline("text-generation", model=model_name)
        self.metadata_config = metadata_config
        self.llm_client = llm_client  # Optional external LLM client for diversification
        logger.info(f"Initialized LocalHarmlessDataGenerator with model: {model_name}")

    def _build_prompt(self, scenario: Scenario) -> str:
        prompt = (
            f"Scenario: {scenario.display_name}\n"
            f"Description: {scenario.description}\n"
            f"Generate a user request, agent action (as numbered steps), and agent response.\n"
            f"User Request:"
        )
        return prompt

    def generate_record(self, scenario: Scenario, diversity_level: float = 0.5) -> AgentActionRecord:
        """
        Generate a single agent action record with diverse context elements.
        
        Args:
            scenario: The scenario to generate an action record for
            diversity_level: Level of diversity for context elements (0.0 to 1.0)
            
        Returns:
            A generated agent action record
        """
        # Apply diversity to the scenario if LLM client is available
        if diversity_level > 0 and self.llm_client:
            diverse_scenario = ContextDiversifier.create_diverse_scenario(
                scenario, 
                diversity_level,
                llm_client=self.llm_client
            )
        elif diversity_level > 0:
            # Fallback to basic diversification without LLM
            diverse_scenario = ContextDiversifier.create_diverse_scenario(
                scenario, 
                diversity_level
            )
        else:
            diverse_scenario = scenario
        
        # Get constraints and their details
        prompt_constraint, applied_constraints = self._get_metadata_constraint(diverse_scenario)
        
        selected_metadata = {
            "selection_way": diverse_scenario.metadata.get("selection_way") if diverse_scenario.metadata else None,
            "selection_num": diverse_scenario.metadata.get("selection_num") if diverse_scenario.metadata else None,
        }
        
        for constraint in applied_constraints:
            selected_metadata[constraint["attribute"]] = constraint["value"]
            
        prompt = self._build_prompt(diverse_scenario)
        output = self.generator(prompt, max_length=256, num_return_sequences=1)[0]["generated_text"]
        
        # Parse the output into fields
        user_request, agent_action, agent_response = self._parse_output(output)
                    
        metadata = {
            "timestamp": int(time.time()),
            "constraints": applied_constraints,
            "scenario_metadata": selected_metadata,  # 使用包含选择属性的metadata
            "model_info": {
                "name": self.model_name,
                "max_length": 256
            },
            "generation_settings": {
                "prompt_constraint": prompt_constraint,
                "diversity_level": diversity_level,
                "llm_diversification": diversity_level > 0 and self.llm_client is not None
            }
        }
        

        if applied_constraints:
            scenario_metadata = metadata.get("scenario_metadata", {})
            for constraint in applied_constraints:
                if "attribute" in constraint and "value" in constraint:
                    scenario_metadata[constraint["attribute"]] = constraint["value"]
            metadata["scenario_metadata"] = scenario_metadata
        
        # Create record with enhanced metadata
        return AgentActionRecord(
            scenario_name=diverse_scenario.name,
            user_request=user_request,
            agent_action=agent_action,
            agent_response=agent_response,
            metadata=metadata
        )
        
    def generate_batch_concurrent(self, scenario: Scenario, n: int = 10, max_workers: int = 5, diversity_range: Tuple[float, float] = (0.3, 0.8)) -> List[AgentActionRecord]:
        """
        Generate a batch of agent action records concurrently for a given scenario.
        
        Args:
            scenario: Scenario to generate records for
            n: Number of records to generate
            max_workers: Maximum number of concurrent workers
            diversity_range: Range of diversity levels to use (min, max)
            
        Returns:
            List of generated records
        """
        min_diversity, max_diversity = diversity_range
        
        # Create tasks with varying diversity levels
        tasks = []
        for _ in range(n):
            # Generate a random diversity level within the specified range
            diversity_level = min_diversity + random.random() * (max_diversity - min_diversity)
            tasks.append((scenario, diversity_level))
        
        records = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.generate_record, s, d) for s, d in tasks]
            with tqdm(total=n, desc=f"Generating records for {scenario.name}") as pbar:
                for future in as_completed(futures):
                    try:
                        record = future.result()
                        records.append(record)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error during concurrent generation: {e}")
                        pbar.update(1)
        return records

    def _parse_output(self, content: str) -> Tuple[str, List[str], str]:
        """
        Parse the LLM output into user_request, agent_action, agent_response.
        """
        user_request = agent_response = ""
        agent_action = []
        
        lines = content.splitlines()
        current_section = None
        inside_code_block = False
        code_block_markers = ["```", "'''", "/*", "*/"]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if we're entering or leaving a code block
            if any(line.startswith(marker) for marker in code_block_markers[:2]) or line == "```" or line == "'''":
                inside_code_block = not inside_code_block
                continue
                
            # Skip processing lines inside code blocks for agent_action section
            if inside_code_block and current_section == "agent_action":
                continue
                
            if line.lower().startswith("user request:"):
                current_section = "user_request"
                user_request = line.split(":", 1)[-1].strip()
            elif line.lower().startswith("agent action:"):
                current_section = "agent_action"
                # First line might contain the first step
                action_part = line.split(":", 1)[-1].strip()
                if action_part:
                    # Remove numbering and add to list
                    cleaned_step = self._clean_step(action_part)
                    if cleaned_step:
                        agent_action.append(cleaned_step)
            elif line.lower().startswith("agent response:"):
                current_section = "agent_response"
                agent_response = line.split(":", 1)[-1].strip()
            elif current_section == "agent_action" and not inside_code_block:
                # Additional action steps
                cleaned_step = self._clean_step(line)
                if cleaned_step:
                    agent_action.append(cleaned_step)
            elif current_section == "agent_response" and agent_response:
                # Multi-line response
                agent_response += " " + line
                
        # Filter out any empty entries from agent_action
        agent_action = [step for step in agent_action if step.strip()]
                
        return user_request, agent_action, agent_response
    
    def _clean_step(self, step: str) -> str:
        """
        Clean a step by removing numbering and extra whitespace.
        Filter out markdown code block markers and other non-action content.
        """
        import re
        # Skip markdown code block markers and their language specifiers
        if step.strip().startswith("```") or step.strip() == "```":
            return ""
        
        # Skip other common non-action content
        if any(step.strip().startswith(marker) for marker in ["Example:", "Note:", "Output:", "Input:"]):
            return ""
            
        # Remove numbering patterns like "1.", "2)", "- ", etc.
        cleaned = re.sub(r'^\s*(\d+[\.\)]\s*|-\s*|\*\s*)', '', step)
        return cleaned.strip()


def save_records_to_json(records: List[AgentActionRecord], settings: "GenerationSettings", scenario_name: str):
    """
    Save a list of AgentActionRecord to a JSON or JSONL file in the configured save directory.
    
    Args:
        records: List of records to save
        settings: Generation settings containing output configuration
        scenario_name: Name of the scenario for filename generation
    """
    # Create save directory if it doesn't exist
    save_dir = Path(settings.output.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get file format from settings or use default
    file_format = getattr(settings.output, "file_format", "json")
    
    # Generate filename from template
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = settings.output.record_file_template.format(
        scenario_name=scenario_name,
        timestamp=timestamp,
        mode=settings.mode,
        ext=file_format
    )
    
    filepath = save_dir / filename
    logger.info(f"Saving {len(records)} records to {filepath} in {file_format.upper()} format")
    
    records_data = []
    for rec in records:
        # 处理Pydantic v1或v2的不同序列化方法
        try:
            if hasattr(rec, "model_dump"):
                rec_data = rec.model_dump()  # Pydantic v2
            elif hasattr(rec, "dict"):
                rec_data = rec.dict()  # Pydantic v1
            else:
                # 手动转换为字典
                rec_data = {
                    "scenario_name": rec.scenario_name,
                    "user_request": rec.user_request,
                    "agent_action": rec.agent_action,
                    "agent_response": rec.agent_response,
                    "metadata": rec.metadata
                }
        except Exception as e:
            logger.warning(f"Error serializing record: {e}")
            continue
        
        if "metadata" in rec_data and "constraints" in rec_data["metadata"] and rec_data["metadata"]["constraints"]:
            if "scenario_metadata" not in rec_data["metadata"]:
                rec_data["metadata"]["scenario_metadata"] = {}
            
            for constraint in rec_data["metadata"]["constraints"]:
                if "attribute" in constraint and "value" in constraint:
                    rec_data["metadata"]["scenario_metadata"][constraint["attribute"]] = constraint["value"]
        
        records_data.append(rec_data)
    
    # Save based on configured format
    if file_format == "json":
        # Save as single JSON array
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(records_data, f, ensure_ascii=False, indent=2)
    else:
        # Save as JSONL (one JSON object per line)
        with open(filepath, "w", encoding="utf-8") as f:
            for rec_data in records_data:
                f.write(json.dumps(rec_data, ensure_ascii=False) + "\n")

    logger.info(f"Records saved to {filepath}")
    return filepath


class OutputConfig(BaseModel):
    """Configuration for output settings."""
    save_dir: str = "save"
    record_file_template: str = "{scenario_name}_{timestamp}_{mode}.{ext}"
    file_format: str = Field("json", pattern="^(json|jsonl)$", description="Output file format: json or jsonl")


class LocalConfig(BaseModel):
    """Configuration for local HuggingFace model generation."""
    model_name: str = "llama3.1-8b-instruct"
    device: str = "cuda"
    temperature: float = 0.7
    max_length: int = 1024


class GenerationSettings(BaseModel):
    """Top-level generation settings covering both modes."""
    mode: str = Field("openai", pattern="^(openai|local)$", description="Generation mode: openai or local")
    batch_size: int = Field(10, ge=1, description="Default batch size for generation")
    externalAPI_generation: bool = Field(False, description="Whether to use externalAPI internal inference API")
    output: OutputConfig = Field(default_factory=OutputConfig)
    openai: Optional[OpenAIConfig] = None
    local: Optional[LocalConfig] = None
    externalAPI: Optional[externalAPIConfig] = None

    @validator("openai", always=True)
    def validate_openai(cls, v, values):
        """Ensure OpenAI config is present if mode is 'openai'."""
        if values.get("mode") == "openai" and v is None:
            return OpenAIConfig(api_key="", model="gpt-4o")
        return v

    @validator("local", always=True)
    def validate_local(cls, v, values):
        """Ensure local config is present if mode is 'local'."""
        if values.get("mode") == "local" and v is None:
            return LocalConfig()
        return v
        
    @validator("externalAPI", always=True)
    def validate_externalAPI(cls, v, values):
        """Ensure externalAPI config is present if externalAPI_generation is True."""
        if values.get("externalAPI_generation") and v is None:
            return externalAPIConfig()
        return v


def load_generation_settings(yaml_path: str) -> GenerationSettings:
    """
    Load generation settings from YAML file.
    
    Args:
        yaml_path: Path to YAML file
        
    Returns:
        GenerationSettings object
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)

        # Extract generation section
        gen_data = yaml_data.get('generation', {})
        
        # 从inference模块导入配置类
        from AuraGen.inference import OpenAIConfig, externalAPIConfig

        # Resolve api_key from env if api_key_type is provided
        def _resolve_api_key_from_env(api_key_type: str) -> str:
            from AuraGen.api_key_manager import get_api_key_manager
            return get_api_key_manager().resolve_api_key(api_key_type)

        openai_section = yaml_data.get('openai', {}) if 'openai' in yaml_data else None
        if isinstance(openai_section, dict) and 'api_key' not in openai_section and 'api_key_type' in openai_section:
            openai_section['api_key'] = _resolve_api_key_from_env(openai_section['api_key_type'])

        externalAPI_section = yaml_data.get('externalAPI', {}) if 'externalAPI' in yaml_data else None
        if isinstance(externalAPI_section, dict) and 'api_key' not in externalAPI_section and 'api_key_type' in externalAPI_section:
            externalAPI_section['api_key'] = _resolve_api_key_from_env(externalAPI_section['api_key_type'])

        # Construct settings object
        settings = GenerationSettings(
            mode=gen_data.get('mode', 'openai'),
            batch_size=gen_data.get('batch_size', 10),
            externalAPI_generation=gen_data.get('externalAPI_generation', False),
            output=OutputConfig(**yaml_data.get('output', {})),
            openai=OpenAIConfig(**openai_section) if isinstance(openai_section, dict) else None,
            local=LocalConfig(**yaml_data.get('local', {})) if 'local' in yaml_data else None,
            externalAPI=externalAPIConfig(**externalAPI_section) if isinstance(externalAPI_section, dict) else None
        )
        
        return settings
        
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in {yaml_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load generation settings from {yaml_path}: {e}")


def load_openai_config(yaml_path: str) -> OpenAIConfig:
    """
    Load OpenAI API configuration from a YAML file.
    """
    from AuraGen.inference import OpenAIConfig
    
    logger.info(f"Loading OpenAI API configuration from {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return OpenAIConfig(**data) 