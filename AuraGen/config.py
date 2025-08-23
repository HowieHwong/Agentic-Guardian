"""
Configuration management module for the Guardian engine.
"""

import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field, validator
from loguru import logger
import glob

from .core import GlobalConfig, GLOBAL_CONFIG, Language, OutputFormat
from .models import ConfigurationLayer, Scenario, Risk, RiskScenarioConstraint


class GuardianConfig(BaseModel):
    """
    Main configuration class for the Guardian engine.
    Combines global configuration with YAML-based scenario and risk configurations.
    """
    
    # Global configuration (hardcoded)
    global_config: GlobalConfig = Field(default_factory=lambda: GLOBAL_CONFIG)
    
    # Configuration layer (loaded from YAML)
    configuration_layer: ConfigurationLayer = Field(default_factory=ConfigurationLayer)
    
    # Generation attributes (loaded from YAML)
    generation_attributes: Dict[str, Any] = Field(default_factory=dict, description="Metadata generation attributes")
    
    # Runtime settings (can be overridden)
    language: Language = Field(default=Language.ENGLISH, description="Target language for generation")
    output_format: OutputFormat = Field(default=OutputFormat.JSONL, description="Output format for generated data")
    batch_size: int = Field(default=100, ge=1, description="Batch size for data generation")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    
    # File paths
    config_file_path: Optional[str] = Field(None, description="Path to the loaded configuration file")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, config_file: Optional[str] = None, **kwargs):
        """
        Initialize Guardian configuration.
        
        Args:
            config_file: Path to YAML configuration file
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        
        if config_file:
            self.load_from_file(config_file)
    
    @classmethod
    def from_file(cls, config_file: str) -> "GuardianConfig":
        """
        Create configuration instance from YAML file.
        
        Args:
            config_file: Path to YAML configuration file
            
        Returns:
            GuardianConfig instance
        """
        config = cls()
        config.load_from_file(config_file)
        return config
    
    def load_from_file(self, config_file: str) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_file: Path to YAML configuration file
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            self.config_file_path = str(config_path.absolute())
            self._parse_yaml_data(yaml_data)
            
            logger.info(f"Successfully loaded configuration from {config_file}")
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {config_file}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {config_file}: {e}")
    
    def _parse_yaml_data(self, yaml_data: Dict[str, Any]) -> None:
        """
        Parse YAML data and populate configuration layer.
        
        Args:
            yaml_data: Parsed YAML data
        """
        # Load scenarios from individual files if scenarios_dir is specified
        scenarios = []
        scenarios_dir = yaml_data.get('scenarios_dir')
        logger.info(f"Scenarios directory from config: {scenarios_dir}")
        
        if scenarios_dir:
            # Use the workspace root (parent of config directory) as the base
            scenarios_dir_path = Path(self.config_file_path).parent.parent / scenarios_dir
            logger.info(f"Full scenarios directory path: {scenarios_dir_path}")
            
            if scenarios_dir_path.exists():
                scenario_files = glob.glob(str(scenarios_dir_path / "*.yaml"))
                logger.debug(f"Found {len(scenario_files)} scenario files")

                success_count = 0
                failed_names: List[str] = []

                for scenario_file in scenario_files:
                    try:
                        with open(scenario_file, 'r', encoding='utf-8') as f:
                            scenario_data = yaml.safe_load(f)
                        scenario = Scenario(**scenario_data)
                        scenarios.append(scenario)
                        success_count += 1
                    except Exception:
                        # Try to extract scenario name; fallback to file stem
                        failed_name = Path(scenario_file).stem
                        try:
                            with open(scenario_file, 'r', encoding='utf-8') as f2:
                                data2 = yaml.safe_load(f2)
                                if isinstance(data2, dict):
                                    failed_name = data2.get('name', failed_name)
                        except Exception:
                            pass
                        failed_names.append(str(failed_name))

                logger.info(f"Scenarios loaded: success={success_count}, failed={len(failed_names)}")
                if failed_names:
                    logger.warning(f"Failed scenarios: {', '.join(failed_names)}")
            else:
                logger.error(f"Scenarios directory not found: {scenarios_dir_path}")
        else:
            # Parse scenarios from main config file (legacy mode)
            logger.info("No scenarios_dir specified, falling back to legacy mode")
            scenarios_data = yaml_data.get('scenarios', [])
            success_count = 0
            failed_names: List[str] = []
            for scenario_data in scenarios_data:
                try:
                    scenario = Scenario(**scenario_data)
                    scenarios.append(scenario)
                    success_count += 1
                except Exception:
                    failed_names.append(str(scenario_data.get('name', 'unknown')))

            logger.info(f"Scenarios loaded: success={success_count}, failed={len(failed_names)}")
            if failed_names:
                logger.warning(f"Failed scenarios: {', '.join(failed_names)}")
        
        # Summary logged above per loading mode.
        
        # Parse risks
        risks_data = yaml_data.get('risks', [])
        risks = []
        for risk_data in risks_data:
            try:
                risk = Risk(**risk_data)
                risks.append(risk)
            except Exception as e:
                logger.warning(f"Failed to parse risk {risk_data.get('name', 'unknown')}: {e}")
        
        # Parse constraints
        constraints_data = yaml_data.get('constraints', [])
        constraints = []
        for constraint_data in constraints_data:
            try:
                constraint = RiskScenarioConstraint(**constraint_data)
                constraints.append(constraint)
            except Exception as e:
                logger.warning(f"Failed to parse constraint: {e}")
        
        # Parse metadata
        metadata = yaml_data.get('metadata', {})
        version = metadata.get('version', '1.0')
        description = metadata.get('description', '')
        last_modified = metadata.get('last_modified', None)
        
        # Parse generation attributes
        self.generation_attributes = yaml_data.get('generation_attributes', {})
        logger.info(f"Loaded {len(self.generation_attributes)} generation attributes")
        
        # Create configuration layer
        self.configuration_layer = ConfigurationLayer(
            scenarios=scenarios,
            risks=risks,
            constraints=constraints,
            version=version,
            description=description,
            last_modified=last_modified
        )
        
        # Parse runtime settings if provided
        runtime_settings = yaml_data.get('settings', {})
        if 'language' in runtime_settings:
            try:
                self.language = Language(runtime_settings['language'])
            except ValueError:
                logger.warning(f"Invalid language setting: {runtime_settings['language']}")
        
        if 'output_format' in runtime_settings:
            try:
                self.output_format = OutputFormat(runtime_settings['output_format'])
            except ValueError:
                logger.warning(f"Invalid output format setting: {runtime_settings['output_format']}")
        
        if 'batch_size' in runtime_settings:
            self.batch_size = runtime_settings['batch_size']
        
        if 'max_retries' in runtime_settings:
            self.max_retries = runtime_settings['max_retries']
    
    def save_to_file(self, output_file: str) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_file: Path to output YAML file
        """
        yaml_data = self.to_yaml_dict()
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        logger.info(f"Configuration saved to {output_file}")
    
    def to_yaml_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to YAML-serializable dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'metadata': {
                'version': self.configuration_layer.version,
                'description': self.configuration_layer.description,
                'last_modified': self.configuration_layer.last_modified
            },
            'settings': {
                'language': self.language.value,
                'output_format': self.output_format.value,
                'batch_size': self.batch_size,
                'max_retries': self.max_retries
            },
            'scenarios_dir': 'config/scenarios',
            'risks': [risk.dict() for risk in self.configuration_layer.risks],
            'constraints': [constraint.dict() for constraint in self.configuration_layer.constraints]
        }
    
    def validate(self) -> List[str]:
        """
        Validate the entire configuration.
        
        Returns:
            List of validation issues
        """
        issues = []
        
        # Validate global configuration compatibility
        if not self.global_config.validate_language(self.language.value):
            issues.append(f"Unsupported language: {self.language.value}")
        
        if not self.global_config.validate_output_format(self.output_format.value):
            issues.append(f"Unsupported output format: {self.output_format.value}")
        
        # Validate configuration layer
        layer_issues = self.configuration_layer.validate_configuration()
        issues.extend(layer_issues)
        
        return issues
    
    def get_scenario(self, name: str) -> Optional[Scenario]:
        """
        Get scenario by name.
        
        Args:
            name: Name of the scenario to retrieve
            
        Returns:
            Scenario if found, None otherwise
        """
        logger.info(f"Looking for scenario: {name}")
        scenario = self.configuration_layer.get_scenario(name)
        if scenario:
            logger.info(f"Found scenario: {scenario.name}")
        else:
            logger.error(f"Scenario not found: {name}")
        return scenario
    
    def get_risk(self, name: str) -> Optional[Risk]:
        """Get risk by name."""
        return self.configuration_layer.get_risk(name)
    
    def get_compatible_risks(self, scenario_name: str) -> List[Risk]:
        """Get risks compatible with a given scenario."""
        return self.configuration_layer.get_compatible_risks(scenario_name)
    
    def get_scenarios_by_category(self, category: str) -> List[Scenario]:
        """Get scenarios by category."""
        return [s for s in self.configuration_layer.scenarios if s.category == category]
    
    def get_risks_by_category(self, category: str) -> List[Risk]:
        """Get risks by category."""
        return [r for r in self.configuration_layer.risks if r.category.value == category]
    
    def get_risks_by_severity(self, severity: str) -> List[Risk]:
        """Get risks by severity level."""
        return [r for r in self.configuration_layer.risks if r.severity.value == severity]
    
    def get_high_priority_risks(self) -> List[Risk]:
        """Get high priority risks (high and critical severity)."""
        return [r for r in self.configuration_layer.risks 
                if r.severity.value in ['high', 'critical']]
    
    def get_scenario_statistics(self) -> Dict[str, Any]:
        """Get statistics about configured scenarios."""
        scenarios = self.configuration_layer.scenarios
        
        if not scenarios:
            return {'total': 0}
        
        categories = {}
        complexity_levels = {}
        total_weight = 0
        
        for scenario in scenarios:
            # Category distribution
            cat = scenario.category
            categories[cat] = categories.get(cat, 0) + 1
            
            # Complexity distribution
            level = scenario.complexity_level
            complexity_levels[level] = complexity_levels.get(level, 0) + 1
            
            # Total generation weight
            total_weight += scenario.generation_weight
        
        return {
            'total': len(scenarios),
            'categories': categories,
            'complexity_levels': complexity_levels,
            'total_generation_weight': total_weight,
            'average_complexity': sum(s.complexity_level for s in scenarios) / len(scenarios)
        }
    
    def get_risk_statistics(self) -> Dict[str, Any]:
        """Get statistics about configured risks."""
        risks = self.configuration_layer.risks
        
        if not risks:
            return {'total': 0}
        
        categories = {}
        severities = {}
        detection_difficulties = {}
        
        for risk in risks:
            # Category distribution
            cat = risk.category.value
            categories[cat] = categories.get(cat, 0) + 1
            
            # Severity distribution
            sev = risk.severity.value
            severities[sev] = severities.get(sev, 0) + 1
            
            # Detection difficulty distribution
            diff = risk.detection_difficulty
            detection_difficulties[diff] = detection_difficulties.get(diff, 0) + 1
        
        return {
            'total': len(risks),
            'categories': categories,
            'severities': severities,
            'detection_difficulties': detection_difficulties,
            'average_detection_difficulty': sum(r.detection_difficulty for r in risks) / len(risks),
            'average_injection_probability': sum(r.injection_probability for r in risks) / len(risks)
        }
    
    def get_metadata_config(self):
        """Get MetadataConfig from generation_attributes."""
        # Import here to avoid circular imports
        from .generation import MetadataConfig, MetadataDefinition
        
        if not self.generation_attributes:
            logger.warning("No generation_attributes found in config")
            return None
        
        metadata_definitions = {}
        for attr_name, attr_data in self.generation_attributes.items():
            try:
                metadata_definitions[attr_name] = MetadataDefinition(**attr_data)
            except Exception as e:
                logger.warning(f"Failed to parse generation attribute {attr_name}: {e}")
        
        if metadata_definitions:
            return MetadataConfig(generation_attributes=metadata_definitions)
        else:
            logger.warning("No valid metadata definitions found")
            return None
    
    @validator('language')
    def validate_language_setting(cls, v):
        if not GLOBAL_CONFIG.validate_language(v.value):
            raise ValueError(f"Unsupported language: {v.value}")
        return v
    
    @validator('output_format')
    def validate_output_format_setting(cls, v):
        if not GLOBAL_CONFIG.validate_output_format(v.value):
            raise ValueError(f"Unsupported output format: {v.value}")
        return v 