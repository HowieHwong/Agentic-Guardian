"""
Utility functions for the Guardian engine.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from loguru import logger
import yaml
import os

from .config import GuardianConfig
from .models import Scenario, Risk, RiskScenarioConstraint
from .core import GLOBAL_CONFIG
from AuraGen.generation import load_generation_settings


console = Console()


def display_config_summary(config: GuardianConfig) -> None:
    """
    Display a formatted summary of the configuration.
    
    Args:
        config: GuardianConfig instance
    """
    console.print(Panel.fit(
        f"[bold blue]Guardian Configuration Summary[/bold blue]\n"
        f"Engine Version: {config.global_config.ENGINE_VERSION}\n"
        f"Config File: {config.config_file_path or 'None'}\n"
        f"Language: {config.language.value}\n"
        f"Output Format: {config.output_format.value}\n"
        f"Batch Size: {config.batch_size}",
        title="Configuration"
    ))
    
    # Scenario statistics
    scenario_stats = config.get_scenario_statistics()
    scenario_table = Table(title="Scenario Statistics")
    scenario_table.add_column("Metric", style="cyan")
    scenario_table.add_column("Value", style="green")
    
    scenario_table.add_row("Total Scenarios", str(scenario_stats.get('total', 0)))
    if scenario_stats.get('total', 0) > 0:
        scenario_table.add_row("Average Complexity", f"{scenario_stats.get('average_complexity', 0):.1f}")
        scenario_table.add_row("Total Generation Weight", f"{scenario_stats.get('total_generation_weight', 0):.1f}")
        
        # Category breakdown
        categories = scenario_stats.get('categories', {})
        for category, count in categories.items():
            scenario_table.add_row(f"  {category.title()}", str(count))
    
    console.print(scenario_table)
    
    # Risk statistics
    risk_stats = config.get_risk_statistics()
    risk_table = Table(title="Risk Statistics")
    risk_table.add_column("Metric", style="cyan")
    risk_table.add_column("Value", style="red")
    
    risk_table.add_row("Total Risks", str(risk_stats.get('total', 0)))
    if risk_stats.get('total', 0) > 0:
        risk_table.add_row("Avg Detection Difficulty", f"{risk_stats.get('average_detection_difficulty', 0):.1f}")
        risk_table.add_row("Avg Injection Probability", f"{risk_stats.get('average_injection_probability', 0):.2f}")
        
        # Severity breakdown
        severities = risk_stats.get('severities', {})
        for severity, count in severities.items():
            risk_table.add_row(f"  {severity.title()}", str(count))
    
    console.print(risk_table)


def display_scenario_details(scenario: Scenario) -> None:
    """
    Display detailed information about a specific scenario.
    
    Args:
        scenario: Scenario instance
    """
    details = f"""
[bold cyan]Name:[/bold cyan] {scenario.display_name}
[bold cyan]ID:[/bold cyan] {scenario.name}
[bold cyan]Category:[/bold cyan] {scenario.category}
[bold cyan]Complexity:[/bold cyan] {scenario.complexity_level}/10
[bold cyan]Estimated Duration:[/bold cyan] {scenario.estimated_duration or 'N/A'} seconds
[bold cyan]Generation Weight:[/bold cyan] {scenario.generation_weight}

[bold cyan]Description:[/bold cyan]
{scenario.description}

[bold cyan]Required Permissions:[/bold cyan]
{', '.join(scenario.required_permissions) or 'None'}

[bold cyan]Dependencies:[/bold cyan]
{', '.join(scenario.dependencies) or 'None'}

[bold cyan]Sample Actions:[/bold cyan]
{', '.join(scenario.sample_actions) or 'None'}

[bold cyan]Tags:[/bold cyan]
{', '.join(scenario.tags) or 'None'}
"""
    
    console.print(Panel(details, title=f"Scenario: {scenario.display_name}"))
    
    if scenario.metadata:
        metadata_json = json.dumps(scenario.metadata, indent=2)
        console.print(Panel(
            Syntax(metadata_json, "json", theme="monokai", line_numbers=True),
            title="Metadata"
        ))


def display_risk_details(risk: Risk) -> None:
    """
    Display detailed information about a specific risk.
    
    Args:
        risk: Risk instance
    """
    severity_color = {
        'low': 'green',
        'medium': 'yellow',
        'high': 'bright_yellow',
        'critical': 'red'
    }.get(risk.severity.value, 'white')
    
    details = f"""
[bold cyan]Name:[/bold cyan] {risk.display_name}
[bold cyan]ID:[/bold cyan] {risk.name}
[bold cyan]Category:[/bold cyan] {risk.category.value}
[bold {severity_color}]Severity:[/bold {severity_color}] {risk.severity.value.upper()}
[bold cyan]Detection Difficulty:[/bold cyan] {risk.detection_difficulty}/10
[bold cyan]Injection Probability:[/bold cyan] {risk.injection_probability:.2f}

[bold cyan]Description:[/bold cyan]
{risk.description}

[bold cyan]Injection Patterns:[/bold cyan]
{chr(10).join(f'• {pattern}' for pattern in risk.injection_patterns) or 'None'}

[bold cyan]Trigger Conditions:[/bold cyan]
{chr(10).join(f'• {condition}' for condition in risk.trigger_conditions) or 'None'}

[bold cyan]Compatible Scenarios:[/bold cyan]
{', '.join(risk.compatible_scenarios) if risk.compatible_scenarios else 'All (unless incompatible)'}

[bold cyan]Incompatible Scenarios:[/bold cyan]
{', '.join(risk.incompatible_scenarios) or 'None'}

[bold cyan]Mitigation Strategies:[/bold cyan]
{chr(10).join(f'• {strategy}' for strategy in risk.mitigation_strategies) or 'None'}

[bold cyan]Detection Signatures:[/bold cyan]
{chr(10).join(f'• {signature}' for signature in risk.detection_signatures) or 'None'}

[bold cyan]Tags:[/bold cyan]
{', '.join(risk.tags) or 'None'}
"""
    
    console.print(Panel(details, title=f"Risk: {risk.display_name}"))
    
    if risk.metadata:
        metadata_json = json.dumps(risk.metadata, indent=2)
        console.print(Panel(
            Syntax(metadata_json, "json", theme="monokai", line_numbers=True),
            title="Metadata"
        ))


def display_compatibility_matrix(config: GuardianConfig) -> None:
    """
    Display a compatibility matrix between scenarios and risks.
    
    Args:
        config: GuardianConfig instance
    """
    scenarios = config.configuration_layer.scenarios
    risks = config.configuration_layer.risks
    
    if not scenarios or not risks:
        console.print("[yellow]No scenarios or risks configured[/yellow]")
        return
    
    # Create compatibility matrix table
    table = Table(title="Risk-Scenario Compatibility Matrix")
    table.add_column("Scenario", style="cyan")
    
    # Add risk columns
    for risk in risks:
        severity_color = {
            'low': 'green',
            'medium': 'yellow',
            'high': 'bright_yellow',
            'critical': 'red'
        }.get(risk.severity.value, 'white')
        table.add_column(risk.name[:15], style=severity_color)
    
    # Add rows for each scenario
    for scenario in scenarios:
        row = [scenario.name[:20]]
        
        for risk in risks:
            compatible_risks = config.get_compatible_risks(scenario.name)
            constraint = config.configuration_layer.get_constraint(risk.name, scenario.name)
            
            if risk in compatible_risks:
                if constraint:
                    symbol = f"✓({constraint.max_injection_rate:.1f})"
                else:
                    symbol = "✓"
            else:
                symbol = "✗"
            
            row.append(symbol)
        
        table.add_row(*row)
    
    console.print(table)


def validate_config_file(config_file: str) -> List[str]:
    """
    Validate a configuration file and return issues.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        List of validation issues
    """
    issues = []
    
    try:
        config = GuardianConfig.from_file(config_file)
        validation_issues = config.validate()
        issues.extend(validation_issues)
        
        if not issues:
            logger.info(f"Configuration file {config_file} is valid")
        else:
            logger.warning(f"Configuration file {config_file} has {len(issues)} issues")
            for issue in issues:
                logger.warning(f"  - {issue}")
    
    except Exception as e:
        issues.append(f"Failed to load configuration: {e}")
        logger.error(f"Error validating {config_file}: {e}")
    
    return issues


def create_sample_config(output_file: str) -> None:
    """
    Create a sample configuration file.
    
    Args:
        output_file: Path to output file
    """
    from .models import ConfigurationLayer, RiskCategory
    from .core import RiskSeverity
    
    # Create sample scenario
    sample_scenario = Scenario(
        name="sample_scenario",
        display_name="Sample Scenario",
        description="A sample scenario for demonstration purposes",
        category="communication",
        complexity_level=3,
        generation_weight=1.0,
        sample_actions=["action1", "action2"],
        tags=["sample", "demo"]
    )
    
    # Create sample risk
    sample_risk = Risk(
        name="sample_risk",
        display_name="Sample Risk",
        description="A sample risk for demonstration purposes",
        category=RiskCategory.HALLUCINATION,
        severity=RiskSeverity.MEDIUM,
        detection_difficulty=5,
        injection_probability=0.3,
        injection_patterns=["pattern1", "pattern2"],
        tags=["sample", "demo"]
    )
    
    # Create sample constraint
    sample_constraint = RiskScenarioConstraint(
        risk_name="sample_risk",
        scenario_name="sample_scenario",
        compatibility=True,
        max_injection_rate=0.5,
        notes="Sample constraint for demonstration"
    )
    
    # Create configuration layer
    config_layer = ConfigurationLayer(
        scenarios=[sample_scenario],
        risks=[sample_risk],
        constraints=[sample_constraint],
        description="Sample configuration for Guardian engine"
    )
    
    # Create Guardian config
    config = GuardianConfig(configuration_layer=config_layer)
    
    # Save to file
    config.save_to_file(output_file)
    console.print(f"[green]Sample configuration created at {output_file}[/green]")


def export_config_stats(config: GuardianConfig, output_file: str) -> None:
    """
    Export configuration statistics to JSON file.
    
    Args:
        config: GuardianConfig instance
        output_file: Path to output JSON file
    """
    stats = {
        'global_config': {
            'engine_version': config.global_config.ENGINE_VERSION,
            'engine_name': config.global_config.ENGINE_NAME,
            'supported_languages': config.global_config.get_supported_languages(),
            'supported_formats': config.global_config.get_supported_formats()
        },
        'configuration_metadata': {
            'version': config.configuration_layer.version,
            'description': config.configuration_layer.description,
            'last_modified': config.configuration_layer.last_modified
        },
        'scenario_statistics': config.get_scenario_statistics(),
        'risk_statistics': config.get_risk_statistics(),
        'validation_issues': config.validate()
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    console.print(f"[green]Configuration statistics exported to {output_file}[/green]")


def merge_configurations(base_config: str, overlay_config: str, output_config: str) -> None:
    """
    Merge two configuration files.
    
    Args:
        base_config: Path to base configuration file
        overlay_config: Path to overlay configuration file
        output_config: Path to output merged configuration file
    """
    try:
        base = GuardianConfig.from_file(base_config)
        overlay = GuardianConfig.from_file(overlay_config)
        
        # Merge scenarios (overlay adds new ones, base keeps existing)
        merged_scenarios = {s.name: s for s in base.configuration_layer.scenarios}
        for scenario in overlay.configuration_layer.scenarios:
            merged_scenarios[scenario.name] = scenario
        
        # Merge risks (overlay adds new ones, base keeps existing)
        merged_risks = {r.name: r for r in base.configuration_layer.risks}
        for risk in overlay.configuration_layer.risks:
            merged_risks[risk.name] = risk
        
        # Merge constraints (overlay overrides base)
        merged_constraints = {
            (c.risk_name, c.scenario_name): c 
            for c in base.configuration_layer.constraints
        }
        for constraint in overlay.configuration_layer.constraints:
            key = (constraint.risk_name, constraint.scenario_name)
            merged_constraints[key] = constraint
        
        # Create merged configuration
        merged_config = GuardianConfig()
        merged_config.configuration_layer.scenarios = list(merged_scenarios.values())
        merged_config.configuration_layer.risks = list(merged_risks.values())
        merged_config.configuration_layer.constraints = list(merged_constraints.values())
        
        # Use overlay settings if available, otherwise base settings
        merged_config.language = overlay.language
        merged_config.output_format = overlay.output_format
        merged_config.batch_size = overlay.batch_size
        merged_config.max_retries = overlay.max_retries
        
        # Save merged configuration
        merged_config.save_to_file(output_config)
        
        console.print(f"[green]Configurations merged successfully to {output_config}[/green]")
        console.print(f"Final stats: {len(merged_scenarios)} scenarios, {len(merged_risks)} risks, {len(merged_constraints)} constraints")
        
    except Exception as e:
        console.print(f"[red]Error merging configurations: {e}[/red]")
        logger.error(f"Failed to merge configurations: {e}")


def find_unused_risks(config: GuardianConfig) -> List[Risk]:
    """
    Find risks that are not compatible with any scenarios.
    
    Args:
        config: GuardianConfig instance
        
    Returns:
        List of unused risks
    """
    unused_risks = []
    
    for risk in config.configuration_layer.risks:
        is_used = False
        for scenario in config.configuration_layer.scenarios:
            compatible_risks = config.get_compatible_risks(scenario.name)
            if risk in compatible_risks:
                is_used = True
                break
        
        if not is_used:
            unused_risks.append(risk)
    
    return unused_risks


def find_orphaned_constraints(config: GuardianConfig) -> List[RiskScenarioConstraint]:
    """
    Find constraints that reference non-existent scenarios or risks.
    
    Args:
        config: GuardianConfig instance
        
    Returns:
        List of orphaned constraints
    """
    scenario_names = {s.name for s in config.configuration_layer.scenarios}
    risk_names = {r.name for r in config.configuration_layer.risks}
    
    orphaned_constraints = []
    
    for constraint in config.configuration_layer.constraints:
        if (constraint.scenario_name not in scenario_names or 
            constraint.risk_name not in risk_names):
            orphaned_constraints.append(constraint)
    
    return orphaned_constraints


def validate_generation_config(config_file: str) -> List[str]:
    """
    Validate generation.yaml configuration and return list of issues.
    """
    issues: List[str] = []
    try:
        _ = load_generation_settings(config_file)
    except Exception as e:
        issues.append(str(e))
        logger.error(f"Error validating generation config {config_file}: {e}")
    return issues


def validate_risk_constraints(default_yaml_path: str, constraint_yaml_path: str) -> None:
    """
    Validate that all risk_name and scenario_name in risk_constraints.yaml exist in default.yaml.
    """
    with open(default_yaml_path, "r", encoding="utf-8") as f:
        default = yaml.safe_load(f)
    with open(constraint_yaml_path, "r", encoding="utf-8") as f:
        constraints = yaml.safe_load(f)
    scenario_names = {s["name"] for s in default.get("scenarios", [])}
    risk_names = {r["name"] for r in default.get("risks", [])}
    errors = []
    for c in constraints.get("constraints", []):
        if c["risk_name"] not in risk_names:
            errors.append(f"Risk '{c['risk_name']}' in risk_constraints.yaml not found in default.yaml.")
        if c["scenario_name"] not in scenario_names:
            errors.append(f"Scenario '{c['scenario_name']}' in risk_constraints.yaml not found in default.yaml.")
    if errors:
        for e in errors:
            logger.error(e)
        raise ValueError("Risk constraint config validation failed. See errors above.")
    logger.info("Risk constraint config validation passed.") 


def resolve_api_key_from_env(api_key_type: str) -> str:
    """
    Resolve API key from environment variables based on a key type.

    Supported types:
      - "openai_api_key" -> env "OPENAI_API_KEY"
      - "deepinfra_api_key" -> env "DEEPINFRA_API_KEY"

    Args:
        api_key_type: Logical API key type identifier from YAML

    Returns:
        The API key string from the corresponding environment variable

    Raises:
        ValueError: If the api_key_type is unknown or the env var is missing/empty
    """
    mapping: Dict[str, str] = {
        "openai_api_key": "OPENAI_API_KEY",
        "deepinfra_api_key": "DEEPINFRA_API_KEY",
    }

    if api_key_type not in mapping:
        raise ValueError(f"Unknown api_key_type: {api_key_type}. Expected one of: {', '.join(mapping.keys())}")

    env_var_name = mapping[api_key_type]
    api_key_value = os.getenv(env_var_name, "").strip()
    if not api_key_value:
        raise ValueError(
            f"Environment variable '{env_var_name}' not set for api_key_type '{api_key_type}'. "
            f"Please export {env_var_name} before running."
        )

    return api_key_value


def load_model_pool_config(model_pool_path: str = "config/model_pool.yaml") -> Dict[str, Any]:
    """
    Load model pool YAML and resolve API keys via environment variables.

    Expects entries in YAML like:
      externalAPI_models:
        - api_url: "https://..."
          api_key_type: "deepinfra_api_key"  # or "openai_api_key"
          model: "..."

    For backward compatibility, if an entry already has 'api_key', it is left as-is.

    Args:
        model_pool_path: Path to YAML file (default: config/model_pool.yaml)

    Returns:
        Dict containing the loaded and resolved model pool configuration
    """
    file_path = Path(model_pool_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Model pool file not found: {model_pool_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    # Currently we support 'externalAPI_models' list; extend if needed later
    for section_key in ["externalAPI_models"]:
        models_list: Optional[List[Dict[str, Any]]] = data.get(section_key)
        if not isinstance(models_list, list):
            continue

        for model_cfg in models_list:
            if not isinstance(model_cfg, dict):
                continue
            # If api_key already present, keep it; otherwise resolve via api_key_type
            if not model_cfg.get("api_key"):
                api_key_type = model_cfg.get("api_key_type")
                if api_key_type:
                    try:
                        resolved_key = resolve_api_key_from_env(api_key_type)
                        model_cfg["api_key"] = resolved_key
                        logger.debug(
                            f"Resolved api_key for model '{model_cfg.get('model', 'unknown')}' "
                            f"via type '{api_key_type}' -> env"
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to resolve API key for model '{model_cfg.get('model', 'unknown')}' "
                            f"(type={api_key_type}): {e}"
                        )
                else:
                    logger.warning(
                        f"Model entry '{model_cfg.get('model', 'unknown')}' in {model_pool_path} has no 'api_key' "
                        f"or 'api_key_type'. It may fail during runtime."
                    )

    return data