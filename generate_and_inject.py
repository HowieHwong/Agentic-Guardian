#!/usr/bin/env python3
"""
Generate harmless records from scenarios, then inject risks and save outputs.
"""

from pathlib import Path
import yaml
import json
from datetime import datetime

from AuraGen.config import GuardianConfig
from AuraGen.models import Scenario
from AuraGen.generation import OpenAIHarmlessDataGenerator, load_generation_settings
from AuraGen.injection import inject_risks_to_file


def get_all_scenarios(scenarios_dir="config/scenarios_generated"):
    """Return all scenario YAML file paths in the directory."""
    scenarios_dir = Path(scenarios_dir)
    return list(scenarios_dir.glob("*.yaml"))


def load_scenario(yaml_path: Path):
    """Load a Scenario from a YAML file."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return Scenario(**data)

# Removed checkpoint utilities to streamline the script.


def main():
    # Prepare output directory and timestamp
    save_dir = Path("generated_records")
    save_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load configs
    config = GuardianConfig.from_file("config/default.yaml")
    settings = load_generation_settings("config/generation.yaml")
    use_internal_inference = settings.externalAPI_generation

    # Initialize generator
    if use_internal_inference:
        if not settings.externalAPI or not settings.externalAPI.api_key:
            raise RuntimeError("Internal inference enabled, but externalAPI API configuration is missing.")
        generator = OpenAIHarmlessDataGenerator(
            config,
            settings.openai,
            metadata_config=config.get_metadata_config(),
            externalAPI_config=settings.externalAPI,
            use_internal_inference=True,
        )
    else:
        if settings.mode != "openai" or not settings.openai:
            raise RuntimeError("OpenAI configuration is not available.")
        generator = OpenAIHarmlessDataGenerator(
            config,
            settings.openai,
            metadata_config=config.get_metadata_config(),
        )

    # Generate records for all scenarios
    all_records = []
    for yaml_file in get_all_scenarios():
        scenario = load_scenario(yaml_file)
        records = generator.generate_batch_concurrent(scenario, n=20, max_workers=10)
        all_records.extend([r.dict() for r in records])

    # Save harmless records
    inference_type = "externalAPI" if use_internal_inference else "openai"
    harmless_file = save_dir / f"all_scenarios_{inference_type}_{timestamp}.json"
    with open(harmless_file, "w", encoding='utf-8') as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    # Inject risks using the just-saved harmless file
    injected_file = save_dir / f"all_injected_{inference_type}_{timestamp}.json"
    inject_risks_to_file(
        input_path=str(harmless_file),
        output_path=str(injected_file),
        config_path="config/risk_injection.yaml",
        constraint_yaml_path="config/risk_constraints.yaml",
        per_record_random_mode=True,
        inject_all_applicable_risks=False,
    )

    print(f"Saved harmless records to: {harmless_file}")
    print(f"Saved injected records to: {injected_file}")


if __name__ == "__main__":
    main()
