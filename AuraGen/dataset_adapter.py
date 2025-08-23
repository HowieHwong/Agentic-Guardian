"""
Dataset Adapter Module

This module provides functionality to adapt external datasets to Guardian's format.
It includes field mapping, format conversion, and LLM-based parsing capabilities.
"""

import json
import yaml
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from pydantic import BaseModel, Field
from loguru import logger
import pandas as pd
from tqdm import tqdm

try:
    import openai
except ImportError:
    openai = None


class FieldMapping(BaseModel):
    """Represents a field mapping from external dataset to Guardian format."""
    external_field: str = Field(..., description="Field name in external dataset")
    guardian_field: str = Field(..., description="Target field name in Guardian format")
    data_type: str = Field(..., description="Expected data type (string, list, dict)")
    requires_parsing: bool = Field(False, description="Whether this field requires LLM parsing")
    default_value: Any = Field(None, description="Default value if field is missing")


class DatasetSchema(BaseModel):
    """Schema information for an external dataset."""
    name: str = Field(..., description="Dataset name")
    fields: List[str] = Field(..., description="Available fields in the dataset")
    sample_record: Dict[str, Any] = Field(..., description="Sample record for analysis")
    total_records: int = Field(..., description="Total number of records")
    
    
class ConversionConfig(BaseModel):
    """Configuration for dataset conversion."""
    field_mappings: List[FieldMapping] = Field(..., description="Field mappings")
    openai_config: Optional[Dict[str, Any]] = Field(None, description="OpenAI configuration for parsing")
    default_scenario_name: str = Field("external_dataset", description="Default scenario name")
    strict_mode: bool = Field(False, description="Whether to fail on missing required fields")


class DatasetAdapter:
    """Main class for adapting external datasets to Guardian format."""
    
    def __init__(self, openai_config: Optional[Dict[str, Any]] = None):
        self.openai_config = openai_config
        if openai_config and openai:
            openai.api_key = openai_config.get("api_key")
    
    def analyze_dataset(self, file_path: str) -> DatasetSchema:
        """
        Analyze an external dataset file and extract schema information.
        
        Args:
            file_path: Path to the dataset file (JSON, JSONL, or CSV)
            
        Returns:
            DatasetSchema with field information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Load data based on file extension with progress bar
        print("📂 Loading dataset...")
        if file_path.suffix.lower() == '.json':
            data = self._load_json(file_path)
        elif file_path.suffix.lower() == '.jsonl':
            data = self._load_jsonl(file_path)
        elif file_path.suffix.lower() == '.csv':
            data = self._load_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        if not data:
            raise ValueError("Dataset is empty")
        
        print(f"✅ Loaded {len(data)} records successfully")
        
        # Analyze schema with progress bar
        print("🔍 Analyzing schema...")
        sample_record = data[0]
        all_fields = set()
        
        # Collect all possible fields from multiple records
        sample_size = min(10, len(data))
        print(f"📊 Analyzing first {sample_size} records to identify all fields...")
        
        for record in tqdm(data[:sample_size], desc="Analyzing record fields", unit="record"):
            if isinstance(record, dict):
                all_fields.update(record.keys())
        
        print(f"🎯 Found {len(all_fields)} unique fields in dataset")
        
        return DatasetSchema(
            name=file_path.stem,
            fields=sorted(list(all_fields)),
            sample_record=sample_record,
            total_records=len(data)
        )
    
    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError("JSON file must contain a list of records or a single record")
    
    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def _load_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load CSV file."""
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    
    def suggest_field_mappings(self, schema: DatasetSchema) -> List[FieldMapping]:
        """
        Use LLM to suggest field mappings from external dataset to Guardian format.
        
        Args:
            schema: Dataset schema information
            
        Returns:
            List of suggested field mappings
        """
        if not self.openai_config or not openai:
            logger.warning("OpenAI not configured, returning basic mappings")
            return self._get_basic_mappings(schema.fields)
        
        print(f"🤖 Generating field mapping suggestions for {len(schema.fields)} fields using LLM...")
        
        # Guardian's expected fields
        guardian_fields = {
            "scenario_name": "Name of the scenario/task type",
            "user_request": "User's input or request",
            "agent_action": "Agent's action plan (should be a list of steps)",
            "agent_response": "Agent's response to the user",
            "context": "Background context or environment information",
            "metadata": "Additional metadata (optional)"
        }
        
        prompt = f"""
You are helping to map fields from an external dataset to Guardian's format.

Guardian's expected fields:
{json.dumps(guardian_fields, indent=2)}

External dataset fields:
{json.dumps(schema.fields, indent=2)}

Sample record from external dataset:
{json.dumps(schema.sample_record, indent=2)}

Please suggest mappings from external fields to Guardian fields. For each mapping, provide:
1. external_field: The field name from the external dataset
2. guardian_field: The target Guardian field name
3. data_type: Expected data type (string, list, dict)
4. requires_parsing: Whether the field needs LLM parsing (especially for agent_action if it's not a list)

Return your response as a JSON array of mappings. Only include mappings for fields that have clear correspondences.

Example format:
[
  {{
    "external_field": "query",
    "guardian_field": "user_request",
    "data_type": "string",
    "requires_parsing": false
  }},
  {{
    "external_field": "actions",
    "guardian_field": "agent_action", 
    "data_type": "list",
    "requires_parsing": true
  }}
]
"""
        
        try:
            # Show progress bar with field-by-field analysis details
            print("💡 Analyzing dataset fields to determine optimal mappings...")
            
            # Analyze each field and show progress
            analyzed_fields = []
            with tqdm(total=len(schema.fields), desc="Analyzing field structure", unit="field") as pbar:
                for i, field in enumerate(schema.fields):
                    # Update progress bar description to show current field
                    pbar.set_description(f"Analyzing field: {field}")
                    
                    # Get sample value for this field from sample record
                    sample_value = schema.sample_record.get(field, "N/A")
                    field_type = type(sample_value).__name__
                    
                    analyzed_fields.append({
                        "field": field,
                        "sample_value": str(sample_value)[:50] + ("..." if len(str(sample_value)) > 50 else ""),
                        "detected_type": field_type
                    })
                    
                    pbar.update(1)
                
                # After analyzing all fields, make the actual LLM call
                pbar.set_description("Generating mapping suggestions with LLM")
                response = openai.chat.completions.create(
                    model=self.openai_config.get("model", "gpt-4"),
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.2,
                    max_tokens=1500,
                )
            
            content = response.choices[0].message.content.strip()
            mappings_data = json.loads(content)
            
            print(f"✅ Generated {len(mappings_data)} field mapping suggestions")
            return [FieldMapping(**mapping) for mapping in mappings_data]
            
        except Exception as e:
            logger.error(f"Failed to generate LLM suggestions: {e}")
            print("⚠️  LLM analysis failed, falling back to basic pattern matching...")
            return self._get_basic_mappings(schema.fields)
    
    def _get_basic_mappings(self, fields: List[str]) -> List[FieldMapping]:
        """Get basic field mappings based on common field names."""
        mappings = []
        
        # Common field name patterns
        patterns = {
            "user_request": ["query", "input", "question", "request", "user_input", "prompt"],
            "agent_action": ["action", "actions", "steps", "plan", "agent_action", "action_sequence"],
            "agent_response": ["response", "output", "answer", "agent_response", "result"],
            "scenario_name": ["scenario", "task", "type", "category", "scenario_name"],
            "context": ["context", "background", "environment", "setting", "situation", "env"]
        }
        
        for guardian_field, possible_names in patterns.items():
            for field in fields:
                if field.lower() in [name.lower() for name in possible_names]:
                    mappings.append(FieldMapping(
                        external_field=field,
                        guardian_field=guardian_field,
                        data_type="list" if guardian_field == "agent_action" else "string",
                        requires_parsing=guardian_field == "agent_action"
                    ))
                    break
        
        return mappings
    
    def convert_dataset(self, file_path: str, config: ConversionConfig) -> List[Dict[str, Any]]:
        """
        Convert external dataset to Guardian format.
        
        Args:
            file_path: Path to the external dataset
            config: Conversion configuration
            
        Returns:
            List of records in Guardian format
        """
        # Load original data
        file_path = Path(file_path)
        if file_path.suffix.lower() == '.json':
            original_data = self._load_json(file_path)
        elif file_path.suffix.lower() == '.jsonl':
            original_data = self._load_jsonl(file_path)
        elif file_path.suffix.lower() == '.csv':
            original_data = self._load_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        print(f"📊 Starting conversion of {len(original_data)} records...")
        converted_records = []
        failed_records = 0
        
        for i, record in enumerate(tqdm(original_data, desc="Converting records to Guardian format", unit="record")):
            try:
                converted_record = self._convert_single_record(record, config)
                converted_records.append(converted_record)
            except Exception as e:
                logger.error(f"Failed to convert record {i}: {e}")
                failed_records += 1
                if config.strict_mode:
                    raise
                continue
        
        print(f"✅ Conversion completed: {len(converted_records)} successful, {failed_records} failed")
        logger.info(f"Successfully converted {len(converted_records)}/{len(original_data)} records")
        return converted_records
    
    def _convert_single_record(self, record: Dict[str, Any], config: ConversionConfig) -> Dict[str, Any]:
        """Convert a single record to Guardian format."""
        converted = {}
        
        # Apply field mappings
        for mapping in config.field_mappings:
            try:
                if mapping.external_field in record:
                    value = record[mapping.external_field]
                    
                    # Parse value if needed
                    if mapping.requires_parsing and value:
                        value = self._parse_field_value(value, mapping)
                    
                    converted[mapping.guardian_field] = value
                else:
                    # Use default value if field is missing
                    if mapping.default_value is not None:
                        converted[mapping.guardian_field] = mapping.default_value
                    elif not config.strict_mode:
                        # Provide sensible defaults for required fields
                        converted[mapping.guardian_field] = self._get_default_value(mapping.guardian_field)
            except Exception as e:
                logger.warning(f"Failed to map field {mapping.external_field}: {e}")
                if config.strict_mode:
                    raise
        
        # Ensure required fields are present
        self._ensure_required_fields(converted, config)
        
        return converted
    
    def _parse_field_value(self, value: Any, mapping: FieldMapping) -> Any:
        """Parse a field value using LLM if needed."""
        if not self.openai_config or not openai:
            logger.warning(f"Cannot parse field {mapping.external_field}: OpenAI not configured")
            return value
        
        if mapping.guardian_field == "agent_action" and not isinstance(value, list):
            return self._parse_agent_action(value)
        
        return value
    
    def _parse_agent_action(self, value: Any) -> List[str]:
        """Parse agent action into a list of steps."""
        if isinstance(value, list):
            return [str(item) for item in value]
        
        if not isinstance(value, str):
            value = str(value)
        
        prompt = f"""
Convert the following agent action into a list of clear, actionable steps.
Each step should be a separate string in the list.

Original action:
{value}

Guidelines:
1. Break down the action into discrete steps
2. Each step should be clear and actionable
3. Maintain the original intent and order
4. Remove any numbering or bullet points
5. Return as a JSON array of strings

Example format:
["search for information", "analyze the results", "prepare response"]
"""
        
        try:
            # Show progress for LLM parsing (only show when verbose or debug mode)
            with tqdm(total=1, desc="Parsing action field with LLM", unit="field", 
                     leave=False, disable=False, position=1, 
                     bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}]") as pbar:
                response = openai.chat.completions.create(
                    model=self.openai_config.get("model", "gpt-4"),
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.2,
                    max_tokens=800,
                )
                pbar.update(1)
            
            content = response.choices[0].message.content.strip()
            parsed_actions = json.loads(content)
            
            if isinstance(parsed_actions, list):
                return [str(action) for action in parsed_actions]
            else:
                logger.warning("LLM did not return a list, falling back to simple split")
                return self._simple_action_split(value)
                
        except Exception as e:
            logger.error(f"Failed to parse agent action with LLM: {e}")
            return self._simple_action_split(value)
    
    def _simple_action_split(self, value: str) -> List[str]:
        """Simple fallback for splitting agent actions."""
        # Try common delimiters
        for delimiter in ['\n', '.', ';', '|']:
            if delimiter in value:
                steps = [step.strip() for step in value.split(delimiter)]
                steps = [step for step in steps if step]  # Remove empty steps
                if len(steps) > 1:
                    return steps
        
        # If no clear delimiter, return as single step
        return [value.strip()]
    
    def _get_default_value(self, field_name: str) -> Any:
        """Get default value for a Guardian field."""
        defaults = {
            "scenario_name": "external_dataset",
            "user_request": "",
            "agent_action": [],
            "agent_response": "",
            "context": "",
            "metadata": {}
        }
        return defaults.get(field_name, None)
    
    def _ensure_required_fields(self, record: Dict[str, Any], config: ConversionConfig):
        """Ensure all required Guardian fields are present."""
        required_fields = ["scenario_name", "user_request", "agent_action", "agent_response"]
        
        for field in required_fields:
            if field not in record:
                record[field] = self._get_default_value(field)
        
        # Ensure metadata exists
        if "metadata" not in record:
            record["metadata"] = {}
        
        # Add conversion metadata
        record["metadata"]["converted_from_external"] = True
        record["metadata"]["conversion_timestamp"] = int(time.time())


def create_web_interface_config(schema: DatasetSchema, suggested_mappings: List[FieldMapping]) -> Dict[str, Any]:
    """
    Create configuration for web interface to let users select field mappings.
    
    Args:
        schema: Dataset schema
        suggested_mappings: LLM-suggested mappings
        
    Returns:
        Configuration dictionary for web interface
    """
    guardian_fields = {
        "scenario_name": {
            "description": "Name of the scenario/task type",
            "required": True,
            "type": "string"
        },
        "user_request": {
            "description": "User's input or request",
            "required": True,
            "type": "string"
        },
        "agent_action": {
            "description": "Agent's action plan (list of steps)",
            "required": True,
            "type": "list"
        },
        "agent_response": {
            "description": "Agent's response to the user",
            "required": True,
            "type": "string"
        },
        "context": {
            "description": "Background context or environment information",
            "required": False,
            "type": "string"
        },
        "metadata": {
            "description": "Additional metadata (optional)",
            "required": False,
            "type": "dict"
        }
    }
    
    return {
        "dataset_info": {
            "name": schema.name,
            "total_records": schema.total_records,
            "available_fields": schema.fields,
            "sample_record": schema.sample_record
        },
        "guardian_fields": guardian_fields,
        "suggested_mappings": [mapping.model_dump() for mapping in suggested_mappings],
        "instructions": {
            "title": "Dataset Field Mapping",
            "description": "Map fields from your external dataset to Guardian's format",
            "steps": [
                "Review the suggested mappings",
                "Modify mappings as needed",
                "Specify which fields require LLM parsing",
                "Set default values for missing fields"
            ]
        }
    } 