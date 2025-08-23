"""
Web Interface Generator for Dataset Adaptation

This module generates HTML interfaces for users to configure field mappings
when adapting external datasets to Guardian format.
"""

import json
from typing import Dict, Any, List
from pathlib import Path


def generate_mapping_interface(config: Dict[str, Any], output_path: str = "field_mapping.html") -> str:
    """
    Generate an HTML interface for field mapping configuration.
    
    Args:
        config: Configuration dictionary from create_web_interface_config
        output_path: Path where to save the HTML file
        
    Returns:
        Path to the generated HTML file
    """
    
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        
        .content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }}
        
        .panel {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        h1, h2, h3 {{
            color: #2c3e50;
            margin-bottom: 20px;
        }}
        
        .dataset-info {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 6px;
            margin-bottom: 20px;
        }}
        
        .field-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }}
        
        .field-tag {{
            background: #3498db;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
        }}
        
        .mapping-form {{
            margin-top: 20px;
        }}
        
        .mapping-row {{
            display: grid;
            grid-template-columns: 2fr 2fr 1fr 1fr;
            gap: 15px;
            align-items: center;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 6px;
            margin-bottom: 15px;
            background: #fafafa;
        }}
        
        .mapping-row.header {{
            background: #34495e;
            color: white;
            font-weight: bold;
        }}
        
        select, input[type="text"] {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
        
        input[type="checkbox"] {{
            transform: scale(1.2);
        }}
        
        .required {{
            color: #e74c3c;
        }}
        
        .sample-preview {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        }}
        
        .actions {{
            margin-top: 30px;
            text-align: center;
        }}
        
        .btn {{
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            margin: 0 10px;
        }}
        
        .btn:hover {{
            background: #2980b9;
        }}
        
        .btn.secondary {{
            background: #95a5a6;
        }}
        
        .btn.secondary:hover {{
            background: #7f8c8d;
        }}
        
        .instructions {{
            background: #e8f4fd;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin-bottom: 20px;
        }}
        
        .instructions ol {{
            margin-left: 20px;
        }}
        
        .instructions li {{
            margin-bottom: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        
        <div class="content">
            <div class="panel">
                <h2>Dataset Information</h2>
                <div class="dataset-info">
                    <h3>Dataset: {dataset_name}</h3>
                    <p><strong>Total Records:</strong> {total_records}</p>
                    <p><strong>Available Fields:</strong></p>
                    <div class="field-list">
                        {field_tags}
                    </div>
                </div>
                
                <div class="instructions">
                    <h3>Instructions</h3>
                    <ol>
                        {instruction_steps}
                    </ol>
                </div>
                
                <h3>Sample Record</h3>
                <div class="sample-preview">{sample_record}</div>
            </div>
            
            <div class="panel">
                <h2>Field Mappings</h2>
                <p>Map your dataset fields to Guardian's required format:</p>
                
                <form id="mappingForm" class="mapping-form">
                    <div class="mapping-row header">
                        <div>External Field</div>
                        <div>Guardian Field</div>
                        <div>Requires Parsing</div>
                        <div>Required</div>
                    </div>
                    
                    {mapping_rows}
                </form>
                
                <div class="actions">
                    <button type="button" class="btn secondary" onclick="addMappingRow()">Add Mapping</button>
                    <button type="button" class="btn" onclick="generateConfig()">Generate Configuration</button>
                    <button type="button" class="btn secondary" onclick="downloadConfig()">Download Config</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const datasetFields = {dataset_fields_json};
        const guardianFields = {guardian_fields_json};
        
        function addMappingRow() {{
            const form = document.getElementById('mappingForm');
            const newRow = createMappingRow('', '', false, false);
            form.appendChild(newRow);
        }}
        
        function createMappingRow(externalField, guardianField, requiresParsing, isRequired) {{
            const row = document.createElement('div');
            row.className = 'mapping-row';
            
            // External field select
            const externalSelect = document.createElement('select');
            externalSelect.name = 'external_field';
            externalSelect.innerHTML = '<option value="">Select field...</option>';
            datasetFields.forEach(field => {{
                const option = document.createElement('option');
                option.value = field;
                option.textContent = field;
                option.selected = field === externalField;
                externalSelect.appendChild(option);
            }});
            
            // Guardian field select
            const guardianSelect = document.createElement('select');
            guardianSelect.name = 'guardian_field';
            guardianSelect.innerHTML = '<option value="">Select target...</option>';
            Object.keys(guardianFields).forEach(field => {{
                const option = document.createElement('option');
                option.value = field;
                option.textContent = field;
                option.selected = field === guardianField;
                guardianSelect.appendChild(option);
            }});
            
            // Requires parsing checkbox
            const parsingCheckbox = document.createElement('input');
            parsingCheckbox.type = 'checkbox';
            parsingCheckbox.name = 'requires_parsing';
            parsingCheckbox.checked = requiresParsing;
            
            // Required indicator
            const requiredDiv = document.createElement('div');
            const isRequiredField = guardianFields[guardianField]?.required || false;
            requiredDiv.textContent = isRequiredField ? 'Yes' : 'No';
            requiredDiv.className = isRequiredField ? 'required' : '';
            
            // Update required indicator when guardian field changes
            guardianSelect.addEventListener('change', () => {{
                const selected = guardianSelect.value;
                const required = guardianFields[selected]?.required || false;
                requiredDiv.textContent = required ? 'Yes' : 'No';
                requiredDiv.className = required ? 'required' : '';
            }});
            
            row.appendChild(externalSelect);
            row.appendChild(guardianSelect);
            row.appendChild(parsingCheckbox);
            row.appendChild(requiredDiv);
            
            return row;
        }}
        
        function generateConfig() {{
            const form = document.getElementById('mappingForm');
            const rows = form.querySelectorAll('.mapping-row:not(.header)');
            const mappings = [];
            
            rows.forEach(row => {{
                const externalField = row.querySelector('select[name="external_field"]').value;
                const guardianField = row.querySelector('select[name="guardian_field"]').value;
                const requiresParsing = row.querySelector('input[name="requires_parsing"]').checked;
                
                if (externalField && guardianField) {{
                    const guardianInfo = guardianFields[guardianField];
                    mappings.push({{
                        external_field: externalField,
                        guardian_field: guardianField,
                        data_type: guardianInfo.type,
                        requires_parsing: requiresParsing
                    }});
                }}
            }});
            
            const config = {{
                field_mappings: mappings,
                default_scenario_name: "external_dataset",
                strict_mode: false
            }};
            
            const configJson = JSON.stringify(config, null, 2);
            
            // Display config in a modal or new window
            const newWindow = window.open('', '_blank');
            newWindow.document.write(`
                <html>
                    <head><title>Generated Configuration</title></head>
                    <body>
                        <h1>Generated Configuration</h1>
                        <p>Copy this configuration for use with the dataset adapter:</p>
                        <pre style="background: #f5f5f5; padding: 20px; border-radius: 4px;">${{configJson}}</pre>
                        <script>
                            window.generatedConfig = ${{configJson}};
                        </script>
                    </body>
                </html>
            `);
        }}
        
        function downloadConfig() {{
            generateConfig();
            // This would be implemented to actually download the config file
            alert('Configuration generated! Check the new window for the JSON config.');
        }}
        
        // Initialize with suggested mappings
        document.addEventListener('DOMContentLoaded', () => {{
            const suggestedMappings = {suggested_mappings_json};
            const form = document.getElementById('mappingForm');
            
            suggestedMappings.forEach(mapping => {{
                const row = createMappingRow(
                    mapping.external_field,
                    mapping.guardian_field,
                    mapping.requires_parsing,
                    guardianFields[mapping.guardian_field]?.required || false
                );
                form.appendChild(row);
            }});
        }});
    </script>
</body>
</html>
    """
    
    # Prepare template variables
    dataset_info = config["dataset_info"]
    guardian_fields = config["guardian_fields"]
    suggested_mappings = config["suggested_mappings"]
    instructions = config["instructions"]
    
    # Generate field tags
    field_tags = "".join([f'<span class="field-tag">{field}</span>' for field in dataset_info["available_fields"]])
    
    # Generate instruction steps
    instruction_steps = "".join([f'<li>{step}</li>' for step in instructions["steps"]])
    
    # Format sample record
    sample_record = json.dumps(dataset_info["sample_record"], indent=2)
    
    # Create mapping rows for suggested mappings
    mapping_rows = ""
    
    # Fill template
    html_content = html_template.format(
        title=instructions["title"],
        description=instructions["description"],
        dataset_name=dataset_info["name"],
        total_records=dataset_info["total_records"],
        field_tags=field_tags,
        instruction_steps=instruction_steps,
        sample_record=sample_record,
        mapping_rows=mapping_rows,
        dataset_fields_json=json.dumps(dataset_info["available_fields"]),
        guardian_fields_json=json.dumps(guardian_fields),
        suggested_mappings_json=json.dumps(suggested_mappings)
    )
    
    # Write to file
    output_path = Path(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return str(output_path) 