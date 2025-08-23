# Agentic-Guardian

## Usage of AuraGen

AuraGen is a data generation engine that produces diverse, high-quality risky trajectories for Agentic Systems. It operates in two stages:

1) Generate harmless trajectories: create clean, task-oriented agent action/response traces across many scenarios.
2) Inject risk: programmatically mutate the harmless trajectories to introduce realistic risks (privacy, misinformation, availability, unauthorized actions, bias, etc.) while keeping them coherent and plausible.

### 1) Environment Setup

- Create and activate with conda:
  ```bash
  conda create -n guardian python=3.11 -y
  conda activate guardian
  ```

- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2) API Configuration

The project supports flexible API key management through `config/api_key_types.yaml`. 

**Built-in API key types:**
- `OPENAI_API_KEY` (openai_api_key)
- `DEEPINFRA_API_KEY` (deepinfra_api_key)

**Interactive CLI for API key management (recommended):**
```bash
python config/configure_api_keys.py
```

**New Features:**
- **Add custom API key types**: Support any API service provider
- **Dynamic configuration management**: Add new API key types without code changes
- **Unified configuration interface**: Manage all API keys with one tool

**CLI Features:**
- [1] Configure API key - Set values for existing API keys
- [2] Remove API key - Delete API key values
- [3] **Add custom API key type** - Add new API key types
- [4] **Remove API key type** - Remove API key types
- [5] Exit

**Custom API key example:**
```yaml
# config/api_key_types.yaml
api_key_types:
  anthropic_api_key:
    env_var: ANTHROPIC_API_KEY
    description: Anthropic Claude API Key
  
  cohere_api_key:
    env_var: COHERE_API_KEY  
    description: Cohere API Key
```

Alternatively, set in your shell for a single session:
- Windows (PowerShell):
  ```powershell
  $env:OPENAI_API_KEY = "<your-openai-key>"
  $env:DEEPINFRA_API_KEY = "<your-deepinfra-key>"
  ```
- macOS/Linux (bash/zsh):
  ```bash
  export OPENAI_API_KEY="<your-openai-key>"
  export DEEPINFRA_API_KEY="<your-deepinfra-key>"
  ```

### 3) Configuration Files

- `config/generation.yaml`
  - **`generation.externalAPI_generation`**: **IMPORTANT SETTING!** Determines which API service to use
    - `true`: Use externalAPI (DeepInfra and other third-party API services)
    - `false`: Use OpenAI API
  - `output.record_file_template`: file naming template (supports `{scenario_name}`, `{timestamp}`, `{mode}`, `{ext}`)
  - `openai`: OpenAI API settings. Use `api_key_type: "openai_api_key"` to resolve from env
  - `externalAPI`: External API settings (e.g., DeepInfra). Use `api_key_type: "deepinfra_api_key"`

- `config/risk_injection.yaml`
  - `injection`: injector runtime settings
    - `mode`: currently only `openai` is supported; can be omitted to default
  - `openai` / `externalAPI`: same API key resolution rules as above
  - `risks`: risk templates and prompts used for injection

- `config/model_pool.yaml`
  - Example pool entries using `api_key_type`:
    ```yaml
    external_models:
      - api_url: "https://api.deepinfra.com/v1/openai/chat/completions"
        api_key_type: "deepinfra_api_key"
        model: "Qwen/Qwen2.5-72B-Instruct"
        temperature: 1.0
        max_tokens: 4096
    ```

**API Switching Examples:**

Switch to OpenAI API:
```yaml
# config/generation.yaml
generation:
  externalAPI_generation: false  # Use OpenAI
```

Switch to DeepInfra API:
```yaml
# config/generation.yaml  
generation:
  externalAPI_generation: true   # Use ExternalAPI (DeepInfra)
```

Notes:
- When `api_key_type` is specified and `api_key` is absent, the loader resolves the key from env or `.env`.
- If the key is missing, an actionable error message will suggest running the configurator.
- Ensure the corresponding API key is properly configured (OPENAI_API_KEY or DEEPINFRA_API_KEY)

### 4) Quick Start

Generate harmless records for all scenarios and then inject risks:
```bash
python generate_and_inject.py
```
Outputs are written under `generated_records/`:
- `all_scenarios_{openai|external}_TIMESTAMP.json`
- `all_injected_{openai|external}_TIMESTAMP.json`

Troubleshooting:
- If you see an error like “Environment variable 'DEEPINFRA_API_KEY' not set…”, run:
  ```bash
  python config/configure_api_keys.py
  ```
  or export the variable in your shell before running.