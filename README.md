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

The project reads API keys from environment variables (or a project `.env` file at the repo root):
- `OPENAI_API_KEY`
- `DEEPINFRA_API_KEY`

Use the interactive CLI to configure keys (recommended):
```bash
python config/configure_api_keys.py
```
- Select the key type (OpenAI/DeepInfra)
- Enter the key (hidden input)
- Choose persistence: write to project `.env` or system environment

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
  - `generation.internal_inference`: whether to route calls to the External API block
  - `output.record_file_template`: file naming template (supports `{scenario_name}`, `{timestamp}`, `{mode}`, `{ext}`)
  - `openai`: OpenAI settings. Use `api_key_type: "openai_api_key"` to resolve from env
  - `externalAPI`: External API settings (formerly RITS). Use `api_key_type: "deepinfra_api_key"`

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

Notes:
- When `api_key_type` is specified and `api_key` is absent, the loader resolves the key from env or `.env`.
- If the key is missing, an actionable error message will suggest running the configurator.

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