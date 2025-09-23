

## Agentic-Guardian

Build a foundational guardrail for general agentic systems via synthetic data. This repository houses:
- AuraGen: a configurable synthetic data generator with risk injection
- Pre-Ex-Bench: a small reference dataset for quick experimentation
- Safiron: a guardian model for pre-execution safety

### Table of Contents
- [AuraGen](#auragen)
- [Pre-Ex-Bench](#pre-ex-bench)
- [Safiron](#safiron)

## AuraGen

AuraGen provides configurable pipelines to generate harmless records from scenarios and then inject controlled risks for safety research and guardrail development.

### Quick Setup
Below is a minimal setup.

1) Prerequisites  
   - Python 3.9+  
   - An API key (e.g., OpenAI) depending on your selected generation mode

2) Install  
   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate  # Windows
   # source .venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt


3. Configure API Keys (recommended)

   ```bash
   python config/configure_api_keys.py
   ```

4. Generate and Inject

   ```bash
   python generate_and_inject.py
   ```

5. Docs
   Full documentation is hosted [here](https://roaring-capybara-053cbe.netlify.app/).

## Pre-Ex-Bench

Pre-Ex-Bench provides a lightweight set of example items for quick tests and demos.

* Location: `Pre-Ex-Bench/dataset.json`
* Format: JSON list of entries (fields vary by use case)

Minimal usage example:

```python
import json
from pathlib import Path

data = json.loads(Path('Pre-Ex-Bench/dataset.json').read_text(encoding='utf-8'))
print(f"Loaded {len(data)} items")
print(data[0])  # peek first item
```

You can pair these examples with AuraGen to prototype generation, filtering, or risk-injection logic.

## Safiron

Safiron is our **guardian model** for **pre-execution safety** in LLM-based agentic systems.
It takes planned agent actions (before execution), detects risks, classifies them, and produces concise explanations. Safiron is trained using AuraGen synthetic data and is evaluated via Pre-Ex-Bench.

* Model release: [Safiron/Safiron](https://huggingface.co/Safiron/Safiron)
* Purpose: prevent unsafe or unintended agent behaviors at the planning stage

### Minimal Usage

```python
from vllm import LLM
from vllm.sampling_params import SamplingParams

instruction = "You are a guardian model... (full instruction here)"
input_text = "{...}"  # shortened example input with agent_action and env_info

messages = [{"role": "user", "content": f\"{instruction}\\n\\nInput:\\n{input_text}\"}]

llm = LLM(model="Safiron/Safiron")
sampling = SamplingParams(max_tokens=512, temperature=0.1, top_p=0.9)
outputs = llm.chat([messages], sampling_params=sampling)

print(outputs[0].outputs[0].text.strip())
```

Safiron is **non-commercial** and licensed under [CC BY-NC 4.0](LICENSE).

