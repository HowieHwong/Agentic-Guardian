# 🛡️ Agentic-Guardian

<div align="center">

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://roaring-capybara-053cbe.netlify.app/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Safiron-yellow)](https://huggingface.co/Safiron/Safiron)

**Build foundational guardrails for general agentic systems via synthetic data**

</div>

---

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [🔧 AuraGen](#-auragen)
- [📊 Pre-Ex-Bench](#-pre-ex-bench)
- [🛡️ Safiron](#️-safiron)
- [📊 Evaluation](#-evaluation)
- [📄 License](#-license)

## 🌟 Overview

This repository provides an integrated ecosystem for developing and testing pre-execution safety guardrails:

| Component | Description | Purpose |
|-----------|-------------|---------|
| **🔧 AuraGen** | Configurable synthetic data generator with risk injection | Generate training data for guardrail research |
| **📊 Pre-Ex-Bench** | Reference dataset for quick experimentation | Evaluate pre-execution safety models |
| **🛡️ Safiron** | Guardian model for pre-execution safety | Detect and explain risks in agent planning |

## 🔧 AuraGen

> **Synthetic data engine with configurable risk injection**

AuraGen generates harmless trajectories from scenarios and then injects controlled risks. These synthetic records are used to train and evaluate safety models.

### 🚀 Quick Setup

#### 📋 Prerequisites
- 🐍 **Python 3.9+**
- 🔑 **API Key** (OpenAI, Anthropic, etc.) depending on generation mode

#### 📦 Installation
```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
````

#### ⚙️ Configure API Keys

```bash
python config/configure_api_keys.py
```

#### 🎯 Generate Data

```bash
python generate_and_inject.py
```

#### 📚 Documentation

📖 Available at: [https://roaring-capybara-053cbe.netlify.app/](https://roaring-capybara-053cbe.netlify.app/)

## 📊 Pre-Ex-Bench

> **Lightweight benchmark for pre-execution safety**

Pre-Ex-Bench provides a small set of examples for evaluating models on detection, classification, explanation, and generalization across different planners.

### 📁 Dataset

* **Location**: `Pre-Ex-Bench/dataset.json`
* **Format**: JSON list of entries

### 💻 Usage Example

```python
import json
from pathlib import Path

data = json.loads(Path('Pre-Ex-Bench/dataset.json').read_text(encoding='utf-8'))
print(f"Loaded {len(data)} items")
print(data[0])
```

## 🛡️ Safiron

> **Guardian model for pre-execution safety in agentic systems**

Safiron is trained on synthetic data from AuraGen and evaluated on Pre-Ex-Bench.
It analyzes planned agent actions *before execution*, detects whether they are risky, classifies the risk type, and generates concise explanations. This enables safer deployment of LLM-based agents by intercepting unsafe plans at the planning stage.

### 🎯 Core Capabilities

| Feature                     | Description                                                |
| --------------------------- | ---------------------------------------------------------- |
| 🔍 **Risk Detection**       | Distinguishes harmless vs. risky plans                     |
| 📊 **Risk Classification**  | Assigns risk to categories (privacy, financial loss, etc.) |
| 📝 **Concise Explanations** | Produces short, interpretable explanations                 |

### 🏆 Model Information

* 🤗 **Model Release**: [Safiron/Safiron](https://huggingface.co/Safiron/Safiron)
* 🎓 **Training**: Supervised fine-tuning + reinforcement learning (GRPO) on AuraGen data
* 📊 **Evaluation**: Benchmarked with Pre-Ex-Bench

### 🚀 Minimal Usage

```python
from vllm import LLM
from vllm.sampling_params import SamplingParams

llm = LLM(model="Safiron/Safiron")
sampling = SamplingParams(max_tokens=512, temperature=0.1, top_p=0.9)

instruction = "You are a guardian model for agentic systems. Assess potential risks..."
input_text = "{agent_action: '...', env_info: '...'}"

messages = [{
    "role": "user", 
    "content": f"{instruction}\n\nInput:\n{input_text}"
}]

outputs = llm.chat([messages], sampling_params=sampling)
print(outputs[0].outputs[0].text.strip())
```


## 🔧 Evaluation

### Overview
Evaluate **risk explanations** only when the model has correctly detected harmful samples.

### Installation
```bash
pip install openai scikit-learn tqdm
export OPENAI_API_KEY="your_api_key_here"
````


### Usage

```bash
python evaluation/eval.py --test-file path/to/results.json --model gpt-4o-mini --out-file evaluation/eval_results.json
```

#### Arguments

| Argument      | Description                                 |
| ------------- | ------------------------------------------- |
| `--test-file` | Path to model prediction file (JSON/JSONL). |
| `--model`     | LLM model for explanation judging.          |
| `--workers`   | Number of parallel threads for LLM calls.   |
| `--out-file`  | Output file with annotated evaluation.      |

#### Input Format

```json
{
  "expected_output": "Expected response text",
  "predicted_output": "Model predicted response"
}
```

#### Output

* Classification accuracy
* Harmful detection accuracy
* Risk category accuracy
* Explanation correctness
* Confusion matrix

Annotated evaluation results are saved to `--out-file`.




## 📄 License

Safiron and related resources are released under the
**Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)** License.

* 🎓 For research and educational purposes
* 🚫 Commercial use prohibited

---

<div align="center">

**🛡️ Building Safer Agentic Systems via Synthetic Data 🛡️**

</div>
