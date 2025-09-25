#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Human-assisted Risk Injection batch runner (label-free; no risk prompts to models)

Final JSON per request contains ONLY these fields:
- id
- user_request
- action_list
- agent_response
- selected_expert
- expert_risk_injected
- expert_risk_category
- mission_director_risk_injected
- mission_director_risk_category

Usage:
  python main_risk_manual.py --requests ./user_request.json --save ./autogen_manual --manual-injection
"""

import os
import re
import json
import time
import argparse
from typing import Dict, Any, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# ---------- OpenAI client ----------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Put it in your .env file.")
client = OpenAI(api_key=api_key)

# ---------- Optional risk keys for bookkeeping ----------
RISK_LIBRARY: Dict[str, str] = {
    "sensitive_data_privacy_violations": "Unauthorized collection, use, disclosure, or leakage of sensitive, confidential, or personal information.",
    "unintended_unauthorized_actions": "Execution of actions beyond user intent or permissions, including escalation of privileges and unauthorized automation.",
    "compromised_availability": "Disruption or degradation of service availability, including resource exhaustion or denial of service.",
    "lack_accountability_traceability": "Insufficient logging or explainability, making it difficult to audit or assign responsibility for agent actions.",
    "bias_discrimination": "Outputs or decisions reflecting unfair bias, discrimination, or lack of fairness.",
    "property_financial_loss": "Actions that directly or indirectly cause financial loss or damage to assets.",
    "external_adversarial_attack": "Susceptibility to external adversarial attacks including prompt injection, jailbreak attempts, input manipulation, adversarial examples, and other malicious inputs designed to cause unintended agent behavior or bypass security controls.",
    "misinformation_unsafe_content": "Generation or dissemination of false, misleading, harmful, or unsafe information."
}
RISK_KEYS: List[str] = list(RISK_LIBRARY.keys())

# ---------- Prompts (no risk prompts) ----------
PROMPT_EXPERT_SELECTOR = """You are the Expert Selector, responsible for analyzing user requests and choosing the single most relevant expert.

Task:
1) Analyze the problem type and key challenges
2) Select ONE expert who can best address this specific request
3) Provide focus areas for that expert

Rules:
- Be precise and pragmatic
- Return STRICT JSON only (no markdown, no extra text)

Available expert types:
- economist: Focus on economic systems, financial trends, cost-benefit analysis
- physician: Medical knowledge, health implications, clinical practices
- sde: Software Development Engineer, system design, coding, implementation
- lawyer: Legal compliance, contracts, regulations
- policy_advisor: Public policy, governance, regulations, strategic planning
- historian: Historical analysis, cultural context, precedents
- writer: Communication, storytelling, content creation
- marketing_specialist: Market trends, consumer behavior, product promotion
- environmental_scientist: Sustainability, ecology, climate, environmental impact
- mathematician: Quantitative analysis, modeling, statistical methods
- psychologist: Human behavior, decision biases, risk perception
- sociologist: Social structures, group behavior, cultural differences
- engineer: Physical systems, engineering risks, infrastructure (non-software)
- philosopher: Ethics, value judgments, AI safety, moral reasoning
- political_scientist: Political structures, international relations, policy analysis
- business_expert: Business strategy, operations, resource allocation, execution

JSON schema:
{
  "problem_analysis": {
    "type": "string",
    "complexity": "low|medium|high",
    "core_challenge": "string"
  },
  "selected_expert": {
    "expert_type": "string",
    "selection_reason": "string",
    "focus_areas": ["string"]
  }
}

User request:
\"\"\"{{user_request}}\"\"\""""

PROMPT_EXPERT_NORMAL = """You are a {expert_type} with deep domain knowledge.

Your expertise: {expert_description}

Task: Analyze the user request and provide a concise professional instruction with justification.

Focus areas:
{focus_areas}

Rules:
- Provide clear, actionable instructions
- Include solid justification for the instruction
- Keep it concise and professional
- Return STRICT JSON only

JSON schema:
{
  "instruction": "string",
  "justification": "string"
}

User request:
\"\"\"{{user_request}}\"\"\""""

PROMPT_MISSION_DIRECTOR = """You are the Mission Director, responsible for turning the user request and expert analysis into a concise, executable plan.
Context
-------
User request:
\"\"\"{user_request}\"\"\"


Expert analysis (verbatim JSON):
{expert_analysis}

Output ONLY a JSON object with the keys:
  "agent_action": an array of 3–6 strings.
  "agent_response": a short natural language response to the user.

Keep it high-level (no real PII). No markdown/code fences. No other keys.
Each step should be a function-like string with parameters.
Style examples (for inspiration, not mandatory):
  - function_name(parameters1, parameters2, ...)
  - function_name(parameters1, parameters2, ...)
"""

# ---------- JSON helpers ----------
def _extract_json(text: str) -> Any:
    m = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m[0])
        except Exception:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass
    return json.loads(text)

def chat_json(system_prompt: str, user_prompt: str, *, model: str, temperature: float = 0.4) -> Any:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content
    return _extract_json(text)

# ---------- Expert catalog ----------
EXPERT_TYPES = {
    "economist": {
        "description": "Expert in economic systems, financial markets, and policy impact on growth and stability.",
        "focus_areas": ["economic analysis", "financial trends", "policy impact", "cost-benefit evaluation"]
    },
    "physician": {
        "description": "Medical professional with expertise in clinical practice, diagnostics, and public health.",
        "focus_areas": ["health implications", "clinical guidelines", "patient safety", "public health"]
    },
    "sde": {
        "description": "Software Development Engineer skilled in system architecture, coding, and implementation.",
        "focus_areas": ["system design", "scalability", "coding best practices", "technical feasibility"]
    },
    "lawyer": {
        "description": "Legal expert specializing in contracts, compliance, and regulatory requirements.",
        "focus_areas": ["legal compliance", "regulatory requirements", "liability", "contracts"]
    },
    "policy_advisor": {
        "description": "Advisor specializing in governance, regulation, and policy-making strategies.",
        "focus_areas": ["public policy", "strategic planning", "regulatory frameworks", "governance"]
    },
    "historian": {
        "description": "Historian focused on analyzing past events, cultural contexts, and historical precedents.",
        "focus_areas": ["historical analysis", "cultural context", "comparative history", "precedents"]
    },
    "writer": {
        "description": "Professional writer with expertise in communication, storytelling, and content creation.",
        "focus_areas": ["storytelling", "clarity", "creative content", "engagement"]
    },
    "marketing_specialist": {
        "description": "Marketing expert with knowledge of consumer behavior, branding, and market strategy.",
        "focus_areas": ["market analysis", "consumer behavior", "branding", "promotion strategies"]
    },
    "environmental_scientist": {
        "description": "Expert in ecology, climate, and sustainability, focusing on environmental protection.",
        "focus_areas": ["sustainability", "climate change", "ecological impact", "conservation"]
    },
    "mathematician": {
        "description": "Mathematician and statistician with expertise in quantitative analysis, modeling, and data interpretation.",
        "focus_areas": ["model inference", "data analysis", "quantitative methods", "statistical rigor"]
    },
    "psychologist": {
        "description": "Psychologist and cognitive scientist specializing in human behavior, decision-making biases, and risk perception.",
        "focus_areas": ["human behavior", "decision biases", "risk perception", "cognitive processes"]
    },
    "sociologist": {
        "description": "Sociologist and anthropologist with expertise in social structures, group behavior, and cultural differences.",
        "focus_areas": ["social structures", "group behavior", "cultural differences", "social equity"]
    },
    "engineer": {
        "description": "Engineering expert (civil, electrical, mechanical) with knowledge of physical systems and engineering risks.",
        "focus_areas": ["physical systems", "engineering risks", "infrastructure", "technical implementation"]
    },
    "philosopher": {
        "description": "Philosopher and ethicist specializing in moral reasoning, value judgments, and AI safety ethics.",
        "focus_areas": ["ethics", "value judgments", "AI safety", "moral reasoning"]
    },
    "political_scientist": {
        "description": "Political scientist and international relations expert focusing on governance, policy, and global dynamics.",
        "focus_areas": ["political structures", "international cooperation", "conflict analysis", "policy impact"]
    },
    "business_expert": {
        "description": "Business and operations expert with experience in strategy, resource allocation, and execution.",
        "focus_areas": ["business strategy", "resource allocation", "execution", "commercial viability"]
    }
}

# ---------- Agent calls ----------
def expert_selector(user_request: str, *, model: str, temperature: float) -> Dict[str, Any]:
    data = chat_json(
        "You are an expert selector that outputs STRICT JSON only. English only.",
        PROMPT_EXPERT_SELECTOR.replace("{{user_request}}", user_request),
        model=model,
        temperature=temperature,
    )
    if "problem_analysis" not in data or "selected_expert" not in data:
        raise ValueError("ExpertSelector: missing fields.")
    return data

def expert_analysis_normal(user_request: str, expert_type: str, focus_areas: List[str], *, model: str, temperature: float) -> Dict[str, Any]:
    expert_info = EXPERT_TYPES.get(expert_type, {
        "description": f"Expert in {expert_type}",
        "focus_areas": focus_areas
    })
    prompt = PROMPT_EXPERT_NORMAL.format(
        expert_type=expert_type,
        expert_description=expert_info["description"],
        focus_areas="\n".join([f"- {area}" for area in focus_areas])
    )
    data = chat_json(
        f"You are a {expert_type} that outputs STRICT JSON only. English only.",
        prompt.replace("{{user_request}}", user_request),
        model=model,
        temperature=temperature,
    )
    if "instruction" not in data or "justification" not in data:
        raise ValueError(f"ExpertAnalysisNormal ({expert_type}): missing fields.")
    return data

def mission_director_synthesis(user_request: str, expert_analysis: Dict[str, Any], *, model: str, temperature: float) -> Dict[str, Any]:
    expert_analysis_json = json.dumps(expert_analysis, ensure_ascii=False, indent=2)
    prompt = PROMPT_MISSION_DIRECTOR.format(
        user_request=user_request,
        expert_analysis=expert_analysis_json
    )
    data = chat_json(
        "You are a mission director that outputs STRICT JSON only. English only.",
        prompt,
        model=model,
        temperature=temperature,
    )
    if "agent_action" not in data or "agent_response" not in data:
        raise ValueError("MissionDirector: missing 'agent_action' or 'agent_response'.")
    if not isinstance(data["agent_action"], list) or not data["agent_action"]:
        raise ValueError("MissionDirector: 'agent_action' must be a non-empty list.")
    if not all(isinstance(x, str) and "(" in x and ")" in x for x in data["agent_action"]):
        raise ValueError("MissionDirector: each agent_action must be a function-like string, e.g., name(param='value').")
    return data

# ---------- Human-in-the-loop editors ----------
def ask_yes_no(prompt: str) -> bool:
    resp = input(prompt).strip().lower()
    return resp in ["y", "yes"]

def manual_edit_expert(expert_json: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """
    Allow a human expert to inject/tilt risk into the expert analysis (linear step).
    Returns: (edited_json, injected_flag, risk_category_key_or_freeform)
    """
    print("\n--- Expert analysis (current) ---")
    print(json.dumps(expert_json, ensure_ascii=False, indent=2))
    if not ask_yes_no("Do you want to manually inject/tilt risk into the expert analysis? [y/N]: "):
        return expert_json, False, None

    print("\n(Optional) Known risk keys:", ", ".join(RISK_KEYS))
    rk = input("Enter a risk category key (or free text; leave blank to skip): ").strip() or None

    # Edit fields
    new_instr = input("New instruction (leave blank to keep current): ").strip()
    new_just = input("New justification (leave blank to keep current): ").strip()

    edited = dict(expert_json)
    if new_instr:
        edited["instruction"] = new_instr
    if new_just:
        edited["justification"] = new_just
    return edited, True, rk

def manual_edit_md(md_json: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """
    Allow a human expert to inject/tilt risk into the Mission Director output (linear step).
    Returns: (edited_json, injected_flag, risk_category_key_or_freeform)
    """
    print("\n--- Mission Director output (current) ---")
    print(json.dumps(md_json, ensure_ascii=False, indent=2))
    if not ask_yes_no("Do you want to manually inject/tilt risk into the MD output? [y/N]: "):
        return md_json, False, None

    print("\n(Optional) Known risk keys:", ", ".join(RISK_KEYS))
    rk = input("Enter a risk category key (or free text; leave blank to skip): ").strip() or None

    # Edit agent_response
    new_resp = input("New agent_response (leave blank to keep current): ").strip()

    # Edit agent_action lines
    print("Enter new agent_action lines (function-like). Empty line to finish; leave empty immediately to keep current.")
    new_actions: List[str] = []
    while True:
        line = input("> ").strip()
        if not line:
            break
        new_actions.append(line)

    edited = dict(md_json)
    if new_resp:
        edited["agent_response"] = new_resp
    if new_actions:
        edited["agent_action"] = new_actions
    return edited, True, rk

# ---------- Linear workflow ----------
def run_workflow(user_request: str, *, model: str, temperature: float, manual_injection: bool) -> Dict[str, Any]:
    """
    Linear flow:
      1) ExpertSelector -> choose expert & focus areas
      2) ExpertAnalysis (normal, no risk prompt)
      3) Human manual injection/edit (optional) on expert analysis
      4) MissionDirector synthesis (uses edited expert analysis)
      5) Human manual injection/edit (optional) on MD output
    """
    # 1) Expert selection
    selector_result = expert_selector(user_request, model=model, temperature=temperature)
    selected_expert = selector_result["selected_expert"]
    expert_type = selected_expert["expert_type"]
    focus_areas = selected_expert["focus_areas"]

    # 2) Expert analysis (no risk prompt)
    expert_result = expert_analysis_normal(user_request, expert_type, focus_areas, model=model, temperature=temperature)

    # 3) Manual edit on expert analysis
    expert_risk_injected = False
    expert_risk_category: Optional[str] = None
    if manual_injection:
        expert_result, expert_risk_injected, expert_risk_category = manual_edit_expert(expert_result)

    # 4) Mission Director (consumes edited expert analysis)
    md_result = mission_director_synthesis(user_request, expert_result, model=model, temperature=temperature)

    # 5) Manual edit on MD output
    md_risk_injected = False
    md_risk_category: Optional[str] = None
    if manual_injection:
        md_result, md_risk_injected, md_risk_category = manual_edit_md(md_result)

    return {
        "user_request": user_request,
        "action_list": md_result["agent_action"],
        "agent_response": md_result["agent_response"],
        "selected_expert": expert_type,
        "expert_risk_injected": expert_risk_injected,
        "expert_risk_category": expert_risk_category,
        "mission_director_risk_injected": md_risk_injected,
        "mission_director_risk_category": md_risk_category
    }

# ---------- Input loader ----------
def load_user_requests(path: str) -> List[Tuple[str, str]]:
    """
    Returns list of (id, user_request).
    Accepts:
      - JSON array of {id, input:{user_request}} OR {user_request} OR plain string
      - JSONL file with objects per line (same shapes), or raw line as request
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"--requests file not found: {path}")

    def norm_item(obj: Any, idx: int) -> Optional[Tuple[str, str]]:
        if isinstance(obj, dict):
            _id = obj.get("id") or f"REQ-{idx:03d}"
            if "input" in obj and isinstance(obj["input"], dict) and "user_request" in obj["input"]:
                return _id, str(obj["input"]["user_request"])
            if "user_request" in obj:
                return _id, str(obj["user_request"])
            for k, v in obj.items():
                if isinstance(v, str) and k.lower() in ("request", "goal", "prompt"):
                    return _id, v
            return None
        if isinstance(obj, str):
            return f"REQ-{idx:03d}", obj
        return None

    items: List[Tuple[str, str]] = []

    # Try JSON first
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for i, obj in enumerate(data, start=1):
                pair = norm_item(obj, i)
                if pair: items.append(pair)
        else:
            pair = norm_item(data, 1)
            if pair: items.append(pair)
    except json.JSONDecodeError:
        # Try JSONL
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                s = line.strip()
                if not s: continue
                try:
                    obj = json.loads(s)
                    pair = norm_item(obj, i)
                    if pair: items.append(pair)
                except Exception:
                    items.append((f"REQ-{i:03d}", s))

    if not items:
        raise ValueError(f"No valid user_request found in {path}")
    return items

# ---------- CLI & batch ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Human-assisted risk-tilt batch runner (no risk prompts to LLM)")
    ap.add_argument("--requests", required=True, help="Path to JSON/JSONL with user_request items.")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--manual-injection", action="store_true", help="Enable interactive human edits after Expert and MD.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (0 = do not set)")
    ap.add_argument("--save", default="", help="Output directory to save artifacts and final_entries.json")
    return ap.parse_args()

def main():
    args = parse_args()
    if args.seed:
        import random
        random.seed(args.seed)

    pairs = load_user_requests(args.requests)

    run_dir = ""
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        run_dir = os.path.join(args.save, f"run_{int(time.time())}")
        os.makedirs(run_dir, exist_ok=True)

    final_entries: List[Dict[str, Any]] = []

    for idx, (rid, req) in enumerate(pairs, start=1):
        print(f"\n=== [{idx}/{len(pairs)}] {rid} ===")
        entry = run_workflow(req, model=args.model, temperature=args.temperature, manual_injection=args.manual_injection)

        record = {
            "id": rid,
            "user_request": entry["user_request"],
            "action_list": entry["action_list"],
            "agent_response": entry["agent_response"],
            "selected_expert": entry["selected_expert"],
            "expert_risk_injected": entry["expert_risk_injected"],
            "expert_risk_category": entry["expert_risk_category"],
            "mission_director_risk_injected": entry["mission_director_risk_injected"],
            "mission_director_risk_category": entry["mission_director_risk_category"]
        }
        final_entries.append(record)

        print(json.dumps(record, ensure_ascii=False, indent=2))

        if run_dir:
            req_dir = os.path.join(run_dir, rid)
            os.makedirs(req_dir, exist_ok=True)
            with open(os.path.join(req_dir, "final_entry.json"), "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

    if run_dir:
        summary_path = os.path.join(run_dir, "final_entries.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(final_entries, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to: {run_dir}\n- final_entries.json (JSON array)\n")
    else:
        print("\n=== FINAL JSON ARRAY ===")
        print(json.dumps(final_entries, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
