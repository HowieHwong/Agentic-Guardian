#!/usr/bin/env python3
"""
Interactive CLI to configure API keys used by Agentic-Guardian.

Features:
- Choose which API key to configure (OpenAI, DeepInfra)
- Enter/update/remove values (input hidden)
- View current (masked)
- Choose how to persist: project .env file or system environment (user scope)
- Exit without saving or save & exit

Note:
- The codebase reads environment variables at runtime:
  OPENAI_API_KEY, DEEPINFRA_API_KEY
- Saving to .env is recommended for project-scoped usage.
"""

from __future__ import annotations

import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple
from getpass import getpass

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.panel import Panel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

console = Console()


KEY_MAP: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "deepinfra": "DEEPINFRA_API_KEY",
}


def mask_value(value: Optional[str]) -> str:
    if not value:
        return "<not set>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"


def read_env_file(env_path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            values[key.strip()] = val.strip().strip('"')
    return values


def write_env_file(env_path: Path, updates: Dict[str, Optional[str]]) -> None:
    existing = read_env_file(env_path)
    for k, v in updates.items():
        if v is None:
            existing.pop(k, None)
        else:
            existing[k] = v
    lines = [f"{k}={v}" for k, v in existing.items()]
    env_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def set_system_env(var: str, value: Optional[str]) -> Tuple[bool, str]:
    system_name = platform.system().lower()
    try:
        if system_name == "windows":
            if value is None:
                # Windows doesn't provide a direct unset via setx; set empty string instead
                value = ""
            # Persist for current user
            subprocess.run(["setx", var, value], check=True, capture_output=True, text=True)
            return True, "Saved to user environment (Windows) via setx. Restart terminal to take effect."
        else:
            # POSIX: append export to the user's shell profile
            shell = os.environ.get("SHELL", "").lower()
            profile_candidates = [
                Path.home() / ".zshrc",
                Path.home() / ".bashrc",
                Path.home() / ".bash_profile",
                Path.home() / ".profile",
            ]
            profile = next((p for p in profile_candidates if p.exists()), profile_candidates[0])
            export_line = f"export {var}='{value or ''}'\n"
            with profile.open("a", encoding="utf-8") as f:
                f.write(f"\n# Added by Agentic-Guardian API key configurator\n{export_line}")
            return True, f"Saved to {profile.name}. Restart your shell or 'source ~/{profile.name}' to take effect."
    except subprocess.CalledProcessError as e:
        return False, e.stderr or str(e)
    except Exception as e:
        return False, str(e)


def show_current_values() -> None:
    table = Table(title="Current API Key Values (masked)")
    table.add_column("Key", style="cyan")
    table.add_column("Env Var", style="magenta")
    table.add_column("Value", style="green")

    env_file_vals = read_env_file(ENV_FILE)
    for key_label, env_var in KEY_MAP.items():
        val = os.getenv(env_var) or env_file_vals.get(env_var)
        table.add_row(key_label, env_var, mask_value(val))

    console.print(table)


def pick_key() -> Optional[str]:
    console.print(Panel.fit("Select which API key to configure:", title="API Key Type"))
    options = ["openai", "deepinfra", "back"]
    choice = Prompt.ask("Choose", choices=options, default="back")
    if choice == "back":
        return None
    return choice


def input_key_value(label: str) -> Optional[str]:
    console.print(Panel.fit(f"Enter value for [bold]{label}[/bold]. Leave empty to cancel.", title="Input"))
    val = getpass("API key: ")
    return val.strip() or None


def choose_persistence() -> Optional[str]:
    console.print(Panel.fit("Choose how to save the key:", title="Persistence"))
    options = {
        "1": "Project .env file (recommended)",
        "2": "System environment (current user)",
        "3": "Cancel",
    }
    for k, v in options.items():
        console.print(f"[{k}] {v}")
    sel = Prompt.ask("Select", choices=list(options.keys()), default="1")
    return sel


def main() -> None:
    console.print(Panel.fit("Agentic-Guardian API Key Configuration", title="Setup", subtitle="Use arrow keys not required; type options."))

    while True:
        console.print()
        show_current_values()
        console.print()
        console.print("Main Menu:")
        console.print("[1] Configure API key")
        console.print("[2] Remove API key")
        console.print("[3] Exit")
        action = Prompt.ask("Select", choices=["1", "2", "3"], default="3")

        if action == "3":
            console.print("Exiting.")
            return

        key_choice = pick_key()
        if not key_choice:
            continue
        env_var = KEY_MAP[key_choice]

        if action == "1":
            new_value = input_key_value(env_var)
            if not new_value:
                console.print("No value entered. Cancelled.")
                continue
            persistence = choose_persistence()
            if persistence == "1":
                write_env_file(ENV_FILE, {env_var: new_value})
                console.print(f"Saved to .env ({ENV_FILE}).")
            elif persistence == "2":
                ok, msg = set_system_env(env_var, new_value)
                console.print(msg if msg else ("Saved to system env" if ok else "Failed to save to system env"))
            else:
                console.print("Cancelled.")

        elif action == "2":
            # Remove from .env and optionally system env
            remove_from_env = Confirm.ask("Remove from project .env?", default=True)
            if remove_from_env:
                write_env_file(ENV_FILE, {env_var: None})
                console.print("Removed from .env.")
            remove_from_system = Confirm.ask("Also clear in system environment? (sets empty)", default=False)
            if remove_from_system:
                ok, msg = set_system_env(env_var, "")
                console.print(msg if msg else ("Cleared system env" if ok else "Failed to clear system env"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\nInterrupted. Goodbye.")
        sys.exit(1)


