#!/usr/bin/env python3
"""
Interactive CLI to configure API keys used by Agentic-Guardian.

Features:
- Choose which API key to configure (supports all types defined in config/api_key_types.yaml)
- Add/remove custom API key types
- Enter/update/remove values (input hidden)
- View current (masked)
- Choose how to persist: project .env file or system environment (user scope)
- Exit without saving or save & exit

Note:
- The codebase reads environment variables at runtime based on api_key_types.yaml
- Saving to .env is recommended for project-scoped usage.
"""

from __future__ import annotations

import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from getpass import getpass

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.panel import Panel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

console = Console()

# Import API key manager
sys.path.insert(0, str(PROJECT_ROOT))
from AuraGen.api_key_manager import get_api_key_manager


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
    table.add_column("Key Type", style="cyan")
    table.add_column("Env Var", style="magenta")
    table.add_column("Description", style="yellow")
    table.add_column("Value", style="green")

    env_file_vals = read_env_file(ENV_FILE)
    manager = get_api_key_manager()
    
    for api_key_type in manager.get_supported_types():
        env_var = manager.get_env_var(api_key_type)
        description = manager.get_description(api_key_type)
        val = os.getenv(env_var) or env_file_vals.get(env_var)
        table.add_row(api_key_type, env_var, description, mask_value(val))

    console.print(table)


def pick_key() -> Optional[str]:
    console.print(Panel.fit("Select which API key to configure:", title="API Key Type"))
    manager = get_api_key_manager()
    supported_types = manager.get_supported_types()
    
    if not supported_types:
        console.print("[yellow]No API key types configured. Please add some first.[/yellow]")
        return None
    
    options = supported_types + ["back"]
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


def add_custom_api_key_type() -> None:
    """Add a new custom API key type"""
    console.print(Panel.fit("Add a new API key type", title="Custom API Key Type"))
    
    # Get API key type name
    api_key_type = Prompt.ask("Enter API key type name (e.g., 'anthropic_api_key')")
    if not api_key_type:
        console.print("No name entered. Cancelled.")
        return
    
    # Check if it already exists
    manager = get_api_key_manager()
    if api_key_type in manager.get_supported_types():
        console.print(f"[red]API key type '{api_key_type}' already exists![/red]")
        return
    
    # Get environment variable name
    env_var = Prompt.ask("Enter environment variable name (e.g., 'ANTHROPIC_API_KEY')")
    if not env_var:
        console.print("No environment variable name entered. Cancelled.")
        return
    
    # Get description
    description = Prompt.ask("Enter description (e.g., 'Anthropic Claude API Key')")
    if not description:
        description = f"API key for {api_key_type}"
    
    try:
        manager.add_api_key_type(api_key_type, env_var, description)
        console.print(f"[green]Successfully added API key type '{api_key_type}'![/green]")
    except Exception as e:
        console.print(f"[red]Failed to add API key type: {e}[/red]")


def remove_custom_api_key_type() -> None:
    """Remove a custom API key type"""
    console.print(Panel.fit("Remove an API key type", title="Remove API Key Type"))
    
    manager = get_api_key_manager()
    supported_types = manager.get_supported_types()
    
    if not supported_types:
        console.print("[yellow]No API key types to remove.[/yellow]")
        return
    
    # Show current types
    console.print("Current API key types:")
    for i, api_key_type in enumerate(supported_types, 1):
        description = manager.get_description(api_key_type)
        console.print(f"[{i}] {api_key_type} - {description}")
    
    # Select type to remove
    choice = Prompt.ask("Enter number to remove (or 'cancel')", choices=[str(i) for i in range(1, len(supported_types) + 1)] + ["cancel"])
    
    if choice == "cancel":
        console.print("Cancelled.")
        return
    
    api_key_type = supported_types[int(choice) - 1]
    
    # Confirm removal
    if not Confirm.ask(f"Are you sure you want to remove '{api_key_type}'?", default=False):
        console.print("Cancelled.")
        return
    
    try:
        manager.remove_api_key_type(api_key_type)
        console.print(f"[green]Successfully removed API key type '{api_key_type}'![/green]")
    except Exception as e:
        console.print(f"[red]Failed to remove API key type: {e}[/red]")


def main() -> None:
    console.print(Panel.fit("Agentic-Guardian API Key Configuration", title="Setup", subtitle="Use arrow keys not required; type options."))

    while True:
        console.print()
        show_current_values()
        console.print()
        console.print("Main Menu:")
        console.print("[1] Configure API key")
        console.print("[2] Remove API key")
        console.print("[3] Add custom API key type")
        console.print("[4] Remove API key type")
        console.print("[5] Exit")
        action = Prompt.ask("Select", choices=["1", "2", "3", "4", "5"], default="5")

        if action == "5":
            console.print("Exiting.")
            return
        
        if action == "3":
            add_custom_api_key_type()
            continue
        
        if action == "4":
            remove_custom_api_key_type()
            continue

        # For actions 1 and 2, we need to pick a key
        key_choice = pick_key()
        if not key_choice:
            continue
        
        manager = get_api_key_manager()
        env_var = manager.get_env_var(key_choice)

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


