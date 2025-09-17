"""
Guardian: Agentic Safety Data Generation Engine

A comprehensive data generation engine for enhancing Guard Model capabilities
in detecting safety issues in agentic AI systems.
"""

__version__ = "0.1.0"
__author__ = "AuraGen Team"
__description__ = "Agentic Safety Data Generation Engine"

from .config import GuardianConfig
from .core import GlobalConfig

__all__ = [
    "GuardianConfig",
    "GlobalConfig",
] 