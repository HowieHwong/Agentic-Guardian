"""
Core module containing global configuration and constants for the Guardian engine.
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class OutputFormat(str, Enum):
    """Supported output formats for generated data."""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"


class Language(str, Enum):
    """Supported languages for data generation."""
    ENGLISH = "en"
    CHINESE = "zh"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"


class RiskSeverity(str, Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GlobalConfig:
    """
    Global configuration constants for the Guardian engine.
    These are hardcoded in the source code and not read from configuration files.
    """
    
    # Engine metadata
    ENGINE_VERSION: str = "0.1.0"
    ENGINE_NAME: str = "Guardian"
    
    # Default settings
    DEFAULT_LANGUAGE: Language = Language.ENGLISH
    DEFAULT_OUTPUT_FORMAT: OutputFormat = OutputFormat.JSONL
    DEFAULT_BATCH_SIZE: int = 100
    DEFAULT_MAX_RETRIES: int = 3
    
    # Supported configurations
    SUPPORTED_OUTPUT_FORMATS: List[OutputFormat] = field(default_factory=lambda: [
        OutputFormat.JSON,
        OutputFormat.JSONL,
        OutputFormat.CSV,
        OutputFormat.PARQUET
    ])
    
    SUPPORTED_LANGUAGES: List[Language] = field(default_factory=lambda: [
        Language.ENGLISH,
        Language.CHINESE,
        Language.SPANISH,
        Language.FRENCH,
        Language.GERMAN,
        Language.JAPANESE
    ])
    
    SUPPORTED_RISK_SEVERITIES: List[RiskSeverity] = field(default_factory=lambda: [
        RiskSeverity.LOW,
        RiskSeverity.MEDIUM,
        RiskSeverity.HIGH,
        RiskSeverity.CRITICAL
    ])
    
    # Quality thresholds
    MIN_QUALITY_SCORE: float = 0.7
    MAX_HALLUCINATION_PROBABILITY: float = 0.3
    
    # Generation limits
    MAX_SCENARIO_DEPTH: int = 10
    MAX_RISK_COMBINATIONS: int = 5
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported output formats as strings."""
        return [fmt.value for fmt in self.SUPPORTED_OUTPUT_FORMATS]
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages as strings."""
        return [lang.value for lang in self.SUPPORTED_LANGUAGES]
    
    def validate_output_format(self, format_str: str) -> bool:
        """Validate if the given format is supported."""
        return format_str in self.get_supported_formats()
    
    def validate_language(self, lang_str: str) -> bool:
        """Validate if the given language is supported."""
        return lang_str in self.get_supported_languages()


# Global instance
GLOBAL_CONFIG = GlobalConfig() 