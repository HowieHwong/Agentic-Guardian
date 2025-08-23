"""
Unified API Key Management Module

This module provides a centralized way to manage API key types and their resolution.
It reads from config/api_key_types.yaml to support custom API key types.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from pydantic import BaseModel


class APIKeyType(BaseModel):
    """Configuration for a single API key type"""
    env_var: str
    description: str


class APIKeyManager:
    """Manages API key types and resolution"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the API key manager
        
        Args:
            config_path: Path to api_key_types.yaml, defaults to config/api_key_types.yaml
        """
        if config_path is None:
            # Default to config/api_key_types.yaml relative to project root
            project_root = Path(__file__).resolve().parents[1]
            config_path = project_root / "config" / "api_key_types.yaml"
        
        self.config_path = Path(config_path)
        self._api_key_types: Dict[str, APIKeyType] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load API key types from configuration file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"API key types config file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            
            api_key_types_data = data.get('api_key_types', {})
            
            for key_type, config in api_key_types_data.items():
                if isinstance(config, dict):
                    self._api_key_types[key_type] = APIKeyType(**config)
                    
        except Exception as e:
            raise RuntimeError(f"Failed to load API key types config: {e}")
    
    def get_supported_types(self) -> List[str]:
        """Get list of all supported API key types"""
        return list(self._api_key_types.keys())
    
    def get_env_var(self, api_key_type: str) -> str:
        """
        Get environment variable name for an API key type
        
        Args:
            api_key_type: The logical API key type
            
        Returns:
            Environment variable name
            
        Raises:
            ValueError: If api_key_type is not supported
        """
        if api_key_type not in self._api_key_types:
            supported = ', '.join(self.get_supported_types())
            raise ValueError(f"Unknown api_key_type: {api_key_type}. Supported types: {supported}")
        
        return self._api_key_types[api_key_type].env_var
    
    def get_description(self, api_key_type: str) -> str:
        """
        Get description for an API key type
        
        Args:
            api_key_type: The logical API key type
            
        Returns:
            Description string
            
        Raises:
            ValueError: If api_key_type is not supported
        """
        if api_key_type not in self._api_key_types:
            supported = ', '.join(self.get_supported_types())
            raise ValueError(f"Unknown api_key_type: {api_key_type}. Supported types: {supported}")
        
        return self._api_key_types[api_key_type].description
    
    def get_type_mapping(self) -> Dict[str, str]:
        """
        Get mapping of api_key_type -> env_var_name
        
        Returns:
            Dictionary mapping logical types to environment variable names
        """
        return {key_type: config.env_var for key_type, config in self._api_key_types.items()}
    
    def get_type_info(self, api_key_type: str) -> APIKeyType:
        """
        Get complete information for an API key type
        
        Args:
            api_key_type: The logical API key type
            
        Returns:
            APIKeyType object with env_var and description
            
        Raises:
            ValueError: If api_key_type is not supported
        """
        if api_key_type not in self._api_key_types:
            supported = ', '.join(self.get_supported_types())
            raise ValueError(f"Unknown api_key_type: {api_key_type}. Supported types: {supported}")
        
        return self._api_key_types[api_key_type]
    
    def resolve_api_key(self, api_key_type: str) -> str:
        """
        Resolve API key from environment variables based on a key type.
        
        Args:
            api_key_type: Logical API key type identifier
            
        Returns:
            The API key string from the corresponding environment variable
            
        Raises:
            ValueError: If the api_key_type is unknown or the env var is missing/empty
        """
        env_var_name = self.get_env_var(api_key_type)
        api_key_value = os.getenv(env_var_name, "").strip()
        
        if not api_key_value:
            # Fallback: read from project .env file
            try:
                project_root = Path(__file__).resolve().parents[1]
                env_path = project_root / ".env"
                if env_path.exists():
                    for line in env_path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        if k.strip() == env_var_name:
                            api_key_value = v.strip().strip('"')
                            break
            except Exception:
                pass
        
        if not api_key_value:
            raise ValueError(
                f"Environment variable '{env_var_name}' not set for api_key_type '{api_key_type}'. "
                f"Consider running: python config/configure_api_keys.py"
            )
        
        return api_key_value
    
    def add_api_key_type(self, api_key_type: str, env_var: str, description: str) -> None:
        """
        Add a new API key type to the configuration
        
        Args:
            api_key_type: Logical name for the API key type
            env_var: Environment variable name
            description: Human-readable description
        """
        if api_key_type in self._api_key_types:
            raise ValueError(f"API key type '{api_key_type}' already exists")
        
        # Update in-memory configuration
        self._api_key_types[api_key_type] = APIKeyType(env_var=env_var, description=description)
        
        # Update configuration file
        self._save_config()
    
    def remove_api_key_type(self, api_key_type: str) -> None:
        """
        Remove an API key type from the configuration
        
        Args:
            api_key_type: Logical name for the API key type to remove
            
        Raises:
            ValueError: If api_key_type doesn't exist
        """
        if api_key_type not in self._api_key_types:
            raise ValueError(f"API key type '{api_key_type}' not found")
        
        # Remove from in-memory configuration
        del self._api_key_types[api_key_type]
        
        # Update configuration file
        self._save_config()
    
    def _save_config(self) -> None:
        """Save current configuration to file"""
        data = {
            'api_key_types': {
                key_type: {'env_var': config.env_var, 'description': config.description}
                for key_type, config in self._api_key_types.items()
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2)


# Global instance for easy access
_global_manager: Optional[APIKeyManager] = None


def get_api_key_manager() -> APIKeyManager:
    """Get the global API key manager instance"""
    global _global_manager
    if _global_manager is None:
        _global_manager = APIKeyManager()
    return _global_manager


def resolve_api_key_from_env(api_key_type: str) -> str:
    """
    Convenience function to resolve API key from environment variables.
    Uses the global API key manager.
    
    Args:
        api_key_type: Logical API key type identifier
        
    Returns:
        The API key string from the corresponding environment variable
        
    Raises:
        ValueError: If the api_key_type is unknown or the env var is missing/empty
    """
    return get_api_key_manager().resolve_api_key(api_key_type)
