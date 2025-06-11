"""
Centralized Configuration Manager

Manages all configuration files in the config/ directory and provides
a unified interface for accessing configuration parameters throughout
the application.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from decimal import Decimal


logger = logging.getLogger(__name__)


@dataclass
class ConfigPaths:
    """Configuration file paths."""
    
    # Get project root directory (assumes this file is in src/bistoury/)
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    
    @property
    def config_dir(self) -> Path:
        """Configuration directory path."""
        return self.project_root / "config"
    
    @property
    def pattern_detection_config(self) -> Path:
        """Pattern detection configuration file."""
        return self.config_dir / "pattern_detection.json"
    
    @property
    def strategy_config(self) -> Path:
        """Strategy configuration file."""
        return self.config_dir / "strategy.json"
    
    @property
    def agents_config(self) -> Path:
        """Agents configuration file."""
        return self.config_dir / "agents.json"
    
    @property
    def trading_config(self) -> Path:
        """Trading configuration file."""
        return self.config_dir / "trading.json"


class ConfigManager:
    """
    Centralized configuration manager.
    
    Provides unified access to all configuration parameters across
    pattern detection, strategy, agents, and trading configurations.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Optional custom config directory path
        """
        self.paths = ConfigPaths()
        if config_dir:
            self.paths.config_dir = config_dir
        
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._load_all_configs()
    
    def _load_all_configs(self) -> None:
        """Load all configuration files."""
        config_files = {
            'pattern_detection': self.paths.pattern_detection_config,
            'strategy': self.paths.strategy_config,
            'agents': self.paths.agents_config,
            'trading': self.paths.trading_config
        }
        
        for config_name, config_path in config_files.items():
            try:
                self._configs[config_name] = self._load_config_file(config_path)
                logger.info(f"Loaded {config_name} configuration from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load {config_name} config from {config_path}: {e}")
                self._configs[config_name] = {}
    
    def _load_config_file(self, path: Path) -> Dict[str, Any]:
        """Load a JSON configuration file."""
        if not path.exists():
            logger.warning(f"Configuration file not found: {path}")
            return {}
        
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file {path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading config file {path}: {e}")
            return {}
    
    def get(self, config_type: str, *keys: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            config_type: Configuration type ('pattern_detection', 'strategy', 'agents', 'trading')
            *keys: Configuration keys (e.g., 'timeframe_analysis', 'primary_timeframe')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            # Get strategy.timeframe_analysis.primary_timeframe
            config.get('strategy', 'timeframe_analysis', 'primary_timeframe')
            
            # Get pattern_detection.engulfing.min_engulfing_ratio
            config.get('pattern_detection', 'engulfing', 'min_engulfing_ratio')
        """
        if config_type not in self._configs:
            logger.warning(f"Unknown config type: {config_type}")
            return default
        
        value = self._configs[config_type]
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_decimal(self, config_type: str, *keys: str, default: Union[str, float, Decimal] = "0") -> Decimal:
        """Get configuration value as Decimal."""
        value = self.get(config_type, *keys, default=default)
        
        try:
            return Decimal(str(value))
        except (ValueError, TypeError):
            logger.warning(f"Could not convert {value} to Decimal, using default {default}")
            return Decimal(str(default))
    
    def get_bool(self, config_type: str, *keys: str, default: bool = False) -> bool:
        """Get configuration value as boolean."""
        value = self.get(config_type, *keys, default=default)
        
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        else:
            return bool(value) if value is not None else default
    
    def get_int(self, config_type: str, *keys: str, default: int = 0) -> int:
        """Get configuration value as integer."""
        value = self.get(config_type, *keys, default=default)
        
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert {value} to int, using default {default}")
            return default
    
    def get_float(self, config_type: str, *keys: str, default: float = 0.0) -> float:
        """Get configuration value as float."""
        value = self.get(config_type, *keys, default=default)
        
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert {value} to float, using default {default}")
            return default
    
    def get_list(self, config_type: str, *keys: str, default: Optional[list] = None) -> list:
        """Get configuration value as list."""
        if default is None:
            default = []
        
        value = self.get(config_type, *keys, default=default)
        
        if isinstance(value, list):
            return value
        else:
            logger.warning(f"Expected list but got {type(value)}, using default {default}")
            return default
    
    def get_section(self, config_type: str, *keys: str) -> Dict[str, Any]:
        """Get an entire configuration section."""
        value = self.get(config_type, *keys, default={})
        
        if isinstance(value, dict):
            return value
        else:
            logger.warning(f"Expected dict but got {type(value)}, returning empty dict")
            return {}
    
    def set(self, config_type: str, keys: list, value: Any) -> None:
        """
        Set configuration value (runtime only, not persisted).
        
        Args:
            config_type: Configuration type
            keys: List of keys for nested access
            value: Value to set
        """
        if config_type not in self._configs:
            self._configs[config_type] = {}
        
        config = self._configs[config_type]
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the final value
        config[keys[-1]] = value
        logger.debug(f"Set {config_type}.{'.'.join(keys)} = {value}")
    
    def reload(self, config_type: Optional[str] = None) -> None:
        """
        Reload configuration files.
        
        Args:
            config_type: Specific config type to reload, or None for all
        """
        if config_type:
            if config_type == 'pattern_detection':
                self._configs[config_type] = self._load_config_file(self.paths.pattern_detection_config)
            elif config_type == 'strategy':
                self._configs[config_type] = self._load_config_file(self.paths.strategy_config)
            elif config_type == 'agents':
                self._configs[config_type] = self._load_config_file(self.paths.agents_config)
            elif config_type == 'trading':
                self._configs[config_type] = self._load_config_file(self.paths.trading_config)
            else:
                logger.warning(f"Unknown config type: {config_type}")
        else:
            self._load_all_configs()
        
        logger.info(f"Reloaded configuration: {config_type or 'all'}")
    
    def save(self, config_type: str) -> bool:
        """
        Save configuration to file.
        
        Args:
            config_type: Configuration type to save
            
        Returns:
            True if successful, False otherwise
        """
        if config_type not in self._configs:
            logger.error(f"Cannot save unknown config type: {config_type}")
            return False
        
        config_files = {
            'pattern_detection': self.paths.pattern_detection_config,
            'strategy': self.paths.strategy_config,
            'agents': self.paths.agents_config,
            'trading': self.paths.trading_config
        }
        
        if config_type not in config_files:
            logger.error(f"No file mapping for config type: {config_type}")
            return False
        
        file_path = config_files[config_type]
        
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write configuration
            with open(file_path, 'w') as f:
                json.dump(self._configs[config_type], f, indent=2, default=str)
            
            logger.info(f"Saved {config_type} configuration to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save {config_type} config to {file_path}: {e}")
            return False
    
    def list_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all loaded configurations."""
        return self._configs.copy()
    
    def validate_config(self, config_type: str) -> Dict[str, list]:
        """
        Validate configuration and return any issues.
        
        Args:
            config_type: Configuration type to validate
            
        Returns:
            Dictionary with 'errors' and 'warnings' lists
        """
        errors = []
        warnings = []
        
        if config_type not in self._configs:
            errors.append(f"Configuration type '{config_type}' not loaded")
            return {'errors': errors, 'warnings': warnings}
        
        config = self._configs[config_type]
        
        if config_type == 'strategy':
            # Validate strategy configuration
            if 'timeframe_analysis' in config:
                tf_config = config['timeframe_analysis']
                
                if 'primary_timeframe' not in tf_config:
                    errors.append("Missing primary_timeframe in timeframe_analysis")
                
                if 'data_quality_threshold' in tf_config:
                    threshold = tf_config['data_quality_threshold']
                    if not (0 <= threshold <= 100):
                        errors.append(f"data_quality_threshold must be 0-100, got {threshold}")
            
            if 'pattern_detection' in config:
                pd_config = config['pattern_detection']
                
                for key in ['min_pattern_confidence', 'min_confluence_score', 'min_quality_score']:
                    if key in pd_config:
                        value = pd_config[key]
                        if not (0 <= value <= 100):
                            errors.append(f"{key} must be 0-100, got {value}")
        
        elif config_type == 'pattern_detection':
            # Validate pattern detection configuration
            required_patterns = ['doji', 'hammer', 'shooting_star', 'engulfing', 'harami']
            
            for pattern in required_patterns:
                if pattern not in config:
                    warnings.append(f"Missing configuration for pattern: {pattern}")
        
        return {'errors': errors, 'warnings': warnings}


# Global configuration manager instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    return _config_manager


# Convenience functions for easy access
def get_config(config_type: str, *keys: str, default: Any = None) -> Any:
    """Convenience function to get config value."""
    return get_config_manager().get(config_type, *keys, default=default)


def get_pattern_config(pattern_name: str, parameter: str, default: Any = None) -> Any:
    """Convenience function to get pattern configuration."""
    return get_config_manager().get('pattern_detection', pattern_name, parameter, default=default)


def get_strategy_config(*keys: str, default: Any = None) -> Any:
    """Convenience function to get strategy configuration."""
    return get_config_manager().get('strategy', *keys, default=default)


def get_agent_config(agent_name: str, *keys: str, default: Any = None) -> Any:
    """Convenience function to get agent configuration."""
    return get_config_manager().get('agents', agent_name, *keys, default=default)


def get_trading_config(*keys: str, default: Any = None) -> Any:
    """Convenience function to get trading configuration."""
    return get_config_manager().get('trading', *keys, default=default) 