"""
Pattern Configuration Manager

This module provides utilities for managing pattern detection configurations,
including loading from files, saving custom configurations, and runtime updates.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import json
import os
from .pattern_config import PatternDetectionConfig, set_pattern_config, get_pattern_config


class PatternConfigManager:
    """Manager for pattern detection configurations."""
    
    def __init__(self):
        self.default_config_path = Path(__file__).parent / "default_pattern_config.json"
        self.user_config_path = Path.cwd() / "pattern_config.json"  # In project root
        
    def load_default_config(self) -> PatternDetectionConfig:
        """Load the default configuration."""
        if not self.default_config_path.exists():
            raise FileNotFoundError(f"Default config not found: {self.default_config_path}")
        
        return PatternDetectionConfig.load_from_file(self.default_config_path)
    
    def load_user_config(self) -> Optional[PatternDetectionConfig]:
        """Load user configuration if it exists."""
        if self.user_config_path.exists():
            return PatternDetectionConfig.load_from_file(self.user_config_path)
        return None
    
    def save_user_config(self, config: PatternDetectionConfig):
        """Save configuration to user config file."""
        config.save_to_file(self.user_config_path)
        
    def create_user_config_from_default(self):
        """Create a user config file from the default configuration."""
        default_config = self.load_default_config()
        self.save_user_config(default_config)
        print(f"Created user configuration file: {self.user_config_path}")
        return default_config
    
    def initialize_config(self) -> PatternDetectionConfig:
        """
        Initialize configuration system.
        
        Loads user config if available, otherwise uses default.
        Sets the global configuration.
        """
        # Try to load user config first
        config = self.load_user_config()
        
        if config is None:
            # Fall back to default config
            config = self.load_default_config()
            print(f"Using default pattern configuration. Create a custom config with:")
            print(f"  python -m bistoury.strategies.patterns.config_manager create-config")
        else:
            print(f"Loaded user pattern configuration from: {self.user_config_path}")
        
        # Set as global config
        set_pattern_config(config)
        return config
    
    def reload_config(self):
        """Reload configuration from files."""
        config = self.initialize_config()
        print("Pattern configuration reloaded successfully")
        return config
    
    def backup_user_config(self) -> Optional[Path]:
        """Create a backup of the current user config."""
        if not self.user_config_path.exists():
            return None
        
        backup_path = self.user_config_path.with_suffix('.json.backup')
        
        # Find next available backup name
        counter = 1
        while backup_path.exists():
            backup_path = self.user_config_path.with_suffix(f'.json.backup.{counter}')
            counter += 1
        
        # Copy the config
        import shutil
        shutil.copy2(self.user_config_path, backup_path)
        print(f"Backed up configuration to: {backup_path}")
        return backup_path
    
    def restore_config_from_backup(self, backup_path: Path):
        """Restore configuration from a backup file."""
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        import shutil
        shutil.copy2(backup_path, self.user_config_path)
        print(f"Restored configuration from: {backup_path}")
        
        # Reload the restored config
        self.reload_config()
    
    def reset_to_default(self):
        """Reset user configuration to default values."""
        # Backup current config if it exists
        if self.user_config_path.exists():
            self.backup_user_config()
        
        # Create new config from default
        self.create_user_config_from_default()
        
        # Reload
        self.reload_config()
        print("Configuration reset to default values")
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get information about current configuration."""
        current_config = get_pattern_config()
        
        info = {
            "default_config_path": str(self.default_config_path),
            "user_config_path": str(self.user_config_path),
            "user_config_exists": self.user_config_path.exists(),
            "current_min_confidence": float(current_config.default_min_confidence),
            "available_patterns": [
                "doji", "hammer", "shooting_star", "spinning_top", "marubozu",
                "engulfing", "harami", "piercing_line", "dark_cloud_cover", 
                "morning_star", "evening_star"
            ]
        }
        
        return info
    
    def validate_config(self, config_path: Optional[Path] = None) -> bool:
        """Validate a configuration file."""
        if config_path is None:
            config_path = self.user_config_path
        
        try:
            if not config_path.exists():
                print(f"Config file does not exist: {config_path}")
                return False
            
            # Try to load the config
            config = PatternDetectionConfig.load_from_file(config_path)
            
            # Basic validation
            if config.default_min_confidence < 0 or config.default_min_confidence > 100:
                print("Invalid default_min_confidence: must be between 0 and 100")
                return False
            
            print(f"Configuration file is valid: {config_path}")
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False


# Global config manager instance
_config_manager = PatternConfigManager()


def get_config_manager() -> PatternConfigManager:
    """Get the global configuration manager."""
    return _config_manager


def initialize_pattern_config() -> PatternDetectionConfig:
    """Initialize the pattern configuration system."""
    return _config_manager.initialize_config()


def main():
    """CLI interface for configuration management."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m bistoury.strategies.patterns.config_manager <command>")
        print("Commands:")
        print("  create-config    - Create a user configuration file from defaults")
        print("  reload          - Reload configuration from files")
        print("  reset           - Reset to default configuration")
        print("  backup          - Create a backup of current configuration")
        print("  validate        - Validate current configuration")
        print("  info            - Show configuration information")
        return
    
    command = sys.argv[1]
    manager = get_config_manager()
    
    if command == "create-config":
        manager.create_user_config_from_default()
        
    elif command == "reload":
        manager.reload_config()
        
    elif command == "reset":
        manager.reset_to_default()
        
    elif command == "backup":
        backup_path = manager.backup_user_config()
        if backup_path:
            print(f"Configuration backed up to: {backup_path}")
        else:
            print("No user configuration to backup")
            
    elif command == "validate":
        manager.validate_config()
        
    elif command == "info":
        info = manager.get_config_info()
        print("Pattern Configuration Info:")
        print(f"  Default config: {info['default_config_path']}")
        print(f"  User config: {info['user_config_path']}")
        print(f"  User config exists: {info['user_config_exists']}")
        print(f"  Current min confidence: {info['current_min_confidence']}")
        print(f"  Available patterns: {', '.join(info['available_patterns'])}")
        
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main() 