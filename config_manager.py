"""
Configuration Management for TPT Scraper
Allows managing multiple named search configurations
"""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

class ConfigManager:
    """Manages multiple named configurations for different searches."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.registry_file = self.config_dir / "registry.json"
        self._load_registry()
    
    def _load_registry(self):
        """Load the registry of all configurations."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                self.registry = json.load(f)
        else:
            self.registry = {"configs": {}}
            self._save_registry()
    
    def _save_registry(self):
        """Save the registry of configurations."""
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2)
    
    def create_config(self, name: str, config_data: Dict[str, Any], description: str = "") -> bool:
        """
        Create a new named configuration.
        
        Args:
            name: Unique name for this configuration
            config_data: Dictionary containing the search parameters
            description: Optional description of what this config searches for
        
        Returns:
            True if created successfully, False if name already exists
        """
        if name in self.registry["configs"]:
            logging.error(f"Configuration '{name}' already exists")
            return False
        
        # Validate required fields
        required_fields = [
            "resource_type", "grade_level", "subject", "format", 
            "price_options", "supports", "sorting_methods"
        ]
        for field in required_fields:
            if field not in config_data:
                config_data[field] = [""]
        
        # Add metadata
        config_data["_metadata"] = {
            "created": datetime.now().isoformat(),
            "description": description,
            "last_used": None
        }
        
        # Save config file
        config_file = self.config_dir / f"{name}.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
        # Update registry
        self.registry["configs"][name] = {
            "file": str(config_file),
            "created": config_data["_metadata"]["created"],
            "description": description,
            "db_file": f"scrape_cache_{name}.db"
        }
        self._save_registry()
        
        logging.info(f"Created configuration '{name}'")
        return True
    
    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a configuration by name."""
        if name not in self.registry["configs"]:
            logging.error(f"Configuration '{name}' not found")
            return None
        
        config_file = Path(self.registry["configs"][name]["file"])
        if not config_file.exists():
            logging.error(f"Configuration file not found: {config_file}")
            return None
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Update last used
        config["_metadata"]["last_used"] = datetime.now().isoformat()
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        return config
    
    def get_db_file(self, name: str) -> Optional[str]:
        """Get the database file path for a configuration."""
        if name not in self.registry["configs"]:
            return None
        return self.registry["configs"][name]["db_file"]
    
    def list_configs(self) -> List[Dict[str, Any]]:
        """List all available configurations."""
        configs = []
        for name, info in self.registry["configs"].items():
            configs.append({
                "name": name,
                "created": info["created"],
                "description": info["description"],
                "db_file": info["db_file"]
            })
        return sorted(configs, key=lambda x: x["created"], reverse=True)
    
    def delete_config(self, name: str, delete_db: bool = False) -> bool:
        """
        Delete a configuration.
        
        Args:
            name: Configuration name to delete
            delete_db: If True, also delete the associated database file
        
        Returns:
            True if deleted successfully
        """
        if name not in self.registry["configs"]:
            logging.error(f"Configuration '{name}' not found")
            return False
        
        # Delete config file
        config_file = Path(self.registry["configs"][name]["file"])
        if config_file.exists():
            config_file.unlink()
        
        # Optionally delete database
        if delete_db:
            db_file = Path(self.registry["configs"][name]["db_file"])
            if db_file.exists():
                db_file.unlink()
                logging.info(f"Deleted database: {db_file}")
        
        # Remove from registry
        del self.registry["configs"][name]
        self._save_registry()
        
        logging.info(f"Deleted configuration '{name}'")
        return True
    
    def update_config(self, name: str, config_data: Dict[str, Any]) -> bool:
        """Update an existing configuration."""
        if name not in self.registry["configs"]:
            logging.error(f"Configuration '{name}' not found")
            return False
        
        config_file = Path(self.registry["configs"][name]["file"])
        
        # Preserve metadata
        old_config = self.get_config(name)
        config_data["_metadata"] = old_config.get("_metadata", {})
        config_data["_metadata"]["last_modified"] = datetime.now().isoformat()
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
        logging.info(f"Updated configuration '{name}'")
        return True
