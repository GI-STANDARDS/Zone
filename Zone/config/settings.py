"""
Advanced Configuration Management for Face Recognition System
Supports environment variables, config files, and command-line arguments
"""
import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False
    debug: bool = False


@dataclass
class ModelConfig:
    """Model configuration"""
    detector: str = "mtcnn"
    encoder: str = "facenet"
    database: str = "sqlite"
    threshold: float = 0.6


@dataclass
class PathConfig:
    """Path configuration"""
    base_dir: str = ""
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    config_dir: str = "config"


@dataclass
class PerformanceConfig:
    """Performance configuration"""
    batch_size: int = 32
    num_workers: int = 4
    device: str = "auto"
    cache_models: bool = True


@dataclass
class SecurityConfig:
    """Security configuration"""
    allowed_extensions: list = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    enable_ethics_warning: bool = True
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]


@dataclass
class AppConfig:
    """Complete application configuration"""
    server: ServerConfig = None
    models: ModelConfig = None
    paths: PathConfig = None
    performance: PerformanceConfig = None
    security: SecurityConfig = None
    
    def __post_init__(self):
        if self.server is None:
            self.server = ServerConfig()
        if self.models is None:
            self.models = ModelConfig()
        if self.paths is None:
            self.paths = PathConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()
        if self.security is None:
            self.security = SecurityConfig()


class ConfigManager:
    """Configuration manager with multiple sources"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = AppConfig()
        
        # Load configuration from multiple sources
        self._load_defaults()
        self._load_config_file()
        self._load_environment_variables()
        
        logger.info("Configuration manager initialized")
    
    def _load_defaults(self):
        """Load default configuration"""
        self.config = AppConfig()
        logger.debug("Default configuration loaded")
    
    def _load_config_file(self):
        """Load configuration from file"""
        if not self.config_file:
            # Look for default config files
            default_paths = [
                "config.yaml",
                "config.json",
                "config/settings.yaml",
                "config/settings.json"
            ]
            
            for path in default_paths:
                if Path(path).exists():
                    self.config_file = path
                    break
        
        if not self.config_file or not Path(self.config_file).exists():
            logger.debug("No configuration file found")
            return
        
        try:
            config_path = Path(self.config_file)
            
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Update configuration
            self._update_config_from_dict(config_data)
            logger.info(f"Configuration loaded from {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load config file {self.config_file}: {e}")
    
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        env_mappings = {
            # Server settings
            'FACE_RECO_HOST': ('server', 'host'),
            'FACE_RECO_PORT': ('server', 'port', int),
            'FACE_RECO_SHARE': ('server', 'share', bool),
            'FACE_RECO_DEBUG': ('server', 'debug', bool),
            
            # Model settings
            'FACE_RECO_DETECTOR': ('models', 'detector'),
            'FACE_RECO_ENCODER': ('models', 'encoder'),
            'FACE_RECO_DATABASE': ('models', 'database'),
            'FACE_RECO_THRESHOLD': ('models', 'threshold', float),
            
            # Path settings
            'FACE_RECO_DATA_DIR': ('paths', 'data_dir'),
            'FACE_RECO_MODELS_DIR': ('paths', 'models_dir'),
            'FACE_RECO_LOGS_DIR': ('paths', 'logs_dir'),
            
            # Performance settings
            'FACE_RECO_BATCH_SIZE': ('performance', 'batch_size', int),
            'FACE_RECO_NUM_WORKERS': ('performance', 'num_workers', int),
            'FACE_RECO_DEVICE': ('performance', 'device'),
            'FACE_RECO_CACHE_MODELS': ('performance', 'cache_models', bool),
            
            # Security settings
            'FACE_RECO_MAX_FILE_SIZE': ('security', 'max_file_size', int),
            'FACE_RECO_ENABLE_ETHICS': ('security', 'enable_ethics_warning', bool),
        }
        
        for env_var, (section, key, *type_hint) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Convert value type
                    if type_hint:
                        type_converter = type_hint[0]
                        if type_converter == bool:
                            value = value.lower() in ['true', '1', 'yes', 'on']
                        elif type_converter == int:
                            value = int(value)
                        elif type_converter == float:
                            value = float(value)
                    
                    # Set configuration value
                    section_obj = getattr(self.config, section)
                    setattr(section_obj, key, value)
                    
                    logger.debug(f"Environment variable {env_var} set {section}.{key} = {value}")
                    
                except (ValueError, AttributeError) as e:
                    logger.error(f"Invalid environment variable {env_var}={value}: {e}")
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section_name, section_data in config_data.items():
            if hasattr(self.config, section_name):
                section_obj = getattr(self.config, section_name)
                
                for key, value in section_data.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                        logger.debug(f"Config file set {section_name}.{key} = {value}")
    
    def get_config(self) -> AppConfig:
        """Get complete configuration"""
        return self.config
    
    def get_server_config(self) -> ServerConfig:
        """Get server configuration"""
        return self.config.server
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration"""
        return self.config.models
    
    def get_path_config(self) -> PathConfig:
        """Get path configuration"""
        return self.config.paths
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration"""
        return self.config.performance
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        return self.config.security
    
    def save_config(self, file_path: str):
        """Save current configuration to file"""
        try:
            config_dict = asdict(self.config)
            
            with open(file_path, 'w') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def print_config(self):
        """Print current configuration"""
        config_dict = asdict(self.config)
        print("Current Configuration:")
        print(json.dumps(config_dict, indent=2))


# Global configuration instance
_config_manager = None


def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager


def get_config() -> AppConfig:
    """Get current application configuration"""
    return get_config_manager().get_config()


if __name__ == "__main__":
    # Test configuration manager
    config_manager = ConfigManager()
    config_manager.print_config()
    
    # Test saving configuration
    config_manager.save_config("test_config.json")
    print("Configuration saved to test_config.json")
