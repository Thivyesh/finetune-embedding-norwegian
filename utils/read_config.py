"""
Configuration loader for embedding model training.

This module loads and validates YAML configuration files, making them easy
to access throughout the training pipeline.
"""

import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """
    Configuration container with dot-notation access.

    This allows you to access config values like:
        config.model.name
    instead of:
        config['model']['name']

    Makes the code cleaner and easier to read!
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize config from dictionary.

        Args:
            config_dict: Dictionary from parsed YAML file
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Recursively convert nested dicts to Config objects
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Config({self.__dict__})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary format."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object with dot-notation access

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML

    Example:
        >>> config = load_config("configs/training_config.yaml")
        >>> print(config.model.name)
        'ltg/norbert3-large'
        >>> print(config.training.num_train_epochs)
        1
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please ensure the config file exists at the specified path."
        )

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Error parsing YAML configuration file: {config_path}\n"
            f"Error details: {e}"
        )

    if config_dict is None:
        raise ValueError(
            f"Configuration file is empty: {config_path}\n"
            f"Please ensure the config file contains valid YAML content."
        )

    return Config(config_dict)


def validate_config(config: Config) -> None:
    """
    Validate that required configuration fields are present.

    This helps catch configuration errors early before training starts.

    Args:
        config: Config object to validate

    Raises:
        ValueError: If required fields are missing or invalid
    """
    # Check required top-level sections
    required_sections = ['model', 'dataset', 'training']
    for section in required_sections:
        if not hasattr(config, section):
            raise ValueError(
                f"Missing required config section: '{section}'\n"
                f"Please add this section to your configuration file."
            )

    # Check model config
    if not hasattr(config.model, 'name'):
        raise ValueError("Missing required field: model.name")

    # Check dataset config
    required_dataset_fields = ['name', 'train_split', 'anchor_column',
                                'positive_column', 'negative_column']
    for field in required_dataset_fields:
        if not hasattr(config.dataset, field):
            raise ValueError(f"Missing required field: dataset.{field}")

    # Check training config
    required_training_fields = ['output_dir', 'num_train_epochs',
                                 'per_device_train_batch_size']
    for field in required_training_fields:
        if not hasattr(config.training, field):
            raise ValueError(f"Missing required field: training.{field}")

    # Validate numeric values
    if config.training.num_train_epochs <= 0:
        raise ValueError("training.num_train_epochs must be > 0")

    if config.training.per_device_train_batch_size <= 0:
        raise ValueError("training.per_device_train_batch_size must be > 0")

    print("✓ Configuration validation passed!")


if __name__ == "__main__":
    """
    Test the config loader.

    Run this file directly to test loading your config:
        python utils/read_config.py
    """
    config_path = "configs/training_config.yaml"

    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    print("\nValidating config...")
    validate_config(config)

    print("\nConfig loaded successfully!")
    print(f"\nModel: {config.model.name}")
    print(f"Dataset: {config.dataset.name}")
    print(f"Output: {config.training.output_dir}")
    print(f"Epochs: {config.training.num_train_epochs}")
    print(f"Batch size: {config.training.per_device_train_batch_size}")
