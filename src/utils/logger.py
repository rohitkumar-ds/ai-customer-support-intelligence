import logging
import logging.config
import os
from pathlib import Path
from typing import Optional
import yaml


def setup_logging(
    config_path: str = "configs/logging_config.yaml",
    default_level: int = logging.INFO,
    logs_dir: str = "logs",
) :
    """Configures application-wide logging from a YAML configuration file.

    Args:
        config_path: Path to the logging configuration YAML file.
        default_level: Fallback logging level if the YAML file cannot be loaded.
        logs_dir: Directory where log files are stored. Ensures the folder exists
            prior to attaching file handlers to prevent FileNotFoundError.
    """
    # WHY: Ensure the log directory exists before initializing RotatingFileHandler.
    # If the directory doesn't exist on a fresh machine or CI environment,
    # logging initialization will fail with FileNotFoundError.
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    config_file = Path(config_path)
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
            logging.config.dictConfig(config_dict)
            return
        except Exception as e:
            # Fallback to basicConfig if YAML parsing fails
            logging.basicConfig(
                level=default_level,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            )
            logging.getLogger(__name__).warning(
                f"Failed to load logging config from {config_path}: {e}. "
                f"Falling back to basicConfig."
            )
    else:
        # Graceful fallback if configuration file is not found
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Returns a standardized logger instance for the given module name.

    Args:
        name: Name of the logger, typically `__name__`.

    Returns:
        A configured logging.Logger instance.
    """
    return logging.getLogger(name or "src")
