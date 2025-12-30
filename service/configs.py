import yaml
import logging
import sys
from pathlib import Path

#Load a YAML file
config_path = Path("config.yaml")

def load_config():
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at path {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

settings = load_config()

#Setup a logger for writing console-level and file-level metrics
def setup(name="ppg_assess"):
    logger = logging.getLogger(name)
    logger.setLevel(settings["logging"]["level"])

    #Format as Time | Level | Message
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    con_handler = logging.StreamHandler(sys.stdout)
    con_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(settings["logging"]["file"])
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(con_handler)
        logger.addHandler(file_handler)

    return logger

logger = setup()