"""YAML-based experiment configuration system."""

import copy
import os
from datetime import datetime
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    """Load a YAML config file and return as a dict."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, path: str):
    """Save a config dict to a YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def make_experiment_dir(config: dict, base_dir: str = "results") -> str:
    """Create a timestamped experiment directory and save the config inside it.

    Returns the path to the experiment directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.get("model", {}).get("name", "unknown")
    data_cfg = config.get("data", {})
    patient = data_cfg.get("patient", data_cfg.get("subject", "unknown"))
    exp_id = f"{timestamp}_{model_name}_{patient}"

    exp_dir = os.path.join(base_dir, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    save_config(config, os.path.join(exp_dir, "config.yaml"))
    return exp_dir
