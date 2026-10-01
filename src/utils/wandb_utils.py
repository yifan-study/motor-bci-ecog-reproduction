"""Shared W&B logging utilities for all training scripts.

All functions are no-ops when W&B is disabled, so training scripts can
call them unconditionally without if-checks.

Usage:
    from src.utils.wandb_utils import init_wandb, log_epoch, log_summary, finish

    init_wandb(config, run_name="unet-lomtev_bci4_sub1", tags=["bci4", "lomtev"])
    ...
    log_epoch({"train/loss": 0.5, "val/loss": 0.4}, step=epoch)
    ...
    log_summary({"test/r_avg": 0.68})
    finish()
"""

import os

import wandb


def init_wandb(config, run_name=None, tags=None, notes=None):
    """Initialize a W&B run with full context.

    Parameters
    ----------
    config : dict
        Full YAML config dict — logged to wandb.config for reproducibility.
    run_name : str, optional
        Run name. Auto-generated as "{model}_{dataset}_{subject}" if None.
    tags : list[str], optional
        Filterable tags. Falls back to config["logging"]["tags"] if None.
    notes : str, optional
        Free-text description for the run.

    Returns
    -------
    wandb.Run or None
        The active run, or None if W&B is disabled.
    """
    logging_cfg = config.get("logging", {})
    if not logging_cfg.get("use_wandb", True):
        return None

    project = logging_cfg.get("wandb_project", "plact-motor-decoding")
    if tags is None:
        tags = logging_cfg.get("tags", [])

    # Auto-generate run name from config
    if run_name is None:
        model = config.get("model", {}).get("name", "model")
        task = config.get("data", {}).get("task", "unknown")
        subject = config.get("data", {}).get("subject",
                  config.get("data", {}).get("patient", "?"))
        run_name = f"{model}_{task}_sub{subject}"

    # Group by dataset for W&B UI organization
    group = config.get("data", {}).get("task", None)

    # Route W&B local files to archive (not home quota)
    wandb_dir = os.path.join("results", "wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = os.path.abspath(wandb_dir)
    os.environ["WANDB_SILENT"] = "true"

    run = wandb.init(
        project=project,
        name=run_name,
        config=config,
        tags=tags,
        notes=notes,
        group=group,
    )
    return run


def log_epoch(metrics_dict, step):
    """Log per-epoch metrics if W&B is active.

    Parameters
    ----------
    metrics_dict : dict
        e.g. {"train/loss": 0.5, "val/loss": 0.4, "val/r_avg": 0.6}
    step : int
        Epoch number.
    """
    if wandb.run is None:
        return
    wandb.log(metrics_dict, step=step)


def log_summary(metrics_dict):
    """Log final summary metrics (test results).

    Parameters
    ----------
    metrics_dict : dict
        e.g. {"test/r_avg": 0.68, "test/loss": 0.3}
    """
    if wandb.run is None:
        return
    for key, value in metrics_dict.items():
        wandb.run.summary[key] = value


def finish():
    """Clean up the W&B run."""
    if wandb.run is None:
        return
    wandb.finish()
