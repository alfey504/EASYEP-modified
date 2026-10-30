"""Common start-up for every script: arguments, configuration, logging, seeding."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Callable

from moe_pruning.config import ConfigError, ExperimentConfig, add_config_arguments, config_from_args
from moe_pruning.utils.logging import setup_logging
from moe_pruning.utils.paths import ArtifactPaths, write_json
from moe_pruning.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


def bootstrap(
    description: str,
    stage: str,
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, ExperimentConfig, ArtifactPaths]:
    parser = argparse.ArgumentParser(description=description)
    add_config_arguments(parser)
    if add_arguments is not None:
        add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        cfg = config_from_args(args)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from None

    paths = ArtifactPaths(cfg)
    setup_logging(cfg.log_level, paths.log_file)
    from moe_pruning.models import available_plugins, load_plugins

    load_plugins(list(cfg.model_plugins))
    logger.debug("Model plugins: %s", sorted(available_plugins()))
    set_seed(cfg.seed)
    logger.info("=== %s | experiment=%s ===", stage, cfg.experiment_name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_json(
        paths.config_snapshot_dir / f"{stage}_{timestamp}.json",
        {"stage": stage, "argv": sys.argv, "config": cfg.to_dict()},
    )
    return args, cfg, paths
