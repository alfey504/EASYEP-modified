"""Rank experts per layer with EASY-EP and select the least important ones for replacement.

Outputs expert_scores.json, expert_scores.csv and expert_selection.json in the
experiment directory.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone

import torch

from moe_pruning.model.architecture import MoELayerInfo, select_layers
from moe_pruning.pruning.expert_selection import score_and_select
from moe_pruning.scoring.statistics import load_statistics
from moe_pruning.utils.cli import bootstrap
from moe_pruning.utils.paths import read_json, write_json

logger = logging.getLogger("score_experts")


def main() -> None:
    _, cfg, paths = bootstrap(__doc__, "score_experts")
    paths.require(paths.expert_stats, "uv run python scripts/collect_stats.py")
    manifest = read_json(paths.require(paths.expert_stats_manifest, "uv run python scripts/collect_stats.py"))
    if manifest["model_source"] != cfg.model_source:
        raise SystemExit(
            f"Statistics were collected for {manifest['model_source']} but the configured model is "
            f"{cfg.model_source}. Re-run collect_stats.py."
        )
    stats = load_statistics(torch.load(paths.expert_stats, weights_only=True)["layers"])
    layers = [MoELayerInfo(**d) for d in manifest["layers"]]
    prune_layers = select_layers(layers, cfg.prune_layers)
    allow_unobserved = cfg.prune_unobserved_policy == "allow"

    rows, selection = score_and_select(
        stats, layers, prune_layers, cfg.prune_experts_per_layer, allow_unobserved
    )

    write_json(paths.expert_scores_json, {"method": "easyep", "experts": [r.to_dict() for r in rows]})
    with paths.expert_scores_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].to_dict()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r.to_dict())

    unobserved = {r.layer_id: [] for r in rows}
    for r in rows:
        if not r.observed:
            unobserved[r.layer_id].append(r.expert_id)
    write_json(
        paths.expert_selection,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "method": cfg.pruning_method,
            "model_source": cfg.model_source,
            "experts_per_layer": cfg.prune_experts_per_layer,
            "prune_layers_spec": cfg.prune_layers,
            "unobserved_policy": cfg.prune_unobserved_policy,
            "calibration_fingerprint": manifest["calibration"]["fingerprint"],
            "layers": {
                layer.layer_id: {
                    "selected_experts": selection[layer.layer_id],
                    "unobserved_experts": unobserved[layer.layer_id],
                    "ranking_least_to_most_important": [
                        r.expert_id for r in sorted(
                            (r for r in rows if r.layer_id == layer.layer_id), key=lambda r: r.rank
                        )
                    ],
                    "layer": layer.to_dict(),
                }
                for layer in prune_layers
            },
        },
    )

    for layer in layers:
        layer_rows = sorted((r for r in rows if r.layer_id == layer.layer_id), key=lambda r: r.rank)
        marker = "PRUNE" if layer.layer_id in selection else "     "
        ranking = " ".join(
            f"{r.expert_id}{'*' if r.selected_for_replacement else ''}:{r.importance_share:.3f}"
            for r in layer_rows
        )
        logger.info("%s %-16s least->most important (expert:share) %s", marker, layer.layer_id, ranking)
    for layer_id, chosen in selection.items():
        logger.info("Selected for replacement in %s: experts %s", layer_id, chosen)
    logger.info("Wrote %s, %s, %s", paths.expert_scores_json, paths.expert_scores_csv, paths.expert_selection)


if __name__ == "__main__":
    main()
