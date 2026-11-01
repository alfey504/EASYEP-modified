"""Save a copy of the configured base model in MODEL_DTYPE (e.g. OLMoE float32 -> float16).

allenai/OLMoE-1B-7B-0125 is stored in float32 (27.7 GB). A float16 copy is
13.8 GB, small enough for a Kaggle Dataset or /kaggle/working. Run once:

    python scripts/convert_model.py --profile configs/kaggle/olmoe-1b-7b-t4x2.env \\
        --output /kaggle/working/olmoe-1b-7b-0125-fp16

then publish the output as a Kaggle Dataset and set
MODEL_PATH=/kaggle/input/<dataset> in later sessions.

The conversion runs on CPU (needs ~1x the target size in RAM). The saved
checkpoint uses the standard Hugging Face layout, so the original-expert
reader used by evaluation works on it unchanged.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from moe_pruning.model.loader import get_model_plugin, load_tokenizer, model_load_kwargs
from moe_pruning.utils.cli import bootstrap
from moe_pruning.utils.device import resolve_dtype
from moe_pruning.utils.paths import write_json

logger = logging.getLogger("convert_model")


def _add_args(parser):
    parser.add_argument("--output", type=Path, required=True, help="Directory for the converted model.")
    parser.add_argument("--max-shard-size", default="2GB", help="Shard size of the saved safetensors files.")


def main() -> None:
    args, cfg, _ = bootstrap(__doc__, "convert_model", _add_args)
    source = cfg.model_source
    dtype = resolve_dtype(cfg.model_dtype)
    plugin = get_model_plugin(cfg)
    logger.info("Converting %s to %s on CPU (plugin %s)", source, cfg.model_dtype, plugin.name)
    model = plugin.auto_model_class.from_pretrained(source, dtype=dtype, **model_load_kwargs(cfg))
    tokenizer = load_tokenizer(cfg)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(out)
    size_gb = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 1e9
    write_json(
        out / "conversion.json",
        {
            "converted_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "revision": cfg.hf_revision,
            "dtype": cfg.model_dtype,
            "parameters": sum(p.numel() for p in model.parameters()),
            "safetensors_gb": round(size_gb, 3),
        },
    )
    logger.info("Saved %s (%.1f GB). Set MODEL_PATH=%s (or the Kaggle Dataset mount).", out, size_gb, out)


if __name__ == "__main__":
    main()
