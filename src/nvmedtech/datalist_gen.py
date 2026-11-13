"""
Generate a MONAI datalist for bundle training.

Many MONAI training bundles expect a datalist JSON with `training` and
`validation` keys, each a list of {"image": <path>, "label": <path>} relative
to a `dataset_dir`. Medical Segmentation Decathlon tasks already ship a
`dataset.json` with a `training` list; this helper splits it into
train/validation and normalizes the paths.

RESEARCH USE ONLY.
"""
import argparse
import json
import logging
import os
import random

logger = logging.getLogger(__name__)


def build_datalist(dataset_dir: str, val_fraction: float = 0.2, seed: int = 42) -> dict:
    """Read <dataset_dir>/dataset.json and produce a train/validation split.

    Paths in the output are relative to dataset_dir (leading "./" stripped),
    matching what `monai.bundle` resolves against `--dataset_dir`.
    """
    ds_json = os.path.join(dataset_dir, "dataset.json")
    if not os.path.isfile(ds_json):
        raise FileNotFoundError(f"dataset.json not found in {dataset_dir}")

    with open(ds_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    training = meta.get("training", [])
    if not training:
        raise ValueError("dataset.json contains no 'training' entries")

    def _norm(p: str) -> str:
        return p[2:] if p.startswith("./") else p.lstrip("/")

    pairs = []
    for entry in training:
        img, lbl = entry.get("image"), entry.get("label")
        if not img or not lbl:
            continue
        pairs.append({"image": _norm(img), "label": _norm(lbl)})

    if not pairs:
        raise ValueError("No valid image/label pairs found in dataset.json")

    # Seeded PRNG for a reproducible train/val split — not a security context.
    rng = random.Random(seed)  # nosec B311
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_fraction))
    validation = pairs[:n_val]
    train = pairs[n_val:]

    logger.info(
        "Datalist: %d training, %d validation (val_fraction=%.2f, seed=%d)",
        len(train), len(validation), val_fraction, seed,
    )
    return {"training": train, "validation": validation, "labels": meta.get("labels", {})}


def main():
    parser = argparse.ArgumentParser(description="Generate MONAI Decathlon datalist")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    datalist = build_datalist(args.dataset_dir, args.val_fraction, args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(datalist, f, indent=2)
    logger.info("Wrote datalist to %s", args.output)


if __name__ == "__main__":
    main()
