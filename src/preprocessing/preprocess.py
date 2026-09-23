"""Batch preprocessing entry point for a SageMaker Processing Job.

One static script, data-driven by a pipeline spec (spec.json) — the SAME
pipeline the local preview runs, via the shared pipeline_builder. It:

  1. Resolves the dataset from the mounted input (dataset.json, or ImageFolder
     class-subfolders for classification) using dataset_resolver.
  2. Builds a monai.transforms.Compose from spec.json (pipeline_builder).
  3. Applies it to every entry (image + label with nearest interp for
     segmentation) and writes processed .nii.gz to the output dir.
  4. Writes an updated dataset.json at the output root pointing at the processed
     files (ready to feed a training Data S3 URI), plus a result.json manifest.

SageMaker mounts:
    input  -> /opt/ml/processing/input   (the dataset root)
    code   -> /opt/ml/processing/code    (this script + helpers + spec.json)
    output -> /opt/ml/processing/output  (processed dataset)

Runs on a CPU instance — no GPU required.

RESEARCH USE ONLY — preprocessing performs no diagnosis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

# Make sibling helpers importable whether run as a module or a flat script
# (SageMaker stages code/ as a flat dir).
_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.dirname(os.path.dirname(_HERE))):
    if p not in sys.path:
        sys.path.insert(0, p)

DISCLAIMER = "RESEARCH USE ONLY — preprocessing performs no diagnosis."


# ── spec loading (works with or without the app's Pydantic models) ─────────────

def _load_spec(spec_path: str):
    with open(spec_path, "r") as f:
        raw = json.load(f)
    try:
        # Prefer the validated model if the app package is importable.
        from app.models.preprocessing import PipelineSpec
        return PipelineSpec(**raw)
    except Exception:
        # Fallback: a lightweight shim exposing the attributes the builder needs.
        from types import SimpleNamespace

        class _Step(SimpleNamespace):
            pass

        keys = raw.get("keys") or (["image", "label"] if raw.get("task_type") == "segmentation" else ["image"])
        steps = [_Step(type=s["type"], params=s.get("params", {})) for s in raw["steps"]]
        return SimpleNamespace(task_type=raw.get("task_type", "segmentation"),
                               keys=keys, steps=steps,
                               dataset_prep=raw.get("dataset_prep"))


def _resolve_dataset(dataset_dir: str):
    from src.preprocessing.dataset_resolver import (
        resolve_from_dir, discover_imagefolder_s3, DatasetJsonMissingError,
    )
    try:
        # Do NOT hard-fail up front if some referenced files are missing — real
        # datasets sometimes list files that aren't present. The per-entry
        # processing loop skips missing/failing files and reports the count.
        return resolve_from_dir(dataset_dir, verify_exists=False)
    except DatasetJsonMissingError:
        # No dataset.json — try ImageFolder classification over the LOCAL dir.
        return _discover_imagefolder_local(dataset_dir)


def _discover_imagefolder_local(dataset_dir: str):
    """Local mirror of discover_imagefolder_s3 for the mounted input dir."""
    from src.preprocessing.dataset_resolver import (
        ResolvedDataset, ResolvedPair, DatasetResolveError, _IMAGE_EXTS,
    )
    classes = {}
    pairs = []
    for name in sorted(os.listdir(dataset_dir)):
        cdir = os.path.join(dataset_dir, name)
        if not os.path.isdir(cdir):
            continue
        imgs = [f for f in _walk_files(cdir) if f.lower().endswith(_IMAGE_EXTS)]
        if imgs:
            classes[name] = len(imgs)
            for rel in imgs:
                pairs.append(ResolvedPair(image=os.path.relpath(rel, dataset_dir),
                                          label=None, split="training"))
    if not classes:
        raise DatasetResolveError("No dataset.json and no class subfolders with images found")
    return ResolvedDataset(
        name=os.path.basename(dataset_dir.rstrip("/")) or "dataset",
        is_segmentation=False, labels={},
        counts={"training": sum(classes.values()), "validation": 0, "test": 0},
        pairs=pairs, source="imagefolder", classes=classes,
    )


def _walk_files(root):
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            yield os.path.join(dirpath, f)


# ── processing ─────────────────────────────────────────────────────────────────

def _process(dataset, spec, input_dir: str, output_dir: str) -> dict:
    from src.preprocessing.pipeline_builder import build_transforms

    is_seg = dataset.is_segmentation

    # Build TWO pipelines: one for image+label pairs (segmentation), and one for
    # image-only entries (classification, or the test split of a seg dataset,
    # which conventionally has no label). Each entry uses whichever matches, so
    # an image-only entry never asks LoadImaged for a missing 'label' key.
    paired_spec = _spec_with_keys(spec, ["image", "label"]) if is_seg else None
    image_spec = _spec_with_keys(spec, ["image"])
    paired_tf = build_transforms(paired_spec, for_preview=True) if paired_spec else None
    image_tf = build_transforms(image_spec, for_preview=True)

    from monai.transforms import SaveImage

    os.makedirs(os.path.join(output_dir, "imagesTr"), exist_ok=True)
    if is_seg:
        os.makedirs(os.path.join(output_dir, "labelsTr"), exist_ok=True)

    img_saver = SaveImage(output_dir=os.path.join(output_dir, "imagesTr"),
                          output_postfix="", output_ext=".nii.gz",
                          separate_folder=False, print_log=False, resample=False)
    lbl_saver = SaveImage(output_dir=os.path.join(output_dir, "labelsTr"),
                          output_postfix="", output_ext=".nii.gz",
                          separate_folder=False, print_log=False, resample=False) if is_seg else None

    new_training, new_test = [], []
    processed, failed, skipped = 0, 0, 0
    t0 = time.time()

    for pair in dataset.pairs:
        has_label = bool(is_seg and pair.label)
        img_path = os.path.join(input_dir, pair.image)
        lbl_path = os.path.join(input_dir, pair.label) if has_label else None

        # Skip entries whose files are absent — real datasets sometimes list
        # files that aren't present. Count and continue instead of aborting.
        if not os.path.isfile(img_path) or (has_label and not os.path.isfile(lbl_path)):
            skipped += 1
            print(f"[preprocess] SKIP (missing file): {pair.image}", flush=True)
            continue

        sample = {"image": img_path}
        if has_label:
            sample["label"] = lbl_path
        transforms = paired_tf if has_label else image_tf
        try:
            out = transforms(sample)
            stem = _stem(pair.image)
            img_saver(out["image"], meta_data=_meta(out["image"], stem))
            entry = {"image": f"./imagesTr/{stem}.nii.gz"}
            if has_label and "label" in out:
                lbl_saver(out["label"], meta_data=_meta(out["label"], stem))
                entry["label"] = f"./labelsTr/{stem}.nii.gz"
            (new_test if pair.split == "test" else new_training).append(entry)
            processed += 1
        except Exception as e:  # keep going; record the failure count
            failed += 1
            print(f"[preprocess] FAILED {pair.image}: {e}", flush=True)

    elapsed = round(time.time() - t0, 2)

    # Updated dataset.json at output root.
    out_dataset = {
        "name": f"{dataset.name}_preprocessed",
        "task_type": dataset.task_type(),
        "labels": dataset.labels,
        "training": new_training,
        "test": new_test,
        "disclaimer": DISCLAIMER,
    }
    with open(os.path.join(output_dir, "dataset.json"), "w") as f:
        json.dump(out_dataset, f, indent=2)

    result = {
        "processed": processed,
        "failed": failed,
        "skipped": skipped,
        "elapsed_seconds": elapsed,
        "task_type": dataset.task_type(),
        "source": dataset.source,
        "spec": _spec_to_dict(spec),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": DISCLAIMER,
    }
    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"[preprocess] done: processed={processed} failed={failed} "
          f"skipped={skipped} in {elapsed}s", flush=True)
    if processed == 0:
        # Nothing produced — make the job fail loudly rather than silently
        # writing an empty dataset.
        raise SystemExit(
            f"No files processed (skipped={skipped}, failed={failed}). "
            "Check that dataset.json paths match the files under the input prefix."
        )
    return result


def _stem(rel_path: str) -> str:
    base = os.path.basename(rel_path)
    for ext in (".nii.gz", ".nii", ".dcm"):
        if base.lower().endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def _meta(img, stem):
    meta = dict(getattr(img, "meta", {}) or {})
    meta["filename_or_obj"] = stem
    return meta


def _spec_with_keys(spec, keys):
    """Return a lightweight spec-like object with the same steps but given keys.

    Lets image-only entries run through an image-only pipeline even inside a
    segmentation dataset (e.g. the test split has no label).
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        task_type=getattr(spec, "task_type", "segmentation"),
        keys=list(keys),
        steps=list(spec.steps),
    )


def _spec_to_dict(spec):
    if hasattr(spec, "model_dump"):
        return spec.model_dump()
    return {"task_type": getattr(spec, "task_type", None),
            "keys": getattr(spec, "keys", None),
            "steps": [{"type": s.type, "params": s.params} for s in spec.steps]}


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description="Batch MONAI preprocessing.")
    ap.add_argument("--spec-path", default="/opt/ml/processing/code/spec.json")
    ap.add_argument("--dataset-input", default="/opt/ml/processing/input")
    ap.add_argument("--output", default="/opt/ml/processing/output")
    args = ap.parse_args(argv)

    print(f"[preprocess] {DISCLAIMER}", flush=True)
    spec = _load_spec(args.spec_path)
    dataset = _resolve_dataset(args.dataset_input)
    print(f"[preprocess] resolved: task={dataset.task_type()} "
          f"counts={dataset.counts} source={dataset.source}", flush=True)
    os.makedirs(args.output, exist_ok=True)
    result = _process(dataset, spec, args.dataset_input, args.output)
    return result


if __name__ == "__main__":
    main()
