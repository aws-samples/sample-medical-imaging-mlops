"""Pydantic models for the preprocessing pipeline.

A single PipelineSpec is the source of truth shared by the local preview runner
and (later) the SageMaker batch job. Step params are validated per-type.

RESEARCH USE ONLY — preprocessing transforms imaging data for research and
development. It performs no diagnosis.
"""
from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

# Supported step types (must match STEP_CATALOG in src/preprocessing/pipeline_builder.py)
StepType = Literal[
    "load",
    "ensure_channel_first",
    "orientation",
    "resample",
    "intensity_window",
    "scale_intensity",
    "scale_intensity_percentiles",
    "clip_percentiles",
    "threshold_intensity",
    "histogram_normalize",
    "normalize",
    "gaussian_smooth",
    "median_smooth",
    "crop_foreground",
    "center_crop",
    "spatial_pad",
    "divisible_pad",
    "resize",
    "zoom",
    "flip",
    "rotate90",
    "cast",
    "ensure_type",
    "save",
]

# Steps that must be first / last when present.
_FIRST_ONLY = "load"
_LAST_ONLY = "save"


class Step(BaseModel):
    """One preprocessing step: a transform type plus its parameters.

    Params are intentionally a free-form dict validated by the builder against
    the catalog, so new MONAI transforms can be added without a schema churn.
    Ordering/enum validity is enforced here and in PipelineSpec.
    """

    type: StepType
    params: dict = Field(default_factory=dict)

    @field_validator("params")
    @classmethod
    def _params_is_mapping(cls, v):
        if not isinstance(v, dict):
            raise ValueError("params must be an object")
        return v


class DatasetPrep(BaseModel):
    enabled: bool = False
    split: dict = Field(default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15})
    seed: int = 42

    @model_validator(mode="after")
    def _check_split(self):
        if self.enabled:
            total = sum(float(v) for v in self.split.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"dataset_prep.split must sum to 1.0 (got {total})")
        return self


class PipelineSpec(BaseModel):
    task_type: Literal["segmentation", "classification"] = "segmentation"
    keys: List[str] = Field(default_factory=lambda: ["image", "label"])
    steps: List[Step]
    dataset_prep: Optional[DatasetPrep] = None

    @field_validator("steps")
    @classmethod
    def _non_empty(cls, v):
        if not v:
            raise ValueError("pipeline must contain at least one step")
        return v

    @model_validator(mode="after")
    def _check_ordering_and_keys(self):
        types = [s.type for s in self.steps]

        # 'load' must be first if present; 'save' must be last if present.
        if _FIRST_ONLY in types and types[0] != _FIRST_ONLY:
            raise ValueError("'load' must be the first step")
        if _LAST_ONLY in types and types[-1] != _LAST_ONLY:
            raise ValueError("'save' must be the last step")
        if types.count(_FIRST_ONLY) > 1:
            raise ValueError("'load' may only appear once")
        if types.count(_LAST_ONLY) > 1:
            raise ValueError("'save' may only appear once")

        # keys must be consistent with task type
        if self.task_type == "classification" and "label" in self.keys:
            raise ValueError("classification pipelines must not include a 'label' key")
        if "image" not in self.keys:
            raise ValueError("keys must include 'image'")
        return self


class PreviewRequest(BaseModel):
    """Local single-image preview request."""

    input_ref: str = Field(..., min_length=1, max_length=4096)
    spec: PipelineSpec


class SubmitRequest(BaseModel):
    """Batch (SageMaker Processing Job) request. Not yet served by the backend."""

    input_s3_uri: str = Field(..., min_length=1, max_length=4096)
    output_s3_uri: str = Field(..., min_length=1, max_length=4096)
    image_uri: str = Field("", max_length=2048)
    role_arn: str = Field("", max_length=2048)
    instance_type: str = Field("ml.m5.xlarge", max_length=64)
    spec: PipelineSpec
