"""blurr.vision._core — Shared helpers for vision tasks."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import torch
from fastai.torch_core import TensorBase
from fastcore.foundation import L
from fasttransform import Transform
from PIL import Image


class VisionInput(TensorBase):
    """Marker type for decoded vision inputs."""


# ===================================================================
# Image loading utilities
# ===================================================================

def _is_url(v: str) -> bool:
    try:
        parsed = urlparse(v)
        return parsed.scheme in {"http", "https"}
    except Exception:
        return False


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu()
    if x.ndim == 3 and x.shape[0] in {1, 3, 4}:  # CHW -> HWC
        x = x.permute(1, 2, 0)
    x = x.numpy()
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 1) if x.max() <= 1.0 else np.clip(x, 0, 255)
        x = (x * 255).astype(np.uint8) if x.max() <= 1.0 else x.astype(np.uint8)
    return Image.fromarray(x)


def load_image(item: Any, *, mode: str = "RGB", timeout: int = 10) -> Image.Image:
    """Load an image from common input formats.

    Supports:
      - local file paths
      - URLs
      - PIL images
      - numpy arrays
      - torch tensors
      - HuggingFace datasets image dicts ({"path": ..., "bytes": ...})
      - raw bytes
    """
    if isinstance(item, Image.Image):
        return item.convert(mode)

    if isinstance(item, torch.Tensor):
        return _tensor_to_pil(item).convert(mode)

    if isinstance(item, np.ndarray):
        if item.dtype != np.uint8:
            item = np.clip(item, 0, 1) if item.max() <= 1.0 else np.clip(item, 0, 255)
            item = (item * 255).astype(np.uint8) if item.max() <= 1.0 else item.astype(np.uint8)
        return Image.fromarray(item).convert(mode)

    if isinstance(item, (bytes, bytearray)):
        return Image.open(BytesIO(item)).convert(mode)

    if isinstance(item, dict):
        if "image" in item:
            return load_image(item["image"], mode=mode, timeout=timeout)
        if item.get("bytes") is not None:
            return Image.open(BytesIO(item["bytes"])).convert(mode)
        if item.get("path"):
            return load_image(item["path"], mode=mode, timeout=timeout)

    if isinstance(item, (str, Path)):
        item = str(item)
        if _is_url(item):
            with urlopen(item, timeout=timeout) as resp:
                return Image.open(BytesIO(resp.read())).convert(mode)
        return Image.open(item).convert(mode)

    raise TypeError(f"Unsupported image type: {type(item)}")


def apply_image_augmentations(image: Image.Image, augmentations: Any | None = None) -> Image.Image:
    """Apply optional augmentation pipelines.

    Supports torchvision-style callables and albumentations-style outputs
    (dict with an `image` key).
    """
    if augmentations is None:
        return image

    pipelines = augmentations if isinstance(augmentations, Sequence) and not callable(augmentations) else [augmentations]
    out: Any = image

    for aug in pipelines:
        if aug is None:
            continue
        out = aug(out)
        if isinstance(out, dict) and "image" in out:
            out = out["image"]

    if isinstance(out, Image.Image):
        return out
    if isinstance(out, torch.Tensor):
        return _tensor_to_pil(out)
    if isinstance(out, np.ndarray):
        return load_image(out)

    raise TypeError(f"Unsupported augmentation output type: {type(out)}")


# ===================================================================
# fastai batch transforms
# ===================================================================

class VisionBatchTransform(Transform):
    """Apply a HuggingFace image processor on-the-fly in ``before_batch``."""

    def __init__(
        self,
        image_processor,
        *,
        include_labels: bool = True,
        is_multilabel: bool = False,
        labels: list[str] | None = None,
        augmentations: Any | None = None,
        processor_call_kwargs: dict[str, Any] | None = None,
    ):
        self.image_processor = image_processor
        self.include_labels = include_labels
        self.is_multilabel = is_multilabel
        self.labels = labels
        self.augmentations = augmentations
        self.processor_call_kwargs = processor_call_kwargs or {}

    def _normalize_sample(self, sample):
        if isinstance(sample, tuple):
            return sample
        if isinstance(sample, list):
            return tuple(sample)
        return (sample,)

    def encodes(self, samples):
        samples = L(samples)
        if len(samples) == 0:
            return samples

        samples = [self._normalize_sample(s) for s in samples]
        raw_images = [s[0] for s in samples]

        images = [
            apply_image_augmentations(load_image(img), self.augmentations)
            for img in raw_images
        ]

        inputs = self.image_processor(
            images=images,
            return_tensors="pt",
            **self.processor_call_kwargs,
        )

        updated = []
        for idx, sample in enumerate(samples):
            inp_dict = {k: v[idx] for k, v in inputs.items()}
            targets = sample[1:]

            if self.include_labels and len(targets) > 0:
                if self.is_multilabel:
                    inp_dict["labels"] = targets[0].float()
                else:
                    inp_dict["labels"] = targets[0]

            if len(targets) > 0:
                updated.append((inp_dict, *targets))
            else:
                updated.append((inp_dict,))

        return updated


class VisionBatchDecodeTransform(Transform):
    """Cast pixel values to ``VisionInput`` for fastai show dispatch."""

    def __init__(self, image_processor):
        self.image_processor = image_processor

    def decodes(self, x: dict) -> VisionInput:
        return VisionInput(x["pixel_values"])
