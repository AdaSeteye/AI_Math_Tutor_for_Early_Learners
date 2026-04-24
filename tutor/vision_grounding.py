"""
Task 5: visual grounding — count objects in rendered count images.

* **Blob baseline** (default, ~0 extra MB): morphology on non-background pixels; tuned for
  ``tutor.visuals.render_count_image`` (cream BG, bottom caption). No curriculum leak.
* **Open-vocab OWL-ViT** (optional): ``google/owlvit-base-patch32`` if ``torch`` +
  ``transformers`` and ``TUTOR_OWLVIT=1`` — **large** download, not in the 75 MB budget.
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import Literal

import numpy as np
from PIL import Image
from scipy import ndimage

Method = Literal["blob", "owlvit", "auto"]

_owl: tuple[object, object] | None = None


def _fg_mask(img: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    bg = np.array([250.0, 248.0, 240.0], dtype=np.float32)
    d = np.sqrt(((img.astype(np.float32) - bg) ** 2).sum(axis=2))
    fg = d > 22.0
    fg[-max(1, int(0.14 * h)) :, :] = False
    return fg


def count_blobs_baseline(png: bytes, object_label: str = "circle") -> int:
    """
    Baseline that does **not** read the curriculum. Uses distance-transform local
    maxima (one peak ≈ one object) on the mask; tuned for our ``render_count_image``
    style (goat with multiple parts, circles, dots). ``object_label`` is reserved
    for future per-label tuning; same path for all.
    """
    _ = object_label
    img = np.array(Image.open(BytesIO(png)).convert("RGB"), dtype=np.float32)
    h, w, _ = img.shape
    fg = _fg_mask(img)
    dt = ndimage.distance_transform_edt(fg)
    foot, thr = 30, 7.0
    mx = ndimage.maximum_filter(
        dt, footprint=np.ones((foot, foot), dtype=bool)
    )
    lm = (dt == mx) & (dt > thr) & fg
    lm[-max(1, int(0.14 * h)) :, :] = False
    _lab, nfeat = ndimage.label(lm)
    return int(nfeat)


def count_owlvit_google(png: bytes, object_label: str) -> int | None:
    """
    OWL-ViT: zero-shot text query, count boxes. Needs ``TUTOR_OWLVIT=1`` to run.
    Returns None if deps missing, import fails, or you prefer to skip the download.
    """
    if os.environ.get("TUTOR_OWLVIT", "").lower() not in ("1", "true", "yes"):
        return None
    try:
        import torch
        from transformers import OwlViTForObjectDetection, OwlViTProcessor
    except Exception:
        return None
    global _owl
    mid = "google/owlvit-base-patch32"
    if _owl is None:
        _owl = (OwlViTProcessor.from_pretrained(mid), OwlViTForObjectDetection.from_pretrained(mid))
    proc, model = _owl
    im = Image.open(BytesIO(png)).convert("RGB")
    lab = (object_label or "object").lower()
    if lab == "goat":
        t = "a photo of a goat"
    elif lab in ("dot", "circle", "shape"):
        t = f"a photo of a {lab}"
    else:
        t = f"a photo of a {lab}"
    with torch.inference_mode():
        inputs = proc(text=[[t]], images=im, return_tensors="pt")
        o = model(**inputs)
    ts = torch.as_tensor([im.size[::-1]]).float()  # H, W
    res = proc.post_process_object_detection(o, target_sizes=ts, threshold=0.2)
    boxes = res[0].get("boxes", [])
    return int(len(boxes)) if boxes is not None else 0


def grounded_count(
    png: bytes, object_label: str, method: Method = "auto"
) -> tuple[int, str]:
    """
    Returns (count, method_tag). *auto* uses OWL-ViT if env+deps and it finds boxes;
    else blob baseline.
    """
    if method == "owlvit":
        o = count_owlvit_google(png, object_label)
        if o is None:
            return int(count_blobs_baseline(png, object_label)), "blob (owl unavailable)"
        return int(o), "owlvit"
    if method == "blob":
        return int(count_blobs_baseline(png, object_label)), "blob"
    o = count_owlvit_google(png, object_label)
    if o is not None and o > 0:
        return int(o), "owlvit"
    return int(count_blobs_baseline(png, object_label)), "blob"
