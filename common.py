"""Shared paths, image I/O and reproducibility helpers for the enhancement study.

Every image (real, fake, or processed) passes through the same crop -> PNG
pipeline so that no format or size difference can explain a detector change.
"""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

# Root folder. On Colab point this at Google Drive so work survives disconnects.
ROOT = Path(os.environ.get("STUDY_ROOT", "/content/drive/MyDrive/aigi_study"))

DIRS = {
    "clean": ROOT / "clean",          # 512x512 PNG crops of real photos
    "variants": ROOT / "variants",    # variants/<variant_name>/<image_id>.png
    "fakes": ROOT / "fakes",          # 512x512 PNG crops of AI-generated images
    "scores": ROOT / "scores",        # one CSV of detector scores per detector
    "results": ROOT / "results",      # tables + figures for the paper
    "weights": ROOT / "weights",      # downloaded model checkpoints
}

CROP = 512            # final image side in pixels
RESIZE_SHORT = 1024   # short side before the centre crop
SEED = 2026
N_CALIB = 300         # clean reals used only to set detector thresholds


def ensure_dirs():
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def image_seed(image_id, tag=""):
    """Stable per-image seed, so noise is identical across reruns."""
    h = hashlib.sha256(f"{image_id}|{tag}|{SEED}".encode()).hexdigest()
    return int(h[:8], 16)


def load_rgb(path):
    """Read an image as an HxWx3 uint8 RGB array."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def save_png(arr, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8)).save(path, format="PNG")


def standard_crop(img):
    """Resize the short side to RESIZE_SHORT (antialiased Lanczos), centre-crop CROP."""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    w, h = img.size
    s = RESIZE_SHORT / min(w, h)
    if s < 1:  # only downsample; never upsample a small source
        img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    w, h = img.size
    if min(w, h) < CROP:
        raise ValueError(f"image too small for a {CROP}px crop: {w}x{h}")
    left, top = (w - CROP) // 2, (h - CROP) // 2
    return np.asarray(img.crop((left, top, left + CROP, top + CROP)), dtype=np.uint8)


def list_clean_ids():
    return sorted(p.stem for p in DIRS["clean"].glob("*.png"))


def get_splits(n_calib=N_CALIB):
    """Fixed calibration/test split, saved once so every stage uses the same one."""
    path = ROOT / "splits.json"
    if path.exists():
        return _limit_test(json.loads(path.read_text()))
    ids = list_clean_ids()
    if len(ids) <= n_calib:
        raise ValueError(f"need more than {n_calib} clean images, found {len(ids)}")
    rng = random.Random(SEED)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    splits = {"calib": sorted(shuffled[:n_calib]), "test": sorted(shuffled[n_calib:])}
    path.write_text(json.dumps(splits, indent=1))
    return _limit_test(splits)


def _limit_test(splits):
    """Smaller study: use only the first N_TEST test images (env STUDY_N_TEST).
    The saved split is unchanged, so the subset is deterministic and nested."""
    n = int(os.environ.get("STUDY_N_TEST", "0"))
    if n:
        splits = {"calib": splits["calib"], "test": splits["test"][:n]}
    return splits


def variant_path(variant, image_id):
    if variant == "clean":
        return DIRS["clean"] / f"{image_id}.png"
    if variant.startswith("fake_"):
        return DIRS["fakes"] / variant[5:] / f"{image_id}.png"
    return DIRS["variants"] / variant / f"{image_id}.png"


def parse_variant(name):
    """'up_realesrgan_4' -> dict(family='up', method='realesrgan', strength=4, kind='neural')."""
    from enhance import METHOD_KIND  # local import avoids a cycle
    if name == "clean":
        return {"family": "clean", "method": "clean", "strength": 0, "kind": "baseline"}
    if name.startswith("fake_"):
        return {"family": "fake", "method": name[5:], "strength": 0, "kind": "fake"}
    family, method, strength = name.split("_")
    return {"family": family, "method": method, "strength": int(strength),
            "kind": METHOD_KIND[(family, method)]}
