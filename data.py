"""Build the clean real-photo set and the (sanity-check) fake set.

Real photos: RAISE-1k (uncompressed camera originals) is the recommended source.
Either point `prepare_from_folder` at images you already have, or give
`download_raise` the RAISE CSV of download links. DIV2K is a fallback.
"""
import csv
import random
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import requests

from common import DIRS, SEED, ensure_dirs, save_png, standard_crop

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _read_any(path):
    """Read 8- or 16-bit images (RAISE TIFFs can be 16-bit) as uint8 RGB."""
    import cv2
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    if arr is None:
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    if arr.dtype == np.uint16:
        arr = (arr / 257.0).round().astype(np.uint8)
    return arr[:, :, ::-1].copy()  # BGR -> RGB


def _safe_id(prefix, stem):
    keep = "".join(c if c.isalnum() else "-" for c in stem)
    return f"{prefix}{keep}"


def prepare_from_folder(src_dir, max_n=1000, prefix="r", out_dir=None):
    """Crop every image in src_dir (recursively) to the standard 512 px PNG."""
    ensure_dirs()
    out_dir = Path(out_dir or DIRS["clean"])
    files = sorted(p for p in Path(src_dir).rglob("*") if p.suffix.lower() in IMG_EXT)
    done = 0
    for p in files:
        if done >= max_n:
            break
        out = out_dir / f"{_safe_id(prefix, p.stem)}.png"
        if out.exists():
            done += 1
            continue
        try:
            save_png(standard_crop(_read_any(p)), out)
            done += 1
        except Exception as e:  # skip unreadable or too-small files
            print("skip", p.name, e)
    print(f"{done} images ready in {out_dir}")
    return done


def download_raise(csv_path, max_n=1000, url_column=None):
    """Stream RAISE TIFFs one at a time: download -> crop -> save -> delete.

    The RAISE download page provides a CSV with one row per image and a column
    of TIFF links. Each 30 MB TIFF is deleted right after cropping, so disk use
    stays small.
    """
    ensure_dirs()
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("empty CSV")
    if url_column is None:
        cands = [c for c in rows[0] if "tif" in c.lower()]
        if not cands:
            raise ValueError(f"no TIFF column found; columns are {list(rows[0])}. Pass url_column=...")
        url_column = cands[0]
    tmp = Path(tempfile.mkdtemp())
    done = 0
    for i, row in enumerate(rows):
        if done >= max_n:
            break
        url = row[url_column].strip()
        name = (row.get("File") or Path(url).stem or f"img{i:04d}").strip()
        out = DIRS["clean"] / f"{_safe_id('r', Path(name).stem)}.png"
        if out.exists():
            done += 1
            continue
        try:
            local = tmp / "img.tif"
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(local, "wb") as fh:
                    shutil.copyfileobj(r.raw, fh)
            save_png(standard_crop(_read_any(local)), out)
            done += 1
            if done % 50 == 0:
                print(done, "done")
        except Exception as e:
            print("skip", url, e)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{done} RAISE images ready")
    return done


DIV2K_URLS = [
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",   # 100 images
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",   # 800 images
]


def download_div2k(max_n=900):
    """Fallback real set if the RAISE server is unavailable (about 900 photos)."""
    tmp = Path(tempfile.mkdtemp())
    for url in DIV2K_URLS:
        z = tmp / Path(url).name
        print("downloading", url)
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(z, "wb") as fh:
                shutil.copyfileobj(r.raw, fh)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(tmp / "x")
        z.unlink()
    n = prepare_from_folder(tmp / "x", max_n=max_n, prefix="d")
    shutil.rmtree(tmp, ignore_errors=True)
    return n


def prepare_fakes(src_dir, per_generator=100):
    """Synthbuster: one sub-folder per generator. Sample and crop like the reals.

    Saved as fakes/<generator>/<id>.png. Only used to show each detector works
    (TPR at the threshold), never to set thresholds.
    """
    rng = random.Random(SEED)
    src_dir = Path(src_dir)
    gens = sorted(d for d in src_dir.iterdir() if d.is_dir())
    for g in gens:
        files = sorted(p for p in g.rglob("*") if p.suffix.lower() in IMG_EXT)
        rng.shuffle(files)
        gname = "".join(c for c in g.name.lower() if c.isalnum())
        n = 0
        for p in files:
            if n >= per_generator:
                break
            out = DIRS["fakes"] / gname / f"{_safe_id('f', p.stem)}.png"
            if not out.exists():
                try:
                    save_png(standard_crop(_read_any(p)), out)
                except Exception as e:
                    print("skip", p.name, e)
                    continue
            n += 1
        print(f"{gname}: {n} fakes")


def fake_generators():
    return sorted(d.name for d in DIRS["fakes"].iterdir() if d.is_dir()) if DIRS["fakes"].exists() else []
