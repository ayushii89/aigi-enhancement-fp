"""Degrade -> restore operators: classical vs neural, at graded strengths.

Every output has the clean crop as its reference, so fidelity (PSNR/SSIM/LPIPS)
can be measured and neural vs classical compared at matched fidelity.

Variant names are '<family>_<method>_<strength>', e.g. 'up_realesrgan_4'
(upscaling, factor 4) or 'dn_swinir_25' (denoising, sigma 25 on a 0-255 scale).
Runs are resumable: an output that already exists on disk is skipped.
"""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from common import CROP, DIRS, ROOT, image_seed, load_rgb, save_png, variant_path

# ---------------------------------------------------------------- registry
UP_SCALES = [2, 4]
DN_SIGMAS = [15, 25, 50]

METHOD_KIND = {
    ("up", "bicubic"): "classical",
    ("up", "lanczosusm"): "classical",
    ("up", "realesrgan"): "neural",
    ("up", "swinir"): "neural",
    ("up", "sdx4"): "neural",
    ("dn", "noisy"): "baseline",     # degraded input, no restoration
    ("dn", "nlm"): "classical",
    ("dn", "bm3d"): "classical",
    ("dn", "swinir"): "neural",
    ("dn", "restormer"): "neural",
}

GH = "https://github.com"
WEIGHTS = {
    ("realesrgan", 2): f"{GH}/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    ("realesrgan", 4): f"{GH}/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    ("swinir_sr", 2): f"{GH}/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth",
    ("swinir_sr", 4): f"{GH}/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
    ("swinir_dn", 15): f"{GH}/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth",
    ("swinir_dn", 25): f"{GH}/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth",
    ("swinir_dn", 50): f"{GH}/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth",
    ("restormer_dn", 0): f"{GH}/swz30/Restormer/releases/download/v1.0/gaussian_color_denoising_blind.pth",
}


BM3D_VARIANTS = ["dn_bm3d_25"]

# Smaller study (about 2 GPU-hours on 300 test images): two upscaling factors and two
# noise levels, two classical and two neural restorers per family.
SMALL_VARIANTS = [
    "up_bicubic_2", "up_lanczosusm_2", "up_realesrgan_2", "up_swinir_2",
    "up_bicubic_4", "up_lanczosusm_4", "up_realesrgan_4", "up_swinir_4",
    "dn_noisy_15", "dn_nlm_15", "dn_swinir_15",
    "dn_noisy_50", "dn_nlm_50", "dn_swinir_50",
]


def all_variants(include_slow=True):
    v = []
    for s in UP_SCALES:
        v += [f"up_bicubic_{s}", f"up_lanczosusm_{s}", f"up_realesrgan_{s}", f"up_swinir_{s}"]
    if include_slow:
        v.append("up_sdx4_4")
    for sg in DN_SIGMAS:
        v += [f"dn_noisy_{sg}", f"dn_nlm_{sg}", f"dn_swinir_{sg}"]
        if include_slow:
            v.append(f"dn_restormer_{sg}")
    # BM3D (~36 s/image on Colab CPU) is run separately on a subset at sigma 25: see BM3D_VARIANTS
    return v


# ------------------------------------------------------------- degradations
def downscale(img, s):
    """Antialiased bicubic downscale (Pillow scales the filter support on reduction)."""
    return np.asarray(Image.fromarray(img).resize((CROP // s, CROP // s), Image.BICUBIC))


def add_noise(img, sigma, image_id):
    """Additive Gaussian noise, same draw for an image on every run, quantised to uint8."""
    rng = np.random.default_rng(image_seed(image_id, f"noise{sigma}"))
    noisy = img.astype(np.float32) + rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(noisy.round(), 0, 255).astype(np.uint8)


# ------------------------------------------------------- classical restorers
def up_bicubic(lr, s):
    return np.asarray(Image.fromarray(lr).resize((CROP, CROP), Image.BICUBIC))


def up_lanczosusm(lr, s):
    """Lanczos upscale + unsharp mask: a strong classical 'enhance' baseline."""
    up = Image.fromarray(lr).resize((CROP, CROP), Image.LANCZOS)
    return np.asarray(up.filter(ImageFilter.UnsharpMask(radius=1.5, percent=80, threshold=2)))


def _nlm(noisy, h):
    import cv2
    bgr = noisy[:, :, ::-1]
    out = cv2.fastNlMeansDenoisingColored(np.ascontiguousarray(bgr), None, h, h, 7, 21)
    return out[:, :, ::-1].copy()


def tune_nlm(calib_ids, n=10):
    """Pick the NLM strength h per sigma that maximises PSNR on calibration images.

    Makes the classical baseline as strong as it can be, so a neural-vs-classical
    gap cannot be blamed on a weak baseline. Cached in nlm_h_v2.json.
    (v1 used a grid whose optimum sat at its lower edge; outputs made with it are removed.)
    """
    from skimage.metrics import peak_signal_noise_ratio as psnr
    path = ROOT / "nlm_h_v2.json"
    old = ROOT / "nlm_h.json"
    if old.exists():  # outputs from the v1 grid used a sub-optimal h: delete and redo
        import shutil
        for d in DIRS["variants"].glob("dn_nlm_*"):
            shutil.rmtree(d)
        old.unlink()
        print("Removed NLM outputs made with the old grid; they will be regenerated.")
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text()).items()}
    best = {}
    for sg in DN_SIGMAS:
        scores = {}
        for mult in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.1):
            h = sg * mult
            vals = []
            for iid in calib_ids[:n]:
                clean = load_rgb(variant_path("clean", iid))
                vals.append(psnr(clean, _nlm(add_noise(clean, sg, iid), h), data_range=255))
            scores[h] = float(np.mean(vals))
        best[sg] = max(scores, key=scores.get)
        print(f"sigma {sg}: best h = {best[sg]:.1f} (PSNR {scores[best[sg]]:.2f} dB)")
    path.write_text(json.dumps(best))
    return best


def _bm3d(noisy, sigma):
    import bm3d
    out = bm3d.bm3d_rgb(noisy.astype(np.float32) / 255.0, sigma_psd=sigma / 255.0)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------- neural restorers
def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _weights(key):
    import torch
    url = WEIGHTS[key]
    dst = DIRS["weights"] / Path(url).name
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(dst))
    return dst


_EXTRA_ARCHES = False


def _install_extra_arches():
    """Register spandrel's extra architectures (Restormer) exactly once.
    Calling install() a second time raises DuplicateArchitectureError."""
    global _EXTRA_ARCHES
    if _EXTRA_ARCHES:
        return
    try:
        import spandrel_extra_arches
        spandrel_extra_arches.install()
    except ImportError:
        pass
    except Exception as e:  # already registered earlier in this session
        if "Duplicate" not in type(e).__name__:
            raise
    _EXTRA_ARCHES = True


class SpandrelModel:
    """Loads Real-ESRGAN / SwinIR / Restormer checkpoints through spandrel."""

    def __init__(self, key, pad_multiple=16):
        import torch
        from spandrel import ModelLoader
        _install_extra_arches()
        self.device = _device()
        self.desc = ModelLoader(device=self.device).load_from_file(str(_weights(key)))
        self.desc.eval()
        self.scale = int(self.desc.scale)
        self.pad = pad_multiple
        self.torch = torch

    def __call__(self, img):
        torch = self.torch
        h, w = img.shape[:2]
        x = torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1).float().div(255.0)[None].to(self.device)
        ph, pw = (-h) % self.pad, (-w) % self.pad
        if ph or pw:
            x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
        with torch.no_grad():
            y = self.desc(x)
        y = y[:, :, : h * self.scale, : w * self.scale]
        y = y.clamp(0, 1).mul(255.0).round().byte()[0].permute(1, 2, 0).cpu().numpy()
        return y


class SDUpscaler:
    """Stable Diffusion x4 upscaler (diffusion-based super-resolution), 128 -> 512."""

    def __init__(self, steps=25, noise_level=20):
        import torch
        from diffusers import StableDiffusionUpscalePipeline
        dev = _device()
        from diffusers.utils import logging as dlog
        dlog.set_verbosity_error()
        dtype = torch.float16 if dev == "cuda" else torch.float32
        name = "stabilityai/stable-diffusion-x4-upscaler"
        try:
            pipe = StableDiffusionUpscalePipeline.from_pretrained(name, dtype=dtype)
        except TypeError:  # older diffusers
            pipe = StableDiffusionUpscalePipeline.from_pretrained(name, torch_dtype=dtype)
        self.pipe = pipe.to(dev)
        self.pipe.set_progress_bar_config(disable=True)
        self.steps, self.noise_level, self.dev, self.torch = steps, noise_level, dev, torch

    def __call__(self, lr, image_id):
        g = self.torch.Generator(self.dev).manual_seed(image_seed(image_id, "sdx4"))
        out = self.pipe(prompt="", image=Image.fromarray(lr), num_inference_steps=self.steps,
                        noise_level=self.noise_level, generator=g).images[0]
        out = out.convert("RGB")
        if out.size != (CROP, CROP):
            out = out.resize((CROP, CROP), Image.BICUBIC)
        return np.asarray(out)


# ------------------------------------------------------------------ runner
def _make_op(variant, nlm_h=None):
    """Return f(clean_img, image_id) -> output image for one variant."""
    family, method, strength = variant.split("_")
    strength = int(strength)
    if family == "up":
        s = strength
        if method == "bicubic":
            return lambda img, iid: up_bicubic(downscale(img, s), s)
        if method == "lanczosusm":
            return lambda img, iid: up_lanczosusm(downscale(img, s), s)
        if method == "realesrgan":
            m = SpandrelModel(("realesrgan", s))
            return lambda img, iid: m(downscale(img, s))
        if method == "swinir":
            m = SpandrelModel(("swinir_sr", s))
            return lambda img, iid: m(downscale(img, s))
        if method == "sdx4":
            assert s == 4, "the SD upscaler is x4 only"
            m = SDUpscaler()
            return lambda img, iid: m(downscale(img, 4), iid)
    if family == "dn":
        sg = strength
        if method == "noisy":
            return lambda img, iid: add_noise(img, sg, iid)
        if method == "nlm":
            h = (nlm_h or {}).get(sg, 0.9 * sg)
            return lambda img, iid: _nlm(add_noise(img, sg, iid), h)
        if method == "bm3d":
            return lambda img, iid: _bm3d(add_noise(img, sg, iid), sg)
        if method == "swinir":
            m = SpandrelModel(("swinir_dn", sg))
            return lambda img, iid: m(add_noise(img, sg, iid))
        if method == "restormer":
            m = SpandrelModel(("restormer_dn", 0), pad_multiple=8)
            return lambda img, iid: m(add_noise(img, sg, iid))
    raise ValueError(f"unknown variant {variant}")


def run_variant(variant, image_ids, nlm_h=None):
    todo = [i for i in image_ids if not variant_path(variant, i).exists()]
    if not todo:
        print(f"{variant}: already complete")
        return
    op = _make_op(variant, nlm_h)
    t0 = time.time()
    for k, iid in enumerate(todo, 1):
        out = op(load_rgb(variant_path("clean", iid)), iid)
        assert out.shape == (CROP, CROP, 3), f"{variant} produced {out.shape}"
        save_png(out, variant_path(variant, iid))
        if k == 5:
            per = (time.time() - t0) / 5
            print(f"{variant}: {per:.2f} s/image, about {per * len(todo) / 60:.0f} min for {len(todo)}")
    print(f"{variant}: done ({len(todo)} new) in {(time.time() - t0) / 60:.1f} min")
    try:
        import torch
        del op
        torch.cuda.empty_cache()
    except Exception:
        pass


def run_variants(variants, image_ids, nlm_h=None):
    failed = {}
    for v in variants:
        try:
            run_variant(v, image_ids, nlm_h)
        except Exception as e:  # keep going; report at the end
            failed[v] = repr(e)
            print(f"!! {v} failed: {e!r}")
    if failed:
        print("\nFailed variants (fix and rerun; finished images are kept):")
        for v, e in failed.items():
            print(" ", v, e)
    return failed
