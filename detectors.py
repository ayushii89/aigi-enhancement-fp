"""AI-generated-image detectors from three families, all scored as 'higher = more fake'.

  UnivFD  - CLIP ViT-L/14 features + linear probe   (Ojha et al., CVPR 2023)
  NPR     - low-level up-sampling artifacts         (Tan et al., CVPR 2024)
  RIGID   - training-free, DINOv2 noise sensitivity (He et al., 2024)

Checkpoints for UnivFD and NPR are the authors' own files, shipped inside their
GitHub repositories (cloned on first use). Each detector writes scores/<name>.csv with
columns image_id, variant, score. Resumable per variant.
"""
import subprocess
import sys

import numpy as np
import pandas as pd

from common import DIRS, load_rgb, set_seed, variant_path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _repo(url):
    """Shallow-clone an authors' repository into weights/ once; return its path."""
    dst = DIRS["weights"] / url.rstrip("/").split("/")[-1]
    if not dst.exists():
        subprocess.run(["git", "clone", "--depth", "1", url, str(dst)], check=True)
    return dst


def _clean_state_dict(sd):
    for key in ("model", "state_dict", "net"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def _batch_tensor(imgs, crop, mean, std):
    """uint8 HxWx3 list -> normalised NCHW tensor; centre crop to `crop` (no resampling)."""
    import torch
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().div(255.0)
    if crop:
        h, w = x.shape[-2:]
        t, l = (h - crop) // 2, (w - crop) // 2
        x = x[:, :, t:t + crop, l:l + crop]
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    return (x - m) / s


class UnivFD:
    name = "univfd"

    def __init__(self):
        import open_clip
        import torch
        self.dev = _device()
        # QuickGELU matches the original OpenAI CLIP that UnivFD was trained on.
        model, _, _ = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai")
        self.clip = model.to(self.dev).eval()
        fc_path = _repo("https://github.com/Yuheng-Li/UniversalFakeDetect") / "pretrained_weights" / "fc_weights.pth"
        sd = _clean_state_dict(torch.load(fc_path, map_location="cpu"))
        sd = {k.split(".")[-1]: v for k, v in sd.items()}  # 'fc.weight' -> 'weight'
        self.fc = torch.nn.Linear(sd["weight"].shape[1], 1)
        self.fc.load_state_dict(sd)
        self.fc = self.fc.to(self.dev).eval()

    def __call__(self, imgs):
        import torch
        x = _batch_tensor(imgs, 224, CLIP_MEAN, CLIP_STD).to(self.dev)
        with torch.no_grad():
            f = self.clip.encode_image(x).float()
            return self.fc(f).flatten().cpu().numpy()


class NPR:
    name = "npr"
    REPO = "https://github.com/chuangchuangtan/NPR-DeepfakeDetection"

    def __init__(self):
        import torch
        self.dev = _device()
        code = _repo(self.REPO)
        sys.path.insert(0, str(code))
        from networks.resnet import resnet50  # from the NPR repository
        self.model = resnet50(num_classes=1)
        sd = _clean_state_dict(torch.load(code / "NPR.pth", map_location="cpu"))
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"NPR weights do not match the network: missing {missing[:5]}")
        if unexpected:
            print("NPR: ignoring unexpected keys", unexpected[:5])
        self.model = self.model.to(self.dev).eval()
        sys.path.remove(str(code))

    def __call__(self, imgs):
        import torch
        # NPR's own test protocol (test.py: no_resize=False, no_crop=True; loadSize=256):
        # PIL bilinear resize to 256x256, no crop, ImageNet normalisation.
        from PIL import Image
        imgs = [np.asarray(Image.fromarray(im).resize((256, 256), Image.BILINEAR)) for im in imgs]
        x = _batch_tensor(imgs, None, IMAGENET_MEAN, IMAGENET_STD).to(self.dev)
        with torch.no_grad():
            return self.model(x).flatten().cpu().numpy()


class RIGID:
    """Score = 1 - cosine(f(x), f(x + lambda*noise)), averaged over a few noise draws."""
    name = "rigid"

    def __init__(self, noise=0.05, draws=3, backbone="dinov2_vitl14"):
        import torch
        self.dev = _device()
        code = _repo("https://github.com/facebookresearch/dinov2")
        # Local clone avoids torch.hub's GitHub API calls; weights still download once.
        self.model = torch.hub.load(str(code), backbone, source="local").to(self.dev).eval()
        self.noise, self.draws = noise, draws

    def __call__(self, imgs):
        import torch
        x = _batch_tensor(imgs, 224, IMAGENET_MEAN, IMAGENET_STD).to(self.dev)
        g = torch.Generator(device=self.dev).manual_seed(2026)  # same noise for every batch
        with torch.no_grad():
            f0 = torch.nn.functional.normalize(self.model(x).float(), dim=-1)
            sims = []
            for _ in range(self.draws):
                eps = torch.randn(x.shape, generator=g, device=self.dev)
                f1 = torch.nn.functional.normalize(self.model(x + self.noise * eps).float(), dim=-1)
                sims.append((f0 * f1).sum(-1))
            return (1 - torch.stack(sims).mean(0)).cpu().numpy()


DETECTORS = {"univfd": UnivFD, "npr": NPR, "rigid": RIGID}


def ids_for(variant, image_ids):
    if variant.startswith("fake_"):
        folder = DIRS["fakes"] / variant[5:]
        return sorted(p.stem for p in folder.glob("*.png"))
    return [i for i in image_ids if variant_path(variant, i).exists()]


def score_all(detector_names, variants, image_ids, batch=16):
    """Score every variant (plus 'clean' and fake_* sets) with each detector."""
    set_seed()
    for name in detector_names:
        out_csv = DIRS["scores"] / f"{name}.csv"
        old = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["image_id", "variant", "score"])
        done = set(old.variant.unique())
        todo = [v for v in variants if v not in done]
        if not todo:
            print(f"{name}: all variants already scored")
            continue
        det = DETECTORS[name]()
        frames = [old]
        for v in todo:
            ids = ids_for(v, image_ids)
            if not ids:
                print(f"{name}: {v} has no images yet, skipped")
                continue
            scores = []
            for k in range(0, len(ids), batch):
                chunk = ids[k:k + batch]
                scores.append(det([load_rgb(variant_path(v, i)) for i in chunk]))
            frames.append(pd.DataFrame({"image_id": ids, "variant": v, "score": np.concatenate(scores)}))
            pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False)
            print(f"{name}: {v} ({len(ids)} images)")
        del det
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def load_scores():
    frames = []
    for p in sorted(DIRS["scores"].glob("*.csv")):
        d = pd.read_csv(p)
        d["detector"] = p.stem
        frames.append(d)
    return pd.concat(frames, ignore_index=True)
