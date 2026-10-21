"""Fidelity of every processed image to its clean original.

PSNR and SSIM (pixel fidelity), LPIPS-AlexNet and DISTS (perceptual distance).
Output: results/fidelity.csv with one row per (image_id, variant). Resumable.
"""
import numpy as np
import pandas as pd

from common import DIRS, load_rgb, variant_path


def compute_fidelity(variants, image_ids, batch=16):
    import lpips
    import piq
    import torch
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    out_csv = DIRS["results"] / "fidelity.csv"
    old = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(
        columns=["image_id", "variant", "psnr", "ssim", "lpips", "dists"])
    have = set(zip(old.image_id, old.variant))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    dists = piq.DISTS(reduction="none").to(dev).eval()

    def to_t(arrs):
        return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).float().div(255.0).to(dev)

    rows = []
    for v in variants:
        ids = [i for i in image_ids if (i, v) not in have and variant_path(v, i).exists()]
        for k in range(0, len(ids), batch):
            chunk = ids[k:k + batch]
            ref = [load_rgb(variant_path("clean", i)) for i in chunk]
            out = [load_rgb(variant_path(v, i)) for i in chunk]
            with torch.no_grad():
                r, o = to_t(ref), to_t(out)
                lp_v = lp(o * 2 - 1, r * 2 - 1).flatten().cpu().numpy()
                di_v = dists(o, r).flatten().cpu().numpy()
            for j, iid in enumerate(chunk):
                rows.append({
                    "image_id": iid, "variant": v,
                    "psnr": peak_signal_noise_ratio(ref[j], out[j], data_range=255),
                    "ssim": structural_similarity(ref[j], out[j], channel_axis=2, data_range=255),
                    "lpips": float(lp_v[j]), "dists": float(di_v[j]),
                })
        if ids:
            print(f"{v}: {len(ids)} images scored")
            # checkpoint after each variant so a disconnect loses little
            pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(out_csv, index=False)
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(out_csv, index=False)
    return df
