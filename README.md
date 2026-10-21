# Which Enhancements Keep a Photo Real?

Code for the paper *"Which Enhancements Keep a Photo Real? A Controlled Study of Neural Restoration and False Positives in AI-Generated Image Detectors"* (IICIT-2027).

Real camera photographs are degraded (downscaling or Gaussian noise), restored with **classical** and **neural** methods, and scored by three AI-generated-image detectors (UnivFD, NPR, RIGID) at thresholds fixed to a 5% false-positive rate on clean photographs. Neural and classical restoration are then compared at matched perceptual fidelity (LPIPS).

## Quick start (Google Colab, T4 GPU)

1. Open `notebooks/aigi_study_small.ipynb` in Colab and select **Runtime → Change runtime type → T4 GPU**.
2. Put the RAISE-1k CSV of download links at `MyDrive/aigi_study/RAISE_1k.csv` (from http://loki.disi.unitn.it/RAISE/download.html).
3. Run the notebook top to bottom. Every stage is resumable: after a disconnect, rerun Step 0 and continue.

Approximate run time on a T4: 2–3 hours for 300 test images.

## Reproducing the paper's numbers

1. Run `notebooks/aigi_study_small.ipynb` top to bottom (Steps 0–5). This writes every table, figure and `per_image.csv` to `MyDrive/aigi_study/results/`.
2. The outputs of the reported run are already in `data/` (per-image scores and flags, fidelity values, all tables and figures, the split and exclusion manifests). RAISE-1k and Synthbuster images are not redistributed; `data/splits.json` and `per_image.csv` list the image identifiers used.
3. Re-run the analysis from these saved scores, without a GPU or any image, and compare with `data/results/`:

```bash
pip install pandas==2.2.3 numpy==2.1.3 scipy==1.16.3 statsmodels==0.15.0 matplotlib
python rebuild_from_results.py data out
```

In a clean environment with these versions, 27 of the 28 analysis files are reproduced exactly. The exception is one supplementary model that the paper does not report (UnivFD, EER threshold, PSNR as matching variable, interaction model): its odds ratios are identical, but its clustered standard error is numerically unstable and may come out infinite.

4. Check every reported number independently of `analyze.py`:

```bash
python verify_results.py data   # 58 checks: splits, thresholds, flags, FPRs, CIs, AUC, McNemar, BH, LPIPS bins, logit ORs, EER
python verify_models.py data     # family-only and common-support logistic models
```

Figures 2–4 are `results/fig_dose.png`, `results/fig_lpips.png` and `results/fig_examples.png`; Tables 4–6 come from `table_sanity.csv`, `logit_neural_effect.csv` and `logit_neural_effect_by_metric.csv`.

## Run history

- NPR was first scored on the full 512×512 crop. Its sanity check failed (AUC 0.31 on Synthbuster), so it was re-scored with the authors' test protocol (bilinear resize to 256×256, no crop). The first run is kept in `scores_archive/npr_native512.csv` and reported in Section 4.5 of the paper. No other detector, threshold rule or analysis was changed after results were seen.
- One RAISE-1k image (`rr0bf7f938t`) did not yield a usable file and was excluded before the split (`exclusions.json`).

## Pipeline

| Module | What it does |
|---|---|
| `common.py` | Paths, seeded split (300 calibration / held-out test), 1024-px resize + 512×512 centre crop, PNG I/O |
| `data.py` | RAISE-1k download and cropping; Synthbuster sampling (100 per generator; GLIDE excluded, 256 px) |
| `enhance.py` | Degradations and restorers: bicubic, Lanczos + unsharp mask, non-local means (h tuned on calibration images), Real-ESRGAN, SwinIR |
| `fidelity.py` | PSNR, SSIM, LPIPS (AlexNet), DISTS against the clean original |
| `detectors.py` | UnivFD, NPR, RIGID with the authors' public weights |
| `analyze.py` | Thresholds (nominal 5% FPR on calibration; tie counts), FPR with bootstrap CIs, McNemar + Benjamini–Hochberg (3 test families), LPIPS overlap and bin tests, logistic regression `flag ~ neural + LPIPS* + condition` with image-clustered SEs (plus interaction, common-support and family-only variants), near-identical subset, leakage checks, figures |

Variant names are `<family>_<method>_<strength>`, e.g. `up_realesrgan_4` (×4 upscaling) or `dn_swinir_50` (σ = 50 denoising). The small study uses `enhance.SMALL_VARIANTS` and `STUDY_N_TEST=300`.

## Reproducibility

- **Seed:** 2026 for the split, per-image noise, Synthbuster sampling and RIGID perturbations.
- **Split:** shuffled once, saved to `splits.json`, reused by every stage. Calibration images are never enhanced.
- **Threshold:** 95th percentile (linear interpolation) of clean calibration scores; flagged if score > τ.
- **Environment:** the notebook writes `results/environment.json` (package versions, GPU, short SHA-256 prefixes of the restorer checkpoints) and `exclusions.json` (unusable RAISE images). Full SHA-256 hashes of all checkpoints, detectors included, are listed below.
- **Outputs:** `results/per_image.csv` holds every detector score, binary flag and fidelity value.

## Checkpoints (all from the original authors)

| Model | Checkpoint | Source |
|---|---|---|
| Real-ESRGAN | `RealESRGAN_x2plus.pth`, `RealESRGAN_x4plus.pth` | github.com/xinntao/Real-ESRGAN releases |
| SwinIR | `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth`, `003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth`, `005_colorDN_DFWB_s128w8_SwinIR-M_noise{15,50}.pth` | github.com/JingyunLiang/SwinIR releases |
| UnivFD | `pretrained_weights/fc_weights.pth` on OpenAI CLIP ViT-L/14 | github.com/Yuheng-Li/UniversalFakeDetect |
| NPR | `NPR.pth` (input resized to 256×256, as in the authors' `test.py`) | github.com/chuangchuangtan/NPR-DeepfakeDetection |
| DINOv2 | `dinov2_vitl14` | github.com/facebookresearch/dinov2 |

## Checkpoint hashes (SHA-256)

TO FILL BEFORE RELEASE: paste the output of the hash cell here.

## Data

- **RAISE-1k:** Dang-Nguyen et al., MMSys 2015. Images are not redistributed; download them from the RAISE site.
- **Synthbuster:** Bammey, IEEE OJSP 2024, https://zenodo.org/records/10066460.

## Citation

```
@inproceedings{yadav2027enhancements,
  title     = {Which Enhancements Keep a Photo Real? A Controlled Study of Neural Restoration and False Positives in AI-Generated Image Detectors},
  author    = {Yadav, Ayushee},
  booktitle = {International Conference on Intelligent Computing and Interactive Technologies (IICIT)},
  year      = {2027}
}
```
