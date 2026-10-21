"""Statistics, tables and figures for the paper.

Thresholds are fixed on clean CALIBRATION reals only (95th percentile = 5% FPR);
every reported rate is measured on the held-out TEST reals.

Main outputs in results/:
  table_sanity.csv           detector AUC/TPR on fakes at the threshold (Table 1)
  fpr_by_variant.csv         FPR + 95% bootstrap CI for every detector x variant
  flip_points.csv            lowest strength with FPR significantly above clean
  neural_vs_classical.csv    paired McNemar tests at equal degradation, BH-corrected
  fidelity_verdict_gap.csv   FPR among near-identical outputs (LPIPS<0.1, SSIM>0.9)
  fpr_by_lpips_bin.csv       neural vs classical inside LPIPS bins (matched fidelity)
  logit_neural_effect.csv    odds ratio of 'neural' holding LPIPS fixed
  fig_dose.(png|pdf), fig_lpips.(png|pdf), fig_examples.(png|pdf)
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from common import DIRS, get_splits, load_rgb, parse_variant, variant_path

RNG = np.random.default_rng(2026)
N_BOOT = 2000
ALPHA = 0.05

# Fixed colour per method (categorical slots, never cycled) + marker for print.
METHOD_STYLE = {
    "bicubic": ("#2a78d6", "o"), "nlm": ("#2a78d6", "o"),
    "lanczosusm": ("#eb6834", "s"), "bm3d": ("#eb6834", "s"),
    "swinir": ("#1baf7a", "^"),
    "realesrgan": ("#6250d6", "D"), "restormer": ("#6250d6", "D"),
    "sdx4": ("#e87ba4", "v"),
    "noisy": ("#8a8a8a", "x"),
}
METHOD_LABEL = {"bicubic": "Bicubic", "lanczosusm": "Lanczos+USM", "realesrgan": "Real-ESRGAN",
                "swinir": "SwinIR", "sdx4": "SD x4 upscaler", "noisy": "Noisy input",
                "nlm": "NLM", "bm3d": "BM3D", "restormer": "Restormer"}
DET_LABEL = {"univfd": "UnivFD", "npr": "NPR", "rigid": "RIGID"}
FAMILY_LABEL = {"up": "Upscaling (factor)", "dn": "Denoising (noise sigma)"}


# ----------------------------------------------------------------- helpers
def boot_ci(flags):
    flags = np.asarray(flags, dtype=float)
    n = len(flags)
    if n == 0:
        return np.nan, np.nan
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    m = flags[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def mcnemar(a, b):
    """Exact McNemar on paired booleans. Returns (b_only_a, b_only_b, p)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p


def thresholds(scores, calib_ids, mode="fpr5"):
    """Per-detector threshold. 'fpr5': 95th pct of clean calibration reals.
    'eer': equal-error point between clean calibration reals and all fakes (robustness check)."""
    out = {}
    for det, d in scores.groupby("detector"):
        real = d[(d.variant == "clean") & d.image_id.isin(calib_ids)].score.values
        if mode == "fpr5":
            out[det] = float(np.quantile(real, 0.95))
        else:
            fake = d[d.variant.str.startswith("fake_")].score.values
            if len(fake) == 0:
                raise ValueError("EER mode needs scored fakes")
            cands = np.unique(np.concatenate([real, fake]))
            fpr = np.array([(real > t).mean() for t in cands])
            fnr = np.array([(fake <= t).mean() for t in cands])
            out[det] = float(cands[np.argmin(np.abs(fpr - fnr))])
    return out


def build_table(scores, fidelity, splits, mode="fpr5"):
    """One row per detector x variant x image, with flag, metadata and fidelity."""
    tau = thresholds(scores, splits["calib"], mode)
    d = scores.copy()
    d["flag"] = d.score > d.detector.map(tau)
    meta = pd.DataFrame([dict(variant=v, **parse_variant(v)) for v in d.variant.unique()])
    d = d.merge(meta, on="variant", how="left")
    if fidelity is not None and len(fidelity):
        d = d.merge(fidelity, on=["image_id", "variant"], how="left")
    if "lpips" in d:
        d.loc[d.variant == "clean", ["lpips", "dists"]] = 0.0
        d.loc[d.variant == "clean", ["ssim"]] = 1.0
    test, calib = set(splits["test"]), set(splits["calib"])
    d["split"] = np.select(
        [d.variant.str.startswith("fake_"), d.image_id.isin(test), d.image_id.isin(calib)],
        ["fake", "test", "calib"], default="unused")
    return d, tau


# ------------------------------------------------------------------ tables
def sanity_table(d):
    rows = []
    for det, g in d.groupby("detector"):
        real = g[(g.variant == "clean") & (g.split == "test")]
        fakes = g[g.family == "fake"]
        row = {"detector": det, "clean_test_fpr": real.flag.mean(), "n_real": len(real)}
        if len(fakes):
            u = mannwhitneyu(fakes.score, real.score, alternative="two-sided").statistic
            row.update(auc=u / (len(fakes) * len(real)), tpr_at_tau=fakes.flag.mean(), n_fake=len(fakes))
            for gen, gg in fakes.groupby("method"):
                row[f"tpr_{gen}"] = gg.flag.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def fpr_table(d):
    t = d[d.split == "test"]
    rows = []
    for (det, v), g in t.groupby(["detector", "variant"]):
        lo, hi = boot_ci(g.flag)
        m = parse_variant(v)
        rows.append({"detector": det, "variant": v, **m, "n": len(g), "fpr": g.flag.mean(),
                     "ci_lo": lo, "ci_hi": hi, "mean_score": g.score.mean(),
                     "median_lpips": g.lpips.median() if "lpips" in g else np.nan})
    return pd.DataFrame(rows).sort_values(["detector", "family", "method", "strength"])


def flip_points(d):
    """Lowest strength at which FPR rises significantly above the same images' clean FPR."""
    t = d[d.split == "test"]
    clean = t[t.variant == "clean"].set_index(["detector", "image_id"]).flag
    tests = []
    for (det, fam, meth, s), g in t[t.kind.isin(["neural", "classical", "baseline"]) & (t.family != "clean")] \
            .groupby(["detector", "family", "method", "strength"]):
        c = clean.loc[det].reindex(g.image_id).values
        n01, n10, p = mcnemar(g.flag.values, c)
        tests.append({"detector": det, "family": fam, "method": meth, "strength": s,
                      "fpr": g.flag.mean(), "clean_fpr": np.mean(c), "newly_flagged": n01,
                      "newly_cleared": n10, "p": p})
    tests = pd.DataFrame(tests)
    if tests.empty:
        return tests, tests
    tests["q"] = multipletests(tests.p, method="fdr_bh")[1]
    tests["sig_increase"] = (tests.q < ALPHA) & (tests.fpr > tests.clean_fpr)
    flips = (tests[tests.sig_increase].groupby(["detector", "family", "method"]).strength.min()
             .rename("flip_strength").reset_index())
    return tests, flips


def neural_vs_classical(d):
    t = d[d.split == "test"]
    rows = []
    for (det, fam, s), g in t[t.kind.isin(["neural", "classical"])].groupby(["detector", "family", "strength"]):
        piv = g.pivot_table(index="image_id", columns="method", values="flag", aggfunc="first")
        kinds = g.drop_duplicates("method").set_index("method").kind
        for nm in [m for m in piv.columns if kinds[m] == "neural"]:
            for cm in [m for m in piv.columns if kinds[m] == "classical"]:
                pair = piv[[nm, cm]].dropna()
                if pair.empty:
                    continue
                n01, n10, p = mcnemar(pair[nm].astype(bool), pair[cm].astype(bool))
                rows.append({"detector": det, "family": fam, "strength": s, "neural": nm, "classical": cm,
                             "n": len(pair), "fpr_neural": pair[nm].mean(), "fpr_classical": pair[cm].mean(),
                             "diff": pair[nm].mean() - pair[cm].mean(),
                             "only_neural_flags": n01, "only_classical_flags": n10, "p": p})
    r = pd.DataFrame(rows)
    if len(r):
        r["q"] = multipletests(r.p, method="fdr_bh")[1]
    return r


def fidelity_verdict_gap(d, lpips_max=0.10, ssim_min=0.90):
    t = d[(d.split == "test") & d.kind.isin(["neural", "classical"]) & (d.lpips < lpips_max) & (d.ssim > ssim_min)]
    rows = []
    for (det, kind), g in t.groupby(["detector", "kind"]):
        lo, hi = boot_ci(g.flag)
        rows.append({"detector": det, "kind": kind, "n": len(g), "fpr": g.flag.mean(), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def lpips_bins(d, n_bins=5):
    """Matched fidelity: bin restored outputs by LPIPS (pooled edges per family),
    then compare neural vs classical FPR inside each bin."""
    t = d[(d.split == "test") & d.kind.isin(["neural", "classical"])].copy()
    rows = []
    for fam, g in t.groupby("family"):
        edges = np.unique(np.quantile(g.lpips.dropna(), np.linspace(0, 1, n_bins + 1)))
        g = g.assign(bin=pd.cut(g.lpips, edges, include_lowest=True, labels=False))
        for (det, b, kind), gg in g.groupby(["detector", "bin", "kind"]):
            lo, hi = boot_ci(gg.flag)
            rows.append({"detector": det, "family": fam, "bin": int(b), "kind": kind,
                         "lpips_lo": edges[int(b)], "lpips_hi": edges[int(b) + 1],
                         "lpips_median": gg.lpips.median(), "n": len(gg),
                         "fpr": gg.flag.mean(), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def lpips_overlap(d):
    """How much neural and classical outputs overlap in LPIPS, per family.
    Matched-fidelity claims are only made inside the common support."""
    t = d[(d.split == "test") & d.kind.isin(["neural", "classical"]) & (d.detector == d.detector.iloc[0])]
    rows = []
    for fam, g in t.groupby("family"):
        q = {k: gk.lpips.dropna() for k, gk in g.groupby("kind")}
        if len(q) < 2:
            continue
        lo = max(q["neural"].min(), q["classical"].min())
        hi = min(q["neural"].max(), q["classical"].max())
        for kind, v in q.items():
            rows.append({"family": fam, "kind": kind, "n": len(v), "min": v.min(), "q25": v.quantile(.25),
                         "median": v.median(), "q75": v.quantile(.75), "max": v.max(),
                         "common_lo": lo, "common_hi": hi,
                         "share_in_common_support": float(((v >= lo) & (v <= hi)).mean()) if hi > lo else 0.0})
    return pd.DataFrame(rows)


def lpips_bin_tests(bins, min_n=20):
    """Neural vs classical inside each LPIPS bin (Fisher exact, BH-corrected).
    Bins where either kind has fewer than min_n outputs are marked and not tested."""
    from scipy.stats import fisher_exact
    rows = []
    for (det, fam, b), g in bins.groupby(["detector", "family", "bin"]):
        k = g.set_index("kind")
        if not {"neural", "classical"} <= set(k.index):
            continue
        nn, nc = int(k.loc["neural", "n"]), int(k.loc["classical", "n"])
        fn, fc = k.loc["neural", "fpr"], k.loc["classical", "fpr"]
        ok = nn >= min_n and nc >= min_n
        p = np.nan
        if ok:
            an, ac = round(fn * nn), round(fc * nc)
            p = fisher_exact([[an, nn - an], [ac, nc - ac]])[1]
        rows.append({"detector": det, "family": fam, "bin": b, "lpips_lo": k.lpips_lo.iloc[0],
                     "lpips_hi": k.lpips_hi.iloc[0], "n_neural": nn, "n_classical": nc,
                     "fpr_neural": fn, "fpr_classical": fc, "diff": fn - fc, "overlap_ok": ok, "p": p})
    r = pd.DataFrame(rows)
    if len(r) and r.overlap_ok.any():
        m = r.overlap_ok
        r.loc[m, "q"] = multipletests(r.loc[m, "p"], method="fdr_bh")[1]
    return r


def _fit(formula, g):
    import statsmodels.formula.api as smf
    return smf.logit(formula, data=g).fit(disp=0, cov_type="cluster",
                                          cov_kwds={"groups": pd.factorize(g.image_id)[0]})


def _or(fit, term):
    ci = fit.conf_int().loc[term]
    return np.exp(fit.params[term]), np.exp(ci[0]), np.exp(ci[1]), fit.pvalues[term]


MODELS = {
    # primary (Eq. 2): neural indicator, standardised LPIPS, degradation condition (family x strength)
    "primary": "y ~ neural + lpips_z + C(cond)",
    # sensitivity: family only, no strength adjustment (the earlier specification)
    "family_only": "y ~ neural + lpips_z + C(family)",
}


def neural_effect(d, metric="lpips"):
    """Logistic regressions per detector with standard errors clustered by image.

    primary      : flag ~ neural + LPIPS* + condition       (condition = family x strength, 4 levels)
    family_only  : flag ~ neural + LPIPS* + family           (sensitivity, no strength adjustment)
    interaction  : flag ~ neural * family + LPIPS* + condition -> neural OR separately per family
    common       : primary model on outputs inside the LPIPS range both kinds cover, per condition
    """
    t = d[(d.split == "test") & d.kind.isin(["neural", "classical"])].dropna(subset=[metric]).copy()
    t["neural"] = (t.kind == "neural").astype(int)
    # the matching variable (LPIPS by default); the column keeps its name so the formulas stay the same
    t["lpips_z"] = (t[metric] - t[metric].mean()) / t[metric].std()
    t["match"] = t[metric]
    t["y"] = t.flag.astype(int)
    t["cond"] = t.family + "_" + t.strength.astype(str)
    rows = []
    for det, g in t.groupby("detector"):
        row = {"detector": det, "matching_metric": metric, "n": len(g), "n_images": g.image_id.nunique()}
        for name, f in MODELS.items():
            try:
                o, lo, hi, p = _or(_fit(f, g), "neural")
                row.update({f"or_{name}": o, f"or_{name}_lo": lo, f"or_{name}_hi": hi, f"p_{name}": p})
            except Exception as e:  # e.g. perfect separation when a detector never fires
                row[f"err_{name}"] = repr(e)
        try:
            fi = _fit("y ~ neural * C(family) + lpips_z + C(cond)", g)
            fams = sorted(g.family.unique())
            b, V = fi.params, fi.cov_params()
            for fam in fams:
                inter = f"neural:C(family)[T.{fam}]"
                est = b["neural"] + (b[inter] if inter in b else 0.0)
                var = V.loc["neural", "neural"] + ((V.loc[inter, inter] + 2 * V.loc["neural", inter]) if inter in b else 0.0)
                se = np.sqrt(var)
                row.update({f"or_neural_{fam}": np.exp(est), f"or_neural_{fam}_lo": np.exp(est - 1.96 * se),
                            f"or_neural_{fam}_hi": np.exp(est + 1.96 * se)})
            inter_terms = [k for k in b.index if k.startswith("neural:")]
            row["p_interaction"] = fi.pvalues[inter_terms[0]] if inter_terms else np.nan
        except Exception as e:
            row["err_interaction"] = repr(e)
        # common LPIPS support within each degradation condition
        keep = []
        for _, gc in g.groupby("cond"):
            lo = max(gc[gc.neural == 1].match.min(), gc[gc.neural == 0].match.min())
            hi = min(gc[gc.neural == 1].match.max(), gc[gc.neural == 0].match.max())
            keep.append(gc[(gc.match >= lo) & (gc.match <= hi)])
        gcs = pd.concat(keep)
        row["n_common"] = len(gcs)
        try:
            o, lo, hi, p = _or(_fit(MODELS["primary"], gcs), "neural")
            row.update(or_common=o, or_common_lo=lo, or_common_hi=hi, p_common=p)
        except Exception as e:
            row["err_common"] = repr(e)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- figures
def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#6b6b6b",
        "axes.linewidth": 0.6, "xtick.color": "#4a4a4a", "ytick.color": "#4a4a4a",
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
        "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(DIRS["results"] / f"{name}.{ext}")


def fig_dose(fpr):
    """FPR vs strength: rows = enhancement family, columns = detector.
    Neural = solid line + filled marker; classical = dashed + hollow; grey band = clean-image FPR CI."""
    import matplotlib.pyplot as plt
    _style()
    dets = sorted(fpr.detector.unique())
    fams = [f for f in ("up", "dn") if f in set(fpr.family)]
    fig, axes = plt.subplots(len(fams), len(dets), figsize=(2.3 * len(dets), 2.1 * len(fams)),
                             squeeze=False, sharey=True)
    for i, fam in enumerate(fams):
        for j, det in enumerate(dets):
            ax = axes[i][j]
            clean = fpr[(fpr.detector == det) & (fpr.variant == "clean")]
            if len(clean):
                ax.axhspan(clean.ci_lo.iloc[0], clean.ci_hi.iloc[0], color="#d9d9d9", lw=0, zorder=0)
                ax.axhline(clean.fpr.iloc[0], color="#8a8a8a", lw=0.8, zorder=1)
            g = fpr[(fpr.detector == det) & (fpr.family == fam)]
            for meth, gm in g.groupby("method"):
                gm = gm.sort_values("strength")
                col, mk = METHOD_STYLE.get(meth, ("#4a4a4a", "o"))
                kind = gm.kind.iloc[0]
                ls = "-" if kind == "neural" else ("--" if kind == "classical" else ":")
                face = col if kind == "neural" else "white"
                ax.fill_between(gm.strength, gm.ci_lo, gm.ci_hi, color=col, alpha=0.12, lw=0)
                ax.plot(gm.strength, gm.fpr, ls=ls, color=col, lw=1.5, marker=mk, ms=4.5,
                        mfc=face, mec=col, mew=1.2, label=METHOD_LABEL.get(meth, meth))
            xs = sorted(g.strength.unique())
            ax.set_xticks(xs)
            ax.set_ylim(0, 1)
            if i == 0:
                ax.set_title(DET_LABEL.get(det, det))
            if j == 0:
                ax.set_ylabel(f"False-positive rate\n{FAMILY_LABEL[fam].split(' (')[0]}")
            ax.set_xlabel(FAMILY_LABEL[fam].split("(")[1].rstrip(")"))
        handles, labels = axes[i][-1].get_legend_handles_labels()
        axes[i][-1].legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_dose")
    return fig


def fig_lpips(bins):
    import matplotlib.pyplot as plt
    _style()
    dets = sorted(bins.detector.unique())
    fams = [f for f in ("up", "dn") if f in set(bins.family)]
    fig, axes = plt.subplots(len(fams), len(dets), figsize=(2.3 * len(dets), 2.1 * len(fams)),
                             squeeze=False, sharey=True)
    styles = {"neural": ("#6250d6", "-", "D", "#6250d6"), "classical": ("#2a78d6", "--", "o", "white")}
    for i, fam in enumerate(fams):
        for j, det in enumerate(dets):
            ax = axes[i][j]
            g = bins[(bins.detector == det) & (bins.family == fam)]
            for kind, gk in g.groupby("kind"):
                col, ls, mk, face = styles[kind]
                gk = gk.sort_values("lpips_median")
                ax.errorbar(gk.lpips_median, gk.fpr, yerr=[gk.fpr - gk.ci_lo, gk.ci_hi - gk.fpr],
                            color=col, ls=ls, lw=1.5, marker=mk, ms=4.5, mfc=face, mec=col,
                            capsize=2, elinewidth=0.8, label=kind.capitalize())
            ax.set_ylim(0, 1)
            if i == 0:
                ax.set_title(DET_LABEL.get(det, det))
            if j == 0:
                ax.set_ylabel(f"False-positive rate\n{FAMILY_LABEL[fam].split(' (')[0]}")
            ax.set_xlabel("LPIPS to original (bin median)")
        axes[i][-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_lpips")
    return fig


def pick_example(d, variant="up_realesrgan_4"):
    """Systematic example choice (no cherry-picking): the test image whose LPIPS for
    `variant` is closest to that variant's median LPIPS. Ties broken by image id."""
    t = d[(d.split == "test") & (d.variant == variant)].drop_duplicates("image_id")
    med = t.lpips.median()
    t = t.assign(dist=(t.lpips - med).abs()).sort_values(["dist", "image_id"])
    return t.image_id.iloc[0]


def leakage_checks(splits, scores):
    """Assertions that calibration and test never mix."""
    calib, test = set(splits["calib"]), set(splits["test"])
    checks = {
        "calib_test_disjoint": len(calib & test) == 0,
        "n_calib": len(calib), "n_test": len(test),
        "no_enhanced_calibration_images": not bool(
            scores[(scores.variant != "clean") & ~scores.variant.str.startswith("fake_")]
            .image_id.isin(calib).any()),
        "fakes_disjoint_from_reals": not scores[scores.variant.str.startswith("fake_")].image_id.isin(calib | test).any(),
    }
    return checks


def fig_examples(d, image_id, variants=("clean", "up_bicubic_4", "up_lanczosusm_4", "up_swinir_4", "up_realesrgan_4")):
    """One test image through several operators, with LPIPS and each detector's verdict."""
    import matplotlib.pyplot as plt
    _style()
    variants = [v for v in variants if variant_path(v, image_id).exists()]
    fig, axes = plt.subplots(1, len(variants), figsize=(2.0 * len(variants), 2.6))
    for ax, v in zip(np.atleast_1d(axes), variants):
        img = load_rgb(variant_path(v, image_id))
        ax.imshow(img[128:384, 128:384])  # central 256 px crop so texture is visible
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        r = d[(d.image_id == image_id) & (d.variant == v)]
        verdict = "  ".join(f"{DET_LABEL.get(x.detector, x.detector)}:{'FAKE' if x.flag else 'real'}"
                            for x in r.sort_values("detector").itertuples())
        lp = r.lpips.iloc[0] if len(r) and "lpips" in r.columns else np.nan
        meta = parse_variant(v)
        unit = "x" if meta["family"] == "up" else "sigma "
        name = "Original" if v == "clean" else f"{METHOD_LABEL.get(meta['method'])} ({unit}{meta['strength']})"
        ax.set_title(f"{name}\nLPIPS {'n/a' if pd.isna(lp) else f'{lp:.3f}'}", fontsize=8)
        ax.set_xlabel(verdict, fontsize=6)
    fig.tight_layout()
    _save(fig, "fig_examples")
    return fig


# ------------------------------------------------------------------- driver
def run_all(mode="fpr5", make_figures=True):
    from detectors import load_scores
    splits = get_splits()
    scores = load_scores()
    fid_path = DIRS["results"] / "fidelity.csv"
    fidelity = pd.read_csv(fid_path) if fid_path.exists() else None
    d, tau = build_table(scores, fidelity, splits, mode)
    suffix = "" if mode == "fpr5" else f"_{mode}"
    out = {
        "thresholds": pd.DataFrame({
            "detector": list(tau), "tau": list(tau.values()),
            "rule": "95th pct of clean calib (linear interp.), flag if score > tau" if mode == "fpr5"
                    else "EER between clean calib and Synthbuster (no test images)",
            "calib_fpr_observed": [d[(d.detector == k) & (d.split == "calib") & (d.variant == "clean")].flag.mean() for k in tau],
            "scores_equal_to_tau": [int((scores[scores.detector == k].score == tau[k]).sum()) for k in tau]}),
        "table_sanity": sanity_table(d),
        "fpr_by_variant": fpr_table(d),
    }
    tests, flips = flip_points(d)
    out["flip_tests"], out["flip_points"] = tests, flips
    out["neural_vs_classical"] = neural_vs_classical(d)
    if fidelity is not None:
        out["fidelity_verdict_gap"] = fidelity_verdict_gap(d)
        out["fpr_by_lpips_bin"] = lpips_bins(d)
        out["lpips_bin_tests"] = lpips_bin_tests(out["fpr_by_lpips_bin"])
        out["lpips_overlap"] = lpips_overlap(d)
        out["logit_neural_effect"] = neural_effect(d)
        # robustness: match on DISTS or PSNR instead of LPIPS (metrics disagree for upscaling)
        out["logit_neural_effect_by_metric"] = pd.concat(
            [neural_effect(d, m) for m in ("lpips", "dists", "psnr", "ssim")], ignore_index=True)
    checks = leakage_checks(splits, scores)
    out["leakage_checks"] = pd.DataFrame([checks])
    # per-image scores and binary flags, for replication
    keep = [c for c in ["detector", "image_id", "variant", "split", "family", "method", "strength", "kind",
                        "score", "flag", "psnr", "ssim", "lpips", "dists"] if c in d.columns]
    out["per_image"] = d[keep]
    for k, v in out.items():
        v.to_csv(DIRS["results"] / f"{k}{suffix}.csv", index=False)
    if make_figures and mode == "fpr5":
        fig_dose(out["fpr_by_variant"])
        if fidelity is not None:
            fig_lpips(out["fpr_by_lpips_bin"])
    print("Leakage checks:", checks)
    print("Thresholds:", {k: round(v, 4) for k, v in tau.items()})
    print("\nSanity check (fakes):\n", out["table_sanity"].round(3).to_string(index=False))
    return d, out
