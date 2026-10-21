"""Independent audit of the IICIT results: recomputes everything from per_image.csv
with code written separately from analyze.py, and compares with the pipeline's CSVs."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact, mannwhitneyu
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

R = Path(sys.argv[1])          # unzipped results root (contains results/, splits.json)
res = R / "results"
p = pd.read_csv(res / "per_image.csv")
splits = json.loads((R / "splits.json").read_text())
log = []
def check(name, ok, detail=""):
    log.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))

# ---------------------------------------------------------------- P3 data / protocol
calib = set(splits["calib"]); test300 = set(sorted(splits["test"])[:300]) if False else None
calib_ids = set(p[p.split == "calib"].image_id)
test_ids = set(p[p.split == "test"].image_id)
check("calib ids == splits.json calib", calib_ids == calib, f"{len(calib_ids)}")
check("300 calib, 300 test, disjoint", len(calib_ids) == 300 and len(test_ids) == 300 and not calib_ids & test_ids)
check("test ids are first 300 of test pool (as listed)", test_ids == set(splits["test"][:300]))
check("no enhanced calibration image", p[(p.split == "calib")].variant.eq("clean").all())
full = p[p.split == "test"].groupby(["detector", "variant"]).image_id.nunique()
check("every detector x condition has 300 unique test images", (full == 300).all() and len(full) == 45)
check("fakes: 8 generators x 100 per detector", (p[p.split == "fake"].groupby(["detector", "variant"]).size() == 100).all())
check("fidelity present for every restored test output",
      p[(p.split == "test") & (p.variant != "clean")][["psnr", "ssim", "lpips", "dists"]].notna().all().all())

# ---------------------------------------------------------------- thresholds and flags
thr = pd.read_csv(res / "thresholds.csv").set_index("detector")
mine = {}
for d, g in p[p.split == "calib"].groupby("detector"):
    tau = np.quantile(g.score.values, 0.95)          # linear interpolation (numpy default)
    mine[d] = tau
    check(f"tau {d} reproduces", np.isclose(tau, thr.loc[d, "tau"], rtol=1e-6), f"{tau:.6g} vs {thr.loc[d,'tau']:.6g}")
    check(f"calib FPR {d} = 5%", np.isclose((g.score > tau).mean(), 0.05))
p["flag2"] = p.score > p.detector.map(mine)
check("every stored flag == score > tau", (p.flag2 == p.flag).all(), f"{(p.flag2 != p.flag).sum()} mismatches")
ties = sum((p[p.detector == d].score == t).sum() for d, t in mine.items())
check("ties at tau reported as 0", ties == 0, f"{ties}")

T = p[p.split == "test"].copy()
rng = np.random.default_rng(7)
def boot(flags, B=2000):
    f = np.asarray(flags, float); n = len(f)
    m = f[rng.integers(0, n, (B, n))].mean(1)
    return np.percentile(m, [2.5, 97.5])

# ---------------------------------------------------------------- P1 FPR + CI
fv = pd.read_csv(res / "fpr_by_variant.csv").set_index(["detector", "variant"])
rows = []
for (d, v), g in T.groupby(["detector", "variant"]):
    k, n = int(g.flag.sum()), len(g)
    lo, hi = boot(g.flag)
    ref = fv.loc[(d, v)]
    rows.append(dict(detector=d, variant=v, k=k, n=n, fpr=k / n, lo=lo, hi=hi,
                     ref_fpr=ref.fpr, ref_lo=ref.ci_lo, ref_hi=ref.ci_hi))
F = pd.DataFrame(rows)
check("FPR identical for all 45 detector x condition cells", np.allclose(F.fpr, F.ref_fpr))
ciw = np.maximum(abs(F.lo - F.ref_lo), abs(F.hi - F.ref_hi)).max()
check("bootstrap CIs agree within resampling noise (<=2.5 pts)", ciw <= 0.025, f"max diff {ciw:.3f}")
F.to_csv(R / "audit_fpr.csv", index=False)

# ---------------------------------------------------------------- AUC / TPR sanity
san = pd.read_csv(res / "table_sanity.csv").set_index("detector")
for d in ["univfd", "npr", "rigid"]:
    real = T[(T.detector == d) & (T.variant == "clean")].score.values
    fake = p[(p.detector == d) & (p.split == "fake")].score.values
    auc = mannwhitneyu(fake, real).statistic / (len(fake) * len(real))
    tpr = (fake > mine[d]).mean()
    check(f"AUC {d}", np.isclose(auc, san.loc[d, "auc"], atol=1e-4), f"{auc:.4f} vs {san.loc[d,'auc']:.4f}")
    check(f"TPR@tau {d}", np.isclose(tpr, san.loc[d, "tpr_at_tau"]), f"{tpr:.4f}")
    check(f"clean test FPR {d} reported separately", np.isclose((real > mine[d]).mean(), san.loc[d, "clean_test_fpr"]))

# ---------------------------------------------------------------- P2 McNemar + BH
W = T.pivot_table(index=["detector", "image_id"], columns="variant", values="flag").astype(int)
def mcn(a, b):
    b01 = int(((a == 1) & (b == 0)).sum()); b10 = int(((a == 0) & (b == 1)).sum())
    pv = 1.0 if b01 + b10 == 0 else binomtest(b01, b01 + b10, 0.5).pvalue
    return b01, b10, pv
pairs = []
for d in ["npr", "rigid", "univfd"]:
    w = W.loc[d]
    for s in (2, 4):
        for nm in ("realesrgan", "swinir"):
            for cm in ("bicubic", "lanczosusm"):
                pairs.append((d, f"up_{nm}_{s}", f"up_{cm}_{s}"))
    for s in (15, 50):
        pairs.append((d, f"dn_swinir_{s}", f"dn_nlm_{s}"))
nv = []
for d, a, b in pairs:
    x, y, pv = mcn(W.loc[d][a], W.loc[d][b])
    nv.append(dict(detector=d, neural=a, classical=b, only_neural=x, only_classical=y, p=pv))
NV = pd.DataFrame(nv); NV["q"] = multipletests(NV.p, method="fdr_bh")[1]
check("family (a) has 30 neural-vs-classical tests", len(NV) == 30)
ref = pd.read_csv(res / "neural_vs_classical.csv")
ref["neural_v"] = ref.family + "_" + ref.neural + "_" + ref.strength.astype(str)
ref["classical_v"] = ref.family + "_" + ref.classical + "_" + ref.strength.astype(str)
m = NV.merge(ref, left_on=["detector", "neural", "classical"], right_on=["detector", "neural_v", "classical_v"])
check("McNemar discordant counts match", (m.only_neural == m.only_neural_flags).all() and (m.only_classical == m.only_classical_flags).all())
check("McNemar p and BH q match", np.allclose(m.p_x, m.p_y, rtol=1e-6, atol=1e-300) and np.allclose(m.q_x, m.q_y, rtol=1e-6, atol=1e-300))
sig = NV.q < 0.05
neural_more = sig & (NV.only_neural > NV.only_classical)
check("28/30 significant; 24 neural-higher, 4 classical-higher",
      sig.sum() == 28 and neural_more.sum() == 24 and (sig & ~neural_more).sum() == 4,
      f"{sig.sum()} sig, {neural_more.sum()} neural, {(sig & ~neural_more).sum()} classical")
NV.to_csv(R / "audit_mcnemar_neural_vs_classical.csv", index=False)

cv = []
for d in ["npr", "rigid", "univfd"]:
    for v in [c for c in W.columns if c != "clean"]:
        x, y, pv = mcn(W.loc[d][v], W.loc[d]["clean"])
        cv.append(dict(detector=d, variant=v, newly=x, cleared=y, p=pv))
CV = pd.DataFrame(cv); CV["q"] = multipletests(CV.p, method="fdr_bh")[1]
check("family (b) has 42 condition-vs-clean tests", len(CV) == 42)
inc = (CV.q < 0.05) & (CV.newly > CV.cleared)
check("33 significant increases over clean", inc.sum() == 33, f"{inc.sum()}")
neu = CV[CV.variant.str.contains("realesrgan|swinir") & CV.variant.str.match(r".*_(2|15)$")]
check("every neural restorer significant at lowest strength (9/9)", ((neu.q < .05) & (neu.newly > neu.cleared)).sum() == 9)

# ---------------------------------------------------------------- overlap, bins, Fisher
RST = T[T.kind.isin(["neural", "classical"])].copy()
one = RST[RST.detector == "npr"]
ov = {}
for fam, g in one.groupby("family"):
    lo = max(g[g.kind == k].lpips.min() for k in ("neural", "classical"))
    hi = min(g[g.kind == k].lpips.max() for k in ("neural", "classical"))
    for k in ("neural", "classical"):
        x = g[g.kind == k].lpips; ov[(fam, k)] = ((x >= lo) & (x <= hi)).mean()
check("overlap shares (up: 1.000/0.887, dn: 0.835/0.930)",
      np.allclose([ov[("up", "neural")], ov[("up", "classical")], ov[("dn", "neural")], ov[("dn", "classical")]],
                  [1.0, 0.8867, 0.835, 0.93], atol=1e-3), str({k: round(v, 3) for k, v in ov.items()}))
binrows = []
for d in ["npr", "rigid", "univfd"]:
    for fam in ("up", "dn"):
        g = RST[(RST.detector == d) & (RST.family == fam)].copy()
        g["bin"] = pd.qcut(g.lpips, 5, labels=False)
        for b, h in g.groupby("bin"):
            a = h[h.kind == "neural"]; c = h[h.kind == "classical"]
            tab = [[a.flag.sum(), len(a) - a.flag.sum()], [c.flag.sum(), len(c) - c.flag.sum()]]
            binrows.append(dict(detector=d, family=fam, bin=b, n_n=len(a), n_c=len(c), f_n=a.flag.mean(), f_c=c.flag.mean(),
                                p=fisher_exact(tab)[1] if min(len(a), len(c)) >= 20 else np.nan))
B = pd.DataFrame(binrows); B["q"] = multipletests(B.p, method="fdr_bh")[1]
check("all 30 bins have >=20 of each kind", (B[["n_n", "n_c"]].min(1) >= 20).all() and len(B) == 30)
rb = pd.read_csv(res / "lpips_bin_tests.csv")
mb = B.merge(rb, on=["detector", "family", "bin"])
check("bin counts, FPRs and Fisher q match", (mb.n_n == mb.n_neural).all() and np.allclose(mb.f_n, mb.fpr_neural)
      and np.allclose(mb.q_x, mb.q_y, rtol=1e-6, atol=1e-300))
B.to_csv(R / "audit_lpips_bins.csv", index=False)

# ---------------------------------------------------------------- logistic models
def fit(df, formula):
    df = df.reset_index(drop=True)
    grp = pd.factorize(df.image_id)[0]
    return smf.logit(formula, df).fit(disp=0, cov_type="cluster", cov_kwds={"groups": grp}, maxiter=200)
def orci(fitres, term):
    b = fitres.params[term]; lo, hi = fitres.conf_int().loc[term]
    return np.exp([b, lo, hi])
L = []
for d in ["univfd", "npr", "rigid"]:
    g = RST[RST.detector == d].copy()
    g["y"] = g.flag.astype(int); g["neural"] = (g.kind == "neural").astype(int)
    g["cond"] = g.family + g.strength.astype(str)
    for metric in ("lpips", "dists", "psnr", "ssim"):
        g["z"] = (g[metric] - g[metric].mean()) / g[metric].std()
        r = dict(detector=d, metric=metric, n=len(g), images=g.image_id.nunique())
        r["primary"] = orci(fit(g, "y ~ neural + z + C(cond)"), "neural")
        fi = fit(g, "y ~ neural:C(family) + z + C(cond)")
        r["up"] = orci(fi, "neural:C(family)[up]"); r["dn"] = orci(fi, "neural:C(family)[dn]")
        r["family_only"] = orci(fit(g, "y ~ neural + z + C(family)"), "neural")
        L.append(r)
    # quasi-separation diagnostic for denoising
    dn = g[g.family == "dn"]
    print(f"   {d} denoising flags: SwinIR {int(dn[dn.neural==1].y.sum())}/{(dn.neural==1).sum()}, "
          f"NLM {int(dn[dn.neural==0].y.sum())}/{(dn.neural==0).sum()}")
LM = pd.DataFrame(L)
ref = pd.read_csv(res / "logit_neural_effect_by_metric.csv")
for _, r in LM.iterrows():
    rr = ref[(ref.detector == r.detector) & (ref.matching_metric == r.metric)].iloc[0]
    ok = np.allclose(r.primary, [rr.or_primary, rr.or_primary_lo, rr.or_primary_hi], rtol=0.02) and \
         np.allclose(r.up, [rr.or_neural_up, rr.or_neural_up_lo, rr.or_neural_up_hi], rtol=0.02)
    if r.detector != "npr":
        ok &= np.allclose(r.dn, [rr.or_neural_dn, rr.or_neural_dn_lo, rr.or_neural_dn_hi], rtol=0.02)
    check(f"logit ORs reproduce ({r.detector}, {r.metric})", ok,
          f"primary {r.primary.round(3)} vs {rr.or_primary:.3f}; up {r.up.round(3)}; dn {r.dn.round(3)}")
LM.to_csv(R / "audit_logit.csv", index=False)

# ---------------------------------------------------------------- near-identical
ni = RST[(RST.lpips < 0.1) & (RST.ssim > 0.9)]
gap = pd.read_csv(res / "fidelity_verdict_gap.csv").set_index(["detector", "kind"])
for (d, k), g in ni.groupby(["detector", "kind"]):
    check(f"near-identical {d}/{k}: n={len(g)}, FPR={g.flag.mean():.3f}",
          len(g) == gap.loc[(d, k), "n"] and np.isclose(g.flag.mean(), gap.loc[(d, k), "fpr"]))

# ---------------------------------------------------------------- EER sensitivity
pe = pd.read_csv(res / "per_image_eer.csv")
te = pd.read_csv(res / "thresholds_eer.csv").set_index("detector")
for d in ["univfd", "npr", "rigid"]:
    cal = p[(p.detector == d) & (p.split == "calib")].score.values
    fk = p[(p.detector == d) & (p.split == "fake")].score.values
    cand = np.unique(np.r_[cal, fk])
    gapv = [abs((cal > t).mean() - (fk <= t).mean()) for t in cand]
    t = cand[int(np.argmin(gapv))]
    check(f"EER tau {d} reproduces (uses no test image)", np.isclose(t, te.loc[d, "tau"], rtol=1e-6), f"{t:.6g} vs {te.loc[d,'tau']:.6g}")
check("EER flags consistent with EER tau", all(((pe[pe.detector == d].score > te.loc[d, "tau"]) == pe[pe.detector == d].flag).all() for d in te.index))

n_fail = sum(1 for _, ok, _ in log if not ok)
print(f"\n{len(log)} checks, {n_fail} failed")
pd.DataFrame(log, columns=["check", "pass", "detail"]).to_csv(R / "audit_log.csv", index=False)
