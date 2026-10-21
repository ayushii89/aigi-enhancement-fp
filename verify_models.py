import sys, numpy as np, pandas as pd, statsmodels.formula.api as smf
from pathlib import Path
R = Path(sys.argv[1]); p = pd.read_csv(R/"results/per_image.csv")
ref = pd.read_csv(R/"results/logit_neural_effect.csv").set_index("detector")
T = p[(p.split=="test") & p.kind.isin(["neural","classical"])].copy()
def fit(df, f):
    df = df.reset_index(drop=True)
    return smf.logit(f, df).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(df.image_id)[0]})
for d in ["univfd","npr","rigid"]:
    g = T[T.detector==d].copy(); g["y"]=g.flag.astype(int); g["neural"]=(g.kind=="neural").astype(int)
    g["cond"]=g.family+g.strength.astype(str); g["z"]=(g.lpips-g.lpips.mean())/g.lpips.std()
    fo = fit(g, "y ~ neural + z + C(family)")
    keep = []
    for c, h in g.groupby("cond"):
        lo = max(h[h.neural==k].lpips.min() for k in (0,1)); hi = min(h[h.neural==k].lpips.max() for k in (0,1))
        keep.append(h[(h.lpips>=lo)&(h.lpips<=hi)])
    cs = pd.concat(keep); cs["z"]=(cs.lpips-cs.lpips.mean())/cs.lpips.std()
    cf = fit(cs, "y ~ neural + z + C(cond)")
    e = lambda r: np.round(np.exp([r.params["neural"], *r.conf_int().loc["neural"]]), 3)
    print(d, "family-only", e(fo), "ref", round(ref.loc[d,"or_family_only"],3),
          "| common", e(cf), "n", len(cs), "ref", round(ref.loc[d,"or_common"],3), ref.loc[d,"n_common"])
