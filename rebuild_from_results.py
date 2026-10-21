"""Re-run the whole analysis from the saved per-image scores (no GPU, no images needed).

    python rebuild_from_results.py data out
writes every table/figure CSV into out/results/ for comparison with data/results/.
"""
import os, shutil, subprocess, sys
from pathlib import Path
import pandas as pd
src, out = Path(sys.argv[1]), Path(sys.argv[2])
(out / "scores").mkdir(parents=True, exist_ok=True); (out / "results").mkdir(exist_ok=True)
for f in ("splits.json", "exclusions.json", "nlm_h_v2.json"):
    shutil.copy(src / f, out / f)
shutil.copy(src / "results" / "fidelity.csv", out / "results" / "fidelity.csv")
p = pd.read_csv(src / "results" / "per_image.csv")
for d, g in p.groupby("detector"):
    g[["image_id", "variant", "score"]].to_csv(out / "scores" / f"{d}.csv", index=False)
env = dict(os.environ, STUDY_ROOT=str(out.resolve()), STUDY_N_TEST="300", MPLBACKEND="Agg")
code = "import analyze; analyze.run_all(mode='fpr5'); analyze.run_all(mode='eer', make_figures=False)"
subprocess.run([sys.executable, "-c", code], env=env, check=True, cwd=Path(__file__).parent)
