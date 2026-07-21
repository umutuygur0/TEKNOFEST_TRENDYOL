"""
61_ensemble_v34_v35a_v36a.py — v34 + v35a + v36a proba ortalaması (ücretsiz, hızlı deneme)
=============================================================================================
Üç ayrı doğrulanmış gerçek skor: v34=0.878, v35a=0.880, v36a=0.881.
Basit ortalama ensemble ek küçük kazanç sağlayabilir mi diye bakılıyor.
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SUBM = BASE / "claude only" / "submissions"
DATA = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"

p34 = np.load(str(SUBM / "v34_test_proba.npy")).astype(np.float64)
p35a = np.load(str(SUBM / "v35a_test_proba.npy")).astype(np.float64)
p36a = np.load(str(SUBM / "v36a_test_proba.npy")).astype(np.float64)
sub = pd.read_csv(DATA / "submission_pairs.csv")

assert len(p34) == len(p35a) == len(p36a) == len(sub)
print(f"p34 mean={p34.mean():.4f} | p35a mean={p35a.mean():.4f} | p36a mean={p36a.mean():.4f}")

ens = (p34 + p35a + p36a) / 3.0

target_pos = round((904565 + 898965) / 2)  # v35a & v36a positive counts, benzer aralık
best_thr, best_diff = 0.5, 1e9
for thr in np.arange(0.5, 0.95, 0.001):
    cnt = (ens > thr).sum()
    diff = abs(cnt - target_pos)
    if diff < best_diff:
        best_diff, best_thr = diff, thr

final = (ens > best_thr).astype(int)
pos_n = final.sum()
print(f"thr={best_thr:.4f} -> Pozitif={pos_n:,} ({100*pos_n/len(sub):.1f}%)")

out_path = SUBM / "submission_ens_v34_v35a_v36a.csv"
pd.DataFrame({"id": sub["id"], "prediction": final}).to_csv(str(out_path), index=False)
print(f"Dosya: {out_path}")
