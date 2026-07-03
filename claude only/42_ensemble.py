"""
Ensemble: v23 + v27 + v29 test probalarını blend et.
v23: Kaggle 0.84 (bilinen referans)
v27: tyembed_cos eklendi, no group feats, OOF=0.9545
v29: NEG_PER_POS=10, no group feats, OOF=0.8664
OOF threshold: blend proba üzerinde F1-eq sweep yok (OOF blend'i yok),
              bireysel thresholdların ortalamasını kullan ve +-0.02 sweep.
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
SUBM = BASE / "claude only" / "submissions"
DATA = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"

print("Proba dosyaları yükleniyor...", flush=True)
p23  = np.load(str(SUBM / "v23_test_proba.npy")).astype(np.float64)
p27  = np.load(str(SUBM / "v27_test_proba.npy")).astype(np.float64)
p29  = np.load(str(SUBM / "v29_test_proba.npy")).astype(np.float64)

sub  = pd.read_csv(DATA / "submission_pairs.csv")
template = pd.read_csv(SUBM / "submission_v27_tyembed_rawcos.csv")

# v23 proba aralığı farklı (64-bit vs 32-bit) — normalize kontrol
print(f"p23: min={p23.min():.4f} max={p23.max():.4f} mean={p23.mean():.4f}")
print(f"p27: min={p27.min():.4f} max={p27.max():.4f} mean={p27.mean():.4f}")
print(f"p29: min={p29.min():.4f} max={p29.max():.4f} mean={p29.mean():.4f}")
print(f"Boyutlar: p23={len(p23)}, p27={len(p27)}, p29={len(p29)}", flush=True)

assert len(p23) == len(p27) == len(p29) == len(template), "Boyut uyuşmazlığı!"

# ---- ENSEMBLEler ----
# Bireysel model OOF thresholdları:
# v23: submission'dan tersine ~0.54-0.56 (calibrated), pozitif: 937617
# v27: OOF thr=0.7300, pozitif: 934208
# v29: OOF thr=0.7200, pozitif: 896554

# Her ensemble için threshold: binary search — ensemble proba üzerinde
# v27 ile aynı pozitif sayısını hedeflemiyoruz, sadece OOF thr ort. kullanıyoruz.

def find_thr_by_positives(proba, target_pos, lo=0.3, hi=0.95, steps=500):
    """Hedef pozitif sayısına en yakın threshold'ı bul."""
    best_thr, best_diff = 0.5, 1e9
    for i in range(steps):
        thr = lo + (hi - lo) * i / (steps - 1)
        cnt = (proba > thr).sum()
        diff = abs(cnt - target_pos)
        if diff < best_diff:
            best_diff, best_thr = diff, thr
    return best_thr

def save_ensemble(proba, name, note=""):
    """Threshold sweep ve submission kaydet."""
    # OOF threshold'u yok, bireysel model ort. (0.73+0.72)/2 = 0.725
    # Bunu referans alıp ±0.03 içinde en iyi "stable" noktayı seç
    # Stable = proba'nın yoğunluk boşluğu (histogram peakinden uzak değil)

    # v27 pozitif sayısını referans al: 934208
    ref_pos = 934208
    thr_ref = find_thr_by_positives(proba, ref_pos)

    pos = (proba > thr_ref).sum()
    pct = 100 * pos / len(proba)
    print(f"\n{name}: thr={thr_ref:.4f} -> {pos:,} pozitif ({pct:.1f}%)")

    out = template.copy()
    out["prediction"] = (proba > thr_ref).astype(np.int8)
    out_path = SUBM / f"{name}.csv"
    out.to_csv(out_path, index=False)
    np.save(str(SUBM / f"{name}_proba.npy"), proba.astype(np.float32))
    print(f"  => Kaydedildi: {out_path}", flush=True)

    # Ayrıca OOF-thr bazlı (0.72 ortalama) tahmin de yaz
    # v27 thr=0.73, v29 thr=0.72  -> ortalama 0.725
    avg_thr = 0.725 if "ens_27_29" in name else (0.54 + 0.73) / 2 if "ens_23_27" in name else 0.66
    pos2 = (proba > avg_thr).sum()
    print(f"  alt_thr={avg_thr:.3f} -> {pos2:,} pozitif ({100*pos2/len(proba):.1f}%)")
    return pos

print("\n" + "="*60)
print("1) ENS_27_29: v27 + v29 eşit ağırlık")
ens_27_29 = 0.5 * p27 + 0.5 * p29
save_ensemble(ens_27_29, "submission_ens_27_29")

print("\n" + "="*60)
print("2) ENS_23_27_29: v23 + v27 + v29 eşit ağırlık (1/3 each)")
ens_3way = (p23 + p27 + p29) / 3.0
save_ensemble(ens_3way, "submission_ens_23_27_29")

print("\n" + "="*60)
print("3) ENS_23_27: v23 + v27 eşit ağırlık")
ens_23_27 = 0.5 * p23 + 0.5 * p27
save_ensemble(ens_23_27, "submission_ens_23_27")

print("\n" + "="*60)
print("4) ENS_23x2_27_29: v23 ağırlık 2, v27 ve v29 ağırlık 1")
ens_w = (2*p23 + p27 + p29) / 4.0
save_ensemble(ens_w, "submission_ens_23x2_27_29")

# Karşılaştırma tablosu
print("\n" + "="*60)
print("KARŞILAŞTIRMA:")
for name, proba in [("v23", p23), ("v27", p27), ("v29", p29),
                    ("ens_27_29", ens_27_29), ("ens_3way", ens_3way),
                    ("ens_23_27", ens_23_27), ("ens_w", ens_w)]:
    # Pozitif sayısını 0.725 üzerinden al (sabit referans, oran değil)
    # Not: v23 proba scale farklı, 0.5 daha uygun
    if name == "v23":
        pos = (proba > 0.50).sum()
        thr_used = 0.50
    else:
        pos = (proba > 0.725).sum()
        thr_used = 0.725
    print(f"  {name:20s}: thr={thr_used:.3f} -> {pos:,} ({100*pos/len(proba):.1f}%)")

print("\nHEPSİ TAMAMLANDI.", flush=True)
