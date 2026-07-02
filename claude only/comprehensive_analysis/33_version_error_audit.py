"""
33_version_error_audit.py — Her Versiyonun Hata Tipi Analizi
=============================================================
v23 OOF verisini kullanarak (cached bert_scores + v23 proba)
hangi hataların hangi kategoride, query tipinde, feature değerinde
yığıldığını ölçer.

v24 ve v23 proba'larını karşılaştırarak:
  - v24'ün nerede kötüleştiğini (flip 1→0)
  - v24'ün nerede iyileştiğini (flip 0→1)
  - Her iki versiyonun da yanlış olduğu çiftler

Çıktı: comprehensive_analysis/version_error_audit.txt
"""

import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE   = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA   = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
CACHE  = BASE / "claude only"
OUT    = BASE / "claude only" / "comprehensive_analysis"
OUT.mkdir(exist_ok=True)

LOG = open(str(OUT / "version_error_audit.txt"), "w", encoding="utf-8")

def pr(*args, **kw):
    print(*args, **kw)
    print(*args, **kw, file=LOG)
    LOG.flush()

t0 = time.time()
pr("=" * 70)
pr("VERSİYON HATA DENETİMİ")
pr("=" * 70)

LOWER = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(s): return str(s).translate(LOWER).lower().strip()

# ─────────────────────────────────────────────────────────────────────
# VERİ YÜKLEMESİ
# ─────────────────────────────────────────────────────────────────────
pr("\n[1] VERİ YÜKLENİYOR...")
items       = pd.read_csv(DATA / "items.csv")
terms       = pd.read_csv(DATA / "terms.csv")
sub_pairs   = pd.read_csv(DATA / "submission_pairs.csv")

for col in ["title","category","brand","gender","age_group","attributes"]:
    items[col] = items[col].fillna("unknown").apply(trl)
terms["query"] = terms["query"].fillna("").apply(trl)

tid_to_q   = dict(zip(terms["term_id"].astype(str), terms["query"]))
iid_to_row = {str(r.item_id): r for r in items.itertuples()}

pr(f"  {len(items):,} ürün | {len(sub_pairs):,} test çifti yüklendi")

# ─────────────────────────────────────────────────────────────────────
# V23 ve V24 PROBA YÜKLE
# ─────────────────────────────────────────────────────────────────────
pr("\n[2] PROBA DOSYALARI...")
v23_path = CACHE / "submissions" / "v23_test_proba.npy"
v24_path = CACHE / "submissions" / "v24_test_proba.npy"

if not v23_path.exists():
    pr(f"  ✗ v23_test_proba.npy bulunamadı → çıkılıyor")
    sys.exit(1)
if not v24_path.exists():
    pr(f"  ✗ v24_test_proba.npy bulunamadı → çıkılıyor")
    sys.exit(1)

v23_proba = np.load(str(v23_path)).astype(np.float32)
v24_proba = np.load(str(v24_path)).astype(np.float32)

V23_THR = 0.775
V24_THR = 0.795

v23_pred = (v23_proba > V23_THR).astype(int)
v24_pred = (v24_proba > V24_THR).astype(int)

pr(f"  v23: {v23_pred.sum():,} pozitif ({100*v23_pred.mean():.1f}%)")
pr(f"  v24: {v24_pred.sum():,} pozitif ({100*v24_pred.mean():.1f}%)")

# ─────────────────────────────────────────────────────────────────────
# FLIP ANALİZİ: V23→V24 DEĞİŞEN TAHMİNLER
# ─────────────────────────────────────────────────────────────────────
pr("\n[3] FLIP ANALİZİ (v23→v24 değişen tahminler)")

flip_01 = (v23_pred == 0) & (v24_pred == 1)  # v23=neg, v24=pos
flip_10 = (v23_pred == 1) & (v24_pred == 0)  # v23=pos, v24=neg
same_1  = (v23_pred == 1) & (v24_pred == 1)  # her ikisi de pos
same_0  = (v23_pred == 0) & (v24_pred == 0)  # her ikisi de neg

pr(f"  v23=0→v24=1 (yeni pozitif)  : {flip_01.sum():,}")
pr(f"  v23=1→v24=0 (kaybedilen pos): {flip_10.sum():,}")
pr(f"  Her ikisi de pozitif        : {same_1.sum():,}")
pr(f"  Her ikisi de negatif        : {same_0.sum():,}")
pr(f"  Net değişim: {flip_01.sum()-flip_10.sum():+,} ({'azaldı' if flip_01.sum()<flip_10.sum() else 'arttı'})")

# ─────────────────────────────────────────────────────────────────────
# KATEGORİ BAZLI FLIP ANALİZİ
# ─────────────────────────────────────────────────────────────────────
pr("\n[4] KATEGORİ BAZLI DEĞİŞİM")

sub_pairs_cp = sub_pairs.copy()
sub_pairs_cp["item_id"] = sub_pairs_cp["item_id"].astype(str)
sub_pairs_cp = sub_pairs_cp.merge(
    items[["item_id","category"]].assign(item_id=items["item_id"].astype(str)),
    on="item_id", how="left"
)
sub_pairs_cp["L1"] = sub_pairs_cp["category"].fillna("unknown").apply(trl).str.split("/").str[0]
sub_pairs_cp["v23"] = v23_pred
sub_pairs_cp["v24"] = v24_pred
sub_pairs_cp["flip_10"] = flip_10  # kaybedilen pozitifler
sub_pairs_cp["flip_01"] = flip_01  # yeni pozitifler

pr("  [Kaybedilen pozitifler v23→v24 (flip 1→0)]")
cat_flip = sub_pairs_cp[sub_pairs_cp["flip_10"]]["L1"].value_counts().head(10)
total_flip = sub_pairs_cp["flip_10"].sum()
for cat, cnt in cat_flip.items():
    pr(f"    {cat:35s}: {cnt:,} ({100*cnt/total_flip:.1f}%)")

pr("\n  [Yeni pozitifler v23→v24 (flip 0→1)]")
cat_new = sub_pairs_cp[sub_pairs_cp["flip_01"]]["L1"].value_counts().head(10)
total_new = sub_pairs_cp["flip_01"].sum()
for cat, cnt in cat_new.items():
    pr(f"    {cat:35s}: {cnt:,} ({100*cnt/total_new:.1f}%)")

# ─────────────────────────────────────────────────────────────────────
# PROBA DAĞILIM KARŞILAŞTIRMASI
# ─────────────────────────────────────────────────────────────────────
pr("\n[5] PROBA DAĞILIM KARŞILAŞTIRMASI")

pr("  Percentile tablosu (v23 vs v24):")
pr(f"  {'Pct':5} {'v23':8} {'v24':8} {'fark':8}")
pr("  " + "-"*30)
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    v23_v = np.percentile(v23_proba, p)
    v24_v = np.percentile(v24_proba, p)
    pr(f"  {p:5d} {v23_v:8.4f} {v24_v:8.4f} {v24_v-v23_v:+8.4f}")

# Belirsiz band (0.3–0.7)
uncertain_v23 = ((v23_proba >= 0.3) & (v23_proba <= 0.7)).sum()
uncertain_v24 = ((v24_proba >= 0.3) & (v24_proba <= 0.7)).sum()
pr(f"\n  Belirsiz band (0.3-0.7):")
pr(f"    v23: {uncertain_v23:,} ({100*uncertain_v23/len(v23_proba):.1f}%)")
pr(f"    v24: {uncertain_v24:,} ({100*uncertain_v24/len(v24_proba):.1f}%)")
pr(f"  → v24 daha az kesin değil ({uncertain_v24-uncertain_v23:+,} fark)")

# ─────────────────────────────────────────────────────────────────────
# BERT SCORE VERİFİKASYONU
# ─────────────────────────────────────────────────────────────────────
pr("\n[6] BERT SCORE COVERAGE ANALİZİ")

bert_test_path = CACHE / "submissions" / "bert_scores_v23_test.npy"
if bert_test_path.exists():
    bert_test = np.load(str(bert_test_path))
    pr(f"  Test BERT scores (n={len(bert_test):,}):")
    pr(f"    <0.1 (çok düşük): {(bert_test<0.1).sum():,} ({100*(bert_test<0.1).mean():.1f}%) ← model kaçırdı")
    pr(f"    0.1-0.5 (orta)  : {((bert_test>=0.1)&(bert_test<0.5)).sum():,} ({100*((bert_test>=0.1)&(bert_test<0.5)).mean():.1f}%)")
    pr(f"    0.5-0.9 (yüksek): {((bert_test>=0.5)&(bert_test<0.9)).sum():,} ({100*((bert_test>=0.5)&(bert_test<0.9)).mean():.1f}%)")
    pr(f"    >0.9 (çok yüksek): {(bert_test>0.9).sum():,} ({100*(bert_test>0.9).mean():.1f}%)")

    # BERT yüksek ama v23'ün negatif tahmin ettiği
    bert_high_v23neg = (bert_test > 0.7) & (v23_pred == 0)
    bert_high_v23pos = (bert_test > 0.7) & (v23_pred == 1)
    pr(f"\n  BERT>0.7 ama v23=0 (potansiyel FP, eşik geçemedi): {bert_high_v23neg.sum():,}")
    pr(f"  BERT>0.7 ve v23=1 (tahmin edilen pozitif): {bert_high_v23pos.sum():,}")

    bert_low_v23pos = (bert_test < 0.3) & (v23_pred == 1)
    pr(f"  BERT<0.3 ama v23=1 (diğer sinyallerle pozitif): {bert_low_v23pos.sum():,}")

    # Sadece BERT düşük olduğu için kaçırılan potansiyel pozitifler
    pr(f"\n  ÖNEMLİ: BERT<0.1 olan {(bert_test<0.1).sum():,} çiftte model gerçekten kör")
    pr(f"    Bu çiftler: semantik alias, synonym, brand-product-type gap")
    pr(f"    TY-ecomm-embed bunları kısmen çözebilir (domain fine-tuned)")
else:
    pr(f"  ⚠ bert_scores_v23_test.npy bulunamadı")

# ─────────────────────────────────────────────────────────────────────
# TÜM VERSİYONLAR ÖZET TABLO
# ─────────────────────────────────────────────────────────────────────
pr("\n[7] TÜM VERSİYONLAR ÖZET TABLO")
pr("""
  ┌────────┬────────────┬────────────────────────────┬────────┬────────┐
  │Versiyon│ Yaklaşım   │ Ana Fark                   │ OOF    │ Kaggle │
  ├────────┼────────────┼────────────────────────────┼────────┼────────┤
  │ v1     │ BM25 top-K │ Retrieval (yanlış çerçeve) │  N/A   │  0.48  │
  │ v2     │ LightGBM   │ 16 feature, threshold opt  │  0.964 │  0.48  │
  │ v3     │ MiniLM     │ Zero-shot embedding        │  N/A   │  0.45  │
  │ v5     │ E5 finetune│ Embedding collapse          │  N/A   │  0.43  │
  │ v6     │ Cat KNN    │ Category + overlap         │  N/A   │  0.47  │
  │ v8     │ Brand+BM25 │ Structured signals         │  N/A   │  0.47  │
  │ v9     │ XLM-R      │ Cross-query negatives      │  N/A   │  0.49  │
  │ v21    │ LambdaRank │ Leak-free + dense embed    │  ~0.87 │  ~0.71 │
  │ v22    │ LGBMClass  │ BERT feat + head features  │  0.9704│  0.83  │
  │ v23    │ LGBMClass  │ bert_v23 + direct thr      │  0.9686│  0.84  │
  │ v24    │ LGBMClass  │ TF-IDF hard neg 180K       │  0.9210│  0.79  │
  └────────┴────────────┴────────────────────────────┴────────┴────────┘
""")

pr("[Toplam süre: {:.1f} dk]".format((time.time()-t0)/60))
pr("=" * 70)
LOG.close()
print(f"\nRapor: {OUT}/version_error_audit.txt")
