"""
S1: Zero-Shot BGE Reranker v2-m3
==================================
Model : BAAI/bge-reranker-v2-m3 (568M param, cross-encoder)
Eğitim: YOK — tamamen zero-shot
Fark  : v9 (XLM-R 278M fine-tuned 0.49) → bge (568M zero-shot) → beklenti 0.60-0.72

Neden bu modelden başlıyoruz:
  - Cross-encoder: (query, item) birlikte encode → semantic interaction
  - 568M param: v3 MiniLM (22M)'den 25× büyük
  - MS-MARCO + multilingual pre-trained → Türkçe anlıyor
  - Zero-shot bile v9'u geçebilir

Çalıştırma:
  python "claude only/11_bge_s1.py"

Süre tahmini:
  CV (500 sorgu × 100 aday = 50K pair): ~4-8 dakika GPU
  Full test (3.36M pair): ~4-6 saat GPU
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import sys
import random
import torch

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
SUBM = BASE / "claude only" / "submissions"
SUBM.mkdir(parents=True, exist_ok=True)

K_PREDICT    = 14
CV_SAMPLES   = 500   # Gerçekçi CV için holdout sorgu sayısı
CV_THRESHOLD = 0.56  # Bu değerin altındaysa test inference başlatma
BATCH_SIZE   = 256
MAX_LENGTH   = 256

LOWER_MAP = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(t): return str(t).translate(LOWER_MAP).lower().strip()

# ─── 1. Veri Yükle ──────────────────────────────────────────────────────────
print("[1] Veri yükleniyor...")
t0 = time.time()
items  = pd.read_csv(DATA / "items.csv")
terms  = pd.read_csv(DATA / "terms.csv")
train  = pd.read_csv(DATA / "training_pairs.csv")
test   = pd.read_csv(DATA / "submission_pairs.csv")
sample = pd.read_csv(DATA / "sample_submission.csv")

iid_to_title    = dict(zip(items["item_id"], items["title"].fillna("")))
iid_to_brand    = dict(zip(items["item_id"], items["brand"].fillna("")))
iid_to_category = dict(zip(items["item_id"], items["category"].fillna("")))
tid_to_query    = dict(zip(terms["term_id"], terms["query"]))

train_pos = defaultdict(set)
for tid, iid in zip(train["term_id"].values, train["item_id"].values):
    train_pos[tid].add(iid)

all_iids  = list(items["item_id"])
all_itxts = [
    trl(iid_to_title.get(i, "") + " " + iid_to_brand.get(i, ""))
    for i in all_iids
]
iid_to_idx = {iid: i for i, iid in enumerate(all_iids)}

print(f"  {time.time()-t0:.1f}s | {len(train_pos):,} train sorgu | {len(all_iids):,} item")

# ─── 2. BGE Reranker Modeli Yükle ───────────────────────────────────────────
print("\n[2] BAAI/bge-reranker-v2-m3 yükleniyor...")
print("  İlk çalıştırmada ~2GB indirme yapacak, lütfen bekle...")
t0 = time.time()

from sentence_transformers import CrossEncoder
model = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    max_length=MAX_LENGTH,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
print(f"  Model yüklendi: {time.time()-t0:.1f}s | Device: {model.model.device}")

# ─── 3. Item Metni Oluşturma Fonksiyonu ──────────────────────────────────────
def item_text(iid):
    """Ürün metnini oluştur: title + brand + kategori L1"""
    t = trl(iid_to_title.get(iid, ""))
    b = trl(iid_to_brand.get(iid, ""))
    c = trl(iid_to_category.get(iid, "").split("/")[0])
    parts = [p for p in [t, b, c] if p and p != "nan"]
    return " | ".join(parts)

# ─── 4. Gerçekçi CV — BM25 Benzeri 100 Aday ile ─────────────────────────────
print(f"\n[3] Gerçekçi CV başlıyor ({CV_SAMPLES} sorgu × 100 BM25 aday)...")
print("   (Test yapısını taklit ediyor: her sorgu için BM25 top-100 aday)")

# BM25 benzetimi için TF-IDF item matrisi
print("  Item TF-IDF matrisi oluşturuluyor...")
t0 = time.time()
item_vect = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(2, 4),
    min_df=3, max_features=200_000, sublinear_tf=True
)
item_mat = item_vect.fit_transform(all_itxts)
print(f"  Item matrix: {item_mat.shape} | {time.time()-t0:.1f}s")

random.seed(42)
all_train_tids = list(train_pos.keys())
random.shuffle(all_train_tids)
holdout_tids = all_train_tids[:CV_SAMPLES]

cv_results = []
cv_t0 = time.time()

for qi, tid in enumerate(holdout_tids):
    q_text = trl(tid_to_query.get(tid, ""))
    true_pos = train_pos[tid]
    if not q_text or not true_pos:
        continue

    # BM25-like top-100 aday seç
    q_vec = item_vect.transform([q_text])
    sims  = (q_vec * item_mat.T).toarray()[0]
    top100_idx = np.argpartition(sims, -100)[-100:]
    top100_idx = top100_idx[np.argsort(sims[top100_idx])[::-1]]
    candidates = [all_iids[i] for i in top100_idx]

    # (query, item) çiftleri oluştur
    pairs = [(q_text, item_text(iid)) for iid in candidates]

    # BGE ile score et
    scores = model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)

    # Top-14 seç
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    predicted = set([iid for iid, _ in ranked[:K_PREDICT]])

    tp = len(predicted & true_pos)
    fp = len(predicted - true_pos)
    fn = len(true_pos - predicted)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    tp_in_cands = len(true_pos & set(candidates))

    cv_results.append({
        "tid": tid, "q": q_text, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "n_true": len(true_pos), "tp_in_cands": tp_in_cands
    })

    if qi % 50 == 0 and qi > 0:
        so_far = [r["f1"] for r in cv_results]
        elapsed = time.time() - cv_t0
        eta = elapsed / (qi + 1) * (CV_SAMPLES - qi - 1)
        print(f"  {qi}/{CV_SAMPLES} | F1 so far: {np.mean(so_far):.3f} | ETA: {eta:.0f}s")

# ─── 5. CV Sonuçları ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"CV SONUÇLARI ({len(cv_results)} sorgu | Model: bge-reranker-v2-m3 zero-shot)")
print(f"{'='*60}")

df_cv = pd.DataFrame(cv_results)
f1_vals = df_cv["f1"].values

# Makro F1 hesapla
tp_t = df_cv["tp"].sum(); fp_t = df_cv["fp"].sum(); fn_t = df_cv["fn"].sum()
prec  = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
rec   = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
f1_1  = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

total_neg = 100 * len(df_cv) - df_cv["n_true"].sum()
fp_neg    = fp_t
tn        = total_neg - fp_neg
fn_neg    = fn_t
prec0 = tn / (tn + fn_neg) if (tn + fn_neg) > 0 else 0
rec0  = tn / (tn + fp_neg) if (tn + fp_neg) > 0 else 0
f1_0  = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0
macro = (f1_0 + f1_1) / 2

print(f"  Precision (class 1): {prec:.3f}")
print(f"  Recall    (class 1): {rec:.3f}")
print(f"  F1        (class 1): {f1_1:.3f}")
print(f"  F1        (class 0): {f1_0:.3f}")
print(f"  MACRO F1  : {macro:.3f}")

tp_rate = df_cv["tp_in_cands"].sum() / df_cv["n_true"].sum()
print(f"\n  BM25 aday coverage: {tp_rate:.1%} (true pos in top-100)")
print(f"  Sorgu başı ort F1 : {np.mean(f1_vals):.3f}")

# F1 dağılımı
print(f"\n  F1 Dağılımı:")
bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
for i in range(len(bins) - 1):
    cnt = ((df_cv["f1"] >= bins[i]) & (df_cv["f1"] < bins[i + 1])).sum()
    bar = "█" * (cnt // 5)
    print(f"    [{bins[i]:.1f}-{bins[i+1]:.1f}): {cnt:3d} {bar}")

# En iyi / kötü sorgular
print(f"\n  En iyi 5 sorgu:")
for _, r in df_cv.nlargest(5, "f1").iterrows():
    print(f"    F1={r['f1']:.2f}  '{r['q']}'  tp={r['tp']}/{r['n_true']}")
print(f"\n  En kötü 5 sorgu (F1=0):")
for _, r in df_cv[df_cv["f1"] == 0].head(5).iterrows():
    print(f"    F1=0.00  '{r['q']}'  tp_in_cands={r['tp_in_cands']}/{r['n_true']}")

print(f"\n  v9 karşılaştırma: v9=0.49 | bge_zero_shot={macro:.3f}")
improvement = macro - 0.49
print(f"  İyileştirme: {'+' if improvement > 0 else ''}{improvement:.3f}")

# ─── 6. Karar: Test Inference Yapılacak mı? ──────────────────────────────────
print(f"\n{'='*60}")
if macro >= CV_THRESHOLD:
    print(f"  ✓ CV={macro:.3f} ≥ {CV_THRESHOLD} → Test inference başlıyor!")
    print(f"  Tahmini süre: ~{3_359_679 / (BATCH_SIZE * 50):.0f} dakika")
    print(f"{'='*60}")

    # ─── 7. Test Inference ───────────────────────────────────────────────────
    print("\n[4] Test inference (3.36M çift)...")
    test_grp  = test.groupby("term_id")
    test_items = test_grp["item_id"].apply(list).to_dict()
    test_ids   = test_grp["id"].apply(list).to_dict()
    test_tids  = list(test_items.keys())

    predictions = {}
    t0 = time.time()

    for i, tid in enumerate(test_tids):
        q_text     = trl(tid_to_query.get(tid, ""))
        candidates = test_items[tid]
        row_ids    = test_ids[tid]

        pairs  = [(q_text, item_text(iid)) for iid in candidates]
        scores = model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)

        ranked = sorted(zip(row_ids, candidates, scores), key=lambda x: x[2], reverse=True)
        for j, (pid, iid, sc) in enumerate(ranked):
            predictions[pid] = 1 if j < K_PREDICT else 0

        if i % 2000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(test_tids) - i - 1)
            pos_so_far = sum(predictions.values())
            print(f"  {i:6d}/{len(test_tids)} | {elapsed/60:.1f}dk | ETA: {eta/60:.1f}dk | pos: {pos_so_far:,}")

    # ─── 8. Submission Oluştur ───────────────────────────────────────────────
    sub = sample[["id"]].copy()
    sub["prediction"] = sub["id"].map(predictions).fillna(0).astype(int)
    pos_count = sub["prediction"].sum()
    pos_rate  = 100 * pos_count / len(sub)
    print(f"\n  Pozitif: {pos_count:,} ({pos_rate:.1f}%) | Beklenen: ~13.4%")

    out_path = SUBM / "submission_v11_bge_zeroshot.csv"
    sub.to_csv(str(out_path), index=False)
    print(f"  Kaydedildi: {out_path}")

    # ─── 9. Kalite Kontrol ───────────────────────────────────────────────────
    print("\n[5] Kalite kontrol — örnek sorgular")
    test_v11 = test.merge(sub, on="id")
    cases = [
        ("TERM_eacb5395", "luck",                   "avon"),
        ("TERM_85856557", "pandora kadın takı",      "pandora"),
        ("TERM_68034abd", "samsung galaxy s24 fe",   "samsung"),
    ]
    for tid_c, label, brand in cases:
        grp = test_v11[test_v11["term_id"] == tid_c]
        pos = grp[grp["prediction"] == 1]["item_id"].tolist()
        brand_cnt = sum(1 for iid in pos if brand in trl(iid_to_brand.get(iid, "")))
        print(f"\n  '{label}' → {brand_cnt}/14 doğru marka ({brand})")
        for iid in pos[:5]:
            t_txt = iid_to_title.get(iid, "?")[:50]
            b_txt = iid_to_brand.get(iid, "?")[:14]
            mark  = "✓" if brand in trl(b_txt) else "✗"
            print(f"    [{b_txt}] {t_txt}  {mark}")

    print(f"\n{'='*60}")
    print(f"SONUÇ:")
    print(f"  CV Macro F1 : {macro:.3f}")
    print(f"  Çıktı       : submission_v11_bge_zeroshot.csv")
    print(f"  Submit et?  : {'EVET — Kaggle slotunu kullan!' if macro > 0.58 else 'Bekle, önce S2/S3 dene'}")
    print(f"{'='*60}")

else:
    print(f"  ✗ CV={macro:.3f} < {CV_THRESHOLD} → Test inference ATLANDI")
    print(f"  Beklenti tutmadı. Sonraki adım: S3 dense hard negative mining")
    print(f"{'='*60}")
    print(f"\n  Sebep analizi:")
    low_cov = (df_cv["tp_in_cands"] == 0).sum()
    print(f"  - BM25 coverage sıfır olan sorgu: {low_cov}/{CV_SAMPLES}")
    print(f"  - Model bu sorguların hiçbirinde pozitif bulamaz")
    print(f"  → Çözüm: Dense retrieval (E5) → S3'e geç")
