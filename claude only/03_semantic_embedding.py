# -*- coding: utf-8 -*-
"""
Submission v3 — Multilingual Semantic Embedding (Zero-Shot)
============================================================
v1/v2 neden 0.48 aldı:
  - Test'teki 100 aday BM25 ile seçilmiş ZATEN (yüksek lexical overlap = zor negatif)
  - BM25/overlap özelliklerimiz bu durumda rastgeleden KÖTÜ çalışıyor
  - "laptop" vs "bilgisayar", "ayakkabı" vs "ayakkabısı" = aynı şey, BM25 anlamaz

v3 çözümü:
  - Multilingual sentence transformer (paraphrase-multilingual-MiniLM-L12-v2)
  - Her item ve query'yi 384-boyutlu embedding vektörüne çevir
  - Cosine similarity ile gerçek semantik benzerlik hesapla
  - RTX 5080 GPU ile hızlı encoding
  - Per-query proportional top-K (~%13.9)

Beklenti: 0.60-0.75 public F1
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm

try:
    import torch
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("HATA: sentence-transformers kurulu degil!")
    print("pip install sentence-transformers")
    exit(1)

# ─── Yollar ───────────────────────────────────────────────────────────────────
DATA_DIR  = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL\trendyol-e-ticaret-yarismasi-2026-kaggle")
SUBM_DIR  = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL\claude only\submissions")
CACHE_DIR = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL\claude only\emb_cache")
CACHE_DIR.mkdir(exist_ok=True)

t0 = time.time()

# ─── Veri Yükle ───────────────────────────────────────────────────────────────
print("=" * 60)
print("Veri yukleniyor...")
items  = pd.read_csv(DATA_DIR / "items.csv")
terms  = pd.read_csv(DATA_DIR / "terms.csv")
train  = pd.read_csv(DATA_DIR / "training_pairs.csv")
test   = pd.read_csv(DATA_DIR / "submission_pairs.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
print(f"  items:{len(items):,}  terms:{len(terms):,}  test:{len(test):,}")

# ─── Item Encode Metni ────────────────────────────────────────────────────────
# Kısa ama bilgi dolu: başlık + marka + kategori L1
# Uzun full_text sinyal gürültüsünü artırır, başlık genelde yeterli
items["encode_text"] = (
    items["title"].fillna("") + " | " +
    items["brand"].fillna("") + " | " +
    items["category"].fillna("").apply(
        lambda x: x.split("/")[0].strip() if isinstance(x, str) and "/" in x else str(x)
    )
)

# item_id → index
item_id_arr  = items["item_id"].values
iid_to_idx   = {iid: i for i, iid in enumerate(item_id_arr)}

# ─── Model Yükle ──────────────────────────────────────────────────────────────
# paraphrase-multilingual-MiniLM-L12-v2:
#   - 384 dim, 66M parametre, Türkçe dahil 50+ dil destekler
#   - Ürün arama için calibrated (paraphrase task)
#   - RTX 5080'de ~3 dk ile tüm 960K item encode edilir
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
print(f"\nModel yukleniyor: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")
if device == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─── Item Embedding (cache ile) ───────────────────────────────────────────────
ITEM_EMB_PATH = CACHE_DIR / "item_embeddings_minilm.npy"
ITEM_ID_PATH  = CACHE_DIR / "item_ids.npy"

if ITEM_EMB_PATH.exists() and ITEM_ID_PATH.exists():
    print("\nCache'den item embedding'leri yukleniyor...")
    item_embeddings = np.load(str(ITEM_EMB_PATH))
    item_ids_saved  = np.load(str(ITEM_ID_PATH), allow_pickle=True)
    # Dogruluk kontrolu
    if len(item_ids_saved) == len(items) and np.array_equal(item_ids_saved, item_id_arr):
        print(f"  Shape: {item_embeddings.shape}  (gecerli cache)")
    else:
        print("  Cache gecersiz, yeniden encode ediliyor...")
        ITEM_EMB_PATH.unlink(); ITEM_ID_PATH.unlink()
        item_embeddings = None
else:
    item_embeddings = None

if item_embeddings is None:
    print(f"\n{len(items):,} item encode ediliyor ({device} uzerinde)...")
    t_enc = time.time()
    item_embeddings = model.encode(
        items["encode_text"].tolist(),
        batch_size=512,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2 normalize → dot product = cosine sim
        device=device,
        convert_to_numpy=True,
    )
    print(f"  Tamamlandi [{time.time()-t_enc:.0f}s]  Shape: {item_embeddings.shape}")
    np.save(str(ITEM_EMB_PATH), item_embeddings)
    np.save(str(ITEM_ID_PATH), item_id_arr)
    print(f"  Cache'e kaydedildi: {CACHE_DIR}")

# item_id → embedding index (yeniden kur, ID sıralaması değişmiş olabilir)
iid_to_emb_idx = {iid: i for i, iid in enumerate(item_id_arr)}

# ─── Query Embedding ──────────────────────────────────────────────────────────
# Sadece test'teki unique termler encode edilir (32K ~ hızlı)
test_tids   = test["term_id"].unique()
tid_to_qstr = dict(zip(terms["term_id"], terms["query"]))
test_qstrs  = [tid_to_qstr.get(tid, "") for tid in test_tids]

QUERY_EMB_PATH = CACHE_DIR / "test_query_embeddings_minilm.npy"
QUERY_ID_PATH  = CACHE_DIR / "test_query_ids.npy"

if QUERY_EMB_PATH.exists() and QUERY_ID_PATH.exists():
    print("\nCache'den query embedding'leri yukleniyor...")
    query_embeddings = np.load(str(QUERY_EMB_PATH))
    q_ids_saved = np.load(str(QUERY_ID_PATH), allow_pickle=True)
    if len(q_ids_saved) == len(test_tids):
        tid_to_q_idx = {tid: i for i, tid in enumerate(q_ids_saved)}
        print(f"  Shape: {query_embeddings.shape}")
    else:
        print("  Cache gecersiz, yeniden encode ediliyor...")
        query_embeddings = None
else:
    query_embeddings = None

if query_embeddings is None:
    print(f"\n{len(test_tids):,} test query encode ediliyor...")
    t_enc = time.time()
    query_embeddings = model.encode(
        test_qstrs,
        batch_size=512,
        show_progress_bar=True,
        normalize_embeddings=True,
        device=device,
        convert_to_numpy=True,
    )
    tid_to_q_idx = {tid: i for i, tid in enumerate(test_tids)}
    np.save(str(QUERY_EMB_PATH), query_embeddings)
    np.save(str(QUERY_ID_PATH), test_tids)
    print(f"  Tamamlandi [{time.time()-t_enc:.0f}s]  Shape: {query_embeddings.shape}")

# ─── Top-K Calibration ────────────────────────────────────────────────────────
# Training ortalama pozitif/query'ye dayalı oran
train_pos_per_term = train.groupby("term_id")["item_id"].count()
MEAN_POS = train_pos_per_term.mean()  # ~13.9
POS_RATE = MEAN_POS / 100            # ~0.139 (100 aday bazında)
print(f"\nPozitif kalibrasyon: mean_pos={MEAN_POS:.1f}, oran={POS_RATE:.3f}")

# ─── Per-Query Similarity + Top-K ─────────────────────────────────────────────
print(f"\nPer-query semantic benzerlik + top-K secimi...")
print(f"  Toplam query: {len(test_tids):,}  |  K = round({POS_RATE:.3f} * n_cands)")

predictions = {}

for tid, grp in tqdm(test.groupby("term_id"), total=len(test_tids), ncols=70):
    q_idx = tid_to_q_idx.get(tid)
    if q_idx is None:
        for row_id in grp["id"].values:
            predictions[row_id] = 0
        continue

    q_emb = query_embeddings[q_idx]  # 384-dim, already L2-normalized

    # Aday itemların embedding indeksleri
    iids = grp["item_id"].values
    ids  = grp["id"].values

    emb_idxs  = np.array([iid_to_emb_idx.get(iid, -1) for iid in iids])
    valid_mask = emb_idxs >= 0

    if not valid_mask.any():
        for row_id in ids:
            predictions[row_id] = 0
        continue

    # Cosine similarity (dot product, embeddings are L2-normalized)
    valid_emb_idxs = emb_idxs[valid_mask]
    cand_embs      = item_embeddings[valid_emb_idxs]  # n_valid × 384
    sims           = cand_embs @ q_emb                 # n_valid

    # Top-K (proportional to n_candidates)
    n_cands = len(grp)
    k = max(1, round(POS_RATE * n_cands))
    k = min(k, valid_mask.sum())

    # Get top-k positions within valid items
    valid_positions = np.where(valid_mask)[0]
    top_k_in_valid  = np.argpartition(sims, -k)[-k:]
    top_k_orig_pos  = valid_positions[top_k_in_valid]

    # Assign predictions
    preds = np.zeros(len(ids), dtype=np.int8)
    preds[top_k_orig_pos] = 1

    for row_id, pred in zip(ids, preds):
        predictions[row_id] = int(pred)

# ─── Submission ───────────────────────────────────────────────────────────────
print("\nSubmission olusturuluyor...")
pred_df = pd.DataFrame({"id": list(predictions.keys()), "prediction": list(predictions.values())})
sub = sample[["id"]].merge(pred_df, on="id", how="left")
sub["prediction"] = sub["prediction"].fillna(0).astype(int)

pos_rate = sub["prediction"].mean()
sub_path = SUBM_DIR / "submission_v3_semantic_minilm.csv"
sub.to_csv(sub_path, index=False)

# ─── Log ──────────────────────────────────────────────────────────────────────
log_path = SUBM_DIR / "submissions_log.csv"
log_df = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
new_row = {
    "version": "v3", "model": f"SentenceTransformer({MODEL_NAME})",
    "val_macro_f1": "N/A (zero-shot)", "threshold": f"top-{round(POS_RATE*100):.0f}%",
    "pos_rate_train": "N/A", "pos_rate_test": f"{pos_rate*100:.1f}%",
    "public_score": "TBD",
    "neg_sampling": "none (zero-shot)",
    "n_features": 384,
    "notes": "Multilingual MiniLM cosine sim, proportional top-K, GPU encoded",
    "file": sub_path.name,
    "runtime_s": f"{time.time()-t0:.0f}",
}
log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
log_df.to_csv(log_path, index=False)

# ─── Ozet ─────────────────────────────────────────────────────────────────────
total_t = time.time() - t0
print("\n" + "=" * 60)
print("OZET")
print("=" * 60)
print(f"  Model           : {MODEL_NAME}")
print(f"  Embedding dim   : {item_embeddings.shape[1]}")
print(f"  Item sayisi     : {item_embeddings.shape[0]:,}")
print(f"  Query sayisi    : {query_embeddings.shape[0]:,}")
print(f"  Test pozitif    : {pos_rate:.3f} ({pos_rate*100:.1f}%)")
print(f"  Toplam sure     : {total_t:.0f}s ({total_t/60:.1f} dk)")
print(f"  Submission      : {sub_path.name}")
print()
print("  Kaggle'a yukle: submission_v3_semantic_minilm.csv")
print("  Beklenen iyilesme: 0.48 → 0.60-0.75 F1")
print("=" * 60)
