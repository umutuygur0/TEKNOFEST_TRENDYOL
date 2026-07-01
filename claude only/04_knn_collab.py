# -*- coding: utf-8 -*-
"""
Submission v4 — KNN Collaborative Filtering (Training Label-Based)
===================================================================
Önceki yöntemlerin neden başarısız olduğu:
  - BM25 / overlap / embedding → test'te anti-korelasyon
  - Test'teki 100 aday büyük ihtimalle BM25 zor negatifleri içeriyor
  - Biz yüksek BM25 = pozitif diyoruz, ama aslında tersi

Bu yöntem:
  1. Her test query'sine semantik olarak benzer TRAINING query'leri bul
  2. O training query'lerinin POZİTİF itemlarını score'la
  3. KNN skoru yüksek = gerçekten pozitif (eğitim labellarından öğrenildi)
  4. Decoy/probing itemlar training'de görünmemiş → skor sıfır → elenir

CUDA kullanımı:
  - Embedding cache var (item_embeddings_minilm.npy) → tekrar encode etme
  - Training queries encode: GPU ile hızlı
  - Similarity matrix: GPU tensor veya numpy

Beklenti: 0.55-0.70 F1 (eğitim labellarından gerçek sinyal)
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ─── Yollar ───────────────────────────────────────────────────────────────────
DATA_DIR  = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL\trendyol-e-ticaret-yarismasi-2026-kaggle")
SUBM_DIR  = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL\claude only\submissions")
CACHE_DIR = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL\claude only\emb_cache")

t0 = time.time()
print("=" * 60)
print("v4: KNN Collaborative Filtering (Training Label-Based)")
print("=" * 60)

# ─── Veri Yükle ───────────────────────────────────────────────────────────────
print("\nVeri yukleniyor...")
items  = pd.read_csv(DATA_DIR / "items.csv")
terms  = pd.read_csv(DATA_DIR / "terms.csv")
train  = pd.read_csv(DATA_DIR / "training_pairs.csv")
test   = pd.read_csv(DATA_DIR / "submission_pairs.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
print(f"  items:{len(items):,}  train:{len(train):,}  test:{len(test):,}")

tid_to_query = dict(zip(terms["term_id"], terms["query"]))

# ─── Sentence Transformer (GPU) ───────────────────────────────────────────────
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nModel: {MODEL_NAME}  |  Device: {device}")
if device == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

model = SentenceTransformer(MODEL_NAME)

# ─── Test Query Embeddings (cache) ────────────────────────────────────────────
TQEMB = CACHE_DIR / "test_query_embeddings_minilm.npy"
TQIDS = CACHE_DIR / "test_query_ids.npy"
if TQEMB.exists():
    print("\nTest query embeddings cache'den yukleniyor...")
    test_q_embs = np.load(str(TQEMB))
    test_q_ids  = np.load(str(TQIDS), allow_pickle=True)
    tid_to_tq_idx = {tid: i for i, tid in enumerate(test_q_ids)}
    print(f"  Shape: {test_q_embs.shape}")
else:
    print("\nTest queries encode ediliyor (GPU)...")
    test_tids  = test["term_id"].unique()
    test_qstrs = [tid_to_query.get(tid, "") for tid in test_tids]
    test_q_embs = model.encode(
        test_qstrs, batch_size=512, show_progress_bar=True,
        normalize_embeddings=True, device=device, convert_to_numpy=True)
    tid_to_tq_idx = {tid: i for i, tid in enumerate(test_tids)}
    np.save(str(TQEMB), test_q_embs)
    np.save(str(TQIDS), test_tids)
    print(f"  Shape: {test_q_embs.shape}")

# ─── Training Query Embeddings (GPU ile encode) ────────────────────────────────
TRAINEMB = CACHE_DIR / "train_query_embeddings_minilm.npy"
TRAINIDS  = CACHE_DIR / "train_query_ids.npy"

train_tids_unique = train["term_id"].unique()

if TRAINEMB.exists():
    print("\nTraining query embeddings cache'den yukleniyor...")
    train_q_embs = np.load(str(TRAINEMB))
    train_q_ids  = np.load(str(TRAINIDS), allow_pickle=True)
else:
    print(f"\n{len(train_tids_unique):,} training query encode ediliyor (GPU)...")
    t_enc = time.time()
    train_qstrs = [tid_to_query.get(tid, "") for tid in train_tids_unique]
    train_q_embs = model.encode(
        train_qstrs, batch_size=512, show_progress_bar=True,
        normalize_embeddings=True, device=device, convert_to_numpy=True)
    train_q_ids = train_tids_unique
    np.save(str(TRAINEMB), train_q_embs)
    np.save(str(TRAINIDS), train_q_ids)
    print(f"  Shape: {train_q_embs.shape}  [{time.time()-t_enc:.1f}s]")

train_q_id_to_idx = {tid: i for i, tid in enumerate(train_q_ids)}
print(f"  Training query embeddings: {train_q_embs.shape}")

# ─── Item → Training Query Set Haritası ────────────────────────────────────────
print("\nItem->training query map olusturuluyor...")
item_to_train_q_idxs = defaultdict(set)
for tid, iid in zip(train["term_id"].values, train["item_id"].values):
    q_idx = train_q_id_to_idx.get(tid)
    if q_idx is not None:
        item_to_train_q_idxs[iid].add(q_idx)

n_items_in_train = sum(1 for iid in test["item_id"].unique() if iid in item_to_train_q_idxs)
print(f"  Training'de olan test itemlari: {n_items_in_train:,} / {test['item_id'].nunique():,}")

# ─── Similarity Matrix: Test × Train (GPU tensor veya numpy) ──────────────────
print("\nTest-Train query similarity hesaplaniyor...")
t_sim = time.time()

# GPU kullan (17GB VRAM var, yeterli)
if device == "cuda":
    tq = torch.from_numpy(test_q_embs).float().to("cuda")   # 32K × 384
    trq = torch.from_numpy(train_q_embs).float().to("cuda") # 18K × 384
    # Batch ile işle (bellek tasarrufu)
    BATCH = 1000
    sim_top_idx_list = []
    sim_top_val_list = []
    K_NEIGHBORS = 50  # Her test query için top-50 training query

    print(f"  Batch similarity (GPU, batch={BATCH})...")
    for i in tqdm(range(0, len(test_q_embs), BATCH), ncols=70):
        batch = tq[i:i+BATCH]             # BATCH × 384
        sims  = batch @ trq.T              # BATCH × 18K
        top_vals, top_idxs = sims.topk(K_NEIGHBORS, dim=1)
        sim_top_idx_list.append(top_idxs.cpu().numpy())
        sim_top_val_list.append(top_vals.cpu().numpy())

    sim_top_idxs = np.vstack(sim_top_idx_list)  # 32K × K_NEIGHBORS
    sim_top_vals = np.vstack(sim_top_val_list)
    del tq, trq
    torch.cuda.empty_cache()
else:
    K_NEIGHBORS = 50
    print("  CPU matmul (yavas olabilir)...")
    # CPU fallback
    sim = test_q_embs @ train_q_embs.T   # 32K × 18K
    sim_top_idxs = np.argsort(sim, axis=1)[:, -K_NEIGHBORS:]
    sim_top_vals  = np.take_along_axis(sim, sim_top_idxs, axis=1)

print(f"  Tamamlandi [{time.time()-t_sim:.1f}s]  Shape: {sim_top_idxs.shape}")

# ─── KNN Scoring ──────────────────────────────────────────────────────────────
print("\nTest ciftleri KNN skoru hesaplaniyor...")
t_score = time.time()

test_copy = test.copy()
test_copy["knn_score"] = 0.0

# Item embedding benzerligini fallback olarak kullan (v3'ten cache var)
IEMB_PATH = CACHE_DIR / "item_embeddings_minilm.npy"
IID_PATH  = CACHE_DIR / "item_ids.npy"
if IEMB_PATH.exists():
    item_embs    = np.load(str(IEMB_PATH))
    item_ids_arr = np.load(str(IID_PATH), allow_pickle=True)
    iid_to_emb_idx = {iid: i for i, iid in enumerate(item_ids_arr)}
    print(f"  Item embeddings yuklendi: {item_embs.shape}")
    USE_ITEM_EMBS = True
else:
    USE_ITEM_EMBS = False
    print("  Item embeddings bulunamadi, sadece KNN skoru kullanilacak")

for tid, grp in tqdm(test_copy.groupby("term_id"), total=test["term_id"].nunique(), ncols=70):
    tq_idx = tid_to_tq_idx.get(tid)
    if tq_idx is None:
        continue

    # Bu test query'nin top-K training query indeksleri
    top_train_q_set = set(sim_top_idxs[tq_idx].tolist())

    iids   = grp["item_id"].values
    scores = np.zeros(len(iids))

    for j, iid in enumerate(iids):
        item_train_qs = item_to_train_q_idxs.get(iid)
        if item_train_qs:
            # Kac training query icin bu item pozitif VE test query'sine benzer
            overlap = len(item_train_qs & top_train_q_set)
            scores[j] = overlap

    # Fallback: KNN skor = 0 olanlar için item-query embedding benzerligini kullan
    if USE_ITEM_EMBS:
        q_emb = test_q_embs[tq_idx]  # 384
        zero_mask = scores == 0
        if zero_mask.any():
            zero_iids = iids[zero_mask]
            emb_idxs  = np.array([iid_to_emb_idx.get(iid, -1) for iid in zero_iids])
            valid_mask = emb_idxs >= 0
            if valid_mask.any():
                cand_embs = item_embs[emb_idxs[valid_mask]]
                emb_sims  = cand_embs @ q_emb
                # KNN skor 0 iken emb sim kullan (negatife normalize et, knn>0 her zaman önce gelsin)
                emb_scores = emb_sims * 0.001  # KNN skoru her zaman > emb_sim
                zero_positions = np.where(zero_mask)[0]
                valid_zero_pos = zero_positions[valid_mask]
                scores[valid_zero_pos] = emb_scores

    test_copy.loc[grp.index, "knn_score"] = scores

print(f"  Tamamlandi [{time.time()-t_score:.1f}s]")
print(f"  KNN>0 tahmin: {(test_copy['knn_score'] > 0.001).sum():,} / {len(test_copy):,}")

# ─── Per-Query Top-K (Calibrated) ─────────────────────────────────────────────
# 50/50 split: training 250K pozitif → test ~250K pozitif
# Test unique queries: 32185 → pozitif per query: 250K/32K = 7.8
# Ama 100 adaydan: ~7.8% pozitif → K_PREDICT ≈ 8
#
# Deneme için IKISI DE dene:
K_OPTIONS = {"k8": 8, "k14": 14}

predictions_by_k = {}
for k_name, K_PREDICT in K_OPTIONS.items():
    preds_this_k = {}
    for tid, grp in test_copy.groupby("term_id"):
        k = max(1, round(K_PREDICT / 100 * len(grp)))
        top_k_ids = grp.nlargest(k, "knn_score").index
        for row_id in grp["id"].values:
            preds_this_k[row_id] = 0
        for row_id in test_copy.loc[top_k_ids, "id"].values:
            preds_this_k[row_id] = 1
    predictions_by_k[k_name] = preds_this_k
    pos_count = sum(predictions_by_k[k_name].values())
    print(f"  K={K_PREDICT}: pozitif={pos_count:,} ({pos_count/len(test)*100:.1f}%)")

# ─── Save Submissions ──────────────────────────────────────────────────────────
print("\nSubmission dosyalari olusturuluyor...")

saved_paths = {}
for k_name, preds in predictions_by_k.items():
    pred_df = pd.DataFrame({"id": list(preds.keys()), "prediction": list(preds.values())})
    sub     = sample[["id"]].merge(pred_df, on="id", how="left")
    sub["prediction"] = sub["prediction"].fillna(0).astype(int)
    fname   = f"submission_v4_knn_{k_name}.csv"
    path    = SUBM_DIR / fname
    sub.to_csv(path, index=False)
    saved_paths[k_name] = path
    print(f"  [{k_name}] {fname}")

# ─── Log ──────────────────────────────────────────────────────────────────────
log_path = SUBM_DIR / "submissions_log.csv"
log_df = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()

for k_name, K_PREDICT in K_OPTIONS.items():
    pos_rate = predictions_by_k[k_name]
    pos_r = sum(pos_rate.values()) / len(pos_rate)
    new_row = {
        "version": f"v4_{k_name}", "model": f"KNN collab (K_neighbors={K_NEIGHBORS})",
        "val_macro_f1": "N/A", "threshold": f"top-{K_PREDICT}",
        "pos_rate_train": "N/A", "pos_rate_test": f"{pos_r*100:.1f}%",
        "public_score": "TBD",
        "neg_sampling": "N/A (training label KNN)",
        "n_features": K_NEIGHBORS,
        "notes": f"KNN collab: top-{K_NEIGHBORS} similar train queries, item overlap score",
        "file": saved_paths[k_name].name,
        "runtime_s": f"{time.time()-t0:.0f}",
    }
    log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)

log_df.to_csv(log_path, index=False)

# ─── Ozet ─────────────────────────────────────────────────────────────────────
total = time.time() - t0
print(f"\n{'='*60}")
print(f"OZET  |  Sure: {total:.0f}s ({total/60:.1f}dk)")
print(f"{'='*60}")
print(f"  Yontem   : KNN Collaborative Filtering")
print(f"  Neighbors: {K_NEIGHBORS} benzer training query")
print(f"  Fallback : item embedding benzerlik (skor=0 durumlar)")
print()
print(f"  YUKLENECEK DOSYALAR (ikisini de dene):")
for k_name, path in saved_paths.items():
    K = K_OPTIONS[k_name]
    pos = sum(predictions_by_k[k_name].values())
    print(f"  -> {path.name}  (K={K}, pozitif=%{pos/len(test)*100:.1f})")
print()
print("  Onceki skorlar: v1=0.48  v2b=0.48  v3=0.45")
print("  Beklenen: 0.55-0.70 (egitim label kullaniyor)")
print(f"{'='*60}")
