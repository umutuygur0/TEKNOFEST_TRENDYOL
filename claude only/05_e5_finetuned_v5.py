# -*- coding: utf-8 -*-
"""
Submission v5 — multilingual-e5-base Fine-tuned + KNN Collaborative
====================================================================
Neden oncekiler basarisiz oldu:
  - BM25/overlap: test'te TERS KORELASYONLU (test adaylari BM25 zor negatifleri)
  - MiniLM paraphrase modeli: asimetrik query-document icin YANLIS model
  - Zero-shot: Turkce e-ticaret icin domain adaptasyonu YOK

Bu yaklasim:
  1. multilingual-e5-base: ozellikle retrieval icin egitilmis (asymmetric query-doc)
  2. Fine-tuning: 250K Turkce e-ticaret pozitif ciftleriyle domain adaptasyonu
  3. KNN collaborative: egitim labellarindan gercek sinyal
  4. Turkce karakter normalizasyonu
  5. Kalibre K=8 (50/50 split matematiğinden)

Beklenti: 0.60-0.80 F1 (onceki max 0.48)
"""

import pandas as pd
import numpy as np
import time
import os
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ─── Sabitler ─────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL\trendyol-e-ticaret-yarismasi-2026-kaggle")
SUBM_DIR   = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL\claude only\submissions")
CACHE_DIR  = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL\claude only\emb_cache")
MODEL_DIR  = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL\claude only\models")
SUBM_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

E5_MODEL   = "intfloat/multilingual-e5-base"
FINETUNED  = str(MODEL_DIR / "e5_base_finetuned_v5")
K_PREDICT  = 8      # 50/50 split: ~250K test pos / 32K queries = 7.8 pos/query
K_NEIGHBORS = 50    # KNN collaborative: top-50 benzer egitim sorgusu

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

t0 = time.time()
print("=" * 65)
print("v5: multilingual-e5-base Fine-tuned + KNN Collaborative")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# BOLUM 1: VERI YUKLEME
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/8] Veri yukleniyor...")
items  = pd.read_csv(DATA_DIR / "items.csv")
terms  = pd.read_csv(DATA_DIR / "terms.csv")
train  = pd.read_csv(DATA_DIR / "training_pairs.csv")
test   = pd.read_csv(DATA_DIR / "submission_pairs.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
print(f"      items:{len(items):,}  terms:{len(terms):,}  train:{len(train):,}  test:{len(test):,}")

tid_to_query = dict(zip(terms["term_id"], terms["query"]))


# ─────────────────────────────────────────────────────────────────────────────
# BOLUM 2: TURKCE METİN NORMALİZASYONU
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/8] Turkce metin normalizasyonu...")

# Python str.lower() Turkce icin yanlis: "I".lower()="i" (ı olmali), "İ".lower()="i̇" (i olmali)
LOWER_MAP = str.maketrans(
    "IiIİŞĞÜÖÇ",
    "iiiışğüöç"
)
# Duzeltme: sadece buyuk -> kucuk harfler
LOWER_MAP = str.maketrans(
    "İŞĞÜÖÇI",   # İŞĞÜÖÇI
    "işğüöçı"    # işğüöçı
)

def tr_normalize(text):
    """Turkce karakter normalizasyonu (I->i, I->i sorununu cozme)."""
    if not isinstance(text, str):
        return ""
    return text.translate(LOWER_MAP).lower().strip()

def make_query_text(query_str):
    """E5 query format: 'query: ' prefix ile."""
    return "query: " + tr_normalize(str(query_str))

def make_item_text(title, brand="", cat=""):
    """E5 passage format: title + brand + kategori L1."""
    parts = []
    if title:
        parts.append(tr_normalize(str(title)))
    if brand and str(brand).strip().lower() not in ("nan", "none", ""):
        parts.append(tr_normalize(str(brand)))
    if cat and str(cat).strip().lower() not in ("nan", "none", ""):
        cat_l1 = str(cat).split("/")[0].strip()
        if cat_l1:
            parts.append(tr_normalize(cat_l1))
    return "passage: " + " ".join(parts)

# Item metin vektorizasyonu (hizli)
items["cat_str"] = items["category"].fillna("") if "category" in items.columns else ""
items["brand_str"] = items["brand"].fillna("") if "brand" in items.columns else ""
items["title_str"] = items["title"].fillna("") if "title" in items.columns else ""

items["e5_text"] = items.apply(
    lambda r: make_item_text(r["title_str"], r["brand_str"], r["cat_str"]), axis=1
)

iid_to_text = dict(zip(items["item_id"].values, items["e5_text"].values))
item_id_arr  = items["item_id"].values
print(f"      Ornek item: {items['e5_text'].iloc[0][:80]}")
print(f"      Ornek query: {make_query_text(tid_to_query.get(list(tid_to_query.keys())[0], ''))[:60]}")


# ─────────────────────────────────────────────────────────────────────────────
# BOLUM 3: MODEL YUKLEME (E5-BASE)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/8] Model yukleniyor / fine-tuning...")
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"      Device: {device}")
if device == "cuda":
    print(f"      GPU: {torch.cuda.get_device_name(0)}")
    print(f"      VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# Fine-tuned model varsa yukle, yoksa fine-tune et
if os.path.isdir(FINETUNED) and len(os.listdir(FINETUNED)) > 0:
    print(f"      Fine-tuned model bulundu: {FINETUNED}")
    model = SentenceTransformer(FINETUNED, device=device)
    print("      Yuklendi.")
else:
    print(f"      Base model indiriliyor: {E5_MODEL}")
    model = SentenceTransformer(E5_MODEL, device=device)

    # ─── Fine-Tuning Dataseti ───────────────────────────────────────────────
    print(f"      Training dataseti hazirlaniyor ({len(train):,} pozitif cift)...")
    train_examples = []
    skipped = 0
    for tid, iid in tqdm(zip(train["term_id"].values, train["item_id"].values),
                         total=len(train), ncols=65):
        q_text = make_query_text(tid_to_query.get(tid, ""))
        i_text = iid_to_text.get(iid, "")
        if q_text and i_text and len(q_text) > 10 and len(i_text) > 15:
            train_examples.append(InputExample(texts=[q_text, i_text]))
        else:
            skipped += 1

    random.shuffle(train_examples)
    print(f"      {len(train_examples):,} gecerli cift hazir ({skipped} atlandi)")

    # ─── Fine-Tuning ────────────────────────────────────────────────────────
    BATCH_SIZE = 128  # 17GB VRAM icin guvenli, 127 in-batch negative
    EPOCHS     = 1    # Domain adaptasyonu icin 1 epoch yeterli
    LR         = 2e-5

    dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    loss_fn    = losses.MultipleNegativesRankingLoss(model, scale=20.0)

    steps_per_epoch = len(dataloader)
    warmup = min(200, int(0.1 * steps_per_epoch))

    print(f"\n      Fine-tuning baslaniyor...")
    print(f"      Batch:{BATCH_SIZE}  Epochs:{EPOCHS}  LR:{LR}  Steps:{steps_per_epoch}")
    print(f"      Warmup:{warmup}  In-batch negatives per step:{BATCH_SIZE-1}")

    t_ft = time.time()
    model.fit(
        train_objectives=[(dataloader, loss_fn)],
        epochs=EPOCHS,
        warmup_steps=warmup,
        optimizer_params={"lr": LR},
        show_progress_bar=True,
        use_amp=(device == "cuda"),   # Mixed precision: 2x hiz
        checkpoint_save_steps=999999, # Checkpoint kaydetme (zaman tasarrufu)
    )
    ft_time = time.time() - t_ft
    print(f"      Fine-tuning tamamlandi [{ft_time:.0f}s = {ft_time/60:.1f}dk]")

    model.save(FINETUNED)
    print(f"      Model kaydedildi: {FINETUNED}")


# ─────────────────────────────────────────────────────────────────────────────
# BOLUM 4: ITEM EMBEDDINGleri (Fine-tuned model ile)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/8] Item embeddinglari encode ediliyor (GPU)...")

ITEM_EMB_PATH = CACHE_DIR / "item_embs_e5_finetuned_v5.npy"
ITEM_IDS_PATH = CACHE_DIR / "item_ids_e5_finetuned_v5.npy"

if ITEM_EMB_PATH.exists() and ITEM_IDS_PATH.exists():
    item_embs    = np.load(str(ITEM_EMB_PATH))
    item_ids_arr = np.load(str(ITEM_IDS_PATH), allow_pickle=True)
    if len(item_ids_arr) == len(items):
        iid_to_emb_idx = {iid: i for i, iid in enumerate(item_ids_arr)}
        print(f"      Cache'den yuklendi: {item_embs.shape}")
    else:
        item_embs = None
else:
    item_embs = None

if item_embs is None:
    t_enc = time.time()
    item_texts_list = items["e5_text"].tolist()
    item_embs = model.encode(
        item_texts_list,
        batch_size=512,
        show_progress_bar=True,
        normalize_embeddings=True,
        device=device,
        convert_to_numpy=True,
    )
    item_ids_arr   = items["item_id"].values
    iid_to_emb_idx = {iid: i for i, iid in enumerate(item_ids_arr)}
    np.save(str(ITEM_EMB_PATH), item_embs)
    np.save(str(ITEM_IDS_PATH), item_ids_arr)
    print(f"      Tamamlandi [{time.time()-t_enc:.0f}s]  Shape: {item_embs.shape}")
    print(f"      Cache'e kaydedildi: {ITEM_EMB_PATH.name}")


# ─────────────────────────────────────────────────────────────────────────────
# BOLUM 5: TEST VE EGITIM QUERY EMBEDDINGleri
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/8] Query embeddingleri encode ediliyor...")

TEST_Q_EMB  = CACHE_DIR / "test_q_embs_e5_v5.npy"
TEST_Q_IDS  = CACHE_DIR / "test_q_ids_e5_v5.npy"
TRAIN_Q_EMB = CACHE_DIR / "train_q_embs_e5_v5.npy"
TRAIN_Q_IDS = CACHE_DIR / "train_q_ids_e5_v5.npy"

test_tids  = test["term_id"].unique()
train_tids = train["term_id"].unique()

# Test queries
if TEST_Q_EMB.exists():
    test_q_embs = np.load(str(TEST_Q_EMB))
    test_q_ids  = np.load(str(TEST_Q_IDS), allow_pickle=True)
    if len(test_q_ids) == len(test_tids):
        tid_to_tq_idx = {tid: i for i, tid in enumerate(test_q_ids)}
        print(f"      Test query cache: {test_q_embs.shape}")
    else:
        test_q_embs = None
else:
    test_q_embs = None

if test_q_embs is None:
    test_qstrs = [make_query_text(tid_to_query.get(tid, "")) for tid in test_tids]
    test_q_embs = model.encode(
        test_qstrs, batch_size=512, show_progress_bar=True,
        normalize_embeddings=True, device=device, convert_to_numpy=True)
    tid_to_tq_idx = {tid: i for i, tid in enumerate(test_tids)}
    np.save(str(TEST_Q_EMB), test_q_embs)
    np.save(str(TEST_Q_IDS), test_tids)
    print(f"      Test queries encoded: {test_q_embs.shape}")

# Training queries (KNN icin)
if TRAIN_Q_EMB.exists():
    train_q_embs = np.load(str(TRAIN_Q_EMB))
    train_q_ids  = np.load(str(TRAIN_Q_IDS), allow_pickle=True)
    if len(train_q_ids) == len(train_tids):
        train_q_id_to_idx = {tid: i for i, tid in enumerate(train_q_ids)}
        print(f"      Train query cache: {train_q_embs.shape}")
    else:
        train_q_embs = None
else:
    train_q_embs = None

if train_q_embs is None:
    train_qstrs = [make_query_text(tid_to_query.get(tid, "")) for tid in train_tids]
    train_q_embs = model.encode(
        train_qstrs, batch_size=512, show_progress_bar=True,
        normalize_embeddings=True, device=device, convert_to_numpy=True)
    train_q_id_to_idx = {tid: i for i, tid in enumerate(train_tids)}
    np.save(str(TRAIN_Q_EMB), train_q_embs)
    np.save(str(TRAIN_Q_IDS), train_tids)
    print(f"      Train queries encoded: {train_q_embs.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# BOLUM 6: KNN COLLABORATIVE — TEST x TRAIN BENZERLIK
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[6/8] KNN similarity (GPU, top-{K_NEIGHBORS} egitim sorgusu)...")
t_sim = time.time()

if device == "cuda":
    tq_gpu  = torch.from_numpy(test_q_embs).float().to("cuda")
    trq_gpu = torch.from_numpy(train_q_embs).float().to("cuda")

    BATCH = 2000  # GPU batching
    knn_top_idxs = []
    for i in range(0, len(test_q_embs), BATCH):
        batch = tq_gpu[i:i+BATCH]
        sims  = batch @ trq_gpu.T
        top_idxs = sims.topk(K_NEIGHBORS, dim=1).indices
        knn_top_idxs.append(top_idxs.cpu().numpy())

    knn_top_idxs = np.vstack(knn_top_idxs)  # 32K × K_NEIGHBORS
    del tq_gpu, trq_gpu
    torch.cuda.empty_cache()
else:
    sim = test_q_embs @ train_q_embs.T
    knn_top_idxs = np.argsort(sim, axis=1)[:, -K_NEIGHBORS:]

print(f"      Tamamlandi [{time.time()-t_sim:.1f}s]  Shape: {knn_top_idxs.shape}")

# Item → egitim query index seti
print("      Item->training query map olusturuluyor...")
item_to_train_q = defaultdict(set)
for tid, iid in zip(train["term_id"].values, train["item_id"].values):
    q_idx = train_q_id_to_idx.get(tid)
    if q_idx is not None:
        item_to_train_q[iid].add(q_idx)

n_test_items_in_train = sum(1 for iid in test["item_id"].unique() if iid in item_to_train_q)
total_test_items = test["item_id"].nunique()
print(f"      Egitimde gorunden test itemlari: {n_test_items_in_train:,}/{total_test_items:,} "
      f"(%{100*n_test_items_in_train/total_test_items:.1f})")


# ─────────────────────────────────────────────────────────────────────────────
# BOLUM 7: SKOR HESAPLAMA (E5 cosine + KNN collaborative)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/8] Test ciftleri skorlaniyor (E5 cosine + KNN hybrid)...")
t_score = time.time()

test_copy = test.copy()
test_copy["final_score"] = 0.0

test_tids_set = set(test_tids)

for tid, grp in tqdm(test_copy.groupby("term_id"), total=len(test_tids), ncols=65):
    tq_idx = tid_to_tq_idx.get(tid)
    if tq_idx is None:
        continue

    iids = grp["item_id"].values
    n    = len(iids)

    # ── E5 Cosine Similarity ──────────────────────────────────────────────
    q_emb     = test_q_embs[tq_idx]       # 768-dim
    emb_idxs  = np.array([iid_to_emb_idx.get(iid, -1) for iid in iids])
    valid_mask = emb_idxs >= 0
    e5_scores  = np.zeros(n)

    if valid_mask.any():
        cand_embs = item_embs[emb_idxs[valid_mask]]
        sims      = cand_embs @ q_emb       # cosine (L2 normalized)
        e5_scores[valid_mask] = sims

    # ── KNN Collaborative Score ───────────────────────────────────────────
    top_train_set = set(knn_top_idxs[tq_idx].tolist())
    knn_scores    = np.zeros(n)

    for j, iid in enumerate(iids):
        item_qs = item_to_train_q.get(iid)
        if item_qs:
            knn_scores[j] = len(item_qs & top_train_set)

    # ── Normalize & Combine ───────────────────────────────────────────────
    # E5: [-1,1] → [0,1]
    e5_min, e5_max = e5_scores.min(), e5_scores.max()
    e5_norm = (e5_scores - e5_min) / (e5_max - e5_min + 1e-8)

    # KNN: [0, K_NEIGHBORS] → [0,1]
    knn_max = knn_scores.max()
    knn_norm = knn_scores / (knn_max + 1e-8) if knn_max > 0 else knn_scores

    # Adaptif agirlik: KNN sinyali varsa -> KNN'e guvven, yoksa -> E5'e guvven
    knn_coverage = (knn_scores > 0).sum() / n
    alpha = min(0.75, 0.15 + 0.6 * knn_coverage)  # 0.15 (coverage=0) → 0.75 (coverage=1)

    combined = alpha * knn_norm + (1.0 - alpha) * e5_norm
    test_copy.loc[grp.index, "final_score"] = combined

elapsed_score = time.time() - t_score
print(f"      Tamamlandi [{elapsed_score:.1f}s]")

# Istatistikler
knn_active = (test_copy["final_score"] > 0.15).sum()
print(f"      Skor > 0.15 olan tahminler: {knn_active:,} ({100*knn_active/len(test_copy):.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# BOLUM 8: PER-QUERY TOP-K TAHMİN + KAYIT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[8/8] Per-query top-{K_PREDICT} tahmin olusturuluyor ve kaydediliyor...")

test_copy["prediction"] = 0
for tid, grp in test_copy.groupby("term_id"):
    k = max(1, round(K_PREDICT / 100 * len(grp)))
    top_k_idx = grp.nlargest(k, "final_score").index
    test_copy.loc[top_k_idx, "prediction"] = 1

pos_count = test_copy["prediction"].sum()
pos_rate  = pos_count / len(test_copy)
print(f"      Pozitif tahmin: {pos_count:,} ({pos_rate*100:.2f}%)")

# Submission dosyasi
pred_df  = test_copy[["id", "prediction"]]
sub      = sample[["id"]].merge(pred_df, on="id", how="left")
sub["prediction"] = sub["prediction"].fillna(0).astype(int)

sub_path = SUBM_DIR / "submission_v5_e5_finetuned_k8.csv"
sub.to_csv(sub_path, index=False)
print(f"      Kaydedildi: {sub_path.name}")

# ─── Log ──────────────────────────────────────────────────────────────────────
log_path = SUBM_DIR / "submissions_log.csv"
log_df = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
total_sec = time.time() - t0
new_row = {
    "version": "v5", "model": "multilingual-e5-base (fine-tuned 1ep)",
    "val_macro_f1": "N/A", "threshold": f"top-{K_PREDICT}",
    "pos_rate_train": "N/A", "pos_rate_test": f"{pos_rate*100:.1f}%",
    "public_score": "TBD",
    "neg_sampling": "in-batch (MultipleNegativesRankingLoss)",
    "n_features": "E5-768 + KNN",
    "notes": (f"E5-base fine-tuned + KNN collab (K_neighbors={K_NEIGHBORS}), "
              f"Turkce normalize, alpha adaptive"),
    "file": sub_path.name,
    "runtime_s": f"{total_sec:.0f}",
}
log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
log_df.to_csv(log_path, index=False)

# ─── Final Ozet ───────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"TAMAMLANDI  |  Sure: {total_sec:.0f}s ({total_sec/60:.1f}dk)")
print(f"{'='*65}")
print(f"  Yontem    : E5-base fine-tuned + KNN Collaborative")
print(f"  Model     : {E5_MODEL} (1 epoch, {len(train):,} pozitif cift)")
print(f"  Loss      : MultipleNegativesRankingLoss (in-batch neg)")
print(f"  KNN K     : {K_NEIGHBORS} benzer egitim sorgusu")
print(f"  Predict K : {K_PREDICT} pozitif/100 aday")
print(f"  Pozitif %%: {pos_rate*100:.2f}%")
print()
print(f"  YUKLE: {sub_path.name}")
print()
print(f"  Onceki skorlar: v1=0.48  v2b=0.48  v3=0.45")
print(f"  Beklenen: 0.60-0.80 (domain-adapt + label sinyal)")
print(f"{'='*65}")
