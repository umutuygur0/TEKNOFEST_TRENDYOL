"""
v7: XLM-RoBERTa Cross-Encoder + BM25 Hard Negatives
=====================================================
Neden 0.70 gerektirir bu yaklaşım:
  - 100 aday zaten BM25-seçili → keyword overlap işe yaramaz
  - Cross-encoder: (query + item) BIRLIKTE encode → ince fark öğrenir
  - BM25 hard negatives: eğitim negatifleri test dağılımına benzer
  - XLM-RoBERTa-base: Türkçe dahil 100 dil, solid multilingual BERT

Adımlar:
  1. TF-IDF index → her eğitim sorgusu için hard negative
  2. Cross-encoder fine-tune (~20-30 dk)
  3. Inference 3.36M pair (~25-30 dk)
  4. K=14 per query (mutlak)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from scipy.sparse import csr_matrix
import time, sys, random

sys.stdout.reconfigure(encoding="utf-8")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE  = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA  = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
SUBM  = BASE / "claude only" / "submissions"
MODEL_DIR = BASE / "claude only" / "models" / "crossencoder_v7"
SUBM.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

K_PREDICT   = 14    # MUTLAK — yüzde değil
HARD_NEG_PER_POS = 3   # Her pozitif için kaç hard negative
MAX_LENGTH  = 128   # Token max length
BATCH_TRAIN = 32    # Training batch size (daha küçük = daha stabil)
BATCH_INFER = 256   # Inference batch size
LR          = 2e-5
EPOCHS      = 1

LOWER_MAP = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def tr_lower(text): return str(text).translate(LOWER_MAP).lower().strip()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ─── 1. Veri Yükle ────────────────────────────────────────────────────────────
print("\n[1] Veri yükleniyor...")
t0 = time.time()
items  = pd.read_csv(DATA / "items.csv")
terms  = pd.read_csv(DATA / "terms.csv")
train  = pd.read_csv(DATA / "training_pairs.csv")
test   = pd.read_csv(DATA / "submission_pairs.csv")
sample = pd.read_csv(DATA / "sample_submission.csv")

iid_to_title  = dict(zip(items["item_id"], items["title"].fillna("")))
iid_to_brand  = dict(zip(items["item_id"], items["brand"].fillna("")))
iid_to_catl1  = {iid: str(cat).split("/")[0].strip()
                 for iid, cat in zip(items["item_id"], items["category"].fillna(""))}
tid_to_query  = dict(zip(terms["term_id"], terms["query"]))
item_id_arr   = items["item_id"].values          # pozisyon → item_id

train_pos = defaultdict(set)
for tid, iid in zip(train["term_id"].values, train["item_id"].values):
    train_pos[tid].add(iid)

print(f"  Yükleme: {time.time()-t0:.1f}s")

# ─── 2. TF-IDF Item Index (Hard Negative Mining) ─────────────────────────────
print("\n[2] TF-IDF index kuruluyor (hard negative mining için)...")
t0 = time.time()
from sklearn.feature_extraction.text import TfidfVectorizer

# İtem metinleri normalize et
item_texts = [
    f"{tr_lower(iid_to_title.get(iid,''))} {tr_lower(iid_to_brand.get(iid,''))} {tr_lower(iid_to_catl1.get(iid,''))}"
    for iid in item_id_arr
]

vectorizer = TfidfVectorizer(
    max_features=80000,
    min_df=2,
    sublinear_tf=True,
    analyzer="word",
    ngram_range=(1, 2)
)
item_tfidf = vectorizer.fit_transform(item_texts)  # (962K, 80K) sparse
print(f"  TF-IDF matrix: {item_tfidf.shape}  Süre: {time.time()-t0:.1f}s")

# ─── 3. Hard Negative Mining ─────────────────────────────────────────────────
print("\n[3] Hard negative mining...")
t0 = time.time()

# iid → idx haritası
iid_to_idx = {iid: i for i, iid in enumerate(item_id_arr)}

all_pairs = []   # (query_text, item_text, label)

BATCH_Q = 50     # Kaçar sorgu işlenecek (bellek için küçük)
train_tids = list(train_pos.keys())

for batch_start in range(0, len(train_tids), BATCH_Q):
    batch_tids = train_tids[batch_start:batch_start+BATCH_Q]

    # Sorgu vektörleri
    q_texts = [tr_lower(tid_to_query.get(tid, "")) for tid in batch_tids]
    q_vecs  = vectorizer.transform(q_texts)        # (BATCH_Q, vocab)

    # TF-IDF benzerlik skorları (batch_q × items)
    sims = (q_vecs @ item_tfidf.T).toarray()       # (BATCH_Q, 962K)

    for i, tid in enumerate(batch_tids):
        pos_iids = train_pos[tid]
        q_text   = q_texts[i]

        # Hard negatives: en yüksek TF-IDF skorlu ama pozitif OLMAYAN itemlar
        scores_i = sims[i]
        sorted_idx = np.argsort(scores_i)[::-1]   # yüksekten düşüğe
        hard_neg_iids = []
        for idx in sorted_idx:
            neg_iid = item_id_arr[idx]
            if neg_iid not in pos_iids:
                hard_neg_iids.append(neg_iid)
                if len(hard_neg_iids) >= len(pos_iids) * HARD_NEG_PER_POS:
                    break

        # Pozitif çiftler ekle
        for pos_iid in pos_iids:
            item_text = (
                f"{tr_lower(iid_to_title.get(pos_iid,''))} "
                f"{tr_lower(iid_to_brand.get(pos_iid,''))} "
                f"{tr_lower(iid_to_catl1.get(pos_iid,''))}"
            )
            all_pairs.append((q_text, item_text.strip(), 1))

        # Hard negative çiftler ekle
        for neg_iid in hard_neg_iids:
            item_text = (
                f"{tr_lower(iid_to_title.get(neg_iid,''))} "
                f"{tr_lower(iid_to_brand.get(neg_iid,''))} "
                f"{tr_lower(iid_to_catl1.get(neg_iid,''))}"
            )
            all_pairs.append((q_text, item_text.strip(), 0))

    if (batch_start // BATCH_Q) % 10 == 0:
        print(f"  Sorgu: {batch_start}/{len(train_tids)}  Çift: {len(all_pairs):,}")

pos_count = sum(1 for _, _, l in all_pairs if l == 1)
neg_count = sum(1 for _, _, l in all_pairs if l == 0)
print(f"  Toplam çift: {len(all_pairs):,}  (pos={pos_count:,}, neg={neg_count:,})")
print(f"  Negatif mining süresi: {time.time()-t0:.1f}s")

random.shuffle(all_pairs)

# ─── 4. Cross-Encoder Fine-Tune ──────────────────────────────────────────────
print("\n[4] XLM-RoBERTa cross-encoder fine-tuning...")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        q, item, label = self.pairs[idx]
        return q, item, label

def collate_fn(batch):
    queries, items, labels = zip(*batch)
    enc = tokenizer(
        list(queries), list(items),
        max_length=MAX_LENGTH,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    return enc, torch.tensor(labels, dtype=torch.float32)

dataset = PairDataset(all_pairs)
loader  = DataLoader(dataset, batch_size=BATCH_TRAIN, shuffle=True,
                     collate_fn=collate_fn, num_workers=0)

model = AutoModelForSequenceClassification.from_pretrained(
    "xlm-roberta-base", num_labels=1
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scaler    = GradScaler()
loss_fn   = nn.BCEWithLogitsLoss()

# LR scheduler
total_steps = len(loader) * EPOCHS
warmup_steps = min(500, total_steps // 10)
from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

print(f"  Toplam adım: {total_steps:,}  Warmup: {warmup_steps}")
t0 = time.time()
model.train()

running_loss = 0.0
for step, (enc, labels) in enumerate(loader):
    enc    = {k: v.to(device) for k, v in enc.items()}
    labels = labels.to(device)

    with autocast():
        logits = model(**enc).logits.squeeze(-1)
        loss   = loss_fn(logits, labels)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad()

    running_loss += loss.item()
    if (step + 1) % 500 == 0:
        avg = running_loss / 500
        elapsed = time.time() - t0
        remaining = elapsed / (step + 1) * (total_steps - step - 1)
        print(f"  Step {step+1}/{total_steps}  loss={avg:.4f}  "
              f"geçen={elapsed/60:.1f}dk  kalan={remaining/60:.1f}dk")
        running_loss = 0.0

model.save_pretrained(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))
print(f"  Eğitim tamamlandı: {(time.time()-t0)/60:.1f} dakika")

# ─── 5. Inference ─────────────────────────────────────────────────────────────
print("\n[5] Test inference başlıyor (3.36M çift)...")
t0 = time.time()
model.eval()

# Test verisi: (query_text, item_text, row_id, term_id)
print("  Test çiftleri hazırlanıyor...")
test_q_texts    = [tr_lower(tid_to_query.get(tid, "")) for tid in test["term_id"]]
test_item_texts = [
    f"{tr_lower(iid_to_title.get(iid,''))} "
    f"{tr_lower(iid_to_brand.get(iid,''))} "
    f"{tr_lower(iid_to_catl1.get(iid,''))}"
    for iid in test["item_id"]
]
test_ids   = test["id"].values
test_tids  = test["term_id"].values

# Batch inference
all_scores = np.zeros(len(test), dtype=np.float32)
interval   = max(1, len(test) // BATCH_INFER // 20)

with torch.no_grad():
    for b_start in range(0, len(test), BATCH_INFER):
        b_end = min(b_start + BATCH_INFER, len(test))
        q_batch    = test_q_texts[b_start:b_end]
        item_batch = [t.strip() for t in test_item_texts[b_start:b_end]]

        enc = tokenizer(
            q_batch, item_batch,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with autocast():
            logits = model(**enc).logits.squeeze(-1)
        scores = torch.sigmoid(logits).cpu().numpy()
        all_scores[b_start:b_end] = scores

        batch_num = b_start // BATCH_INFER
        if batch_num % interval == 0:
            elapsed = time.time() - t0
            pct = 100 * b_end / len(test)
            remaining = elapsed / max(b_end, 1) * (len(test) - b_end)
            print(f"  {pct:.0f}%  {elapsed/60:.1f}dk geçti  ~{remaining/60:.1f}dk kaldı")

print(f"  Inference tamamlandı: {(time.time()-t0)/60:.1f} dakika")

# ─── 6. Top-K Selection ───────────────────────────────────────────────────────
print("\n[6] Top-K tahmin (K=14 MUTLAK)...")
t0 = time.time()

# test_copy: id, term_id, item_id, score, prediction
test_copy = test.copy()
test_copy["score"] = all_scores
test_copy["prediction"] = 0

for tid, grp in test_copy.groupby("term_id"):
    top_k_idx = grp.nlargest(K_PREDICT, "score").index
    test_copy.loc[top_k_idx, "prediction"] = 1

print(f"  Tahmin tamamlandı: {time.time()-t0:.1f}s")

# ─── 7. Submission ────────────────────────────────────────────────────────────
print("\n[7] Submission oluşturuluyor...")
pred_df = test_copy[["id", "prediction"]]
sub = sample[["id"]].merge(pred_df, on="id", how="left")
sub["prediction"] = sub["prediction"].fillna(0).astype(int)

pos_count_total = sub["prediction"].sum()
print(f"  Toplam: {len(sub):,}  Pozitif: {pos_count_total:,} ({100*pos_count_total/len(sub):.1f}%)")
print(f"  Query başına ort: {pos_count_total/test['term_id'].nunique():.1f}")

out_path = SUBM / "submission_v7_crossencoder.csv"
sub.to_csv(str(out_path), index=False)
print(f"  Kaydedildi: {out_path}")

# Log
log_path = SUBM / "submissions_log.csv"
log = pd.read_csv(str(log_path))
new_row = pd.DataFrame([{
    "version": "v7",
    "description": "XLM-RoBERTa cross-encoder + BM25 hard negatives, K=14",
    "positive_rate": f"{100*pos_count_total/len(sub):.1f}%",
    "public_score": "TBD",
    "notes": "Cross-encoder jointly encodes (query, item) for fine-grained reranking"
}])
log = pd.concat([log, new_row], ignore_index=True)
log.to_csv(str(log_path), index=False)

print("\n=== TAMAMLANDI ===")
