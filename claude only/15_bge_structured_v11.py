"""
15_bge_structured_v11.py — BGE Reranker + Yapısal Negatifler → v11 Submission
===============================================================================
Önceki hatalar:
  v9: cross-query neg (çok kolay) → 0.49
  S1: BGE zero-shot → 0.502 (sadece +0.012)

Bu versiyon:
  Model : BAAI/bge-reranker-v2-m3 (568M)
  Veri  : 250K pozitif + yapısal negatifler (brand_swap, gender_swap, age_swap)
  Neden : Model artık "avon luck" vs "oriflame love" ayrımını öğreniyor
  Hedef : CV > 0.60

Submission: submissions/submission_v11_bge_structured.csv

Çalıştır:
  python "claude only/15_bge_structured_v11.py"
"""

import random
import time
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.feature_extraction.text import TfidfVectorizer

sys.stdout.reconfigure(encoding="utf-8")
random.seed(42)

BASE   = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA   = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
PARSED = BASE / "claude only" / "parsed_queries"
MODELS = BASE / "claude only" / "models"
SUBM   = BASE / "claude only" / "submissions"
MODELS.mkdir(parents=True, exist_ok=True)
SUBM.mkdir(parents=True, exist_ok=True)

# ─── Hiperparametreler ──────────────────────────────────────────────────────
MODEL_NAME   = "BAAI/bge-reranker-v2-m3"
MAX_LENGTH   = 256
BATCH_TRAIN  = 32        # gradient_accumulation ile efektif 32
GRAD_ACC     = 1
EPOCHS       = 1
LR           = 2e-5
WARMUP_RATIO = 0.05
K_PREDICT    = 14
CV_SAMPLES   = 500
CV_THRESHOLD = 0.58      # bu altındaysa test inference ATLA
BATCH_INFER  = 256

LOWER_MAP = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(t): return str(t).translate(LOWER_MAP).lower().strip()

# ─── 1. Veri Yükle ──────────────────────────────────────────────────────────
print("[1] Veri yükleniyor...")
t0 = time.time()
items    = pd.read_csv(DATA / "items.csv")
terms    = pd.read_csv(DATA / "terms.csv")
train_df = pd.read_csv(DATA / "training_pairs.csv")
test_df  = pd.read_csv(DATA / "submission_pairs.csv")
sample   = pd.read_csv(DATA / "sample_submission.csv")
struct   = pd.read_csv(PARSED / "training_pairs_structured.csv")

iid_to_title    = dict(zip(items["item_id"], items["title"].fillna("").apply(trl)))
iid_to_brand    = dict(zip(items["item_id"], items["brand"].fillna("").apply(trl)))
iid_to_catl1    = {iid: c.split("/")[0] for iid, c in
                   zip(items["item_id"], items["category"].fillna(""))}
tid_to_query    = dict(zip(terms["term_id"], terms["query"]))

train_pos = defaultdict(set)
for tid, iid in zip(train_df["term_id"].values, train_df["item_id"].values):
    train_pos[tid].add(iid)

print(f"  {time.time()-t0:.1f}s | Yapısal çift: {len(struct):,} "
      f"(pos={( struct['label']==1).sum():,}, neg={(struct['label']==0).sum():,})")

# ─── 2. Item Metin Fonksiyonu ────────────────────────────────────────────────
def item_text(iid):
    t = iid_to_title.get(iid, "")
    b = iid_to_brand.get(iid, "")
    c = iid_to_catl1.get(iid, "").split("/")[0]
    parts = [p for p in [t, b, c] if p and p != "nan"]
    return " | ".join(parts)

# ─── 3. Dataset ─────────────────────────────────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.queries = df["query_text"].tolist()
        self.items   = df["item_text"].tolist()
        self.labels  = df["label"].tolist()
        self.tok     = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tok(
            self.queries[idx], self.items[idx],
            max_length=MAX_LENGTH, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.float),
        }

# ─── 4. Model Yükle + Fine-Tune ─────────────────────────────────────────────
print(f"\n[2] Model yükleniyor: {MODEL_NAME}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=1
).to(device)

# Eğitim verisi (tüm yapısal çiftler)
train_set = PairDataset(struct, tokenizer)
loader    = DataLoader(train_set, batch_size=BATCH_TRAIN, shuffle=True,
                       num_workers=0, pin_memory=True)

total_steps   = (len(loader) // GRAD_ACC) * EPOCHS
warmup_steps  = int(total_steps * WARMUP_RATIO)

optimizer = AdamW(model.parameters(), lr=LR)
from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"\n[3] Fine-tuning...")
print(f"  Çift: {len(struct):,} | Batch: {BATCH_TRAIN} | Steps: {total_steps:,} | Warmup: {warmup_steps}")
t0     = time.time()
model.train()
loss_fn = torch.nn.BCEWithLogitsLoss()
step    = 0
running_loss = 0.0

for epoch in range(EPOCHS):
    for bi, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["label"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits.squeeze(-1)
        loss   = loss_fn(logits, labels)
        loss.backward()
        running_loss += loss.item()

        if (bi + 1) % GRAD_ACC == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % 200 == 0:
                elapsed  = time.time() - t0
                eta      = elapsed / step * (total_steps - step)
                avg_loss = running_loss / 200
                running_loss = 0.0
                print(f"  step {step:5d}/{total_steps} | loss={avg_loss:.4f} | "
                      f"{elapsed/60:.1f}dk geçti | ETA {eta/60:.1f}dk")

    print(f"  Epoch {epoch+1} bitti | {(time.time()-t0)/60:.1f} dakika")

# Model kaydet
save_path = MODELS / "bge_structured_v11"
model.save_pretrained(str(save_path))
tokenizer.save_pretrained(str(save_path))
print(f"  Model kaydedildi: {save_path}")

# ─── 5. CV — Gerçekçi Değerlendirme ─────────────────────────────────────────
print(f"\n[4] Gerçekçi CV ({CV_SAMPLES} sorgu, BM25 top-100 aday)...")

# BM25 aday indeksi
all_iids  = items["item_id"].tolist()
all_itxts = [
    trl(iid_to_title.get(i, "") + " " + iid_to_brand.get(i, ""))
    for i in all_iids
]
item_vect = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(2, 4),
    min_df=3, max_features=200_000, sublinear_tf=True
)
item_mat = item_vect.fit_transform(all_itxts)

random.seed(42)
all_tids   = list(train_pos.keys())
random.shuffle(all_tids)
holdout    = all_tids[:CV_SAMPLES]

model.eval()

def score_pairs(pairs):
    all_scores = []
    for i in range(0, len(pairs), BATCH_INFER):
        batch_pairs = pairs[i:i + BATCH_INFER]
        enc = tokenizer(
            [p[0] for p in batch_pairs],
            [p[1] for p in batch_pairs],
            max_length=MAX_LENGTH, truncation=True,
            padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits.squeeze(-1)
        all_scores.extend(torch.sigmoid(logits).cpu().tolist())
    return all_scores

cv_results = []
cv_t0 = time.time()

for qi, tid in enumerate(holdout):
    q_text   = trl(tid_to_query.get(tid, ""))
    true_pos = train_pos[tid]
    if not q_text or not true_pos:
        continue

    q_vec    = item_vect.transform([q_text])
    sims     = (q_vec * item_mat.T).toarray()[0]
    top100   = np.argpartition(sims, -100)[-100:]
    top100   = top100[np.argsort(sims[top100])[::-1]]
    cands    = [all_iids[i] for i in top100]

    pairs   = [(q_text, item_text(iid)) for iid in cands]
    scores  = score_pairs(pairs)

    ranked  = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)
    pred    = set(iid for iid, _ in ranked[:K_PREDICT])

    tp = len(pred & true_pos)
    fp = len(pred - true_pos)
    fn = len(true_pos - pred)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    cv_results.append({"f1": f1, "tp": tp, "fp": fp, "fn": fn,
                        "n_true": len(true_pos),
                        "tp_in_cands": len(true_pos & set(cands))})

    if qi % 50 == 0 and qi > 0:
        so_far = [r["f1"] for r in cv_results]
        print(f"  {qi}/{CV_SAMPLES} | F1 so far: {np.mean(so_far):.3f} | "
              f"{time.time()-cv_t0:.0f}s")

df_cv = pd.DataFrame(cv_results)
tp_t  = df_cv["tp"].sum(); fp_t = df_cv["fp"].sum(); fn_t = df_cv["fn"].sum()
prec  = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
rec   = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
f1_1  = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
total_neg = 100 * len(df_cv) - df_cv["n_true"].sum()
tn    = total_neg - fp_t
prec0 = tn / (tn + fn_t) if (tn + fn_t) > 0 else 0
rec0  = tn / (tn + fp_t) if (tn + fp_t) > 0 else 0
f1_0  = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0
macro = (f1_0 + f1_1) / 2

print(f"\n{'='*60}")
print(f"CV SONUÇLARI — v11 (BGE + yapısal neg)")
print(f"{'='*60}")
print(f"  Precision: {prec:.3f} | Recall: {rec:.3f}")
print(f"  F1 class1: {f1_1:.3f} | F1 class0: {f1_0:.3f}")
print(f"  MACRO F1 : {macro:.3f}")
print(f"\n  Karşılaştırma:")
print(f"    v9  (cross-query neg) : 0.490")
print(f"    S1  (BGE zero-shot)   : 0.502")
print(f"    v11 (BGE + struct neg): {macro:.3f}  ← YENİ")
improvement = macro - 0.502
print(f"    S1'e göre: {'+' if improvement >= 0 else ''}{improvement:.3f}")

# ─── 6. Test Inference ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
if macro >= CV_THRESHOLD:
    print(f"  ✓ CV={macro:.3f} ≥ {CV_THRESHOLD} → Test inference başlıyor!")
    print(f"{'='*60}")

    test_grp   = test_df.groupby("term_id")
    test_items = test_grp["item_id"].apply(list).to_dict()
    test_ids   = test_grp["id"].apply(list).to_dict()
    test_tids  = list(test_items.keys())

    predictions = {}
    t0 = time.time()

    for i, tid in enumerate(test_tids):
        q_text  = trl(tid_to_query.get(tid, ""))
        cands   = test_items[tid]
        row_ids = test_ids[tid]

        pairs  = [(q_text, item_text(iid)) for iid in cands]
        scores = score_pairs(pairs)

        ranked = sorted(zip(row_ids, cands, scores), key=lambda x: x[2], reverse=True)
        for j, (pid, iid, sc) in enumerate(ranked):
            predictions[pid] = 1 if j < K_PREDICT else 0

        if i % 2000 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (len(test_tids) - i - 1)
            print(f"  {i:6d}/{len(test_tids)} | {elapsed/60:.1f}dk | ETA {eta/60:.1f}dk")

    sub = sample[["id"]].copy()
    sub["prediction"] = sub["id"].map(predictions).fillna(0).astype(int)
    pos_count = sub["prediction"].sum()
    print(f"\n  Pozitif: {pos_count:,} ({100*pos_count/len(sub):.1f}%) | Beklenen ~13.4%")

    out_path = SUBM / "submission_v11_bge_structured.csv"
    sub.to_csv(str(out_path), index=False)
    print(f"  → {out_path}")
    print(f"\n  STATUS.md'yi güncelle: v11 CV={macro:.3f}")
    print(f"  Kaggle'a yükle ve public skoru buraya yaz!")

else:
    print(f"  ✗ CV={macro:.3f} < {CV_THRESHOLD} → Test inference atlandı.")
    print(f"  Sonraki adım: 16_qwen_zeroshot_v12.py (LLM)")
    print(f"{'='*60}")
