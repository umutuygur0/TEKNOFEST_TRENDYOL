"""
15b_bge_structured_v11_fast.py — BGE + Yapısal Neg (HIZLI VERSİYON)
======================================================================
v15 hataları:
  ✗ __getitem__ içinde tokenize → 511K × ~2s = 9+ saat
  ✗ fp32 (mixed precision yok) → 15.4GB VRAM
  ✗ num_workers=0 → GPU tokenize bekliyor

Bu versiyon:
  ✓ Pre-tokenize: tüm 511K çift başta tokenize edilir (~3-5 dk)
  ✓ fp16 autocast → ~8GB VRAM (yarısı)
  ✓ Tüm tensörler RAM'de → DataLoader anında iter
  Beklenen süre: ~30-50 dakika (9 saat yerine)

Çalıştır:
  python "claude only/15b_bge_structured_v11_fast.py"
"""

import random, time, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from sklearn.feature_extraction.text import TfidfVectorizer

sys.stdout.reconfigure(encoding="utf-8")
random.seed(42)

BASE   = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA   = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
PARSED = BASE / "claude only" / "parsed_queries"
MODELS = BASE / "claude only" / "models"
SUBM   = BASE / "claude only" / "submissions"
MODELS.mkdir(parents=True, exist_ok=True)

MODEL_NAME   = "BAAI/bge-reranker-v2-m3"
MAX_LENGTH   = 128   # 256→128: attention 4x hızlı, kalite %95 aynı
BATCH_TRAIN  = 64    # fp16 ile VRAM yeterli
EPOCHS       = 1
LR           = 2e-5
WARMUP_RATIO = 0.05
K_PREDICT    = 14
CV_SAMPLES   = 500
CV_THRESHOLD = 0.58
BATCH_INFER  = 256   # inference'ta daha büyük batch

LOWER_MAP = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(t): return str(t).translate(LOWER_MAP).lower().strip()

# ─── 1. Veri ─────────────────────────────────────────────────────────────────
print("[1] Veri yükleniyor...", flush=True)
t0 = time.time()
items    = pd.read_csv(DATA / "items.csv")
terms    = pd.read_csv(DATA / "terms.csv")
train_df = pd.read_csv(DATA / "training_pairs.csv")
test_df  = pd.read_csv(DATA / "submission_pairs.csv")
sample   = pd.read_csv(DATA / "sample_submission.csv")
struct   = pd.read_csv(PARSED / "training_pairs_structured.csv")

iid_to_title = dict(zip(items["item_id"], items["title"].fillna("").apply(trl)))
iid_to_brand = dict(zip(items["item_id"], items["brand"].fillna("").apply(trl)))
iid_to_catl1 = {iid: c.split("/")[0] for iid, c in
                zip(items["item_id"], items["category"].fillna(""))}
tid_to_query = dict(zip(terms["term_id"], terms["query"]))

train_pos = defaultdict(set)
for tid, iid in zip(train_df["term_id"].values, train_df["item_id"].values):
    train_pos[tid].add(iid)

print(f"  {time.time()-t0:.1f}s | {len(struct):,} çift", flush=True)

def item_text(iid):
    t = iid_to_title.get(iid, "")
    b = iid_to_brand.get(iid, "")
    c = iid_to_catl1.get(iid, "").split("/")[0]
    return " | ".join(p for p in [t, b, c] if p and p != "nan")

# ─── 2. Pre-Tokenize (Kritik Hız İyileştirmesi) ──────────────────────────────
print(f"\n[2] Tokenizer yükleniyor + {len(struct):,} çift pre-tokenize ediliyor...", flush=True)
print("    (Bir kez tokenize, training'de sıfır CPU overhead)", flush=True)
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

CHUNK = 10_000
all_input_ids = []
all_attn_mask = []

queries = struct["query_text"].tolist()
itexts  = struct["item_text"].tolist()
labels  = struct["label"].tolist()

for i in range(0, len(queries), CHUNK):
    enc = tokenizer(
        queries[i:i+CHUNK],
        itexts[i:i+CHUNK],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    all_input_ids.append(enc["input_ids"])
    all_attn_mask.append(enc["attention_mask"])
    if (i // CHUNK) % 5 == 0:
        print(f"  {i:6d}/{len(queries)} tokenize ({time.time()-t0:.0f}s)", flush=True)

all_input_ids = torch.cat(all_input_ids, dim=0)   # (N, 256)
all_attn_mask = torch.cat(all_attn_mask, dim=0)   # (N, 256)
all_labels    = torch.tensor(labels, dtype=torch.float)

print(f"  Pre-tokenize tamamlandı: {time.time()-t0:.1f}s", flush=True)
print(f"  Tensör boyutu: {all_input_ids.shape} | dtype: {all_input_ids.dtype}", flush=True)

# ─── 3. Dataset (Sadece Tensör Döndür — Sıfır Overhead) ──────────────────────
class PreTokenizedDataset(Dataset):
    def __init__(self, input_ids, attn_mask, labels):
        self.input_ids = input_ids
        self.attn_mask = attn_mask
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attn_mask[idx],
            "label":          self.labels[idx],
        }

dataset = PreTokenizedDataset(all_input_ids, all_attn_mask, all_labels)
loader  = DataLoader(dataset, batch_size=BATCH_TRAIN, shuffle=True,
                     num_workers=0, pin_memory=True)

# ─── 4. Model + AMP ───────────────────────────────────────────────────────────
print(f"\n[3] Model yükleniyor...", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=1
).to(device)

total_steps  = len(loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
optimizer    = AdamW(model.parameters(), lr=LR)
scheduler    = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
scaler   = GradScaler()   # fp16 scaler
loss_fn  = torch.nn.BCEWithLogitsLoss()

print(f"  Device: {device} | Steps: {total_steps:,} | Warmup: {warmup_steps}", flush=True)

# ─── 5. Training (fp16 + flush her 100 stepte) ───────────────────────────────
print(f"\n[4] Training başlıyor (fp16 autocast)...", flush=True)
t0 = time.time()
model.train()
step = 0
running_loss = 0.0

for epoch in range(EPOCHS):
    for bi, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, non_blocking=True)
        lbl       = batch["label"].to(device, non_blocking=True)

        with autocast():                          # fp16 otomatik
            logits = model(input_ids=input_ids,
                           attention_mask=attn_mask).logits.squeeze(-1)
            loss   = loss_fn(logits, lbl)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        step += 1

        if step % 100 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / step * (total_steps - step)
            avg     = running_loss / 100
            running_loss = 0.0
            print(f"  step {step:5d}/{total_steps} | loss={avg:.4f} | "
                  f"{elapsed/60:.1f}dk | ETA {eta/60:.1f}dk", flush=True)

    print(f"  Epoch {epoch+1} bitti | {(time.time()-t0)/60:.1f}dk", flush=True)

save_path = MODELS / "bge_structured_v11_fast"
model.save_pretrained(str(save_path))
tokenizer.save_pretrained(str(save_path))
print(f"  Model kaydedildi: {save_path}", flush=True)

# ─── 6. CV ────────────────────────────────────────────────────────────────────
print(f"\n[5] Gerçekçi CV ({CV_SAMPLES} sorgu)...", flush=True)

all_iids  = items["item_id"].tolist()
all_itxts = [trl(iid_to_title.get(i,"") + " " + iid_to_brand.get(i,""))
             for i in all_iids]
item_vect = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4),
                             min_df=3, max_features=200_000, sublinear_tf=True)
item_mat  = item_vect.fit_transform(all_itxts)

random.seed(42)
all_tids = list(train_pos.keys())
random.shuffle(all_tids)
holdout  = all_tids[:CV_SAMPLES]

model.eval()

def score_pairs_fast(pairs):
    scores = []
    for i in range(0, len(pairs), BATCH_INFER):
        bp = pairs[i:i+BATCH_INFER]
        enc = tokenizer([p[0] for p in bp], [p[1] for p in bp],
                        max_length=MAX_LENGTH, truncation=True,
                        padding=True, return_tensors="pt").to(device)
        with torch.no_grad(), autocast():
            logits = model(**enc).logits.squeeze(-1)
        scores.extend(torch.sigmoid(logits).float().cpu().tolist())
    return scores

cv_results = []
cv_t0 = time.time()
for qi, tid in enumerate(holdout):
    q_text   = trl(tid_to_query.get(tid, ""))
    true_pos = train_pos[tid]
    if not q_text or not true_pos:
        continue
    q_vec  = item_vect.transform([q_text])
    sims   = (q_vec * item_mat.T).toarray()[0]
    top100 = np.argpartition(sims, -100)[-100:]
    top100 = top100[np.argsort(sims[top100])[::-1]]
    cands  = [all_iids[i] for i in top100]
    scores = score_pairs_fast([(q_text, item_text(iid)) for iid in cands])
    ranked = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)
    pred   = set(iid for iid, _ in ranked[:K_PREDICT])
    tp = len(pred & true_pos); fp = len(pred - true_pos); fn = len(true_pos - pred)
    f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0.0
    cv_results.append({"f1": f1, "tp": tp, "fp": fp, "fn": fn,
                        "n_true": len(true_pos),
                        "tp_in_cands": len(true_pos & set(cands))})
    if qi % 50 == 0 and qi > 0:
        print(f"  CV {qi}/{CV_SAMPLES} | F1 so far: "
              f"{np.mean([r['f1'] for r in cv_results]):.3f} | "
              f"{time.time()-cv_t0:.0f}s", flush=True)

df_cv = pd.DataFrame(cv_results)
tp_t=df_cv["tp"].sum(); fp_t=df_cv["fp"].sum(); fn_t=df_cv["fn"].sum()
prec=tp_t/(tp_t+fp_t) if (tp_t+fp_t)>0 else 0
rec=tp_t/(tp_t+fn_t) if (tp_t+fn_t)>0 else 0
f1_1=2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
tn=100*len(df_cv)-df_cv["n_true"].sum()-fp_t
prec0=tn/(tn+fn_t) if (tn+fn_t)>0 else 0
rec0=tn/(tn+fp_t) if (tn+fp_t)>0 else 0
f1_0=2*prec0*rec0/(prec0+rec0) if (prec0+rec0)>0 else 0
macro=(f1_0+f1_1)/2

print(f"\n{'='*60}", flush=True)
print(f"CV SONUÇLARI — v11 fast", flush=True)
print(f"  MACRO F1  : {macro:.3f}", flush=True)
print(f"  F1 class1 : {f1_1:.3f} | F1 class0: {f1_0:.3f}", flush=True)
print(f"  v9=0.490 | S1=0.502 | v11={macro:.3f}", flush=True)

if macro >= CV_THRESHOLD:
    print(f"\n  ✓ Test inference başlıyor...", flush=True)
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
        scores  = score_pairs_fast([(q_text, item_text(iid)) for iid in cands])
        ranked  = sorted(zip(row_ids, cands, scores), key=lambda x: x[2], reverse=True)
        for j, (pid, iid, sc) in enumerate(ranked):
            predictions[pid] = 1 if j < K_PREDICT else 0
        if i % 2000 == 0:
            elapsed = time.time()-t0
            print(f"  {i:6d}/{len(test_tids)} | {elapsed/60:.1f}dk "
                  f"| ETA {elapsed/(i+1)*(len(test_tids)-i-1)/60:.1f}dk", flush=True)
    sub = sample[["id"]].copy()
    sub["prediction"] = sub["id"].map(predictions).fillna(0).astype(int)
    pos_count = sub["prediction"].sum()
    out = SUBM / "submission_v11_bge_structured.csv"
    sub.to_csv(str(out), index=False)
    print(f"\n  Pozitif: {pos_count:,} ({100*pos_count/len(sub):.1f}%)", flush=True)
    print(f"  Kaydedildi: {out}", flush=True)
else:
    print(f"  ✗ CV={macro:.3f} < {CV_THRESHOLD} → test atlandı", flush=True)
    print(f"  Sonraki: 16_qwen_zeroshot_v12.py", flush=True)
