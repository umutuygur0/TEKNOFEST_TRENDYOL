"""
43_v30_bert_hardneg.py — BERT v30 Fine-Tune + Test Inference
=============================================================
- bert_v23'ten basla (scratch degil) → 2 epoch, 100K sample, LR=8e-6
- Negatives: FAISS NN rank 11-60 (hard) + same_brand_diff_main/sub
- Cikti: claude only/models/bert_v30/ + submissions/bert_scores_v30_test.npy
- Sonraki: 44_v30_lgbm.py (v27 pipeline + bert_v30 scores)

Tahmin sure: ~1.5-2 saat
"""

import gc, re, os, sys, time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import faiss
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # RTX 5080 sm_100 incompatible
sys.stdout.reconfigure(encoding="utf-8")

BASE       = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA       = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
SUBM       = BASE / "claude only" / "submissions"
MODELS_DIR = BASE / "claude only" / "models"
BERT_V23   = MODELS_DIR / "bert_v23"
BERT_V30   = MODELS_DIR / "bert_v30"
EMB_DIR    = BASE / "claude only" / "emb_cache"

UNKNOWN = "unknown"
device  = torch.device("cpu")
print(f"[INFO] Device: {device}", flush=True)

t_start = time.time()

# ─── helpers ────────────────────────────────────────────────
def parse_attrs(s):
    if not s or str(s) == UNKNOWN: return {}
    d = {}
    for part in str(s).split(","):
        if ":" in part:
            k, _, v = part.partition(":")
            d[k.strip()] = v.strip()
    return d

def build_bert_item_text(row):
    if row is None: return ""
    d   = parse_attrs(getattr(row, "attributes", "") or "")
    parts = []
    t = getattr(row, "title",    "") or ""
    b = getattr(row, "brand",    "") or ""
    c = getattr(row, "category", "") or ""
    if t and t != UNKNOWN: parts.append(t)
    if b and b != UNKNOWN: parts.append(b)
    mc = c.split("/")[0] if c and c != UNKNOWN else ""
    if mc: parts.append(mc)
    renk = d.get("renk", "")
    if renk and renk != UNKNOWN: parts.append(f"renk:{renk}")
    return " | ".join(parts[:4])


# ─── A0: Embeddings + FAISS ─────────────────────────────────
print("="*65, flush=True)
print("[A0] FAISS NN POOL...", flush=True)
item_embs    = np.load(str(EMB_DIR / "item_embs_tyembed.npy"))   # (962873, 768)
train_q_embs = np.load(str(EMB_DIR / "train_q_embs_tyembed.npy"))  # (17968, 768)

faiss_idx = faiss.IndexFlatIP(item_embs.shape[1])
faiss_idx.add(item_embs.astype(np.float32))
print(f"  FAISS: {faiss_idx.ntotal:,} items indexed", flush=True)

# ─── A: Veri ────────────────────────────────────────────────
print("[A] VERI YUKLENIYOR...", flush=True)
items       = pd.read_csv(DATA / "items.csv")
terms       = pd.read_csv(DATA / "terms.csv")
train_pairs = pd.read_csv(DATA / "training_pairs.csv")
sub_pairs   = pd.read_csv(DATA / "submission_pairs.csv")

for df in [items, train_pairs, sub_pairs, terms]:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna(UNKNOWN)

items["item_id"]       = items["item_id"].astype(str)
train_pairs["term_id"] = train_pairs["term_id"].astype(str)
train_pairs["item_id"] = train_pairs["item_id"].astype(str)
sub_pairs["term_id"]   = sub_pairs["term_id"].astype(str)
sub_pairs["item_id"]   = sub_pairs["item_id"].astype(str)

# sub_category / main_category'nin varligini kontrol et
if "main_category" not in items.columns:
    items["main_category"] = items["category"].apply(
        lambda x: x.split("/")[0] if x and x != UNKNOWN else UNKNOWN)
if "sub_category" not in items.columns:
    items["sub_category"] = items["category"].apply(
        lambda x: x.split("/")[1] if x and "/" in str(x) else UNKNOWN)

iid_to_row = {row.item_id: row for row in items.itertuples(index=False)}
tid_to_q   = dict(zip(terms["term_id"].astype(str), terms["query"].astype(str)))
print(f"  Items: {len(items):,} | Train pos: {len(train_pairs):,} | Test: {len(sub_pairs):,}", flush=True)

# Unique train term_ids (sorted, same order as train_q_embs)
unique_train_tids = sorted(set(train_pairs["term_id"].tolist()))
assert len(unique_train_tids) == train_q_embs.shape[0], \
    f"Mismatch: {len(unique_train_tids)} train tids vs {train_q_embs.shape[0]} embs"

# NN hard neg pool
print("  FAISS search (top-60)...", flush=True)
_, I_nn = faiss_idx.search(train_q_embs.astype(np.float32), 60)
item_ids_arr    = items["item_id"].values
positive_keys   = set(train_pairs["term_id"] + "\t" + train_pairs["item_id"])

hard_neg_pool = {}
for qi, tid in enumerate(unique_train_tids):
    pool = []
    for rank in range(10, 60):          # skip top-10 (unlabeled pos risk)
        idx = int(I_nn[qi, rank])
        if idx < 0: continue
        iid = str(item_ids_arr[idx])
        if tid + "\t" + iid not in positive_keys:
            pool.append(iid)
    hard_neg_pool[tid] = pool

avg_pool = np.mean([len(v) for v in hard_neg_pool.values()])
print(f"  NN pool: avg {avg_pool:.1f} items/query", flush=True)

# Index for regular negatives
by_brand = {}; by_main = {}; by_bm = {}; by_bs = {}
for row in items.itertuples(index=False):
    iid = str(row.item_id); br = str(row.brand)
    mc  = str(row.main_category); sc = str(row.sub_category)
    by_brand.setdefault(br, []).append(iid)
    by_main.setdefault(mc, []).append(iid)
    by_bm.setdefault((br, mc), []).append(iid)       # same brand, same main
    by_bs.setdefault((br, mc, sc), []).append(iid)   # same brand, same main, same sub

rng = np.random.default_rng(42)
used = set()

def pick(pool, pos_id):
    if not pool: return None
    idxs = rng.permutation(min(len(pool), 50))
    for i in idxs:
        iid = pool[i]
        k = pos_id + "|" + iid
        if iid != pos_id and k not in used:
            used.add(k); return iid
    return None

# ─── B: Sampling ─────────────────────────────────────────────
print("[B] NEGATIVE SAMPLING (NN hard + regular)...", flush=True)
nn_ptr = {tid: 0 for tid in unique_train_tids}
neg_tids = []; neg_iids = []; neg_src = []

for row in train_pairs.itertuples(index=False):
    tid = str(row.term_id); pos = str(row.item_id)
    item = iid_to_row.get(pos)
    if item is None: continue
    mc   = str(item.main_category); sc = str(item.sub_category)
    br   = str(item.brand)

    selected = []

    # 1. NN hard negatives (2 slots)
    pool_nn = hard_neg_pool.get(tid, [])
    ptr = nn_ptr.get(tid, 0)
    while ptr < len(pool_nn) and len(selected) < 2:
        iid = pool_nn[ptr]; ptr += 1
        if tid + "\t" + iid not in positive_keys:
            selected.append((iid, "nn_hard"))
    nn_ptr[tid] = ptr

    # 2. same brand, diff sub_cat (same main)
    if len(selected) < 5:
        # get all brand+main items, exclude same sub
        bm_pool = [i for i in by_bm.get((br, mc), [])
                   if i not in by_bs.get((br, mc, sc), [])]
        iid = pick(bm_pool, pos)
        if iid and tid + "\t" + iid not in positive_keys:
            selected.append((iid, "same_brand_diff_sub"))

    # 3. same brand, diff main
    if len(selected) < 5:
        bm_all = by_bm.get((br, mc), [])
        brand_pool = [i for i in by_brand.get(br, []) if i not in bm_all]
        iid = pick(brand_pool, pos)
        if iid and tid + "\t" + iid not in positive_keys:
            selected.append((iid, "same_brand_diff_main"))

    # 4. same main fallback
    while len(selected) < 5:
        iid = pick(by_main.get(mc, []), pos)
        if iid and tid + "\t" + iid not in positive_keys:
            selected.append((iid, "same_main"))
        else:
            break

    for iid, src in selected[:5]:
        neg_tids.append(tid); neg_iids.append(iid); neg_src.append(src)

src_counts = Counter(neg_src)
print(f"  Kaynak: {dict(src_counts)}", flush=True)
print(f"  NN hard neg orani: {100*src_counts.get('nn_hard',0)/max(sum(src_counts.values()),1):.1f}%", flush=True)

negatives = pd.DataFrame({"term_id": neg_tids, "item_id": neg_iids, "label": 0})
bert_df   = pd.concat([
    train_pairs[["term_id","item_id"]].assign(label=1),
    negatives
], ignore_index=True)
bert_df["term_id"] = bert_df["term_id"].astype(str)
bert_df["item_id"] = bert_df["item_id"].astype(str)
print(f"  BERT train pool: {len(bert_df):,} | pos={bert_df.label.sum():,}", flush=True)

# ─── C: BERT Fine-Tune ──────────────────────────────────────
print("[C] BERT FINE-TUNE (bert_v23 → bert_v30)...", flush=True)

class PairDataset(Dataset):
    def __init__(self, tids, iids, labels, tok, max_len=96):
        self.queries = [tid_to_q.get(str(t), "") for t in tids]
        self.items   = [build_bert_item_text(iid_to_row.get(str(i))) for i in iids]
        self.labels  = list(labels)
        self.tok     = tok
        self.max_len = max_len

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tok(self.queries[idx], self.items[idx],
                       max_length=self.max_len, truncation=True,
                       padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(self.labels[idx], dtype=torch.float)


def bert_inference(tids, iids, model, tok, batch=512, desc=""):
    ds = PairDataset(tids, iids, [0]*len(tids), tok, max_len=96)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    out_scores = []
    model.eval()
    n = len(tids)
    with torch.no_grad():
        for i, (enc, _) in enumerate(dl):
            logits = model(**enc).logits.squeeze(-1)
            out_scores.extend(torch.sigmoid(logits).cpu().tolist())
            if (i+1) % 1000 == 0:
                print(f"    {desc}: {(i+1)*batch:,}/{n:,} ({100*(i+1)*batch/n:.0f}%)", flush=True)
    return np.array(out_scores, dtype=np.float32)


BERT_V30.mkdir(parents=True, exist_ok=True)
print(f"  bert_v23 yukleniyor...", flush=True)
tokenizer  = AutoTokenizer.from_pretrained(str(BERT_V23))
bert_model = AutoModelForSequenceClassification.from_pretrained(str(BERT_V23)).to(device)

BERT_SAMPLE = 100_000
EPOCHS = 2; BATCH = 32; LR = 8e-6

sample_df = bert_df.sample(BERT_SAMPLE, random_state=99)
print(f"  Fine-tune: {BERT_SAMPLE:,} sample | {EPOCHS} epoch | LR={LR}", flush=True)

ds_ft = PairDataset(sample_df["term_id"].tolist(), sample_df["item_id"].tolist(),
                    sample_df["label"].tolist(), tokenizer, max_len=96)
dl_ft = DataLoader(ds_ft, batch_size=BATCH, shuffle=True, num_workers=0)

optimizer  = torch.optim.AdamW(bert_model.parameters(), lr=LR, weight_decay=0.01)
n_steps    = len(dl_ft) * EPOCHS
scheduler  = get_linear_schedule_with_warmup(optimizer, int(0.05*n_steps), n_steps)
criterion  = nn.BCEWithLogitsLoss()

t_ft = time.time()
for epoch in range(EPOCHS):
    bert_model.train()
    total_loss = 0
    for step, (enc, lbl) in enumerate(dl_ft):
        optimizer.zero_grad()
        loss = criterion(bert_model(**enc).logits.squeeze(-1), lbl)
        loss.backward()
        nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        total_loss += loss.item()
        if step % 200 == 0:
            print(f"  Epoch {epoch+1} step {step}/{len(dl_ft)} loss={total_loss/(step+1):.4f}", flush=True)
    print(f"  Epoch {epoch+1} done | {(time.time()-t_ft)/60:.1f} dk", flush=True)

bert_model.save_pretrained(str(BERT_V30))
tokenizer.save_pretrained(str(BERT_V30))
print(f"  bert_v30 kaydedildi: {BERT_V30}", flush=True)

del ds_ft, dl_ft, sample_df, bert_df, negatives
gc.collect()

# ─── D: Test Inference (3.36M) ──────────────────────────────
print("[D] TEST BERT INFERENCE (3.36M cift)...", flush=True)
out_path = SUBM / "bert_scores_v30_test.npy"
t_test = time.time()
test_scores = bert_inference(
    sub_pairs["term_id"].tolist(), sub_pairs["item_id"].tolist(),
    bert_model, tokenizer, batch=512, desc="test"
)
np.save(str(out_path), test_scores)
print(f"  Test scores: {len(test_scores):,} | min={test_scores.min():.4f} max={test_scores.max():.4f}", flush=True)
print(f"  Kaydedildi: {out_path} | {(time.time()-t_test)/60:.1f} dk", flush=True)

total = (time.time() - t_start) / 60
print(f"\n{'='*65}", flush=True)
print(f"TAMAMLANDI -- v30 BERT fine-tune + test inference", flush=True)
print(f"  Toplam sure  : {total:.1f} dk", flush=True)
print(f"  bert_v30     : {BERT_V30}", flush=True)
print(f"  Test scores  : {out_path}", flush=True)
print(f"  Sonraki      : 44_v30_lgbm.py (v27 pipeline + bert_v30)", flush=True)
