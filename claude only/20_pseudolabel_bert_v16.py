"""
20_pseudolabel_bert_v16.py — Pseudo-Label + Turkish BERT Cross-Encoder
=======================================================================
NEDEN: LightGBM tavan ~0.72. Semantic anlama için neural model şart.

YAKLAŞIM (3 aşama):
  Aşama A: LightGBM → test seti için probability scores (~5 dk)
  Aşama B: Pseudo-labeling → high-confidence neg/pos seç (~1 dk)
  Aşama C: Turkish BERT fine-tune → test'e uygula (~60-90 dk GPU)

NEDEN ÇALIŞACAK:
  - Test seti = Trendyol retrieval'ından geçmiş çiftler (hard cases)
  - LightGBM score < 0.05 → neredeyse kesin negatif (test dağılımından)
  - Bu pseudo-neg'lerle eğitilen BERT gerçek hard case'leri öğreniyor
  - bgeden farklı: BGE wrong negatives (cross-query) → bu doğru distribution

Model: dbmdz/bert-base-turkish-cased (110M, Türkçe-özel, hızlı)
Beklenen: 0.68 → 0.76+
Süre: ~90 dk (LightGBM 5 dk + BERT 85 dk)
"""

import sys, time, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from rapidfuzz import fuzz as rfuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

sys.stdout.reconfigure(encoding="utf-8")

BASE  = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA  = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
SUBM  = BASE / "claude only" / "submissions"
MODELS= BASE / "claude only" / "models"
SUBM.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

LOWER_MAP = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(t): return str(t).translate(LOWER_MAP).lower().strip()

BERT_MODEL  = "dbmdz/bert-base-turkish-cased"
MAX_LEN     = 128
BATCH_TRAIN = 64
BATCH_INFER = 256
LR          = 2e-5
EPOCHS      = 1
PSEUDO_NEG_THRESH = 0.10   # LightGBM score < 0.10 → pseudo-negatif
PSEUDO_POS_THRESH = 0.90   # LightGBM score > 0.90 → pseudo-pozitif

# ════════════════════════════════════════════════════════════
# AŞAMA A: LightGBM → Test Scores
# ════════════════════════════════════════════════════════════
print("=" * 60, flush=True)
print("AŞAMA A: LightGBM → Test Probability Scores", flush=True)
print("=" * 60, flush=True)
t0 = time.time()

items       = pd.read_csv(DATA / "items.csv")
terms       = pd.read_csv(DATA / "terms.csv")
train_pairs = pd.read_csv(DATA / "training_pairs.csv")
sub_pairs   = pd.read_csv(DATA / "submission_pairs.csv")

for col in ["title","category","brand","gender","age_group","attributes"]:
    items[col] = items[col].fillna("unknown").apply(trl)
terms["query"] = terms["query"].fillna("").apply(trl)
print(f"  Veri: {time.time()-t0:.1f}s", flush=True)

# Hard negatives (hızlı versiyon)
pool = sub_pairs.merge(terms, on="term_id", how="left").merge(items, on="item_id", how="left")
for col in ["query","title","gender","category"]:
    pool[col] = pool[col].fillna("unknown").apply(trl)
mask_gender = (
    (pool["query"].str.contains("erkek") & (pool["gender"] == "kadın")) |
    (pool["query"].str.contains("kadın") & (pool["gender"] == "erkek"))
)
mask_zero = [len(set(q.split()) & set((t+" "+c.replace("/", " ")).split())) == 0
             for q, t, c in zip(pool["query"], pool["title"], pool["category"])]
pool["mask_zero"] = mask_zero
hard_neg = pool[mask_gender | pool["mask_zero"]][["term_id","item_id"]].copy()
hard_neg["label"] = 0
train_pairs["label"] = 1
train_ready = pd.concat([train_pairs[["term_id","item_id","label"]],
                         hard_neg.sample(n=min(250_000, len(hard_neg)), random_state=42)],
                         ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

def build_features(df_in, tfidf_vect, fit=False, all_texts=None):
    df = df_in.merge(terms, on="term_id", how="left").merge(items, on="item_id", how="left")
    for col in ["query","title","brand","category","gender","age_group","attributes"]:
        df[col] = df[col].fillna("unknown").apply(trl)

    if fit:
        corpus = pd.concat([df["title"], df["query"], pd.Series(all_texts or [])])
        tfidf_vect.fit(corpus)

    qs = df["query"].tolist(); ts = df["title"].tolist()

    def tfidf_cos(q_list, t_list, chunk=50_000):
        n = len(q_list); out = np.zeros(n, dtype=np.float32)
        for i in range(0, n, chunk):
            qm = normalize(tfidf_vect.transform(q_list[i:i+chunk]), "l2")
            tm = normalize(tfidf_vect.transform(t_list[i:i+chunk]), "l2")
            out[i:i+chunk] = np.array(qm.multiply(tm).sum(axis=1)).flatten()
        return out

    def jac(a, b):
        sa, sb = set(a.split()), set(b.split())
        return len(sa & sb) / len(sa | sb) if sa and sb else 0.0

    def q_cov(q, t):
        qw = set(q.split())
        return len(qw & set(t.split())) / len(qw) if qw else 0.0

    feats = pd.DataFrame()
    feats["tfidf_cos"]       = tfidf_cos(qs, ts)
    feats["jaccard"]         = [jac(q, t) for q, t in zip(qs, ts)]
    feats["fuzz_set"]        = [rfuzz.token_set_ratio(q, t)/100.0 for q, t in zip(qs, ts)]
    feats["fuzz_partial"]    = [rfuzz.partial_ratio(q, t)/100.0 for q, t in zip(qs, ts)]
    feats["fuzz_sort"]       = [rfuzz.token_sort_ratio(q, t)/100.0 for q, t in zip(qs, ts)]
    feats["fuzz_basic"]      = [rfuzz.ratio(q, t)/100.0 for q, t in zip(qs, ts)]
    feats["cat_overlap"]     = [jac(q, c.replace("/", " ")) for q, c in zip(qs, df["category"].tolist())]
    feats["is_brand"]        = [1 if b != "unknown" and b in q else 0 for q, b in zip(qs, df["brand"].tolist())]
    feats["is_gender"]       = [1 if g in q else 0 for q, g in zip(qs, df["gender"].tolist())]
    feats["is_age"]          = [1 if a not in ("unknown","") and a in q else 0 for q, a in zip(qs, df["age_group"].tolist())]
    feats["attr_match"]      = [jac(q, a) for q, a in zip(qs, df["attributes"].tolist())]
    feats["len_diff"]        = [abs(len(q)-len(t)) for q, t in zip(qs, ts)]
    feats["q_cov_in_title"]  = [q_cov(q, t) for q, t in zip(qs, ts)]
    feats["exact_match"]     = [1 if q in t else 0 for q, t in zip(qs, ts)]
    feats["token_overlap"]   = [len(set(q.split()) & set(t.split())) for q, t in zip(qs, ts)]
    feats["ana_kat"]         = pd.Categorical(df["category"].apply(lambda x: x.split("/")[0]))
    return feats, df

tfidf_vect = TfidfVectorizer(ngram_range=(1,2), max_features=50_000, sublinear_tf=True, min_df=2)
FEATURES   = ["tfidf_cos","jaccard","fuzz_set","fuzz_partial","fuzz_sort","fuzz_basic",
              "cat_overlap","is_brand","is_gender","is_age","attr_match","len_diff",
              "q_cov_in_title","exact_match","token_overlap","ana_kat"]

print("  LightGBM özellikleri (train)...", flush=True)
X_tr, df_tr = build_features(train_ready, tfidf_vect, fit=True,
                              all_texts=items["title"].tolist()+terms["query"].tolist())
y = train_ready["label"].values
groups = df_tr["term_id"].values

print("  LightGBM 5-fold...", flush=True)
oof = np.zeros(len(train_ready))
models_lgbm = []
gkf = GroupKFold(n_splits=5)
for fold, (tri, vli) in enumerate(gkf.split(X_tr, y, groups)):
    m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, max_depth=7,
                            num_leaves=128, random_state=42, class_weight="balanced",
                            n_jobs=-1, verbose=-1)
    m.fit(X_tr[FEATURES].iloc[tri], y[tri],
          eval_set=[(X_tr[FEATURES].iloc[vli], y[vli])],
          categorical_feature=["ana_kat"],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    oof[vli] = m.predict_proba(X_tr[FEATURES].iloc[vli])[:,1]
    models_lgbm.append(m)
    f = f1_score(y[vli], (oof[vli]>0.5).astype(int), average="macro")
    print(f"  Fold {fold+1} | F1={f:.4f}", flush=True)

print("\n  LightGBM özellikleri (test)...", flush=True)
X_te, df_te = build_features(sub_pairs, tfidf_vect)
test_scores = sum(m.predict_proba(X_te[FEATURES])[:,1] for m in models_lgbm) / len(models_lgbm)
print(f"  Test score dist: min={test_scores.min():.3f} mean={test_scores.mean():.3f} max={test_scores.max():.3f}", flush=True)
print(f"  AŞAMA A bitti: {(time.time()-t0)/60:.1f} dk", flush=True)

# ════════════════════════════════════════════════════════════
# AŞAMA B: Pseudo-Labeling
# ════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("AŞAMA B: Pseudo-Labeling", flush=True)
print("="*60, flush=True)

# Test setinden pseudo-label seç
pseudo_neg_mask = test_scores < PSEUDO_NEG_THRESH
pseudo_pos_mask = test_scores > PSEUDO_POS_THRESH

print(f"  Pseudo-neg (score<{PSEUDO_NEG_THRESH}): {pseudo_neg_mask.sum():,}", flush=True)
print(f"  Pseudo-pos (score>{PSEUDO_POS_THRESH}): {pseudo_pos_mask.sum():,}", flush=True)

pseudo_neg = sub_pairs[pseudo_neg_mask].copy()
pseudo_neg["label"] = 0
pseudo_pos = sub_pairs[pseudo_pos_mask].copy()
pseudo_pos["label"] = 1

TARGET_PSEUDO_NEG = 300_000
TARGET_PSEUDO_POS = 100_000

pneg = pseudo_neg.sample(n=min(TARGET_PSEUDO_NEG, len(pseudo_neg)), random_state=42)
ppos = pseudo_pos.sample(n=min(TARGET_PSEUDO_POS, len(pseudo_pos)), random_state=42)

# Real positives + pseudo labels
real_pos = train_pairs[["term_id","item_id"]].copy()
real_pos["label"] = 1

bert_train = pd.concat([real_pos, pneg[["term_id","item_id","label"]],
                         ppos[["term_id","item_id","label"]]],
                         ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n  BERT eğitim seti: {len(bert_train):,} çift", flush=True)
print(f"    Real pos  : {len(real_pos):,}", flush=True)
print(f"    Pseudo neg: {len(pneg):,}", flush=True)
print(f"    Pseudo pos: {len(ppos):,}", flush=True)

# ════════════════════════════════════════════════════════════
# AŞAMA C: Turkish BERT Cross-Encoder Fine-Tune
# ════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("AŞAMA C: Turkish BERT Cross-Encoder", flush=True)
print("="*60, flush=True)

# Metinleri birleştir
iid_to_title = dict(zip(items["item_id"], items["title"]))
iid_to_brand = dict(zip(items["item_id"], items["brand"]))
iid_to_cat   = dict(zip(items["item_id"], items["category"].apply(lambda x: x.split("/")[0])))
tid_to_q     = dict(zip(terms["term_id"], terms["query"]))

def make_text(tid, iid):
    q  = tid_to_q.get(tid, "")
    t  = iid_to_title.get(iid, "")
    b  = iid_to_brand.get(iid, "")
    c  = iid_to_cat.get(iid, "")
    prod = " | ".join(p for p in [t, b, c] if p and p != "unknown")
    return q, prod

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
print(f"  Model: {BERT_MODEL} | Device: {device}", flush=True)

# Pre-tokenize
print(f"  Pre-tokenize {len(bert_train):,} çift...", flush=True)
t1 = time.time()
CHUNK = 10_000
all_ids, all_mask, all_labels = [], [], []

tids = bert_train["term_id"].tolist()
iids = bert_train["item_id"].tolist()
labs = bert_train["label"].tolist()

for i in range(0, len(tids), CHUNK):
    qs_b  = [make_text(t, ii)[0] for t, ii in zip(tids[i:i+CHUNK], iids[i:i+CHUNK])]
    ps_b  = [make_text(t, ii)[1] for t, ii in zip(tids[i:i+CHUNK], iids[i:i+CHUNK])]
    enc = tokenizer(qs_b, ps_b, max_length=MAX_LEN, truncation=True,
                    padding="max_length", return_tensors="pt")
    all_ids.append(enc["input_ids"])
    all_mask.append(enc["attention_mask"])
    if i % 50_000 == 0 and i > 0:
        print(f"  {i:,}/{len(tids)} tokenize", flush=True)

all_ids   = torch.cat(all_ids, dim=0)
all_mask  = torch.cat(all_mask, dim=0)
all_labels= torch.tensor(labs, dtype=torch.float)
print(f"  Pre-tokenize: {time.time()-t1:.1f}s", flush=True)

class PairDS(Dataset):
    def __init__(self, ids, mask, labels):
        self.ids, self.mask, self.labels = ids, mask, labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": self.mask[i], "label": self.labels[i]}

loader = DataLoader(PairDS(all_ids, all_mask, all_labels),
                    batch_size=BATCH_TRAIN, shuffle=True, pin_memory=True)

model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=1).to(device)
total_steps  = len(loader) * EPOCHS
warmup_steps = int(total_steps * 0.05)
optimizer    = AdamW(model.parameters(), lr=LR)
from transformers import get_linear_schedule_with_warmup
scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
scaler       = GradScaler("cuda") if device.type == "cuda" else None
loss_fn      = torch.nn.BCEWithLogitsLoss()

print(f"\n  Fine-tune: {total_steps:,} steps | warmup={warmup_steps}", flush=True)
t1 = time.time()
model.train()
step = running_loss = 0

for batch in loader:
    inp  = batch["input_ids"].to(device, non_blocking=True)
    mask = batch["attention_mask"].to(device, non_blocking=True)
    lbl  = batch["label"].to(device, non_blocking=True)

    if scaler:
        with autocast("cuda"):
            logits = model(input_ids=inp, attention_mask=mask).logits.squeeze(-1)
            loss   = loss_fn(logits, lbl)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
    else:
        logits = model(input_ids=inp, attention_mask=mask).logits.squeeze(-1)
        loss   = loss_fn(logits, lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    running_loss += loss.item()
    step += 1

    if step % 200 == 0:
        el  = time.time() - t1
        eta = el / step * (total_steps - step)
        print(f"  step {step:5d}/{total_steps} | loss={running_loss/200:.4f} | "
              f"{el/60:.1f}dk | ETA {eta/60:.1f}dk", flush=True)
        running_loss = 0.0

save_path = MODELS / "bert_pseudolabel_v16"
model.save_pretrained(str(save_path))
tokenizer.save_pretrained(str(save_path))
print(f"  BERT kaydedildi: {save_path} | {(time.time()-t1)/60:.1f} dk", flush=True)

# ════════════════════════════════════════════════════════════
# AŞAMA D: Test Inference
# ════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print(f"AŞAMA D: Test Inference ({len(sub_pairs):,} çift)...", flush=True)
print("="*60, flush=True)

model.eval()
all_bert_scores = []

test_tids = sub_pairs["term_id"].tolist()
test_iids = sub_pairs["item_id"].tolist()

for i in range(0, len(test_tids), BATCH_INFER):
    bt = test_tids[i:i+BATCH_INFER]
    bi = test_iids[i:i+BATCH_INFER]
    qs_b = [make_text(t, ii)[0] for t, ii in zip(bt, bi)]
    ps_b = [make_text(t, ii)[1] for t, ii in zip(bt, bi)]
    enc = tokenizer(qs_b, ps_b, max_length=MAX_LEN, truncation=True,
                    padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        ctx = autocast("cuda") if device.type == "cuda" else torch.no_grad()
        with ctx if device.type == "cuda" else torch.no_grad():
            logits = model(**enc).logits.squeeze(-1)
    all_bert_scores.extend(torch.sigmoid(logits).float().cpu().tolist())

    if i % 500_000 == 0 and i > 0:
        print(f"  {i:,}/{len(test_tids)}", flush=True)

bert_scores = np.array(all_bert_scores)

# Ensemble: BERT 70% + LightGBM 30%
ensemble = 0.7 * bert_scores + 0.3 * test_scores

# Threshold optimizasyonu (LightGBM OOF'u referans al)
best_thr, best_f1_oof = 0.5, 0.0
for thr in np.arange(0.3, 0.71, 0.01):
    s = f1_score(y, (oof > thr).astype(int), average="macro")
    if s > best_f1_oof: best_f1_oof, best_thr = s, thr

final_bert     = (bert_scores > best_thr).astype(int)
final_ensemble = (ensemble    > best_thr).astype(int)

pos_bert = final_bert.sum()
pos_ens  = final_ensemble.sum()

# İki submission kaydet
out_bert = SUBM / "submission_v16_bert_pseudolabel.csv"
out_ens  = SUBM / "submission_v16_ensemble.csv"
pd.DataFrame({"id": sub_pairs["id"], "prediction": final_bert}).to_csv(str(out_bert), index=False)
pd.DataFrame({"id": sub_pairs["id"], "prediction": final_ensemble}).to_csv(str(out_ens), index=False)

print(f"\n{'='*60}", flush=True)
print(f"TAMAMLANDI — v16 Pseudo-Label + Turkish BERT", flush=True)
print(f"  BERT pozitif    : {pos_bert:,} ({100*pos_bert/len(sub_pairs):.1f}%)", flush=True)
print(f"  Ensemble pozitif: {pos_ens:,} ({100*pos_ens/len(sub_pairs):.1f}%)", flush=True)
print(f"  Toplam süre     : {(time.time()-t0)/60:.1f} dk", flush=True)
print(f"  BERT dosya      : {out_bert}", flush=True)
print(f"  Ensemble dosya  : {out_ens}", flush=True)
print(f"{'='*60}", flush=True)
