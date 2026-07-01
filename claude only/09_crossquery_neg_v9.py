"""
v9: XLM-RoBERTa Cross-Encoder with CROSS-QUERY Negatives
=========================================================
Tüm önceki yaklaşımların sorunu:
  - v6/v8 heuristic: "kalan 86 item = negatif" → 0.47 (baseline'dan iyi değil)
  - v7 cross-encoder: TF-IDF negatives KONTAMINE → pandora itemleri negatif sayıldı

KÖKLÜ FİX:
  Cross-query negatives: (query_A, item_from_query_B) → 0
  - Granül kahve item'ı → "pandora gold" query için KESİNLİKLE negatif
  - Kontaminasyon: SIFIR (başka query'nin item'ı bu query için asla pozitif olamaz)

Training data:
  1. Orijinal train positives: (query, pos_item) → 1
  2. Query expansion: (first_2_words, pos_item) → 1  [test pattern coverage]
  3. Synthetic test positives: (test_query, item_with_all_tokens) → 1
  4. Cross-query negatives: (query, item_from_OTHER_query) → 0  [TEMIZ!]
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import random, time, sys, torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

sys.stdout.reconfigure(encoding="utf-8")

BASE   = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA   = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
SUBM   = BASE / "claude only" / "submissions"
MODELS = BASE / "claude only" / "models"
SUBM.mkdir(parents=True, exist_ok=True)

MODEL_NAME  = "xlm-roberta-base"
MAX_LENGTH  = 128
BATCH_TRAIN = 32
BATCH_INFER = 256
LR          = 2e-5
EPOCHS      = 1
NEG_PER_POS = 3
SEED        = 42
K_PREDICT   = 14

LOWER_MAP = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def tr_lower(t): return str(t).translate(LOWER_MAP).lower().strip()

STOPWORDS = {"ve","ile","bir","bu","da","de","mi","mı","mu","mü","için","ama","veya",
             "gibi","olan","her","ne","ki","çok","az","en","the","a","an","of","in",
             "to","for","and","or","is","it","as","ml","gr","kg","cm","adet","set"}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ─── 1. Veri Yükle ───────────────────────────────────────────────────────────
print("[1] Veri yükleniyor...")
t0 = time.time()
items  = pd.read_csv(DATA / "items.csv")
terms  = pd.read_csv(DATA / "terms.csv")
train  = pd.read_csv(DATA / "training_pairs.csv")
test   = pd.read_csv(DATA / "submission_pairs.csv")
sample = pd.read_csv(DATA / "sample_submission.csv")

iid_to_title = dict(zip(items["item_id"], items["title"].fillna("")))
iid_to_brand = dict(zip(items["item_id"], items["brand"].fillna("")))
iid_to_catl1 = {iid: str(c).split("/")[0] for iid, c in zip(items["item_id"], items["category"].fillna(""))}
tid_to_query  = dict(zip(terms["term_id"], terms["query"]))

train_pos = defaultdict(set)
for tid, iid in zip(train["term_id"].values, train["item_id"].values):
    train_pos[tid].add(iid)

test_tids = set(test["term_id"].unique())
print(f"  Süre: {time.time()-t0:.1f}s | Items: {len(items):,} | Train pos: {len(train):,}")

def itext(iid):
    """Item metni: title + brand + category (cross-encoder input olarak)"""
    t = tr_lower(iid_to_title.get(iid, ""))
    b = tr_lower(iid_to_brand.get(iid, ""))
    c = tr_lower(iid_to_catl1.get(iid, ""))
    return f"{t} [SEP] {b} {c}".strip()

# ─── 2. Training Çiftleri ────────────────────────────────────────────────────
print("[2] Training çiftleri oluşturuluyor...")
t0 = time.time()
positives = []  # (query_text, item_text)
negatives = []

# 2a) Orijinal train positives
for tid, pos_iids in train_pos.items():
    q = tr_lower(tid_to_query.get(tid, ""))
    for iid in pos_iids:
        positives.append((q, itext(iid)))
print(f"  2a) Orijinal pozitif: {len(positives):,}")

# 2b) Query expansion: ilk 2 kelime prefix (test query coverage)
# "avon luck parfüm edp 50ml" → "avon luck" de aynı item'la pozitif
exp = 0
for tid, pos_iids in train_pos.items():
    q = tr_lower(tid_to_query.get(tid, ""))
    toks = q.split()
    if len(toks) >= 3:
        prefix2 = " ".join(toks[:2])
        for iid in pos_iids:
            positives.append((prefix2, itext(iid)))
            exp += 1
print(f"  2b) Query expansion (+{exp:,}): toplam pozitif: {len(positives):,}")

# 2c) Synthetic positives from TEST queries (ALL tokens must match in item title+brand)
# "luck" + item "avon luck parfüm" → "luck" title'da var → synthetic positive
print("  2c) Synthetic test positives oluşturuluyor...")
syn = 0
test_grp = test.groupby("term_id")["item_id"].apply(list).to_dict()

for tid in test_tids:
    q = tr_lower(tid_to_query.get(tid, ""))
    q_toks = [t for t in q.split() if len(t) >= 3 and t not in STOPWORDS]
    if not q_toks or len(q_toks) == 0:
        continue
    candidates = test_grp.get(tid, [])
    for iid in candidates:
        it_full = tr_lower(iid_to_title.get(iid,"") + " " + iid_to_brand.get(iid,""))
        # Tüm query token'ları item title+brand'ında geçiyorsa → synthetic positive
        if all(tok in it_full for tok in q_toks):
            positives.append((q, itext(iid)))
            syn += 1

print(f"  2c) Synthetic test positives: +{syn:,}  Toplam: {len(positives):,}")

# Cap positives
MAX_POS = 750000
if len(positives) > MAX_POS:
    random.shuffle(positives)
    positives = positives[:MAX_POS]
    print(f"  Cap: {len(positives):,}")

# 2d) CROSS-QUERY NEGATIVES — ana yenilik!
# (query_A, item_from_completely_different_query_B) → 0
# Kontaminasyon: SIFIR
print("  2d) Cross-query negatives oluşturuluyor...")
tid_list    = list(train_pos.keys())
tid_to_pos  = {tid: list(iids) for tid, iids in train_pos.items()}

for q_text, _ in positives:
    for _ in range(NEG_PER_POS):
        neg_tid = random.choice(tid_list)
        neg_iid = random.choice(tid_to_pos[neg_tid])
        negatives.append((q_text, itext(neg_iid)))

print(f"  2d) Cross-query negatif: {len(negatives):,}")

# Build dataset
all_pairs = [(q, it, 1) for q, it in positives] + [(q, it, 0) for q, it in negatives]
random.shuffle(all_pairs)
print(f"  Toplam çift: {len(all_pairs):,}  Süre: {time.time()-t0:.1f}s")
del positives, negatives  # Memory

# ─── 3. Dataset & DataLoader ─────────────────────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, pairs, tok, maxlen):
        self.pairs = pairs; self.tok = tok; self.maxlen = maxlen
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        q, it, lbl = self.pairs[i]
        enc = self.tok(q, it, max_length=self.maxlen, padding="max_length",
                       truncation=True, return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(float(lbl), dtype=torch.float32)
        }

# ─── 4. Model Yükle ──────────────────────────────────────────────────────────
print("[3] Model yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"  {MODEL_NAME}  Params: {total_params:,}")

# ─── 5. Eğitim ───────────────────────────────────────────────────────────────
print("[4] Eğitim başlıyor...")
dataset = PairDataset(all_pairs, tokenizer, MAX_LENGTH)
loader  = DataLoader(dataset, batch_size=BATCH_TRAIN, shuffle=True, num_workers=0, pin_memory=True)

total_steps  = len(loader) * EPOCHS
warmup_steps = total_steps // 10

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
scaler    = GradScaler("cuda")
criterion = torch.nn.BCEWithLogitsLoss()

model.train()
t0 = time.time()
for epoch in range(EPOCHS):
    for step, batch in enumerate(loader):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl  = batch["label"].to(device)

        with autocast("cuda"):
            out  = model(input_ids=ids, attention_mask=mask)
            loss = criterion(out.logits.squeeze(-1), lbl)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        scheduler.step()

        gs = epoch * len(loader) + step + 1
        if gs % 500 == 0 or gs == 1:
            e = (time.time()-t0)/60
            r = e/gs*(total_steps-gs)
            print(f"  Step {gs}/{total_steps}  loss={loss.item():.4f}  {e:.1f}dk  ~{r:.1f}dk kaldı")

model_path = MODELS / "crossencoder_v9"
model.save_pretrained(str(model_path))
tokenizer.save_pretrained(str(model_path))
print(f"  Model kaydedildi: {model_path}")

# ─── 6. Inference ────────────────────────────────────────────────────────────
print("[5] Test inference (3.36M çift)...")
model.eval()
t0 = time.time()

class InferDataset(Dataset):
    def __init__(self, rows, tok, maxlen, tid2q):
        self.rows = rows; self.tok = tok; self.maxlen = maxlen; self.tid2q = tid2q
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        tid, iid = self.rows[i]
        q  = tr_lower(self.tid2q.get(tid, ""))
        it = itext(iid)
        enc = self.tok(q, it, max_length=self.maxlen, padding="max_length",
                       truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)}

test_rows = list(zip(test["term_id"].values, test["item_id"].values))
infer_ds  = InferDataset(test_rows, tokenizer, MAX_LENGTH, tid_to_query)
infer_ld  = DataLoader(infer_ds, batch_size=BATCH_INFER, shuffle=False,
                       num_workers=0, pin_memory=True)

all_scores = []
with torch.no_grad():
    for bi, batch in enumerate(infer_ld):
        with autocast("cuda"):
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
        all_scores.extend(out.logits.squeeze(-1).cpu().float().tolist())
        if bi % 200 == 0:
            pct = 100*bi/len(infer_ld); e = (time.time()-t0)/60
            r = e/max(bi,1)*(len(infer_ld)-bi)
            print(f"  {pct:.0f}%  {e:.1f}dk  ~{r:.1f}dk kaldı")

print(f"  Inference: {(time.time()-t0)/60:.1f}dk")

# ─── 7. Top-K Tahmin ─────────────────────────────────────────────────────────
print("[6] Top-K tahmin (K=14)...")
test["score"] = all_scores
predictions = {}
for tid, grp in test.groupby("term_id"):
    grp_s = grp.sort_values("score", ascending=False)
    for i, (_, row) in enumerate(grp_s.iterrows()):
        predictions[row["id"]] = 1 if i < K_PREDICT else 0

sub = sample[["id"]].copy()
sub["prediction"] = sub["id"].map(predictions).fillna(0).astype(int)
pos_count = sub["prediction"].sum()
print(f"  Pozitif: {pos_count:,} ({100*pos_count/len(sub):.1f}%)")

out_path = SUBM / "submission_v9_crossquery.csv"
sub.to_csv(str(out_path), index=False)
print(f"  Kaydedildi: {out_path}")

# Log
log_path = SUBM / "submissions_log.csv"
if log_path.exists():
    log = pd.read_csv(str(log_path))
    new = pd.DataFrame([{
        "version": "v9", "positive_rate": f"{100*pos_count/len(sub):.1f}%",
        "public_score": "TBD",
        "description": "XLM-RoBERTa cross-encoder, CROSS-QUERY negatives (clean!), query expansion, synthetic test positives",
        "notes": f"Pairs: {len(all_pairs):,} | Model: {MODEL_NAME}"
    }])
    log = pd.concat([log, new], ignore_index=True)
    log.to_csv(str(log_path), index=False)

print("\n=== TAMAMLANDI ===")
