"""
21_bert_threshold_v16b.py — BERT threshold tuning
BERT 54% pozitif tahmin etti (bias). Doğru eşik: ~%44 pozitif.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.amp import autocast

sys.stdout.reconfigure(encoding="utf-8")

BASE   = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA   = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
MODEL  = BASE / "claude only" / "models" / "bert_pseudolabel_v16"
SUBM   = BASE / "claude only" / "submissions"

LOWER_MAP = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(t): return str(t).translate(LOWER_MAP).lower().strip()

print("Veri yükleniyor...", flush=True)
items     = pd.read_csv(DATA / "items.csv")
terms     = pd.read_csv(DATA / "terms.csv")
sub_pairs = pd.read_csv(DATA / "submission_pairs.csv")

for col in ["title","brand","category"]:
    items[col] = items[col].fillna("unknown").apply(trl)
terms["query"] = terms["query"].fillna("").apply(trl)

iid_to_title = dict(zip(items["item_id"], items["title"]))
iid_to_brand = dict(zip(items["item_id"], items["brand"]))
iid_to_cat   = dict(zip(items["item_id"], items["category"].apply(lambda x: x.split("/")[0])))
tid_to_q     = dict(zip(terms["term_id"], terms["query"]))

def make_text(tid, iid):
    q    = tid_to_q.get(tid, "")
    t    = iid_to_title.get(iid, "")
    b    = iid_to_brand.get(iid, "")
    c    = iid_to_cat.get(iid, "")
    prod = " | ".join(p for p in [t, b, c] if p and p != "unknown")
    return q, prod

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
model     = AutoModelForSequenceClassification.from_pretrained(str(MODEL)).to(device)
model.eval()
print(f"Model yüklendi | Device: {device}", flush=True)

# Inference
BATCH = 512
test_tids = sub_pairs["term_id"].tolist()
test_iids = sub_pairs["item_id"].tolist()
bert_scores = []
t0 = time.time()

for i in range(0, len(test_tids), BATCH):
    bt = test_tids[i:i+BATCH]
    bi = test_iids[i:i+BATCH]
    qs = [make_text(t, ii)[0] for t, ii in zip(bt, bi)]
    ps = [make_text(t, ii)[1] for t, ii in zip(bt, bi)]
    enc = tokenizer(qs, ps, max_length=128, truncation=True,
                    padding=True, return_tensors="pt").to(device)
    with torch.no_grad(), autocast("cuda"):
        logits = model(**enc).logits.squeeze(-1)
    bert_scores.extend(torch.sigmoid(logits).float().cpu().tolist())
    if i % 500_000 == 0 and i > 0:
        pct = 100 * i / len(test_tids)
        el  = time.time() - t0
        eta = el / (i / len(test_tids)) - el
        print(f"  {i:,}/{len(test_tids)} ({pct:.0f}%) | {el/60:.1f}dk | ETA {eta/60:.1f}dk", flush=True)

bert_scores = np.array(bert_scores)
np.save(str(SUBM / "bert_scores_v16.npy"), bert_scores)
print(f"\nBERT score dist: min={bert_scores.min():.3f} mean={bert_scores.mean():.3f} max={bert_scores.max():.3f}", flush=True)
print(f"Inference: {(time.time()-t0)/60:.1f} dk", flush=True)

# Threshold analizi
print("\n--- THRESHOLD ANALİZİ ---", flush=True)
for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    pos = (bert_scores > thr).sum()
    print(f"  thr={thr:.2f} → {pos:,} pozitif ({100*pos/len(bert_scores):.1f}%)", flush=True)

# Hedef: ~44% pozitif (önceki başarılı submission'lar)
# Percentile-based threshold
target_pos_rate = 0.44
thr_p = float(np.percentile(bert_scores, 100 - target_pos_rate * 100))
pos_p = (bert_scores > thr_p).sum()
print(f"\nHedef %44 → threshold={thr_p:.4f} → {pos_p:,} pozitif ({100*pos_p/len(bert_scores):.1f}%)", flush=True)

# Ayrıca %40, %42, %44, %46 deneyelim
for rate in [0.40, 0.42, 0.44, 0.46, 0.48]:
    thr_r = float(np.percentile(bert_scores, 100 - rate * 100))
    pos_r = (bert_scores > thr_r).sum()
    fname = f"submission_v16b_bert_{int(rate*100)}pct.csv"
    pd.DataFrame({"id": sub_pairs["id"], "prediction": (bert_scores > thr_r).astype(int)}).to_csv(str(SUBM / fname), index=False)
    print(f"  %{rate*100:.0f} → thr={thr_r:.4f} → {pos_r:,} pos | {fname}", flush=True)

print("\nBitti!", flush=True)
