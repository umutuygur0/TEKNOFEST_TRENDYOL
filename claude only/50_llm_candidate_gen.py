"""
50_llm_candidate_gen.py — v35 Aşama 1: LLM doğrulaması için aday üretimi
==========================================================================
Amaç: v28/v29'un yaptığı "TY-embed ile en yakın komşuları hard negative say"
yaklaşımını YAPMIYORUZ (bu v24/v28'i mahvetmişti — %29 unlabeled positive
kirlenmesi). Bunun yerine adayları SADECE üretiyoruz, etiketlemeyi Claude'a
(bir sonraki aşamada) bırakıyoruz.

Yöntem:
  1. training_pairs'ten rastgele ~2.500 sorgu seç (seed=42, deterministik).
  2. Her sorgu için TY-ecomm-embed embedding'i ile TÜM katalogdaki (962K)
     item'lara karşı cosine similarity hesapla.
  3. training_pairs'te zaten pozitif olarak işaretli item'ları ele.
  4. Kalan en yakın 5 item'ı aday olarak al (yani query'ye çok benziyor ama
     hiç etiketlenmemiş — ya gerçek pozitif ya da gerçek zor negatif, ikisi
     de mümkün, bu yüzden LLM'e soracağız).
  5. Çıktı: query + item metadata (title, brand, category, attributes) içeren
     bir CSV — Claude'un okuyup karar verebileceği kompakt formatta.

Çıktı: claude only/51_llm_labels/candidates.csv
  Kolonlar: pair_id, term_id, item_id, query, title, brand, category,
            gender, age_group, attributes_short, cosine
"""

import re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE      = Path(__file__).resolve().parents[1]
DATA      = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
EMB_CACHE = BASE / "claude only" / "emb_cache"
OUT_DIR   = BASE / "claude only" / "51_llm_labels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNKNOWN = "unknown"
LOWER = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(s): return str(s).translate(LOWER).lower().strip()

N_QUERIES   = 2500   # kaç training sorgusu örnekleyelim
TOP_K_RAW   = 30     # ham en-yakın kaç item bakılsın (pozitifleri elemek için buffer)
N_CANDIDATE = 5      # sorgu başına kaç aday (pozitif olmayan) alınsın

print("=" * 65, flush=True)
print("[A] Veri + embedding cache yükleniyor...", flush=True)
t0 = time.time()

items       = pd.read_csv(DATA / "items.csv")
terms       = pd.read_csv(DATA / "terms.csv")
train_pairs = pd.read_csv(DATA / "training_pairs.csv")

for col in ["title","category","brand","gender","age_group","attributes"]:
    items[col] = items[col].fillna(UNKNOWN).apply(trl)
terms["query"] = terms["query"].fillna("").apply(trl)

item_ids_raw = np.load(str(EMB_CACHE / "item_ids_tyembed.npy"), allow_pickle=True)
item_embs    = np.load(str(EMB_CACHE / "item_embs_tyembed.npy"))
train_q_ids  = np.load(str(EMB_CACHE / "train_q_ids_tyembed.npy"), allow_pickle=True)
train_q_embs = np.load(str(EMB_CACHE / "train_q_embs_tyembed.npy"))

print(f"  item_embs: {item_embs.shape} | train_q_embs: {train_q_embs.shape} | {time.time()-t0:.1f}s", flush=True)

iid_to_row = {str(r.item_id): r for r in items.itertuples(index=False)}
tid_to_q   = dict(zip(terms["term_id"], terms["query"]))
tid_to_emb_idx = {str(tid): i for i, tid in enumerate(train_q_ids)}

# term_id -> pozitif item_id seti (leak/duplicate önleme)
positive_map: dict[str, set] = {}
for tid, iid in zip(train_pairs["term_id"].astype(str), train_pairs["item_id"].astype(str)):
    positive_map.setdefault(tid, set()).add(iid)

print(f"  {len(positive_map):,} benzersiz training sorgusu | {len(items):,} ürün", flush=True)

# ── Rastgele sorgu örneklemi (deterministik) ────────────────────────────
rng = np.random.default_rng(42)
all_train_tids = np.array(list(positive_map.keys()))
sample_tids = rng.choice(all_train_tids, size=min(N_QUERIES, len(all_train_tids)), replace=False)
print(f"  Örneklenen sorgu sayısı: {len(sample_tids):,}", flush=True)

# ── Attribute özetleme (kompakt gösterim için) ──────────────────────────
def parse_attrs(s):
    if not s or s in (UNKNOWN, ""): return {}
    d = {}
    for part in s.split(","):
        if ":" in part:
            k, _, v = part.partition(":")
            d[k.strip()] = v.strip()
    return d

def attrs_short(s, max_keys=4):
    d = parse_attrs(s)
    if not d: return ""
    items_ = list(d.items())[:max_keys]
    return "; ".join(f"{k}:{v}" for k, v in items_)

# ── Aday üretimi (batch halinde, bellek dostu) ──────────────────────────
print("\n[B] Aday üretimi (embedding top-K)...", flush=True)
t1 = time.time()

rows = []
BATCH = 50
n_batches = (len(sample_tids) + BATCH - 1) // BATCH

for bi in range(0, len(sample_tids), BATCH):
    batch_tids = sample_tids[bi:bi+BATCH]
    q_idx = [tid_to_emb_idx.get(str(t)) for t in batch_tids]
    valid_mask = [i is not None for i in q_idx]
    batch_tids = [t for t, v in zip(batch_tids, valid_mask) if v]
    q_idx = [i for i in q_idx if i is not None]
    if not q_idx:
        continue
    q_embs_batch = train_q_embs[q_idx]                       # (B, 768)
    sims = q_embs_batch @ item_embs.T                          # (B, 962873)

    for row_i, tid in enumerate(batch_tids):
        tid_s = str(tid)
        pos_set = positive_map.get(tid_s, set())
        sim_row = sims[row_i]
        top_idx = np.argpartition(-sim_row, TOP_K_RAW)[:TOP_K_RAW]
        top_idx = top_idx[np.argsort(-sim_row[top_idx])]
        n_found = 0
        for idx in top_idx:
            iid = str(item_ids_raw[idx])
            if iid in pos_set:
                continue
            rows.append((tid_s, iid, float(sim_row[idx])))
            n_found += 1
            if n_found >= N_CANDIDATE:
                break

    if (bi // BATCH) % 10 == 0:
        print(f"    {min(bi+BATCH, len(sample_tids)):,}/{len(sample_tids):,} sorgu işlendi | "
              f"{len(rows):,} aday | {time.time()-t1:.0f}s", flush=True)

print(f"  Toplam aday: {len(rows):,} | {(time.time()-t1)/60:.1f} dk", flush=True)

# ── Metadata ekle + CSV yaz ──────────────────────────────────────────────
print("\n[C] Metadata ekleniyor + CSV yazılıyor...", flush=True)

out_records = []
for pair_id, (tid, iid, cos) in enumerate(rows):
    q = tid_to_q.get(tid, tid_to_q.get(int(tid) if tid.isdigit() else tid, ""))
    item_row = iid_to_row.get(iid)
    if item_row is None:
        continue
    out_records.append({
        "pair_id": pair_id,
        "term_id": tid,
        "item_id": iid,
        "query": q,
        "title": item_row.title,
        "brand": item_row.brand,
        "category": item_row.category,
        "gender": item_row.gender,
        "age_group": item_row.age_group,
        "attributes_short": attrs_short(item_row.attributes),
        "cosine": round(cos, 4),
    })

out_df = pd.DataFrame(out_records)
out_path = OUT_DIR / "candidates.csv"
out_df.to_csv(str(out_path), index=False, encoding="utf-8")

print(f"\n{'='*65}", flush=True)
print(f"TAMAMLANDI — {len(out_df):,} aday çift üretildi", flush=True)
print(f"  Sorgu sayısı        : {out_df['term_id'].nunique():,}", flush=True)
print(f"  Ortalama cosine     : {out_df['cosine'].mean():.3f}", flush=True)
print(f"  Dosya               : {out_path}", flush=True)
print(f"  Toplam süre         : {(time.time()-t0)/60:.1f} dk", flush=True)
print(f"{'='*65}", flush=True)
print("\nÖrnek adaylar:", flush=True)
print(out_df.head(10).to_string(), flush=True)
