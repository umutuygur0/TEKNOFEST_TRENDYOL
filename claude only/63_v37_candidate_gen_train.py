"""
63_v37_candidate_gen_train.py — v37 Aşama a: HEDEFLİ (stratified) train-side aday üretimi
==============================================================================================
50_llm_candidate_gen.py'nin stratified versiyonu. v34-tabanı OOF hata analizinde
(analysis/oof_error_analysis_v34base.txt) bulunan iki net zayıf nokta hedefleniyor:
  1. 1-kelimelik sorgular en düşük F1'e sahip (0.926 vs 2-7 kelime 0.97-0.99).
  2. Zayıf kategoriler: ayakkabı (0.950), kırtasiye & ofis malzemeleri (0.957),
     anne & bebek & çocuk (0.963), giyim (0.968).

Rastgele 2.500 yerine STRATIFIED 2.500 örnekleniyor:
  %40 tek-kelimelik sorgular
  %40 zayıf kategori pozitiflerinden gelen sorgular
  %20 rastgele (temel çeşitlilik + kontrol grubu)

Yöntem (v35 ile aynı — sadece ÖRNEKLEME değişti): her sorgu için TY-ecomm-embed ile
tüm katalogdaki item'lara karşı cosine similarity hesapla, training_pairs'te zaten
pozitif olan item'ları ele, kalan en yakın 5'i aday olarak al. LLM (bir sonraki aşamada)
gerçekten alakalı mı diye doğrulayacak.

Çıktı: claude only/63_llm_labels_v37_train/candidates.csv
"""

import re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE      = Path(__file__).resolve().parents[1]
DATA      = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
EMB_CACHE = BASE / "claude only" / "emb_cache"
OUT_DIR   = BASE / "claude only" / "63_llm_labels_v37_train"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNKNOWN = "unknown"
LOWER = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(s): return str(s).translate(LOWER).lower().strip()

N_QUERIES   = 2500
TOP_K_RAW   = 30
N_CANDIDATE = 5
WEAK_CATEGORIES = {"ayakkabı", "kırtasiye & ofis malzemeleri", "anne & bebek & çocuk", "giyim"}
FRAC_SHORT  = 0.40
FRAC_WEAK   = 0.40
FRAC_RANDOM = 0.20

print("=" * 65, flush=True)
print("[A] Veri + embedding cache yükleniyor...", flush=True)
t0 = time.time()

items       = pd.read_csv(DATA / "items.csv")
terms       = pd.read_csv(DATA / "terms.csv")
train_pairs = pd.read_csv(DATA / "training_pairs.csv")

for col in ["title","category","brand","gender","age_group","attributes"]:
    items[col] = items[col].fillna(UNKNOWN).apply(trl)
items["main_category"] = items["category"].str.split("/").str[0].fillna(UNKNOWN).apply(trl)
terms["query"] = terms["query"].fillna("").apply(trl)

item_ids_raw = np.load(str(EMB_CACHE / "item_ids_tyembed.npy"), allow_pickle=True)
item_embs    = np.load(str(EMB_CACHE / "item_embs_tyembed.npy"))
train_q_ids  = np.load(str(EMB_CACHE / "train_q_ids_tyembed.npy"), allow_pickle=True)
train_q_embs = np.load(str(EMB_CACHE / "train_q_embs_tyembed.npy"))

print(f"  item_embs: {item_embs.shape} | train_q_embs: {train_q_embs.shape} | {time.time()-t0:.1f}s", flush=True)

iid_to_row = {str(r.item_id): r for r in items.itertuples(index=False)}
iid_to_main = {str(r.item_id): r.main_category for r in items.itertuples(index=False)}
tid_to_q   = dict(zip(terms["term_id"], terms["query"]))
tid_to_emb_idx = {str(tid): i for i, tid in enumerate(train_q_ids)}

positive_map: dict[str, set] = {}
positive_first_item: dict[str, str] = {}
for tid, iid in zip(train_pairs["term_id"].astype(str), train_pairs["item_id"].astype(str)):
    positive_map.setdefault(tid, set()).add(iid)
    positive_first_item.setdefault(tid, iid)

print(f"  {len(positive_map):,} benzersiz training sorgusu | {len(items):,} ürün", flush=True)

# ── STRATIFIED ÖRNEKLEME ────────────────────────────────────────────────
rng = np.random.default_rng(42)
all_train_tids = np.array(list(positive_map.keys()))

def q_word_count(tid):
    q = tid_to_q.get(tid, tid_to_q.get(int(tid) if str(tid).isdigit() else tid, ""))
    return len(str(q).split())

short_tids, weak_tids = [], []
for tid in all_train_tids:
    tid_s = str(tid)
    if q_word_count(tid) == 1:
        short_tids.append(tid)
    pos_iid = positive_first_item.get(tid_s)
    if pos_iid and iid_to_main.get(pos_iid, "") in WEAK_CATEGORIES:
        weak_tids.append(tid)

short_tids = np.array(short_tids)
weak_tids  = np.array(weak_tids)
print(f"  1-kelimelik sorgu havuzu: {len(short_tids):,} | zayıf-kategori sorgu havuzu: {len(weak_tids):,}", flush=True)

n_short  = int(N_QUERIES * FRAC_SHORT)
n_weak   = int(N_QUERIES * FRAC_WEAK)
n_random = N_QUERIES - n_short - n_weak

picked = set()
def sample_pool(pool, n):
    pool = np.array([t for t in pool if t not in picked])
    if len(pool) == 0: return np.array([])
    n = min(n, len(pool))
    chosen = rng.choice(pool, size=n, replace=False)
    picked.update(chosen.tolist())
    return chosen

sel_short  = sample_pool(short_tids, n_short)
sel_weak   = sample_pool(weak_tids, n_weak)
sel_random = sample_pool(all_train_tids, n_random)

sample_tids = np.concatenate([sel_short, sel_weak, sel_random])
print(f"  Seçilen: 1-kelime={len(sel_short):,} | zayıf-kategori={len(sel_weak):,} | rastgele={len(sel_random):,} | TOPLAM={len(sample_tids):,}", flush=True)

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

print("\n[B] Aday üretimi (embedding top-K)...", flush=True)
t1 = time.time()

rows = []
BATCH = 50
for bi in range(0, len(sample_tids), BATCH):
    batch_tids = sample_tids[bi:bi+BATCH]
    q_idx = [tid_to_emb_idx.get(str(t)) for t in batch_tids]
    valid_mask = [i is not None for i in q_idx]
    batch_tids = [t for t, v in zip(batch_tids, valid_mask) if v]
    q_idx = [i for i in q_idx if i is not None]
    if not q_idx:
        continue
    q_embs_batch = train_q_embs[q_idx]
    sims = q_embs_batch @ item_embs.T

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
        print(f"    {min(bi+BATCH, len(sample_tids)):,}/{len(sample_tids):,} sorgu işlendi | {len(rows):,} aday | {time.time()-t1:.0f}s", flush=True)

print(f"  Toplam aday: {len(rows):,} | {(time.time()-t1)/60:.1f} dk", flush=True)

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
print(f"TAMAMLANDI — {len(out_df):,} hedefli aday çifti üretildi", flush=True)
print(f"  Sorgu sayısı        : {out_df['term_id'].nunique():,}", flush=True)
print(f"  Ortalama cosine     : {out_df['cosine'].mean():.3f}", flush=True)
print(f"  Dosya               : {out_path}", flush=True)
print(f"  Toplam süre         : {(time.time()-t0)/60:.1f} dk", flush=True)
print(f"{'='*65}", flush=True)
