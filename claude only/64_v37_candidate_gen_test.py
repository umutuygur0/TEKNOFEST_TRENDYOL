"""
64_v37_candidate_gen_test.py — v37 Aşama a: HEDEFLİ (stratified) test-side aday üretimi
==============================================================================================
57_v36_test_candidate_gen.py'nin stratified versiyonu — aynı iki zayıf noktayı hedefler
(1-kelimelik sorgular, zayıf kategoriler: ayakkabı/kırtasiye & ofis/anne&bebek&çocuk/giyim)
ama bu sefer GERÇEK test-dağılımından (`submission_pairs.csv`) örnekleniyor.

Test sorgularının gerçek etiketi yok, o yüzden "zayıf kategori" o sorgunun GERÇEK aday
havuzundaki (submission_pairs.csv) ÇOĞUNLUK kategorisine bakılarak belirleniyor.

Stratifikasyon: %40 1-kelime, %40 zayıf-kategori-çoğunluklu, %20 rastgele.
Ölçek öncekiyle aynı (~2.750 sorgu, ~5 aday/sorgu ≈ 13.7K çift) — sadece hedef değişti.

Çıktı: claude only/64_llm_labels_v37_test/candidates.csv
"""

import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE      = Path(__file__).resolve().parents[1]
DATA      = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
EMB_CACHE = BASE / "claude only" / "emb_cache"
OUT_DIR   = BASE / "claude only" / "64_llm_labels_v37_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNKNOWN = "unknown"
LOWER = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(s): return str(s).translate(LOWER).lower().strip()

N_QUERIES   = 2750
N_CANDIDATE = 5
WEAK_CATEGORIES = {"ayakkabı", "kırtasiye & ofis malzemeleri", "anne & bebek & çocuk", "giyim"}
FRAC_SHORT  = 0.40
FRAC_WEAK   = 0.40
FRAC_RANDOM = 0.20

print("=" * 65, flush=True)
print("[A] Veri + embedding cache yükleniyor...", flush=True)
t0 = time.time()

items     = pd.read_csv(DATA / "items.csv")
terms     = pd.read_csv(DATA / "terms.csv")
sub_pairs = pd.read_csv(DATA / "submission_pairs.csv")

for col in ["title","category","brand","gender","age_group","attributes"]:
    items[col] = items[col].fillna(UNKNOWN).apply(trl)
items["main_category"] = items["category"].str.split("/").str[0].fillna(UNKNOWN).apply(trl)
terms["query"] = terms["query"].fillna("").apply(trl)

item_ids_raw = np.load(str(EMB_CACHE / "item_ids_tyembed.npy"), allow_pickle=True)
item_embs    = np.load(str(EMB_CACHE / "item_embs_tyembed.npy"))
test_q_ids   = np.load(str(EMB_CACHE / "test_q_ids_tyembed.npy"), allow_pickle=True)
test_q_embs  = np.load(str(EMB_CACHE / "test_q_embs_tyembed.npy"))

print(f"  item_embs: {item_embs.shape} | test_q_embs: {test_q_embs.shape} | {time.time()-t0:.1f}s", flush=True)

iid_to_row = {str(r.item_id): r for r in items.itertuples(index=False)}
iid_to_main = {str(r.item_id): r.main_category for r in items.itertuples(index=False)}
tid_to_q   = dict(zip(terms["term_id"], terms["query"]))
iid_to_emb_idx = {str(iid): i for i, iid in enumerate(item_ids_raw)}
tid_to_emb_idx = {str(tid): i for i, tid in enumerate(test_q_ids)}

real_candidates: dict[str, list] = {}
for tid, iid in zip(sub_pairs["term_id"].astype(str), sub_pairs["item_id"].astype(str)):
    real_candidates.setdefault(tid, []).append(iid)

print(f"  {len(real_candidates):,} benzersiz test sorgusu | ortalama {sub_pairs.shape[0]/len(real_candidates):.1f} aday/sorgu | {len(items):,} ürün", flush=True)

# ── STRATIFIED ÖRNEKLEME ────────────────────────────────────────────────
rng = np.random.default_rng(42)
all_test_tids = np.array(list(real_candidates.keys()))

def q_word_count(tid):
    q = tid_to_q.get(tid, tid_to_q.get(int(tid) if str(tid).isdigit() else tid, ""))
    return len(str(q).split())

def majority_category(tid):
    cats = [iid_to_main.get(iid, UNKNOWN) for iid in real_candidates.get(tid, [])]
    if not cats: return UNKNOWN
    return pd.Series(cats).mode()[0]

short_tids, weak_tids = [], []
for tid in all_test_tids:
    if q_word_count(tid) == 1:
        short_tids.append(tid)
    if majority_category(tid) in WEAK_CATEGORIES:
        weak_tids.append(tid)

short_tids = np.array(short_tids)
weak_tids  = np.array(weak_tids)
print(f"  1-kelimelik sorgu havuzu: {len(short_tids):,} | zayıf-kategori-çoğunluklu sorgu havuzu: {len(weak_tids):,}", flush=True)

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
sel_random = sample_pool(all_test_tids, n_random)

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

print("\n[B] Gerçek adaylardan top-K seçiliyor (cosine benzerliği)...", flush=True)
t1 = time.time()

rows = []
for i, tid in enumerate(sample_tids):
    tid_s = str(tid)
    q_idx = tid_to_emb_idx.get(tid_s)
    if q_idx is None:
        continue
    cand_iids = real_candidates.get(tid_s, [])
    if not cand_iids:
        continue
    valid_iids, valid_idx = [], []
    for iid in cand_iids:
        e_idx = iid_to_emb_idx.get(iid)
        if e_idx is not None:
            valid_iids.append(iid); valid_idx.append(e_idx)
    if not valid_idx:
        continue
    q_emb = test_q_embs[q_idx]
    sims  = item_embs[valid_idx] @ q_emb
    order = np.argsort(-sims)[:N_CANDIDATE]
    for oi in order:
        rows.append((tid_s, valid_iids[oi], float(sims[oi])))

    if (i + 1) % 250 == 0:
        print(f"    {i+1:,}/{len(sample_tids):,} sorgu işlendi | {len(rows):,} aday | {time.time()-t1:.0f}s", flush=True)

print(f"  Toplam aday: {len(rows):,} | {(time.time()-t1)/60:.1f} dk", flush=True)

sub_keys = set(sub_pairs["term_id"].astype(str) + "\t" + sub_pairs["item_id"].astype(str))
n_bad = sum(1 for tid, iid, _ in rows if (tid + "\t" + iid) not in sub_keys)
print(f"  Sahte (submission_pairs'te olmayan) aday sayısı: {n_bad} (0 olmalı)", flush=True)
assert n_bad == 0, f"HATA! {n_bad} aday gerçek submission_pairs'te yok!"

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
print(f"TAMAMLANDI — {len(out_df):,} hedefli GERÇEK test-adayı çifti üretildi", flush=True)
print(f"  Sorgu sayısı        : {out_df['term_id'].nunique():,}", flush=True)
print(f"  Ortalama cosine     : {out_df['cosine'].mean():.3f}", flush=True)
print(f"  Dosya               : {out_path}", flush=True)
print(f"  Toplam süre         : {(time.time()-t0)/60:.1f} dk", flush=True)
print(f"{'='*65}", flush=True)
