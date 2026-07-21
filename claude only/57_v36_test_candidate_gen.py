"""
57_v36_test_candidate_gen.py — v36 Aşama a: GERÇEK test-dağılımından aday üretimi
====================================================================================
v35'ten (50_llm_candidate_gen.py) FARKI: adaylar embedding-benzerliğiyle tüm katalogdan
MADENCİLİK yapılarak üretilmiyor — doğrudan `submission_pairs.csv`'deki GERÇEK aday
havuzundan (Trendyol'un kendi retrieval sisteminin ürettiği ~104 aday/sorgu) örnekleniyor.

Bu, modelin training sırasında hiç görmediği ama test'te bol miktarda karşılaşacağı
zorluk seviyesini (gerçek "yakın ama alakasız" adaylar) yakalamayı hedefliyor.

Yöntem:
  1. submission_pairs'ten rastgele ~2.750 test sorgusu seç (seed=42, deterministik).
  2. Her sorgu için, submission_pairs'teki GERÇEK adayları (~104/sorgu) al.
  3. TY-ecomm-embed cosine benzerliğine göre en yüksek 5'ini seç (en belirsiz/zor
     adaylar — çok düşük benzerlikli adaylar zaten bariz alakasız, LLM'e sormaya gerek yok).
  4. Çıktı: query + item metadata içeren CSV — Claude'un okuyup karar verebileceği format.

ÖNEMLİ (uyumluluk): Bu adaylar LLM tarafından SADECE eğitim sinyali (distillation) olarak
doğrulanacak — test satırları için LLM kararı asla doğrudan final tahmine yazılmıyor.

Çıktı: claude only/57_llm_labels_test/candidates.csv
  Kolonlar: pair_id, term_id, item_id, query, title, brand, category,
            gender, age_group, attributes_short, cosine
"""

import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE      = Path(__file__).resolve().parents[1]
DATA      = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
EMB_CACHE = BASE / "claude only" / "emb_cache"
OUT_DIR   = BASE / "claude only" / "57_llm_labels_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNKNOWN = "unknown"
LOWER = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(s): return str(s).translate(LOWER).lower().strip()

N_QUERIES   = 2750   # kaç test sorgusu örnekleyelim
N_CANDIDATE = 5       # sorgu başına kaç aday (en yüksek cosine) alınsın

print("=" * 65, flush=True)
print("[A] Veri + embedding cache yükleniyor...", flush=True)
t0 = time.time()

items     = pd.read_csv(DATA / "items.csv")
terms     = pd.read_csv(DATA / "terms.csv")
sub_pairs = pd.read_csv(DATA / "submission_pairs.csv")

for col in ["title","category","brand","gender","age_group","attributes"]:
    items[col] = items[col].fillna(UNKNOWN).apply(trl)
terms["query"] = terms["query"].fillna("").apply(trl)

item_ids_raw = np.load(str(EMB_CACHE / "item_ids_tyembed.npy"), allow_pickle=True)
item_embs    = np.load(str(EMB_CACHE / "item_embs_tyembed.npy"))
test_q_ids   = np.load(str(EMB_CACHE / "test_q_ids_tyembed.npy"), allow_pickle=True)
test_q_embs  = np.load(str(EMB_CACHE / "test_q_embs_tyembed.npy"))

print(f"  item_embs: {item_embs.shape} | test_q_embs: {test_q_embs.shape} | {time.time()-t0:.1f}s", flush=True)

iid_to_row = {str(r.item_id): r for r in items.itertuples(index=False)}
tid_to_q   = dict(zip(terms["term_id"], terms["query"]))
iid_to_emb_idx = {str(iid): i for i, iid in enumerate(item_ids_raw)}
tid_to_emb_idx = {str(tid): i for i, tid in enumerate(test_q_ids)}

# term_id -> GERÇEK aday item_id listesi (submission_pairs'ten)
real_candidates: dict[str, list] = {}
for tid, iid in zip(sub_pairs["term_id"].astype(str), sub_pairs["item_id"].astype(str)):
    real_candidates.setdefault(tid, []).append(iid)

print(f"  {len(real_candidates):,} benzersiz test sorgusu | ortalama {sub_pairs.shape[0]/len(real_candidates):.1f} aday/sorgu | {len(items):,} ürün", flush=True)

# ── Rastgele sorgu örneklemi (deterministik) ────────────────────────────
rng = np.random.default_rng(42)
all_test_tids = np.array(list(real_candidates.keys()))
sample_tids = rng.choice(all_test_tids, size=min(N_QUERIES, len(all_test_tids)), replace=False)
print(f"  Örneklenen sorgu sayısı: {len(sample_tids):,}", flush=True)

# ── Attribute özetleme ──────────────────────────────────────────────────
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

# ── Aday seçimi: her sorgu için GERÇEK adaylardan en yüksek cosine 5 tane ─
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

# ── Doğrulama: her aday GERÇEKTEN submission_pairs'te var mı? ───────────
sub_keys = set(sub_pairs["term_id"].astype(str) + "\t" + sub_pairs["item_id"].astype(str))
n_bad = sum(1 for tid, iid, _ in rows if (tid + "\t" + iid) not in sub_keys)
print(f"  Sahte (submission_pairs'te olmayan) aday sayısı: {n_bad} (0 olmalı)", flush=True)
assert n_bad == 0, f"HATA! {n_bad} aday gerçek submission_pairs'te yok!"

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
print(f"TAMAMLANDI — {len(out_df):,} GERÇEK test-adayı çifti üretildi", flush=True)
print(f"  Sorgu sayısı        : {out_df['term_id'].nunique():,}", flush=True)
print(f"  Ortalama cosine     : {out_df['cosine'].mean():.3f}", flush=True)
print(f"  Dosya               : {out_path}", flush=True)
print(f"  Toplam süre         : {(time.time()-t0)/60:.1f} dk", flush=True)
print(f"{'='*65}", flush=True)
print("\nÖrnek adaylar:", flush=True)
print(out_df.head(10).to_string(), flush=True)
