"""
37_v26_tyembed_feature.py — v26: TY-ecomm-embed Cosine Features
================================================================
v25 tabanı + TY-ecomm-embed bi-encoder özellikleri

v25'ten farklılıklar:
  + Phase D (YENİ): TY-ecomm-embed cosine similarity + group features
  + 5 yeni feature:
      tyembed_cos       : cosine(query_emb, item_emb)  — normalize dot product
      tyembed_grp_pct   : query grubu içinde yüzdelik sıra
      tyembed_grp_zscore: query grubu içinde z-score
      tyembed_grp_max   : query grubunda en yüksek cosine
      tyembed_rel_max   : tyembed_cos / grp_max
  + 44 feature toplam (39 v25 + 5 tyembed)

Cache gereksinimi (36_tyembed_cpuonly.py ile üretilmiş olmalı):
  emb_cache/item_embs_tyembed.npy     : (962873, 768) float32
  emb_cache/item_ids_tyembed.npy      : item_id dizisi
  emb_cache/train_q_embs_tyembed.npy  : (17968, 768) float32
  emb_cache/train_q_ids_tyembed.npy   : train term_id dizisi
  emb_cache/test_q_embs_tyembed.npy   : (32185, 768) float32
  emb_cache/test_q_ids_tyembed.npy    : test term_id dizisi
"""

import gc, re, sys, time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from rapidfuzz import fuzz as rfuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

sys.stdout.reconfigure(encoding="utf-8")

BASE       = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA       = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
SUBM       = BASE / "claude only" / "submissions"
CACHE_DIR  = BASE / "claude only"
BERT_V23   = CACHE_DIR / "models" / "bert_v23"
EMB_CACHE  = BASE / "claude only" / "emb_cache"
SUBM.mkdir(exist_ok=True)

LOWER = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(s): return str(s).translate(LOWER).lower().strip()
UNKNOWN = "unknown"

# ── TÜRKÇE STEM + METRİKLER (v23/v25 ile aynı) ───────────────────────
TR_SUFFIXES = sorted([
    "ları","leri","lar","ler","nın","nin","nun","nün","ın","in","un","ün",
    "daki","deki","taki","teki","dan","den","tan","ten","da","de","ta","te",
    "ya","ye","yı","yi","yu","yü","la","le","ça","çe","ca","ce",
    "lik","lık","luk","lük","cı","ci","cu","cü","çı","çi","çu","çü",
    "sı","si","su","sü","sal","sel","li","lı","lu","lü","ki",
    "a","e","ı","i","u","ü",
], key=len, reverse=True)

def stem(w):
    for s in TR_SUFFIXES:
        if w.endswith(s) and len(w)-len(s) >= 3: return w[:-len(s)]
    return w

def jac(a, b):
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb)/len(sa | sb) if (sa | sb) else 0.0

def stem_jac(a, b):
    sa = {stem(w) for w in a.split()}; sb = {stem(w) for w in b.split()}
    return len(sa & sb)/len(sa | sb) if (sa | sb) else 0.0

def q_cov(q, t):
    qw = set(q.split())
    return len(qw & set(t.split()))/len(qw) if qw else 0.0

TR_STOP_FUNC = {'ve','ile','için','bu','bir','de','da','den','dan','te','ta',
                'ki','mi','mu','mü','mı','ne','ya','veya','bazı','her','tüm',
                'çok','az','en','daha','gibi'}

def query_head(q):
    words = [w for w in q.split() if w not in TR_STOP_FUNC and len(w)>1]
    return words[-1] if words else (q.split()[-1] if q else "")

def _word_match(w, t_words, t_stems):
    if w in t_words: return True
    ws = stem(w)
    if ws in t_stems: return True
    if len(ws)>=5 and any(tw.startswith(ws[:5]) for tw in t_words): return True
    return False

def head_in_title(q, t):
    h = query_head(q)
    if not h: return 0.0
    t_words = set(t.split()); t_stems = {stem(w) for w in t_words}
    if h in t_words: return 1.0
    hs = stem(h)
    if hs in t_stems: return 0.8
    if len(hs)>=5 and any(tw.startswith(hs[:5]) for tw in t_words): return 0.4
    return 0.0

def weighted_q_cov(q, t):
    words = [w for w in q.split() if w not in TR_STOP_FUNC and len(w)>1]
    if not words: return q_cov(q, t)
    t_words = set(t.split()); t_stems = {stem(w) for w in t_words}
    total_w = hit_w = 0.0
    for i, w in enumerate(words):
        weight = 2.0 if i==len(words)-1 else 1.0; total_w += weight
        if _word_match(w, t_words, t_stems): hit_w += weight
    return hit_w/total_w if total_w>0 else 0.0

def head_in_cat(q, cat):
    h = query_head(q)
    if not h: return 0.0
    c_words = set(cat.replace("/"," ").split()); c_stems = {stem(w) for w in c_words}
    if h in c_words: return 1.0
    hs = stem(h)
    if hs in c_stems: return 0.8
    if len(hs)>=5 and any(cw.startswith(hs[:5]) for cw in c_words): return 0.4
    return 0.0

RENKLER = {"kırmızı","mavi","beyaz","siyah","sarı","yeşil","pembe","mor","gri","turuncu",
           "lacivert","bej","kahverengi","altın","gold","gümüş","silver","rose","ekru","krem",
           "bordo","haki","füme","antrasit","indigo","petrol"}
COLOR_NORM = {"gold":"altın","silver":"gümüş","rose":"pembe","krem":"bej","kiremit":"kırmızı"}
COLOR_FAMILY = {"antrasit":"gri","füme":"gri","platin":"gri","lacivert":"mavi","indigo":"mavi",
                "petrol":"mavi","bordo":"kırmızı","altın":"sarı","gold":"sarı","krem":"bej","ekru":"bej"}

def norm_color(c): return COLOR_NORM.get(c, c)
def color_family(c): return COLOR_FAMILY.get(c, c)

def get_query_color(q):
    for tok in q.split():
        if tok in RENKLER: return norm_color(tok)
    return None

def color_typed_match(q_color, item_renk):
    if q_color is None: return 0.0
    if not item_renk: return 0.0
    if q_color == item_renk: return 2.0
    if color_family(q_color) == color_family(item_renk): return 1.0
    return -1.0

MATERIAL_MAP_FULL = {"pamuk":"pamuk","pamuklu":"pamuk","deri":"deri","hakiki":"deri",
                     "polyester":"polyester","yün":"yün","keten":"keten","naylon":"naylon",
                     "çelik":"çelik","plastik":"plastik","ahşap":"ahşap"}

def parse_attrs(s):
    if not s or s in (UNKNOWN, ""): return {}
    d = {}
    for part in s.split(","):
        if ":" in part:
            k, _, v = part.partition(":")
            d[k.strip()] = v.strip()
    return d

def attr_typed_jac(q, a):
    d = parse_attrs(a)
    return jac(q, " ".join(d.values())) if d else jac(q, a)

def brand_tok_overlap(q, brand):
    if not brand or brand == UNKNOWN: return 0.0
    b_toks = set(t for t in re.sub(r'[^a-z0-9çğıöşü\s]', ' ', brand).split() if len(t) > 1)
    q_toks = set(re.sub(r'[^a-z0-9çğıöşü\s]', ' ', q).split())
    return len(b_toks & q_toks) / len(b_toks) if b_toks else 0.0

def material_match(q, attr_mat):
    if not attr_mat: return 0.0
    for tok in q.split():
        mat = MATERIAL_MAP_FULL.get(tok)
        if mat and mat in attr_mat: return 1.0
    return 0.0

def gender_cross(q, g):
    q_k = bool(re.search(r'\b(kadın|bayan)\b', q))
    q_e = bool(re.search(r'\berkek\b', q))
    g_k = g in ("kadın","bayan","kız")
    g_e = g == "erkek"
    if q_k and g_e: return -1.0
    if q_e and g_k: return -1.0
    if q_k and g_k: return  1.0
    if q_e and g_e: return  1.0
    return 0.0

BRAND_TOKENS = set()
GENDER_STOP  = {"kadın","erkek","bayan","kız","unisex","bay","bayan"}
AGE_STOP     = {"bebek","çocuk","yetişkin"}
COLOR_STOP   = RENKLER
MAT_STOP     = set(MATERIAL_MAP_FULL.keys())
FUNC_STOP    = TR_STOP_FUNC

def extract_product_type_words(q, brand):
    b_toks = set()
    if brand and brand != UNKNOWN:
        b_toks = set(re.sub(r'[^a-z0-9çğıöşü\s]', ' ', brand).split())
    stop = b_toks | GENDER_STOP | AGE_STOP | COLOR_STOP | MAT_STOP | FUNC_STOP
    return [w for w in q.split() if w not in stop and len(w) > 1]

def product_type_cover(q, t, brand):
    core = extract_product_type_words(q, brand)
    if not core: return 0.0
    t_words = set(t.split()); t_stems = {stem(w) for w in t_words}
    hits = sum(_word_match(w, t_words, t_stems) for w in core)
    return hits / len(core)

def brand_weight(q):
    global BRAND_TOKENS
    toks = q.split()
    if not toks: return 0.0
    brand_hits = sum(1 for t in toks if t in BRAND_TOKENS)
    return brand_hits / len(toks)

# ═══════════════════════════════════════════════════════════════════════
# PHASE A: VERİ + NEGATİFLER (v23/v25 ile birebir aynı)
# ═══════════════════════════════════════════════════════════════════════
t0 = time.time()
print("=" * 65, flush=True)
print("[A] VERİ + NEGATİFLER (v23 ile aynı, rng=42)...", flush=True)

items       = pd.read_csv(DATA / "items.csv")
terms       = pd.read_csv(DATA / "terms.csv")
train_pairs = pd.read_csv(DATA / "training_pairs.csv")
sub_pairs   = pd.read_csv(DATA / "submission_pairs.csv")

for col in ["title","category","brand","gender","age_group","attributes"]:
    items[col] = items[col].fillna(UNKNOWN).apply(trl)
terms["query"] = terms["query"].fillna("").apply(trl)
items["main_category"] = items["category"].str.split("/").str[0].fillna(UNKNOWN)
items["L1"] = items["main_category"]

from collections import Counter as _Counter
_brand_freq = _Counter()
for b in items["brand"]:
    if b and b != UNKNOWN:
        for tok in re.sub(r'[^a-z0-9çğıöşü\s]', ' ', b).split():
            if len(tok) > 1:
                _brand_freq[tok] += 1
BRAND_TOKENS = {tok for tok, cnt in _brand_freq.items() if cnt >= 3}

item_ids_arr    = items["item_id"].values
item_mains_arr  = items["main_category"].values
item_brands_arr = items["brand"].values

iid_to_str = {str(row.item_id): row for row in items.itertuples()}
tid_to_q   = dict(zip(terms["term_id"], terms["query"]))

train_tids = set(train_pairs["term_id"].unique())
test_tids  = set(sub_pairs["term_id"].unique())
assert len(train_tids & test_tids) == 0, "LEAK: train/test term_id kesişimi var!"
print(f"  {len(items):,} ürün | {len(sub_pairs):,} test çifti", flush=True)
print(f"  ✓ train/test term_id disjoint ({len(train_tids):,} train, {len(test_tids):,} test)", flush=True)

def build_group_idx(df, cols):
    df_r = df.reset_index(drop=True)
    return {k: g.index.values for k, g in df_r.groupby(cols if len(cols)>1 else cols[0], sort=False)}

by_main   = build_group_idx(items, ["main_category"])
by_gender = build_group_idx(items, ["gender"])
by_mg     = build_group_idx(items, ["main_category","gender"])
by_age    = build_group_idx(items, ["age_group"])
by_ma     = build_group_idx(items, ["main_category","age_group"])
by_brand  = build_group_idx(items, ["brand"])

items["sub_category"] = items["category"].str.split("/").str[:2].str.join("/").fillna(UNKNOWN)
by_brand_sub = {}
for (brand, sub), grp in items.reset_index(drop=True).groupby(["brand","sub_category"]):
    by_brand_sub[(brand, sub)] = grp.index.values

positive_keys = set(train_pairs["term_id"].astype(str) + "\t" + train_pairs["item_id"].astype(str))
used_keys: set = set()
rng = np.random.default_rng(42)

def sample_pool(pool, term_id, pos_iid, max_tries=40):
    if pool is None or len(pool) == 0: return None
    for _ in range(max_tries):
        idx = int(pool[rng.integers(0, len(pool))])
        iid = str(item_ids_arr[idx])
        k = term_id + "\t" + iid
        if iid != pos_iid and k not in positive_keys and k not in used_keys:
            return iid
    return None

def sample_diff_main(cur_main, term_id, pos_iid, max_tries=80):
    n = len(item_ids_arr)
    for _ in range(max_tries):
        idx = int(rng.integers(0, n))
        if item_mains_arr[idx] == cur_main: continue
        iid = str(item_ids_arr[idx])
        k = term_id + "\t" + iid
        if iid != pos_iid and k not in positive_keys and k not in used_keys:
            return iid
    return None

def sample_same_brand_diff_main(brand, main, term_id, pos_iid, max_tries=60):
    if not brand or brand == UNKNOWN: return None
    pool = by_brand.get(brand)
    if pool is None or len(pool) == 0: return None
    for _ in range(max_tries):
        idx = int(pool[rng.integers(0, len(pool))])
        if item_mains_arr[idx] == main: continue
        iid = str(item_ids_arr[idx])
        k = term_id + "\t" + iid
        if iid != pos_iid and k not in positive_keys and k not in used_keys:
            return iid
    return None

def sample_same_brand_diff_sub(brand, sub, term_id, pos_iid, max_tries=60):
    if not brand or brand == UNKNOWN: return None
    pool = by_brand_sub.get((brand, sub))
    if pool is None or len(pool) == 0: return None
    for _ in range(max_tries):
        idx = int(pool[rng.integers(0, len(pool))])
        iid = str(item_ids_arr[idx])
        k = term_id + "\t" + iid
        if iid != pos_iid and k not in positive_keys and k not in used_keys:
            return iid
    return None

items["sub_category"] = items["sub_category"].apply(trl)
pos_with_info = train_pairs.merge(terms, on="term_id", how="left").merge(
    items[["item_id","main_category","sub_category","gender","age_group","brand"]], on="item_id", how="left"
)
for col in ["main_category","sub_category","gender","age_group","brand"]:
    pos_with_info[col] = pos_with_info[col].fillna(UNKNOWN).apply(trl)

neg_tids, neg_iids, neg_src = [], [], []
NEG_PER_POS = 5

for row in pos_with_info.itertuples(index=False):
    tid    = str(row.term_id)
    pos_id = str(row.item_id)
    main   = str(row.main_category)
    sub    = str(row.sub_category)
    query  = str(row.query) if isinstance(row.query, str) else ""
    brand  = str(row.brand)
    selected = []

    iid = sample_same_brand_diff_main(brand, main, tid, pos_id)
    if iid: selected.append((iid, "same_brand_diff_main"))

    if len(selected) < NEG_PER_POS:
        iid = sample_same_brand_diff_sub(brand, sub, tid, pos_id)
        if iid: selected.append((iid, "same_brand_diff_sub"))

    if len(selected) < NEG_PER_POS:
        if re.search(r'\berkek\b', query):
            pool = by_mg.get((main, "kadın"))
            if pool is None: pool = by_gender.get("kadın")
            iid = sample_pool(pool, tid, pos_id)
            if iid: selected.append((iid, "gender_conflict"))
        elif re.search(r'\b(kadın|bayan)\b', query):
            pool = by_mg.get((main, "erkek"))
            if pool is None: pool = by_gender.get("erkek")
            iid = sample_pool(pool, tid, pos_id)
            if iid: selected.append((iid, "gender_conflict"))

    if len(selected) < NEG_PER_POS:
        if re.search(r'\b(bebek|çocuk)\b', query):
            pool = by_ma.get((main, "yetişkin"))
            if pool is None: pool = by_age.get("yetişkin")
            iid = sample_pool(pool, tid, pos_id)
            if iid: selected.append((iid, "age_conflict"))

    if len(selected) < NEG_PER_POS:
        iid = sample_pool(by_main.get(main), tid, pos_id)
        if iid: selected.append((iid, "same_main_category"))

    while len(selected) < NEG_PER_POS:
        iid = sample_diff_main(main, tid, pos_id)
        if iid: selected.append((iid, "diff_main_category"))
        else: break

    for iid, src in selected[:NEG_PER_POS]:
        k = tid + "\t" + iid
        if k in used_keys or k in positive_keys: continue
        used_keys.add(k)
        neg_tids.append(tid); neg_iids.append(iid); neg_src.append(src)

negatives = pd.DataFrame({"term_id": neg_tids, "item_id": neg_iids, "label": 0})
train_pairs["label"] = 1
train_df = pd.concat([
    train_pairs[["term_id","item_id","label"]],
    negatives[["term_id","item_id","label"]]
], ignore_index=True).sort_values("term_id").reset_index(drop=True)
train_df["term_id"] = train_df["term_id"].astype(str)
train_df["item_id"] = train_df["item_id"].astype(str)
sub_pairs["term_id"] = sub_pairs["term_id"].astype(str)
sub_pairs["item_id"] = sub_pairs["item_id"].astype(str)

src_counts = Counter(neg_src)
print(f"  Train: {len(train_df):,} | pos={train_df.label.sum():,}", flush=True)
print(f"  Neg kaynak: {dict(src_counts)}", flush=True)
print(f"  Süre: {time.time()-t0:.0f}s", flush=True)

# ═══════════════════════════════════════════════════════════════════════
# PHASE B: BERT SCORES (v23 cache)
# ═══════════════════════════════════════════════════════════════════════
print("\n[B] BERT SCORES (v23 cache)...", flush=True)

train_bert_path = CACHE_DIR / "bert_scores_v23_train.npy"
test_bert_path  = CACHE_DIR / "submissions" / "bert_scores_v23_test.npy"

if not train_bert_path.exists() or not test_bert_path.exists():
    print("  ✗ bert cache bulunamadı!", flush=True); sys.exit(1)

train_bert = np.load(str(train_bert_path))
test_bert  = np.load(str(test_bert_path))

if len(train_bert) != len(train_df):
    print(f"  ✗ BERT train boyutu uyuşmuyor: {len(train_bert)} vs {len(train_df)}", flush=True)
    if len(train_bert) < len(train_df):
        pad = np.zeros(len(train_df) - len(train_bert), dtype=np.float32)
        train_bert = np.concatenate([train_bert, pad])
    else:
        train_bert = train_bert[:len(train_df)]

print(f"  Train BERT: {len(train_bert):,} ✓ | Test BERT: {len(test_bert):,} ✓", flush=True)

# ═══════════════════════════════════════════════════════════════════════
# PHASE C: BERT GROUP FEATURES (v25 ile aynı)
# ═══════════════════════════════════════════════════════════════════════
print("\n[C] BERT GROUP FEATURES...", flush=True)
t1 = time.time()

def add_group_features(df, score_arr, prefix, tid_col="term_id"):
    """Bir score dizisinden group rank/zscore/max/rel_max feature'ları üret."""
    df = df.copy()
    col = f"__{prefix}__"
    df[col] = score_arr.astype(np.float32)
    df[f"{prefix}_grp_pct"]    = df.groupby(tid_col)[col].rank(pct=True, ascending=True)
    gm = df.groupby(tid_col)[col].transform("mean")
    gs = df.groupby(tid_col)[col].transform("std").fillna(1e-8)
    df[f"{prefix}_grp_zscore"] = (df[col] - gm) / gs
    df[f"{prefix}_grp_max"]    = df.groupby(tid_col)[col].transform("max")
    df[f"{prefix}_rel_max"]    = df[col] / df[f"{prefix}_grp_max"].clip(lower=1e-8)
    df.drop(columns=[col], inplace=True)
    return df

train_df      = add_group_features(train_df, train_bert, prefix="bert", tid_col="term_id")
sub_pairs_aug = add_group_features(sub_pairs.copy(), test_bert, prefix="bert", tid_col="term_id")

print(f"  bert_grp_pct train mean: {train_df['bert_grp_pct'].mean():.3f} | "
      f"test mean: {sub_pairs_aug['bert_grp_pct'].mean():.3f}", flush=True)
print(f"  → {time.time()-t1:.0f}s", flush=True)
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# PHASE D: TY-ECOMM-EMBED COSINE + GROUP FEATURES  (YENİ)
# ═══════════════════════════════════════════════════════════════════════
print("\n[D] TY-ECOMM-EMBED FEATURES...", flush=True)
t1 = time.time()

# Embedding cache yolları
ITEM_EMB    = EMB_CACHE / "item_embs_tyembed.npy"
ITEM_IDS    = EMB_CACHE / "item_ids_tyembed.npy"
TRAIN_Q_EMB = EMB_CACHE / "train_q_embs_tyembed.npy"
TRAIN_Q_IDS = EMB_CACHE / "train_q_ids_tyembed.npy"
TEST_Q_EMB  = EMB_CACHE / "test_q_embs_tyembed.npy"
TEST_Q_IDS  = EMB_CACHE / "test_q_ids_tyembed.npy"

for p in [ITEM_EMB, ITEM_IDS, TRAIN_Q_EMB, TRAIN_Q_IDS, TEST_Q_EMB, TEST_Q_IDS]:
    if not p.exists():
        print(f"  ✗ Eksik cache: {p}", flush=True)
        print("    → 36_tyembed_cpuonly.py çalıştır!", flush=True)
        sys.exit(1)

print("  Embedding cache yükleniyor...", flush=True)
item_embs    = np.load(str(ITEM_EMB))                    # (962873, 768) float32
item_ids_raw = np.load(str(ITEM_IDS), allow_pickle=True) # item_id dizisi
train_q_embs = np.load(str(TRAIN_Q_EMB))                 # (17968, 768) float32
train_q_ids  = np.load(str(TRAIN_Q_IDS), allow_pickle=True)
test_q_embs  = np.load(str(TEST_Q_EMB))                  # (32185, 768) float32
test_q_ids   = np.load(str(TEST_Q_IDS), allow_pickle=True)

print(f"  item_embs: {item_embs.shape} | "
      f"train_q: {train_q_embs.shape} | test_q: {test_q_embs.shape}", flush=True)

# Lookup dictionaries: id -> embedding index
iid_to_emb_idx      = {str(iid): i for i, iid in enumerate(item_ids_raw)}
train_tid_to_emb_idx = {str(tid): i for i, tid in enumerate(train_q_ids)}
test_tid_to_emb_idx  = {str(tid): i for i, tid in enumerate(test_q_ids)}

def compute_cosine(df, q_embs, q_tid_map, item_embs, iid_map, chunk=50_000):
    """
    L2-normalize edilmiş embedding'ler için cosine = dot product.
    Chunk ile hafıza tasarrufu: her seferinde CHUNK×768×4×2 byte.
    """
    n = len(df)
    cos = np.zeros(n, dtype=np.float32)
    tids = df["term_id"].values
    iids = df["item_id"].values

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        q_idx = np.array([q_tid_map.get(str(t), -1) for t in tids[start:end]])
        i_idx = np.array([iid_map.get(str(i), -1)   for i in iids[start:end]])
        valid = (q_idx >= 0) & (i_idx >= 0)
        if valid.any():
            q_chunk = q_embs[q_idx[valid]]      # (k, 768)
            i_chunk = item_embs[i_idx[valid]]   # (k, 768)
            cos[start:end][valid] = (q_chunk * i_chunk).sum(axis=1)
        if (start // chunk) % 20 == 0 or end == n:
            print(f"    {end:,}/{n:,} ({100*end/n:.0f}%)", flush=True)
    return cos

print("  Train cosine hesaplanıyor...", flush=True)
train_cos = compute_cosine(
    train_df, train_q_embs, train_tid_to_emb_idx, item_embs, iid_to_emb_idx
)
print(f"  train_cos: min={train_cos.min():.3f} max={train_cos.max():.3f} "
      f"mean={train_cos.mean():.3f}", flush=True)

print("  Test cosine hesaplanıyor...", flush=True)
test_cos = compute_cosine(
    sub_pairs_aug, test_q_embs, test_tid_to_emb_idx, item_embs, iid_to_emb_idx
)
print(f"  test_cos:  min={test_cos.min():.3f} max={test_cos.max():.3f} "
      f"mean={test_cos.mean():.3f}", flush=True)

# Group features (percentile, zscore, max, rel_max)
train_df      = add_group_features(train_df,      train_cos, prefix="tyembed", tid_col="term_id")
sub_pairs_aug = add_group_features(sub_pairs_aug, test_cos,  prefix="tyembed", tid_col="term_id")

# Direkt cosine'ı da feature olarak tut
train_df["tyembed_cos"]      = train_cos
sub_pairs_aug["tyembed_cos"] = test_cos

print(f"  tyembed_cos pozitifler için mean: "
      f"{train_df.loc[train_df['label']==1,'tyembed_cos'].mean():.3f} "
      f"(negatifler: {train_df.loc[train_df['label']==0,'tyembed_cos'].mean():.3f})", flush=True)
print(f"  → TY-ecomm-embed features hazır | {(time.time()-t1)/60:.1f} dk", flush=True)

# item_embs RAM'den serbest bırak (artık gerekmez)
del item_embs, train_q_embs, test_q_embs
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# PHASE E: FEATURE ENGINEERING (44 feature)
# ═══════════════════════════════════════════════════════════════════════
print("\n[E] FEATURE ENGINEERING (44 feature)...", flush=True)

def build_features(df_in, tfidf_vect, bert_arr, fit=False, all_texts=None):
    df = df_in.copy()
    df["term_id"] = df["term_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    terms_local = terms.copy(); terms_local["term_id"] = terms_local["term_id"].astype(str)
    items_local = items.copy(); items_local["item_id"] = items_local["item_id"].astype(str)

    df = df.merge(terms_local[["term_id","query"]], on="term_id", how="left")
    df = df.merge(items_local[["item_id","title","category","brand","gender",
                                "age_group","attributes","main_category","L1"]],
                  on="item_id", how="left")
    for col in ["query","title","brand","category","gender","age_group",
                "attributes","main_category","L1"]:
        df[col] = df[col].fillna(UNKNOWN).apply(trl)

    if fit:
        corpus = list(df["title"]) + list(df["query"]) + (all_texts or [])
        tfidf_vect.fit(corpus)

    qs    = df["query"].tolist()
    ts    = df["title"].tolist()
    cats  = df["category"].tolist()
    brs   = df["brand"].tolist()
    gens  = df["gender"].tolist()
    ages  = df["age_group"].tolist()
    attrs = df["attributes"].tolist()

    def tfidf_cos_fn(ql, tl, chunk=50_000):
        n = len(ql); out = np.zeros(n, dtype=np.float32)
        for i in range(0, n, chunk):
            qm = normalize(tfidf_vect.transform(ql[i:i+chunk]), "l2")
            tm = normalize(tfidf_vect.transform(tl[i:i+chunk]), "l2")
            out[i:i+chunk] = np.array(qm.multiply(tm).sum(axis=1)).flatten()
        return out

    parsed    = [parse_attrs(a) for a in attrs]
    attr_renk = [d.get("renk","") for d in parsed]
    attr_mat  = [d.get("materyal bileşeni", d.get("materyal","")) for d in parsed]
    attr_kol  = [d.get("kol boyu", d.get("kol tipi","")) for d in parsed]
    q_colors  = [get_query_color(q) for q in qs]

    f = pd.DataFrame()
    # ── v23 35 feature
    f["fuzz_partial"]    = [rfuzz.partial_ratio(q,t)/100    for q,t in zip(qs,ts)]
    f["fuzz_set"]        = [rfuzz.token_set_ratio(q,t)/100  for q,t in zip(qs,ts)]
    f["fuzz_sort"]       = [rfuzz.token_sort_ratio(q,t)/100 for q,t in zip(qs,ts)]
    f["fuzz_basic"]      = [rfuzz.ratio(q,t)/100            for q,t in zip(qs,ts)]
    f["jaccard"]         = [jac(q,t)                        for q,t in zip(qs,ts)]
    f["tfidf_cos"]       = tfidf_cos_fn(qs, ts)
    f["q_cov_title"]     = [q_cov(q,t)                     for q,t in zip(qs,ts)]
    f["t_cov_query"]     = [q_cov(t,q)                     for q,t in zip(qs,ts)]
    f["cat_overlap"]     = [jac(q, c.replace("/"," "))      for q,c in zip(qs,cats)]
    f["exact_in_title"]  = [(q in t)*1.0                    for q,t in zip(qs,ts)]
    f["token_overlap"]   = [len(set(q.split())&set(t.split())) for q,t in zip(qs,ts)]
    f["age_in_q"]        = [(a not in (UNKNOWN,"") and a in q)*1.0 for a,q in zip(ages,qs)]
    f["stem_jaccard"]    = [stem_jac(q,t)                   for q,t in zip(qs,ts)]
    f["stem_cat_jac"]    = [stem_jac(q, c.replace("/"," ")) for q,c in zip(qs,cats)]
    f["gender_cross"]    = [gender_cross(q,g)               for q,g in zip(qs,gens)]
    f["first_tok_title"] = [(q.split()[0] in t if q.split() else False)*1.0 for q,t in zip(qs,ts)]
    f["first_tok_brand"] = [(q.split()[0] in b if q.split() and b!=UNKNOWN else False)*1.0 for q,b in zip(qs,brs)]
    f["q_len"]           = [len(q.split()) for q in qs]
    f["t_len"]           = [len(t.split()) for t in ts]
    f["bert_score"]      = bert_arr
    f["ana_kategori"]    = pd.Categorical([c.split("/")[0] for c in cats])
    f["color_typed"]     = [color_typed_match(qc,ir) for qc,ir in zip(q_colors,attr_renk)]
    f["has_q_color"]     = [(qc is not None)*1.0 for qc in q_colors]
    f["attr_has_renk"]   = [(ar!="")*1.0 for ar in attr_renk]
    f["brand_tok_ovlp"]  = [brand_tok_overlap(q,b) for q,b in zip(qs,brs)]
    f["material_match"]  = [material_match(q,am)  for q,am in zip(qs,attr_mat)]
    f["kol_boyu_match"]  = [( ("uzun" in q and "uzun" in ak) or
                               ("kısa" in q and "kısa" in ak) or
                               ("kolsuz" in q and "kolsuz" in ak) )*1.0
                             for q,ak in zip(qs,attr_kol)]
    f["attr_jac_fixed"]  = [attr_typed_jac(q,a) for q,a in zip(qs,attrs)]
    f["attr_q_cov"]      = [q_cov(q," ".join(d.values())) if (d:=parse_attrs(a)) else 0.0
                             for q,a in zip(qs,attrs)]
    f["head_in_title"]   = [head_in_title(q,t) for q,t in zip(qs,ts)]
    f["weighted_q_cov"]  = [weighted_q_cov(q,t) for q,t in zip(qs,ts)]
    f["head_in_cat"]     = [head_in_cat(q,c)    for q,c in zip(qs,cats)]
    f["product_type_cover"] = [product_type_cover(q,t,b) for q,t,b in zip(qs,ts,brs)]
    f["all_q_in_title"]  = [(all(tok in t.split() for tok in q.split()) if q.split() else False)*1.0
                             for q,t in zip(qs,ts)]
    f["brand_weight"]    = [brand_weight(q) for q in qs]

    # ── BERT group features (4 adet — v25'ten)
    for feat in ["bert_grp_pct","bert_grp_zscore","bert_grp_max","bert_rel_max"]:
        f[feat] = df_in[feat].values if feat in df_in.columns else (0.5 if "pct" in feat or "rel" in feat else 0.0)

    # ── TY-ecomm-embed features (5 adet — v26 YENİ)
    f["tyembed_cos"]        = df_in["tyembed_cos"].values        if "tyembed_cos"        in df_in.columns else 0.0
    f["tyembed_grp_pct"]    = df_in["tyembed_grp_pct"].values    if "tyembed_grp_pct"    in df_in.columns else 0.5
    f["tyembed_grp_zscore"] = df_in["tyembed_grp_zscore"].values if "tyembed_grp_zscore" in df_in.columns else 0.0
    f["tyembed_grp_max"]    = df_in["tyembed_grp_max"].values    if "tyembed_grp_max"    in df_in.columns else 0.0
    f["tyembed_rel_max"]    = df_in["tyembed_rel_max"].values    if "tyembed_rel_max"    in df_in.columns else 0.5

    return f

t1 = time.time()
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=60_000, sublinear_tf=True, min_df=2)

FEATS = [
    # v23 35 feature
    "fuzz_partial","fuzz_set","fuzz_sort","fuzz_basic",
    "jaccard","tfidf_cos","q_cov_title","t_cov_query",
    "cat_overlap","exact_in_title","token_overlap","age_in_q",
    "stem_jaccard","stem_cat_jac","gender_cross",
    "first_tok_title","first_tok_brand","q_len","t_len",
    "bert_score","ana_kategori",
    "color_typed","has_q_color","attr_has_renk","brand_tok_ovlp",
    "material_match","kol_boyu_match","attr_jac_fixed","attr_q_cov",
    "head_in_title","weighted_q_cov","head_in_cat",
    "product_type_cover","all_q_in_title","brand_weight",
    # v25 BERT group (4 adet)
    "bert_grp_pct","bert_grp_zscore","bert_grp_max","bert_rel_max",
    # v26 TY-ecomm-embed (5 adet — YENİ)
    "tyembed_cos","tyembed_grp_pct","tyembed_grp_zscore",
    "tyembed_grp_max","tyembed_rel_max",
]

X_tr = build_features(
    train_df, tfidf, train_bert, fit=True,
    all_texts=items["title"].tolist() + terms["query"].tolist()
)
y    = train_df["label"].values
tids = train_df["term_id"].values
print(f"  Train features: {X_tr[FEATS].shape} | {(time.time()-t1):.0f}s", flush=True)

# ═══════════════════════════════════════════════════════════════════════
# PHASE F: BINARY LGBM 5-FOLD (iter=8000, early_stop=200)
# ═══════════════════════════════════════════════════════════════════════
print("\n[F] BINARY LGBM 5-FOLD (iter=8000, early_stop=200)...", flush=True)

oof    = np.zeros(len(train_df), dtype=np.float32)
models = []
gkf    = GroupKFold(n_splits=5)

for fold, (tri, vli) in enumerate(gkf.split(X_tr, y, tids)):
    X_train_f = X_tr[FEATS].iloc[tri]
    y_train_f = y[tri]
    X_val_f   = X_tr[FEATS].iloc[vli]
    y_val_f   = y[vli]

    model = LGBMClassifier(
        n_estimators=8000,
        learning_rate=0.03,
        num_leaves=127,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train_f, y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)]
    )

    val_proba = model.predict_proba(X_val_f)[:, 1]
    oof[vli]  = val_proba.astype(np.float32)
    models.append(model)

    best_f = 0.0
    for thr in np.arange(0.20, 0.90, 0.02):
        fs = f1_score(y_val_f, (val_proba > thr).astype(int), average="macro")
        if fs > best_f: best_f = fs
    print(f"  Fold {fold+1} | iter={model.best_iteration_} | val F1≈{best_f:.4f}", flush=True)

best_thr, best_f1 = 0.5, 0.0
for thr in np.arange(0.20, 0.90, 0.005):
    s = f1_score(y, (oof > thr).astype(int), average="macro")
    if s > best_f1: best_f1, best_thr = s, thr
print(f"\n  OOF F1={best_f1:.5f} | OOF thr={best_thr:.4f}", flush=True)

fi = pd.Series(
    sum(m.feature_importances_ for m in models) / len(models),
    index=FEATS
).sort_values(ascending=False)
print("\n  Feature Önemleri (top 20):", flush=True)
YENI = {"bert_grp_pct","bert_grp_zscore","bert_grp_max","bert_rel_max",
        "tyembed_cos","tyembed_grp_pct","tyembed_grp_zscore","tyembed_grp_max","tyembed_rel_max"}
for feat, imp in fi.head(20).items():
    marker = " ← YENİ" if feat in YENI else ""
    print(f"    {feat:25s}: {imp:6.0f}{marker}", flush=True)

# ═══════════════════════════════════════════════════════════════════════
# PHASE G: TEST INFERENCE + SUBMISSION
# ═══════════════════════════════════════════════════════════════════════
print("\n[G] TEST INFERENCE + SUBMISSION...", flush=True)

X_te = build_features(sub_pairs_aug, tfidf, test_bert)
test_proba = sum(m.predict_proba(X_te[FEATS])[:, 1] for m in models) / len(models)

np.save(str(SUBM / "v26_test_proba.npy"), test_proba.astype(np.float32))

final = (test_proba > best_thr).astype(int)
pos_n = final.sum()
print(f"  Direct thr={best_thr:.4f} → Pozitif={pos_n:,} ({100*pos_n/len(sub_pairs):.1f}%)", flush=True)

out_path = SUBM / "submission_v26_tyembed.csv"
pd.DataFrame({"id": sub_pairs["id"], "prediction": final}).to_csv(str(out_path), index=False)

print(f"\n{'='*65}", flush=True)
print(f"TAMAMLANDI — v26 TY-ecomm-embed Features", flush=True)
print(f"  OOF F1        : {best_f1:.5f}", flush=True)
print(f"  OOF threshold : {best_thr:.4f}", flush=True)
print(f"  Pozitif       : {pos_n:,} ({100*pos_n/len(sub_pairs):.1f}%)", flush=True)
print(f"  Train çifti   : {len(train_df):,}", flush=True)
print(f"  Feature sayısı: {len(FEATS)} (35 v23 + 4 bert_grp + 5 tyembed)", flush=True)
print(f"  Toplam süre   : {(time.time()-t0)/60:.1f} dk", flush=True)
print(f"  Dosya         : {out_path}", flush=True)
print(f"{'='*65}", flush=True)
