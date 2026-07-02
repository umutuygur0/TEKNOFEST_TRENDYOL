"""
28_v23_combined.py — v23 Combined All Improvements
====================================================
v22'den farklar (3 iyileştirme tek versiyonda):

1. BERT HARD NEGATIVE RETRAINING
   - v22: BERT kolay negatiflerle (same_cat + diff_cat) eğitilmişti
   - v23: BERT'i same_brand_diff_main + same_brand_diff_sub ile yeniden eğit
   - BERT artık "marka eşleşir ama ürün tipi farklı = negatif" kuralını KENDİ öğrenir
   - 5 epoch (v21'de 3 idi), 500K sample (v21'de 300K idi)

2. DAHA ÇEŞİTLİ NEGATİFLER (NEG_PER_POS=5)
   - same_brand_diff_main   : marka eşleşir, ana kategori farklı (v22'den)
   - same_brand_diff_sub    : marka eşleşir, aynı ana kat, farklı alt kat (YENİ, daha sert)
   - gender_conflict        : cinsiyet uyuşmazlığı
   - age_conflict           : yaş uyuşmazlığı
   - same_main_cat          : aynı ana kategoriden rastgele (fallback)
   - diff_main_cat          : farklı kategoriden (son fallback)

3. YENİ LGBM ÖZELLİKLERİ (3 adet)
   - product_type_cover : marka/cinsiyet/renk/malzeme/stop çıkarıldıktan sonra
                          kalan "ürün tipi" kelimelerinin title'da kapsamı
                          → "karaca çaydanlık" vs "karaca bardak seti" = 0.0 ✓
   - all_q_in_title     : tüm query kelimeleri title'da var mı? (0/1 binary)
   - brand_weight       : query'nin kaçta kaçı marka tokenları?

LEAK KONTROL LİSTESİ:
   ✓ Negatifler SADECE item kataloğundan → submission_pairs YOK
   ✓ positive_keys seti: bilinen pozitifler negatife düşmüyor
   ✓ used_keys seti: aynı (term_id, item_id) çifti iki kez eklenmiyor
   ✓ GroupKFold(term_id): query bazlı CV bölme, sızdırma yok
   ✓ Train ve test term_id'leri disjoint (0 örtüşme) → doğrudan leak imkansız
   ✓ BERT test inference: sadece sub_pairs → training bilgisi karışmıyor
"""

import gc, re, sys, time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
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
MODELS_DIR = BASE / "claude only" / "models"
BERT_V23   = MODELS_DIR / "bert_v23"
BERT_BASE  = "dbmdz/bert-base-turkish-cased"
CACHE_DIR  = BASE / "claude only"

LOWER = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(s): return str(s).translate(LOWER).lower().strip()

UNKNOWN = "unknown"

# ─────────────────────────────────────────────
# TÜRKÇE STEM
# ─────────────────────────────────────────────
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
        if w.endswith(s) and len(w) - len(s) >= 3:
            return w[:-len(s)]
    return w

def jac(a, b):
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0

def stem_jac(a, b):
    sa = {stem(w) for w in a.split()}
    sb = {stem(w) for w in b.split()}
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0

def q_cov(q, t):
    qw = set(q.split())
    return len(qw & set(t.split())) / len(qw) if qw else 0.0

# ─────────────────────────────────────────────
# HEAD NOUN (v22'den)
# ─────────────────────────────────────────────
TR_STOP_FUNC = {
    've', 'ile', 'için', 'bu', 'bir', 'de', 'da', 'den', 'dan',
    'te', 'ta', 'ki', 'mi', 'mu', 'mü', 'mı', 'ne', 'ya', 'veya',
    'bazı', 'her', 'tüm', 'çok', 'az', 'en', 'daha', 'gibi',
}

def query_head(q):
    words = [w for w in q.split() if w not in TR_STOP_FUNC and len(w) > 1]
    return words[-1] if words else (q.split()[-1] if q else "")

def _word_match(w, t_words, t_stems):
    if w in t_words: return True
    ws = stem(w)
    if ws in t_stems: return True
    if len(ws) >= 5 and any(tw.startswith(ws[:5]) for tw in t_words): return True
    return False

def head_in_title(q, t):
    h = query_head(q)
    if not h: return 0.0
    t_words = set(t.split())
    t_stems = {stem(w) for w in t_words}
    if h in t_words: return 1.0
    hs = stem(h)
    if hs in t_stems: return 0.8
    if len(hs) >= 5 and any(tw.startswith(hs[:5]) for tw in t_words): return 0.4
    return 0.0

def weighted_q_cov(q, t):
    words = [w for w in q.split() if w not in TR_STOP_FUNC and len(w) > 1]
    if not words: return q_cov(q, t)
    t_words = set(t.split())
    t_stems = {stem(w) for w in t_words}
    total_w = hit_w = 0.0
    for i, w in enumerate(words):
        weight = 2.0 if i == len(words) - 1 else 1.0
        total_w += weight
        if _word_match(w, t_words, t_stems): hit_w += weight
    return hit_w / total_w if total_w > 0 else 0.0

def head_in_cat(q, cat):
    h = query_head(q)
    if not h: return 0.0
    c_words = set(cat.replace("/", " ").split())
    c_stems = {stem(w) for w in c_words}
    if h in c_words: return 1.0
    hs = stem(h)
    if hs in c_stems: return 0.8
    if len(hs) >= 5 and any(cw.startswith(hs[:5]) for cw in c_words): return 0.4
    return 0.0

# ─────────────────────────────────────────────
# YENİ v23: PRODUCT TYPE COVER
# ─────────────────────────────────────────────
GENDER_WORDS = {"erkek","kadın","bayan","kız","unisex","erkekler","kadınlar","kızlar"}
AGE_WORDS    = {"bebek","çocuk","yetişkin","çocuklar","bebekler"}

MATERIAL_MAP = {
    "pamuk","pamuklu","deri","hakiki","polyester","yün","keten","naylon",
    "çelik","plastik","ahşap","akrilik","viskon","modal","bambu","ipek",
}

RENKLER = {
    "kırmızı","mavi","beyaz","siyah","sarı","yeşil","pembe","mor","gri","turuncu",
    "lacivert","bej","kahverengi","altın","gold","gümüş","silver","rose","ekru","krem",
    "bordo","haki","füme","antrasit","indigo","petrol",
}

def brand_tok_set(brand):
    if not brand or brand == UNKNOWN: return set()
    return set(t for t in re.sub(r'[^a-z0-9çğıöşü\s]', ' ', brand).split() if len(t) > 1)

def product_type_cover(q, t, brand, btok_ovlp):
    """
    Query'den marka (eğer eşleşiyorsa) / cinsiyet / renk / malzeme / stop çıkar.
    Kalan 'ürün tipi' kelimeleri title'da ne kadar kaplıyor?

    Örnek:
      "karaca çaydanlık" vs title="karaca bardak seti" (brand="karaca", ovlp=1.0)
      → b_toks={"karaca"}, q_words=["çaydanlık"]
      → "çaydanlık" title'da yok → 0.0  ✓ (doğru negatif sinyali)

      "karaca çaydanlık" vs title="karaca çaydanlık takımı" (brand="karaca", ovlp=1.0)
      → q_words=["çaydanlık"] → title'da var → 1.0  ✓
    """
    b_toks = brand_tok_set(brand) if btok_ovlp > 0.3 else set()

    q_words = [w for w in q.split()
               if w not in GENDER_WORDS
               and w not in AGE_WORDS
               and w not in RENKLER
               and w not in MATERIAL_MAP
               and w not in TR_STOP_FUNC
               and w not in b_toks
               and len(w) > 1]

    if not q_words: return q_cov(q, t)

    t_words = set(t.split())
    t_stems = {stem(w) for w in t_words}
    hits = sum(1 for w in q_words if _word_match(w, t_words, t_stems))
    return hits / len(q_words)

def all_q_in_title(q, t):
    """Tüm query kelimeleri title'da mevcut mu? (strict binary)"""
    q_words = set(q.split())
    t_words = set(t.split())
    return 1.0 if q_words.issubset(t_words) else 0.0

def brand_weight(q, brand):
    """Query'nin kaçta kaçı marka tokenları?"""
    b_toks = brand_tok_set(brand)
    if not b_toks: return 0.0
    q_toks = q.split()
    if not q_toks: return 0.0
    return sum(1 for t in q_toks if t in b_toks) / len(q_toks)

# ─────────────────────────────────────────────
# RENK + ATTRIBUTE (v22'den)
# ─────────────────────────────────────────────
COLOR_NORM = {"gold":"altın","silver":"gümüş","rose":"pembe","krem":"bej","kiremit":"kırmızı"}
COLOR_FAMILY = {
    "antrasit":"gri","füme":"gri","platin":"gri","koyu gri":"gri","açık gri":"gri",
    "kurşun":"gri","gri melanj":"gri","lacivert":"mavi","indigo":"mavi","petrol":"mavi",
    "saks mavi":"mavi","bebe mavisi":"mavi","açık mavi":"mavi",
    "bordo":"kırmızı","kiremit":"kırmızı","altın":"sarı","gold":"sarı",
    "gümüş":"metalik","silver":"metalik","krem":"bej","kırık beyaz":"bej","ekru":"bej",
}

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
    b_toks = brand_tok_set(brand)
    q_toks = set(re.sub(r'[^a-z0-9çğıöşü\s]', ' ', q).split())
    return len(b_toks & q_toks) / len(b_toks) if b_toks else 0.0

MATERIAL_MAP_FULL = {
    "pamuk":"pamuk","pamuklu":"pamuk","deri":"deri","hakiki":"deri",
    "polyester":"polyester","yün":"yün","keten":"keten","naylon":"naylon",
    "çelik":"çelik","plastik":"plastik","ahşap":"ahşap",
}

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

# ─────────────────────────────────────────────────────────────────────
# PHASE A: VERİ YÜKLEMESİ + ENHANCED NEGATİFLER
# ─────────────────────────────────────────────────────────────────────
print("=" * 65, flush=True)
print("[A] VERİ YÜKLENİYOR...", flush=True)
t0 = time.time()

items       = pd.read_csv(DATA / "items.csv")
terms       = pd.read_csv(DATA / "terms.csv")
train_pairs = pd.read_csv(DATA / "training_pairs.csv")
sub_pairs   = pd.read_csv(DATA / "submission_pairs.csv")

# LEAK KONTROL: sub_pairs hiçbir zaman negatif kaynağı olarak kullanılmıyor
# Sadece Phase E'de test inference için kullanılıyor

for col in ["title","category","brand","gender","age_group","attributes"]:
    items[col] = items[col].fillna(UNKNOWN).apply(trl)
terms["query"] = terms["query"].fillna("").apply(trl)

# Kategori hiyerarşisi
items["main_category"] = items["category"].str.split("/").str[0].fillna(UNKNOWN).apply(trl)
items["sub_category"]  = items["category"].str.split("/").str[1].fillna(UNKNOWN).apply(trl)

item_ids_arr   = items["item_id"].values
item_mains_arr = items["main_category"].values
item_subs_arr  = items["sub_category"].values

iid_to_str = {str(row.item_id): row for row in items.itertuples()}
tid_to_q   = {row.term_id: row.query for row in terms.itertuples()}

print(f"  {len(items):,} ürün | {len(train_pairs):,} pozitif | {len(sub_pairs):,} test | {time.time()-t0:.1f}s", flush=True)

# LEAK KONTROL: Eğitim ve test term_id'leri disjoint mu?
train_tids_set = set(train_pairs["term_id"].unique())
sub_tids_set   = set(sub_pairs["term_id"].unique())
overlap        = len(train_tids_set & sub_tids_set)
print(f"  Train-test term_id örtüşmesi: {overlap} (0 olmalı)", flush=True)
assert overlap == 0, f"LEAK! {overlap} ortak term_id var!"


def build_group_idx(df, cols):
    df_reset = df.reset_index(drop=True)
    if len(cols) == 1:
        return {k: g.index.values for k, g in df_reset.groupby(cols[0], sort=False)}
    return {k: g.index.values for k, g in df_reset.groupby(cols, sort=False)}


print("\n[A2] Enhanced negatives (NEG_PER_POS=5)...", flush=True)
by_main   = build_group_idx(items, ["main_category"])
by_gender = build_group_idx(items, ["gender"])
by_mg     = build_group_idx(items, ["main_category","gender"])
by_age    = build_group_idx(items, ["age_group"])
by_ma     = build_group_idx(items, ["main_category","age_group"])
by_brand  = build_group_idx(items, ["brand"])

# LEAK KONTROL: positive_keys ile bilinen pozitifler negatife düşmüyor
positive_keys = set(
    train_pairs["term_id"].astype(str) + "\t" + train_pairs["item_id"].astype(str)
)
used_keys: set = set()
rng = np.random.default_rng(42)


def sample_pool(pool, term_id, pos_iid, max_tries=40):
    if pool is None or len(pool) == 0: return None
    for _ in range(max_tries):
        idx = int(pool[rng.integers(0, len(pool))])
        iid = str(item_ids_arr[idx])
        k   = term_id + "\t" + iid
        if iid != pos_iid and k not in positive_keys and k not in used_keys:
            return iid
    return None


def sample_diff_main(cur_main, term_id, pos_iid, max_tries=80):
    n = len(item_ids_arr)
    for _ in range(max_tries):
        idx = int(rng.integers(0, n))
        if item_mains_arr[idx] == cur_main: continue
        iid = str(item_ids_arr[idx])
        k   = term_id + "\t" + iid
        if iid != pos_iid and k not in positive_keys and k not in used_keys:
            return iid
    return None


def sample_same_brand_diff_main(brand, main, term_id, pos_iid, max_tries=60):
    """[v22] Aynı marka, farklı ana kategori → hard negative"""
    if not brand or brand == UNKNOWN: return None
    pool = by_brand.get(brand)
    if pool is None or len(pool) == 0: return None
    for _ in range(max_tries):
        idx = int(pool[rng.integers(0, len(pool))])
        if item_mains_arr[idx] == main: continue  # ana kat aynıysa atla
        iid = str(item_ids_arr[idx])
        k   = term_id + "\t" + iid
        if iid != pos_iid and k not in positive_keys and k not in used_keys:
            return iid
    return None


def sample_same_brand_diff_sub(brand, main, sub, term_id, pos_iid, max_tries=80):
    """[YENİ v23] Aynı marka + aynı ana kat, farklı alt kat → daha sert hard negative

    Örnek: "karaca çaydanlık" sorgusu için pozitif "karaca çaydanlık" (Ev&Yaşam/Çay Setleri)
    Bu fonksiyon: "karaca tava" (Ev&Yaşam/Pişirme Araçları) → negatif
    → Marka + ana kategori eşleşir ama ürün tipi farklı: en konfüze edici negatif
    """
    if not brand or brand == UNKNOWN: return None
    pool = by_brand.get(brand)
    if pool is None or len(pool) == 0: return None
    for _ in range(max_tries):
        idx = int(pool[rng.integers(0, len(pool))])
        if item_mains_arr[idx] != main: continue  # ana kat eşleşmeli
        if item_subs_arr[idx] == sub: continue    # ama alt kat farklı olmalı
        iid = str(item_ids_arr[idx])
        k   = term_id + "\t" + iid
        if iid != pos_iid and k not in positive_keys and k not in used_keys:
            return iid
    return None


pos_with_info = train_pairs.merge(terms, on="term_id", how="left")
pos_with_info = pos_with_info.merge(
    items[["item_id","main_category","sub_category","gender","age_group","brand"]],
    on="item_id", how="left"
)
for col in ["main_category","sub_category","gender","age_group","brand","query"]:
    pos_with_info[col] = pos_with_info[col].fillna(UNKNOWN).apply(trl)

neg_tids, neg_iids, neg_src = [], [], []
NEG_PER_POS = 5

for row in pos_with_info.itertuples(index=False):
    tid    = str(row.term_id)
    pos_id = str(row.item_id)
    main   = str(row.main_category)
    sub    = str(row.sub_category)
    query  = str(row.query)
    brand  = str(row.brand)
    selected = []

    # ── 1. SAME BRAND, DIFFERENT MAIN CAT (v22의 핵심 fix)
    if len(selected) < NEG_PER_POS:
        iid = sample_same_brand_diff_main(brand, main, tid, pos_id)
        if iid: selected.append((iid, "same_brand_diff_main"))

    # ── 2. SAME BRAND, SAME MAIN, DIFFERENT SUB (YENİ — en sert hard negative)
    if len(selected) < NEG_PER_POS:
        iid = sample_same_brand_diff_sub(brand, main, sub, tid, pos_id)
        if iid: selected.append((iid, "same_brand_diff_sub"))

    # ── 3. Gender conflict
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

    # ── 4. Age conflict
    if len(selected) < NEG_PER_POS:
        if re.search(r'\b(bebek|çocuk)\b', query):
            pool = by_ma.get((main, "yetişkin"))
            if pool is None: pool = by_age.get("yetişkin")
            iid = sample_pool(pool, tid, pos_id)
            if iid: selected.append((iid, "age_conflict"))

    # ── 5. Same main category (fallback 1)
    if len(selected) < NEG_PER_POS:
        iid = sample_pool(by_main.get(main), tid, pos_id)
        if iid: selected.append((iid, "same_main_category"))

    # ── 6. Different category (fallback 2 — fill remaining slots)
    while len(selected) < NEG_PER_POS:
        iid = sample_diff_main(main, tid, pos_id)
        if iid: selected.append((iid, "different_main_category"))
        else:   break

    for iid, src in selected[:NEG_PER_POS]:
        k = tid + "\t" + iid
        if k in used_keys or k in positive_keys: continue
        used_keys.add(k)
        neg_tids.append(tid); neg_iids.append(iid); neg_src.append(src)

negatives = pd.DataFrame({"term_id": neg_tids, "item_id": neg_iids, "label": 0})
train_pairs["label"] = 1

src_counts = Counter(neg_src)
print(f"  Negatif kaynak: {dict(src_counts)}", flush=True)

train_df = pd.concat([
    train_pairs[["term_id","item_id","label"]],
    negatives[["term_id","item_id","label"]]
], ignore_index=True)
train_df = train_df.sort_values("term_id").reset_index(drop=True)
train_df["term_id"] = train_df["term_id"].astype(str)
train_df["item_id"] = train_df["item_id"].astype(str)

print(f"  Train: {len(train_df):,} | pos={train_df.label.sum():,} neg={(train_df.label==0).sum():,}", flush=True)

# ─────────────────────────────────────────────────────────────────────
# PHASE B: BERT HARD NEGATIVE RETRAINING
# YENİ: same_brand_diff_main + same_brand_diff_sub negatiflerini görür
# ─────────────────────────────────────────────────────────────────────
print("\n[B] BERT RETRAINING (hard negatives)...", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}", flush=True)

BERT_V23.mkdir(parents=True, exist_ok=True)
v23_train_bert_path = CACHE_DIR / "bert_scores_v23_train.npy"
v23_test_bert_path  = SUBM / "bert_scores_v23_test.npy"

bert_needs_finetune = not (BERT_V23 / "config.json").exists()
bert_needs_train    = (not v23_train_bert_path.exists() or
                       len(np.load(str(v23_train_bert_path))) != len(train_df))
bert_needs_test     = not v23_test_bert_path.exists()


def build_bert_item_text(item_row):
    if item_row is None: return ""
    d    = parse_attrs(getattr(item_row, "attributes", "") or "")
    renk = d.get("renk", "")
    mat  = d.get("materyal bileşeni", d.get("materyal", ""))
    parts = []
    t = getattr(item_row, "title", "") or ""
    b = getattr(item_row, "brand", "") or ""
    c = getattr(item_row, "category", "") or ""
    if t and t != UNKNOWN: parts.append(t)
    if b and b != UNKNOWN: parts.append(b)
    cat_main = c.split("/")[0] if c else ""
    if cat_main and cat_main != UNKNOWN: parts.append(cat_main)
    if renk and renk not in (UNKNOWN, ""): parts.append(f"renk:{renk}")
    if mat  and mat  not in (UNKNOWN, ""): parts.append(f"mat:{mat[:25]}")
    return " | ".join(parts[:5])


class PairDataset(Dataset):
    def __init__(self, tids, iids, labels, tokenizer, max_len=128):
        self.queries = [tid_to_q.get(int(t) if str(t).isdigit() else t, "") for t in tids]
        self.items   = [build_bert_item_text(iid_to_str.get(str(i))) for i in iids]
        self.labels  = labels
        self.tok     = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tok(self.queries[idx], self.items[idx],
                       max_length=self.max_len, truncation=True,
                       padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(self.labels[idx], dtype=torch.float)


def bert_inference(tids, iids, model, tokenizer, batch=256, desc=""):
    ds = PairDataset(tids, iids, [0]*len(tids), tokenizer)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)
    scores = []
    model.eval()
    n = len(tids)
    with torch.no_grad():
        for i, (batch_enc, _) in enumerate(dl):
            batch_enc = {k: v.to(device) for k, v in batch_enc.items()}
            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                out = model(**batch_enc).logits.squeeze(-1)
            scores.extend(torch.sigmoid(out).float().cpu().tolist())
            if (i+1) % 500 == 0:
                print(f"    {desc} {(i+1)*batch:,}/{n:,}", flush=True)
    return np.array(scores, dtype=np.float32)


if bert_needs_finetune:
    print("  BERT fine-tune başlıyor (hard negatives ile)...", flush=True)
    tokenizer  = AutoTokenizer.from_pretrained(BERT_BASE)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_BASE, num_labels=1).to(device)

    # 500K sample (v21'de 300K idi) — daha fazla hard negative görmesi için
    BERT_SAMPLE = min(500_000, len(train_df))
    EPOCHS, BATCH, LR = 5, 32, 2e-5
    bert_train = train_df.sample(BERT_SAMPLE, random_state=42)

    # Hard negative dağılımı kontrol: kaçta kaçı same_brand_diff_sub?
    train_src_check = negatives[negatives["item_id"].isin(bert_train["item_id"].astype(str))]
    print(f"  BERT train sample: {BERT_SAMPLE:,} | pos~{bert_train.label.sum():,}", flush=True)

    ds_train = PairDataset(bert_train["term_id"].tolist(), bert_train["item_id"].tolist(),
                           bert_train["label"].tolist(), tokenizer)
    dl_train = DataLoader(ds_train, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)

    optimizer    = torch.optim.AdamW(bert_model.parameters(), lr=LR, weight_decay=0.01)
    total_steps  = len(dl_train) * EPOCHS
    scheduler    = get_linear_schedule_with_warmup(optimizer, int(0.1*total_steps), total_steps)
    scaler       = GradScaler() if torch.cuda.is_available() else None
    criterion    = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        bert_model.train()
        total_loss = 0
        for step, (enc, lbl) in enumerate(dl_train):
            enc = {k: v.to(device) for k, v in enc.items()}
            lbl = lbl.to(device)
            optimizer.zero_grad()
            if scaler:
                with autocast("cuda"):
                    loss = criterion(bert_model(**enc).logits.squeeze(-1), lbl)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(bert_model(**enc).logits.squeeze(-1), lbl)
                loss.backward()
                nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if step % 300 == 0:
                print(f"  Epoch {epoch+1} step {step}/{len(dl_train)} loss={total_loss/(step+1):.4f}", flush=True)

    bert_model.save_pretrained(str(BERT_V23))
    tokenizer.save_pretrained(str(BERT_V23))
    print(f"  BERT v23 saved → {BERT_V23}", flush=True)

else:
    print(f"  BERT v23 cache'de var → yükleniyor", flush=True)
    tokenizer  = AutoTokenizer.from_pretrained(str(BERT_V23))
    bert_model = AutoModelForSequenceClassification.from_pretrained(str(BERT_V23)).to(device)

if bert_needs_train:
    print(f"  Train BERT inference ({len(train_df):,} çift)...", flush=True)
    t1 = time.time()
    train_bert = bert_inference(train_df["term_id"].tolist(), train_df["item_id"].tolist(),
                                bert_model, tokenizer, desc="train")
    np.save(str(v23_train_bert_path), train_bert)
    print(f"  → {v23_train_bert_path.name} | {(time.time()-t1)/60:.1f} dk", flush=True)
else:
    train_bert = np.load(str(v23_train_bert_path))
    print(f"  Train BERT cache'den: {len(train_bert):,}", flush=True)

if bert_needs_test:
    print(f"  Test BERT inference ({len(sub_pairs):,} çift)...", flush=True)
    t1 = time.time()
    test_bert = bert_inference(sub_pairs["term_id"].tolist(), sub_pairs["item_id"].tolist(),
                               bert_model, tokenizer, desc="test")
    np.save(str(v23_test_bert_path), test_bert)
    print(f"  → {v23_test_bert_path.name} | {(time.time()-t1)/60:.1f} dk", flush=True)
else:
    test_bert = np.load(str(v23_test_bert_path))
    print(f"  Test BERT cache'den: {len(test_bert):,}", flush=True)

del bert_model; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────
# PHASE C: FEATURE ENGINEERING (32 v22 + 3 yeni = 35 özellik)
# ─────────────────────────────────────────────────────────────────────
print("\n[C] FEATURE ENGINEERING...", flush=True)


def build_features(df_in, tfidf_vect, bert_arr, fit=False, all_texts=None):
    df = df_in.copy()
    df["term_id"] = df["term_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    terms_l = terms.copy(); terms_l["term_id"] = terms_l["term_id"].astype(str)
    items_l = items.copy(); items_l["item_id"] = items_l["item_id"].astype(str)

    df = df.merge(terms_l, on="term_id", how="left").merge(items_l, on="item_id", how="left")
    for col in ["query","title","brand","category","gender","age_group","attributes","main_category"]:
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

    def tfidf_cos(ql, tl, chunk=50_000):
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

    # Önce brand overlap hesapla (product_type_cover için gerekli)
    btok_ovlp = [brand_tok_overlap(q, b) for q, b in zip(qs, brs)]

    f = pd.DataFrame()
    # ── v21/v22'den gelen özellikler
    f["fuzz_partial"]    = [rfuzz.partial_ratio(q,t)/100    for q,t in zip(qs,ts)]
    f["fuzz_set"]        = [rfuzz.token_set_ratio(q,t)/100  for q,t in zip(qs,ts)]
    f["fuzz_sort"]       = [rfuzz.token_sort_ratio(q,t)/100 for q,t in zip(qs,ts)]
    f["fuzz_basic"]      = [rfuzz.ratio(q,t)/100            for q,t in zip(qs,ts)]
    f["jaccard"]         = [jac(q,t)                        for q,t in zip(qs,ts)]
    f["tfidf_cos"]       = tfidf_cos(qs, ts)
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
    f["brand_tok_ovlp"]  = btok_ovlp
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
    # ── YENİ v23 ÖZELLİKLERİ
    f["product_type_cover"] = [product_type_cover(q,t,b,bo)
                                for q,t,b,bo in zip(qs,ts,brs,btok_ovlp)]
    f["all_q_in_title"]     = [all_q_in_title(q,t)  for q,t in zip(qs,ts)]
    f["brand_weight"]       = [brand_weight(q,b)     for q,b in zip(qs,brs)]

    return f, df


t1 = time.time()
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=60_000, sublinear_tf=True, min_df=2)

FEATS = [
    # v21/v22 özellikleri
    "fuzz_partial","fuzz_set","fuzz_sort","fuzz_basic",
    "jaccard","tfidf_cos","q_cov_title","t_cov_query",
    "cat_overlap","exact_in_title","token_overlap","age_in_q",
    "stem_jaccard","stem_cat_jac","gender_cross",
    "first_tok_title","first_tok_brand","q_len","t_len",
    "bert_score","ana_kategori",
    "color_typed","has_q_color","attr_has_renk","brand_tok_ovlp",
    "material_match","kol_boyu_match","attr_jac_fixed","attr_q_cov",
    "head_in_title","weighted_q_cov","head_in_cat",
    # YENİ v23
    "product_type_cover","all_q_in_title","brand_weight",
]

X_tr, df_tr = build_features(
    train_df, tfidf, train_bert, fit=True,
    all_texts=items["title"].tolist() + terms["query"].tolist()
)
y    = train_df["label"].values
tids = train_df["term_id"].values
print(f"  Train features: {X_tr[FEATS].shape} | {(time.time()-t1):.1f}s", flush=True)

# ─────────────────────────────────────────────────────────────────────
# PHASE D: BINARY LGBMClassifier 5-FOLD
# ─────────────────────────────────────────────────────────────────────
print("\n[D] BINARY LGBM 5-FOLD...", flush=True)

oof    = np.zeros(len(train_df), dtype=np.float32)
models = []
gkf    = GroupKFold(n_splits=5)

for fold, (tri, vli) in enumerate(gkf.split(X_tr, y, tids)):
    X_train_f = X_tr[FEATS].iloc[tri]
    y_train_f = y[tri]
    X_val_f   = X_tr[FEATS].iloc[vli]
    y_val_f   = y[vli]

    model = LGBMClassifier(
        n_estimators=2000,
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
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
    )

    val_proba = model.predict_proba(X_val_f)[:, 1]
    oof[vli]  = val_proba.astype(np.float32)
    models.append(model)

    best_f = 0.0
    for thr in np.arange(0.20, 0.90, 0.02):
        fs = f1_score(y_val_f, (val_proba > thr).astype(int), average="macro")
        if fs > best_f: best_f = fs
    print(f"  Fold {fold+1} | iter={model.best_iteration_} | val F1≈{best_f:.4f}", flush=True)

# OOF threshold
best_thr, best_f1 = 0.5, 0.0
for thr in np.arange(0.20, 0.90, 0.005):
    s = f1_score(y, (oof > thr).astype(int), average="macro")
    if s > best_f1: best_f1, best_thr = s, thr
print(f"\n  OOF F1={best_f1:.5f} | thr={best_thr:.4f}", flush=True)

# Feature önemleri
fi = pd.Series(
    sum(m.feature_importances_ for m in models) / len(models),
    index=FEATS
).sort_values(ascending=False)
print("\n  Feature Önemleri (top 20):", flush=True)
for feat, imp in fi.head(20).items():
    print(f"    {feat:25s}: {imp:6.0f}", flush=True)

# ─────────────────────────────────────────────────────────────────────
# PHASE E: TEST INFERENCE + SUBMISSION
# ─────────────────────────────────────────────────────────────────────
print("\n[E] TEST INFERENCE + SUBMISSION...", flush=True)
X_te, _ = build_features(sub_pairs, tfidf, test_bert)
test_proba = sum(m.predict_proba(X_te[FEATS])[:, 1] for m in models) / len(models)

# OOF quantile → test threshold
oof_q     = np.mean(oof < best_thr)
final_thr = np.quantile(test_proba, oof_q)
final     = (test_proba > final_thr).astype(int)
pos_n     = final.sum()

print(f"  Pozitif={pos_n:,} ({100*pos_n/len(sub_pairs):.1f}%) | thr={final_thr:.4f}", flush=True)

out_path = SUBM / "submission_v23_combined.csv"
pd.DataFrame({"id": sub_pairs["id"], "prediction": final}).to_csv(str(out_path), index=False)

print(f"\n{'='*65}", flush=True)
print(f"TAMAMLANDI — v23 Combined", flush=True)
print(f"  OOF F1      : {best_f1:.5f}", flush=True)
print(f"  Pozitif     : {pos_n:,} ({100*pos_n/len(sub_pairs):.1f}%)", flush=True)
print(f"  Toplam süre : {(time.time()-t0)/60:.1f} dk", flush=True)
print(f"  Dosya       : {out_path}", flush=True)
print(f"{'='*65}", flush=True)
