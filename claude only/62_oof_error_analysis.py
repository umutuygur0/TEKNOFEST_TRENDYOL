"""
59_v36a_quicktest.py — v36a: GERÇEK test-dağılımından LLM-doğrulamalı veri ekleme (hızlı test)
====================================================================================================
Hedef: 0.880 (v35a, doğrulanmış) -> mümkün olduğunca yükseğe (liderlik tablosu 0.95-0.97'de).

v35a sonucu: gerçek skor 0.880 (+0.002), OOF pratik olarak aynı (0.96996 vs 0.96999) —
eğitim-sorgusu kaynaklı LLM temizliğinin marjinal katkısı doğrulandı. Yeni bulgu:
`submission_pairs.csv` incelendiğinde 32.185 test sorgusu için ortalama ~104 aday/sorgu
olduğu görüldü — bunlar Trendyol'un kendi retrieval sisteminin GERÇEK aday havuzu.
Eğitim negatiflerimiz (kural-tabanlı) bu gerçek zorluk dağılımını hiç temsil etmiyor.

v36'nın FİKRİ: v35 ile AYNI LLM doğrulama yöntemi, ama adaylar bu sefer embedding-benzerliğiyle
tüm katalogdan madenciliyle DEĞİL, doğrudan `submission_pairs.csv`'deki GERÇEK aday havuzundan
örnekleniyor (57_v36_test_candidate_gen.py + 58_merge_test_llm_labels.py). Sonuç:
claude only/57_llm_labels_test/merged_test_llm_labels.csv — 13.750 doğrulanmış çift,
%79.2 llm_recovered_positive_test, %20.8 llm_verified_negative_test (v35'in eğitim-sorgusu
kaynaklı %10.3'ünden çok daha yüksek negatif oranı — beklenen, çünkü bunlar gerçek retrieval
sisteminin ürettiği daha zor/daha alakasız adaylar).

UYUMLULUK: Test sorgularından gelen LLM etiketleri SADECE eğitim sinyali (distillation) —
test satırları için LLM kararı ASLA doğrudan final tahmine yazılmıyor. Bu pseudo-etiketli
satırlar train_df'e eklenip ELECTRA+LGBM ile yeniden eğitiliyor; final tahmin yine bu
modelden geliyor.

v36a'nın DEĞİŞİKLİĞİ: Phase A (negatif üretimi) v34 ile BİREBİR AYNI kalıyor (aynı seed=42,
aynı 7 kaynak) — bu orijinal train_df'i oluşturuyor. Sonra HEM v35'in `merged_llm_labels.csv`
(eğitim-sorgusu kaynaklı) HEM v36'nın `merged_test_llm_labels.csv` (test-dağılımı kaynaklı)
satırları train_df'e EKLENİYOR. BERT skorları için: orijinal satırlar cache'den
(bert_scores_v34_train.npy) okunuyor, SADECE yeni LLM satırları (v35+v36 toplamı) için
mevcut (yeniden eğitilmeyen) bert_v34 ELECTRA modeliyle hızlı inference yapılıyor — tam
BERT fine-tune YOK, bu yüzden "hızlı test" (quick test).

KESİN KAÇINILACAKLAR (altı kez doğrulandı, değişmedi):
  ✗ Query-grubu-göreli özellik (v25/v26: 0.84->0.68 çöküşü)
  ✗ DOĞRULANMAMIŞ embedding/TF-IDF benzerliğiyle hard-negative mining (v24, v28/v29)
  ✗ BERT'i similarity-mined (doğrulanmamış) negatiflerle eğitmek (v30)
  ✗ Final inference adımında paid/LLM model kullanmak (yasak) — LLM SADECE eğitim
    verisi üretiminde (train-side VE test-side pseudo-label) kullanıldı, nihai tahmin
    hâlâ ELECTRA+LGBM.

DEĞİŞMEYENLER: v34'ün TAMAMI — color_conflict dahil 7 negatif kaynağı, stratified BERT
  sample mantığı (ilk fine-tune için), 41 feature'ın tamamı, LGBM ayarları
  (n_estimators=4000, early_stop=150), threshold yöntemi (direct), NEG_PER_POS=5, seed=42.

SAĞLIK KONTROLÜ (script sonunda otomatik basılır):
  - OOF F1 v34'ün 0.96999'una yakın olmalı (LLM verisiyle biraz değişebilir, aşırı sıçrama şüpheli).
  - Hiçbir feature tek başına baskın olmamalı.
  - LLM kaynak dağılımı (her iki kaynak için ayrı ayrı) makul olmalı.
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

BASE       = Path(__file__).resolve().parents[1]
DATA       = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
SUBM       = BASE / "claude only" / "submissions"
MODELS_DIR = BASE / "claude only" / "models"
EMB_CACHE  = BASE / "claude only" / "emb_cache"
BERT_V34   = MODELS_DIR / "bert_v34"          # YENİ: bert_v23/bert_v33'e dokunmuyoruz
BERT_BASE  = "dbmdz/electra-base-turkish-cased-discriminator"
CACHE_DIR  = BASE / "claude only"
SUBM.mkdir(parents=True, exist_ok=True)

LOWER = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(s): return str(s).translate(LOWER).lower().strip()

UNKNOWN = "unknown"

# ─────────────────────────────────────────────
# TÜRKÇE STEM (v23 ile aynı)
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
# v23: PRODUCT TYPE COVER
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
    q_words = set(q.split())
    t_words = set(t.split())
    return 1.0 if q_words.issubset(t_words) else 0.0

def brand_weight(q, brand):
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
# age_cross (v32'den)
# ─────────────────────────────────────────────────────────────────────
def age_cross(q, age_group):
    q_baby  = bool(re.search(r'\b(bebek|bebekler)\b', q))
    q_child = bool(re.search(r'\b(çocuk|çocuklar)\b', q))
    q_adult = bool(re.search(r'\byetişkin\b', q))
    if not (q_baby or q_child or q_adult): return 0.0
    if age_group == UNKNOWN or not age_group: return 0.0
    a_baby  = age_group in ("bebek", "bebek & çocuk")
    a_child = age_group in ("çocuk", "bebek & çocuk", "genç")
    a_adult = age_group == "yetişkin"
    if q_baby  and a_baby:  return 1.0
    if q_child and a_child: return 1.0
    if q_adult and a_adult: return 1.0
    if q_baby  and a_adult: return -1.0
    if q_adult and a_baby:  return -1.0
    if q_child and a_adult: return -1.0
    if q_adult and a_child: return -1.0
    return 0.0

# ─────────────────────────────────────────────────────────────────────
# material_mismatch (v32'den)
# ─────────────────────────────────────────────────────────────────────
def material_mismatch(q, attr_mat):
    q_mat = None
    for tok in q.split():
        m = MATERIAL_MAP_FULL.get(tok)
        if m: q_mat = m; break
    if q_mat is None: return 0.0
    if not attr_mat: return 0.0
    if q_mat in attr_mat: return 1.0
    return -1.0

# ─────────────────────────────────────────────────────────────────────
# N-GRAM YARDIMCI (v31'den)
# ─────────────────────────────────────────────────────────────────────
def query_ngrams(q, maxn=3):
    toks = q.split()
    grams = []
    for n in range(1, maxn + 1):
        for i in range(len(toks) - n + 1):
            grams.append(" ".join(toks[i:i + n]))
    return grams

# ─────────────────────────────────────────────────────────────────────
# PHASE A: VERİ YÜKLEMESİ + ENHANCED NEGATİFLER (+ YENİ color_conflict)
# ─────────────────────────────────────────────────────────────────────
print("=" * 65, flush=True)
print("[A] VERİ YÜKLENİYOR...", flush=True)
t0 = time.time()

items       = pd.read_csv(DATA / "items.csv")
terms       = pd.read_csv(DATA / "terms.csv")
train_pairs = pd.read_csv(DATA / "training_pairs.csv")
sub_pairs   = pd.read_csv(DATA / "submission_pairs.csv")

for col in ["title","category","brand","gender","age_group","attributes"]:
    items[col] = items[col].fillna(UNKNOWN).apply(trl)
terms["query"] = terms["query"].fillna("").apply(trl)

items["main_category"] = items["category"].str.split("/").str[0].fillna(UNKNOWN).apply(trl)
items["sub_category"]  = items["category"].str.split("/").str[1].fillna(UNKNOWN).apply(trl)

# YENİ v33: her item için renk ailesini önceden çıkar (color_conflict negatif üretimi için)
def extract_item_color_family(attrs_str):
    d = parse_attrs(attrs_str)
    renk = d.get("renk", "")
    if not renk: return ""
    return color_family(norm_color(renk))

items["color_family"] = items["attributes"].apply(extract_item_color_family)

item_ids_arr    = items["item_id"].values
item_mains_arr  = items["main_category"].values
item_subs_arr   = items["sub_category"].values
item_colors_arr = items["color_family"].values

iid_to_str = {str(row.item_id): row for row in items.itertuples()}
tid_to_q   = {row.term_id: row.query for row in terms.itertuples()}

n_with_color = int((items["color_family"] != "").sum())
print(f"  {len(items):,} ürün | {len(train_pairs):,} pozitif | {len(sub_pairs):,} test | "
      f"renkli ürün: {n_with_color:,} ({100*n_with_color/len(items):.1f}%) | {time.time()-t0:.1f}s", flush=True)

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


print("\n[A2] Enhanced negatives (NEG_PER_POS=5, + YENİ color_conflict)...", flush=True)
by_main   = build_group_idx(items, ["main_category"])
by_gender = build_group_idx(items, ["gender"])
by_mg     = build_group_idx(items, ["main_category","gender"])
by_age    = build_group_idx(items, ["age_group"])
by_ma     = build_group_idx(items, ["main_category","age_group"])
by_brand  = build_group_idx(items, ["brand"])

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
    if not brand or brand == UNKNOWN: return None
    pool = by_brand.get(brand)
    if pool is None or len(pool) == 0: return None
    for _ in range(max_tries):
        idx = int(pool[rng.integers(0, len(pool))])
        if item_mains_arr[idx] == main: continue
        iid = str(item_ids_arr[idx])
        k   = term_id + "\t" + iid
        if iid != pos_iid and k not in positive_keys and k not in used_keys:
            return iid
    return None


def sample_same_brand_diff_sub(brand, main, sub, term_id, pos_iid, max_tries=80):
    if not brand or brand == UNKNOWN: return None
    pool = by_brand.get(brand)
    if pool is None or len(pool) == 0: return None
    for _ in range(max_tries):
        idx = int(pool[rng.integers(0, len(pool))])
        if item_mains_arr[idx] != main: continue
        if item_subs_arr[idx] == sub: continue
        iid = str(item_ids_arr[idx])
        k   = term_id + "\t" + iid
        if iid != pos_iid and k not in positive_keys and k not in used_keys:
            return iid
    return None


def sample_color_conflict(q_color, main, term_id, pos_iid, max_tries=60):
    """YENİ v33: aynı ana kategori + KESİN FARKLI renk ailesi → negatif.
    gender_conflict/age_conflict ile birebir aynı desen: %100 kural tabanlı,
    benzerlik-madenli DEĞİL (items.attributes'taki renk alanına doğrudan bakılıyor)."""
    if q_color is None: return None
    target_family = color_family(q_color)
    pool = by_main.get(main)
    if pool is None or len(pool) == 0: return None
    for _ in range(max_tries):
        idx = int(pool[rng.integers(0, len(pool))])
        item_cf = item_colors_arr[idx]
        if not item_cf or item_cf == target_family: continue  # renksiz veya aynı aile → atla
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

    if len(selected) < NEG_PER_POS:
        iid = sample_same_brand_diff_main(brand, main, tid, pos_id)
        if iid: selected.append((iid, "same_brand_diff_main"))

    if len(selected) < NEG_PER_POS:
        iid = sample_same_brand_diff_sub(brand, main, sub, tid, pos_id)
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

    # YENİ v33: color_conflict
    if len(selected) < NEG_PER_POS:
        q_color = get_query_color(query)
        if q_color is not None:
            iid = sample_color_conflict(q_color, main, tid, pos_id)
            if iid: selected.append((iid, "color_conflict"))

    if len(selected) < NEG_PER_POS:
        iid = sample_pool(by_main.get(main), tid, pos_id)
        if iid: selected.append((iid, "same_main_category"))

    while len(selected) < NEG_PER_POS:
        iid = sample_diff_main(main, tid, pos_id)
        if iid: selected.append((iid, "different_main_category"))
        else:   break

    for iid, src in selected[:NEG_PER_POS]:
        k = tid + "\t" + iid
        if k in used_keys or k in positive_keys: continue
        used_keys.add(k)
        neg_tids.append(tid); neg_iids.append(iid); neg_src.append(src)

negatives = pd.DataFrame({"term_id": neg_tids, "item_id": neg_iids, "label": 0, "src": neg_src})
train_pairs["label"] = 1

src_counts = Counter(neg_src)
print(f"  Negatif kaynak: {dict(src_counts)}", flush=True)
print(f"  (v23/v31/v32'nin diğer 6 kaynağıyla karşılaştır — yakın olmalı; color_conflict "
      f"eklendiği için same_main/diff_main biraz azalmış olabilir, bu BEKLENEN ve ZARARSIZ)", flush=True)

train_df = pd.concat([
    train_pairs[["term_id","item_id","label"]].assign(src="positive"),
    negatives[["term_id","item_id","label","src"]]
], ignore_index=True)
train_df = train_df.sort_values("term_id").reset_index(drop=True)
train_df["term_id"] = train_df["term_id"].astype(str)
train_df["item_id"] = train_df["item_id"].astype(str)
sub_pairs["term_id"] = sub_pairs["term_id"].astype(str)
sub_pairs["item_id"] = sub_pairs["item_id"].astype(str)

print(f"  Train: {len(train_df):,} | pos={train_df.label.sum():,} neg={(train_df.label==0).sum():,}", flush=True)

# ─────────────────────────────────────────────────────────────────────
# 62: OOF HATA ANALİZİ — v34 SAF TABANI (LLM verisi YOK, GPU YOK — arkada v36b çalışıyor)
# LLM augmentation atlanıyor: bu analiz gerçek 0.878 skoruna en yakın konfigürasyonu
# (v34 birebir aynı) inceliyor. BERT skorları cache'den okunuyor, hiç GPU/inference yok.
# ─────────────────────────────────────────────────────────────────────
print("\n[A2] ATLANDI — v34 ile birebir aynı devam ediliyor (LLM verisi eklenmiyor)", flush=True)
original_train_df = train_df.copy()
n_original = len(original_train_df)
llm_df = pd.DataFrame(columns=["term_id", "item_id", "label", "src"])

# ─────────────────────────────────────────────────────────────────────
# PHASE B: BERT — SADECE cache'den oku, GPU/inference YOK (v36b arka planda GPU kullanıyor)
# ─────────────────────────────────────────────────────────────────────
print("\n[B] BERT v34 skorları cache'den okunuyor (GPU kullanılmıyor)...", flush=True)
device = torch.device("cpu")
v34_train_bert_path = CACHE_DIR / "bert_scores_v34_train.npy"
train_bert = np.load(str(v34_train_bert_path))
assert len(train_bert) == len(train_df), \
    f"Cache boyutu ({len(train_bert)}) train_df ({len(train_df)}) ile eşleşmiyor!"
print(f"  Train BERT cache'den: {len(train_bert):,}", flush=True)


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


def build_query_item_texts(tids, iids):
    queries = [tid_to_q.get(int(t) if str(t).isdigit() else t, "") for t in tids]
    items_t = [build_bert_item_text(iid_to_str.get(str(i))) for i in iids]
    return queries, items_t


def bulk_tokenize(queries, items_t, tokenizer, max_len=128, chunk=20_000):
    all_ids, all_mask, all_type = [], [], []
    n = len(queries)
    for i in range(0, n, chunk):
        enc = tokenizer(queries[i:i+chunk], items_t[i:i+chunk],
                        max_length=max_len, truncation=True,
                        padding="max_length", return_tensors="pt")
        all_ids.append(enc["input_ids"])
        all_mask.append(enc["attention_mask"])
        all_type.append(enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])))
    return (torch.cat(all_ids), torch.cat(all_mask), torch.cat(all_type))


class TensorPairDataset(Dataset):
    def __init__(self, input_ids, attention_mask, token_type_ids, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = torch.as_tensor(labels, dtype=torch.float)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        enc = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
        }
        return enc, self.labels[idx]


def bert_inference(tids, iids, model, tokenizer, batch=256, desc=""):
    queries, items_t = build_query_item_texts(tids, iids)
    scores = []
    model.eval()
    n = len(tids)
    tok_chunk = 50_000
    with torch.no_grad():
        for start in range(0, n, tok_chunk):
            end = min(start + tok_chunk, n)
            ids, mask, ttype = bulk_tokenize(queries[start:end], items_t[start:end], tokenizer)
            for b in range(0, end - start, batch):
                be = min(b + batch, end - start)
                batch_enc = {
                    "input_ids": ids[b:be].to(device),
                    "attention_mask": mask[b:be].to(device),
                    "token_type_ids": ttype[b:be].to(device),
                }
                with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                    out = model(**batch_enc).logits.squeeze(-1)
                scores.extend(torch.sigmoid(out).float().cpu().tolist())
            print(f"    {desc} {end:,}/{n:,}", flush=True)
    return np.array(scores, dtype=np.float32)


# v36b arka planda GPU kullandığı için fine-tune/inference tamamen ATLANIYOR.
# train_bert zaten yukarıda cache'den yüklendi. Test seti için de inference YOK —
# bu analiz sadece OOF (train) üzerinde çalışıyor, submission üretmiyor.
print("  (GPU/fine-tune adımları bilinçli olarak atlandı — sadece cache kullanılıyor)", flush=True)

# ─────────────────────────────────────────────────────────────────────
# PHASE C: TY-ECOMM-EMBED COSINE (ham, grup feature YOK — v27/v31/v32 yöntemi)
# ─────────────────────────────────────────────────────────────────────
print("\n[C] TY-ECOMM-EMBED COSINE (grup feature olmadan)...", flush=True)
t1 = time.time()

ITEM_EMB    = EMB_CACHE / "item_embs_tyembed.npy"
ITEM_IDS    = EMB_CACHE / "item_ids_tyembed.npy"
TRAIN_Q_EMB = EMB_CACHE / "train_q_embs_tyembed.npy"
TRAIN_Q_IDS = EMB_CACHE / "train_q_ids_tyembed.npy"
TEST_Q_EMB  = EMB_CACHE / "test_q_embs_tyembed.npy"
TEST_Q_IDS  = EMB_CACHE / "test_q_ids_tyembed.npy"

for p in [ITEM_EMB, ITEM_IDS, TRAIN_Q_EMB, TRAIN_Q_IDS, TEST_Q_EMB, TEST_Q_IDS]:
    if not p.exists():
        print(f"  ✗ Eksik cache: {p}", flush=True)
        print("    → önce 36_tyembed_cpuonly.py çalıştırılmalı!", flush=True)
        sys.exit(1)

item_embs    = np.load(str(ITEM_EMB))
item_ids_raw = np.load(str(ITEM_IDS), allow_pickle=True)
train_q_embs = np.load(str(TRAIN_Q_EMB))
train_q_ids  = np.load(str(TRAIN_Q_IDS), allow_pickle=True)
test_q_embs  = np.load(str(TEST_Q_EMB))
test_q_ids   = np.load(str(TEST_Q_IDS), allow_pickle=True)

print(f"  item_embs: {item_embs.shape} | train_q: {train_q_embs.shape} | test_q: {test_q_embs.shape}", flush=True)

iid_to_emb_idx       = {str(iid): i for i, iid in enumerate(item_ids_raw)}
train_tid_to_emb_idx = {str(tid): i for i, tid in enumerate(train_q_ids)}
test_tid_to_emb_idx  = {str(tid): i for i, tid in enumerate(test_q_ids)}

def compute_cosine(df, q_embs, q_tid_map, it_embs, iid_map, chunk=50_000):
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
            cos[start:end][valid] = (q_embs[q_idx[valid]] * it_embs[i_idx[valid]]).sum(axis=1)
        if (start // chunk) % 20 == 0 or end == n:
            print(f"    {end:,}/{n:,} ({100*end/n:.0f}%)", flush=True)
    return cos

print("  Train cosine hesaplanıyor...", flush=True)
train_tyembed_cos = compute_cosine(train_df, train_q_embs, train_tid_to_emb_idx, item_embs, iid_to_emb_idx)
print(f"  train_cos: pos mean={train_tyembed_cos[train_df['label'].values==1].mean():.3f} "
      f"neg mean={train_tyembed_cos[train_df['label'].values==0].mean():.3f}", flush=True)

print("  Test cosine hesaplanıyor...", flush=True)
test_tyembed_cos = compute_cosine(sub_pairs, test_q_embs, test_tid_to_emb_idx, item_embs, iid_to_emb_idx)
print(f"  test_cos: mean={test_tyembed_cos.mean():.3f}", flush=True)
print(f"  → {(time.time()-t1)/60:.1f} dk", flush=True)

del item_embs, train_q_embs, test_q_embs
gc.collect()

# ─────────────────────────────────────────────────────────────────────
# PHASE C2: TRUSTED BRAND + CATEGORY PHRASE SÖZLÜKLERİ (v31'den)
# ─────────────────────────────────────────────────────────────────────
print("\n[C2] Trusted brand + category phrase sözlükleri kuruluyor...", flush=True)

_category_words = set()
_category_phrases = set()
for cat in items["category"].unique():
    segs = [s.strip() for s in cat.split("/") if s.strip() and s.strip() != UNKNOWN]
    for seg in segs:
        _category_phrases.add(seg)
        for w in seg.split():
            _category_words.add(w)

_brand_noise = GENDER_WORDS | AGE_WORDS | RENKLER | set(MATERIAL_MAP_FULL.keys()) | _category_words

_brand_counts = items.loc[items["brand"] != UNKNOWN, "brand"].value_counts()
TRUSTED_BRANDS = set()
for b, cnt in _brand_counts.items():
    if " " in b:
        TRUSTED_BRANDS.add(b)
    elif cnt >= 3 and b not in _brand_noise and len(b) > 2:
        TRUSTED_BRANDS.add(b)

print(f"  Trusted brand sayısı: {len(TRUSTED_BRANDS):,} (toplam benzersiz brand: {items['brand'].nunique():,})", flush=True)
print(f"  Category phrase sayısı: {len(_category_phrases):,}", flush=True)


def query_trusted_brand(q):
    best = None
    for g in query_ngrams(q, maxn=3):
        if g in TRUSTED_BRANDS and (best is None or len(g) > len(best)):
            best = g
    return best


def trusted_brand_signal(q, item_brand):
    qb = query_trusted_brand(q)
    if qb is None: return 0.0
    if item_brand != UNKNOWN and (qb == item_brand or qb in item_brand or item_brand in qb):
        return 1.0
    return -1.0


def query_category_phrase(q):
    best = None
    for g in query_ngrams(q, maxn=3):
        if g in _category_phrases and (best is None or len(g) > len(best)):
            best = g
    return best


def category_phrase_signal(q, item_category):
    phrase = query_category_phrase(q)
    if phrase is None: return 0.0
    return 1.0 if phrase in item_category else -1.0


def query_type(q):
    qb = query_trusted_brand(q)
    qc = query_category_phrase(q)
    has_brand = qb is not None
    has_cat   = qc is not None
    if has_brand and qb == q: return "brand_only"
    if has_brand and has_cat: return "brand_category"
    if has_brand: return "brand_other"
    if has_cat and qc == q: return "category_only"
    if has_cat: return "category_attr"
    return "other"

# ─────────────────────────────────────────────────────────────────────
# PHASE D: FEATURE ENGINEERING (41 v32 feature'ı, DEĞİŞMEDİ)
# ─────────────────────────────────────────────────────────────────────
print("\n[D] FEATURE ENGINEERING...", flush=True)


def build_features(df_in, tfidf_vect, bert_arr, tyembed_arr, fit=False, all_texts=None):
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
    # ── v23 özellikleri
    f["product_type_cover"] = [product_type_cover(q,t,b,bo)
                                for q,t,b,bo in zip(qs,ts,brs,btok_ovlp)]
    f["all_q_in_title"]     = [all_q_in_title(q,t)  for q,t in zip(qs,ts)]
    f["brand_weight"]       = [brand_weight(q,b)     for q,b in zip(qs,brs)]
    # ── v31 özellikleri
    f["tyembed_cos"]             = tyembed_arr
    f["trusted_brand_mismatch"]  = [trusted_brand_signal(q,b)    for q,b in zip(qs,brs)]
    f["category_phrase_match"]  = [category_phrase_signal(q,c)  for q,c in zip(qs,cats)]
    # ── v32 özellikleri
    f["age_cross"]           = [age_cross(q,a)              for q,a in zip(qs,ages)]
    f["material_mismatch"]   = [material_mismatch(q,am)     for q,am in zip(qs,attr_mat)]
    f["query_type"]          = pd.Categorical([query_type(q) for q in qs])

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
    # v23
    "product_type_cover","all_q_in_title","brand_weight",
    # v31
    "tyembed_cos","trusted_brand_mismatch","category_phrase_match",
    # v32
    "age_cross","material_mismatch","query_type",
]

X_tr, df_tr = build_features(
    train_df, tfidf, train_bert, train_tyembed_cos, fit=True,
    all_texts=items["title"].tolist() + terms["query"].tolist()
)
y    = train_df["label"].values
tids = train_df["term_id"].values
print(f"  Train features: {X_tr[FEATS].shape} | {(time.time()-t1):.1f}s", flush=True)

# ─────────────────────────────────────────────────────────────────────
# PHASE E: BINARY LGBMClassifier 5-FOLD (v31/v32 ile aynı ayarlar)
# ─────────────────────────────────────────────────────────────────────
print("\n[E] BINARY LGBM 5-FOLD (n_estimators=4000, early_stop=150)...", flush=True)

oof    = np.zeros(len(train_df), dtype=np.float32)
models = []
gkf    = GroupKFold(n_splits=5)

for fold, (tri, vli) in enumerate(gkf.split(X_tr, y, tids)):
    X_train_f = X_tr[FEATS].iloc[tri]
    y_train_f = y[tri]
    X_val_f   = X_tr[FEATS].iloc[vli]
    y_val_f   = y[vli]

    model = LGBMClassifier(
        n_estimators=4000,
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
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)]
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

# ── SAĞLIK KONTROLÜ (otomatik) ──────────────────────────────────────
print("\n" + "=" * 65, flush=True)
print("SAĞLIK KONTROLÜ", flush=True)
print("=" * 65, flush=True)
V34_OOF = 0.96999
if best_f1 > V34_OOF + 0.02:
    print(f"  ⚠ UYARI: OOF F1 ({best_f1:.4f}) v34'ün {V34_OOF}'ından çok yüksek!", flush=True)
    print(f"    v25/v26'da bu durum gerçek Kaggle skorunun 0.84->0.68 çökmesiyle sonuçlanmıştı.", flush=True)
    print(f"    Feature importance tablosuna bak: tek bir feature domine ediyor mu?", flush=True)
else:
    print(f"  ✓ OOF F1 ({best_f1:.4f}) v34'e ({V34_OOF}) yakın/altında — makul bir sinyal.", flush=True)

if len(llm_df) > 0:
    print(f"\n  LLM veri kaynak dağılımı (toplam): pozitif={n_llm_pos:,} "
          f"| negatif={n_llm_neg:,}", flush=True)
    print(f"  Kaynak bazında: {dict(llm_df['src'].value_counts())}", flush=True)

top_feat, top_imp = fi.index[0], fi.iloc[0]
top_share = top_imp / fi.sum()
if top_share > 0.30:
    print(f"  ⚠ UYARI: '{top_feat}' tüm importance'ın %{100*top_share:.0f}'ini oluşturuyor — tek feature'a aşırı bağımlılık riski.", flush=True)
else:
    print(f"  ✓ En baskın feature '{top_feat}' importance'ın sadece %{100*top_share:.0f}'ini oluşturuyor.", flush=True)
print("=" * 65, flush=True)

# ─────────────────────────────────────────────────────────────────────
# PHASE G: OOF HATA ANALİZİ — hangi sorguları anlamlandıramadık?
# (Submission YOK, sadece train/OOF üzerinde analiz — GPU/test seti gerekmiyor)
# ─────────────────────────────────────────────────────────────────────
print("\n[G] OOF HATA ANALİZİ BAŞLIYOR...", flush=True)
from sklearn.metrics import precision_score, recall_score

OUT_DIR = BASE / "claude only" / "analysis"
OUT_DIR.mkdir(exist_ok=True)

adf = df_tr.copy()
adf["label"]     = y
adf["oof_proba"] = oof
adf["pred"]      = (oof > best_thr).astype(int)
adf["fp"] = ((adf["pred"] == 1) & (adf["label"] == 0)).astype(int)
adf["fn"] = ((adf["pred"] == 0) & (adf["label"] == 1)).astype(int)
adf["tp"] = ((adf["pred"] == 1) & (adf["label"] == 1)).astype(int)
adf["tn"] = ((adf["pred"] == 0) & (adf["label"] == 0)).astype(int)
for feat in FEATS:
    if feat in ("ana_kategori", "query_type"): continue
    adf[feat] = X_tr[feat].values

fp_df = adf[adf["fp"] == 1]
fn_df = adf[adf["fn"] == 1]
tp_df = adf[adf["tp"] == 1]
tn_df = adf[adf["tn"] == 1]

lines = []
lines.append("=" * 70)
lines.append("v34 TABANI — DETAYLI OOF HATA ANALİZİ (v36b arka planda çalışırken üretildi)")
lines.append("=" * 70)

total = len(adf)
n_fp, n_fn, n_tp, n_tn = len(fp_df), len(fn_df), len(tp_df), len(tn_df)
lines.append(f"\n[1] GENEL ÖZET")
lines.append(f"  Toplam OOF çift    : {total:>10,}")
lines.append(f"  False Positive (FP): {n_fp:>10,} ({100*n_fp/total:.2f}%) ← model 1 dedi, gerçek 0")
lines.append(f"  False Negative (FN): {n_fn:>10,} ({100*n_fn/total:.2f}%) ← model 0 dedi, gerçek 1")
prec = n_tp/(n_tp+n_fp) if (n_tp+n_fp) else 0
rec  = n_tp/(n_tp+n_fn) if (n_tp+n_fn) else 0
lines.append(f"  Precision={prec:.4f} | Recall={rec:.4f} | OOF F1={best_f1:.5f}")

# ── PER-QUERY F1 ──
lines.append(f"\n[2] PER-QUERY F1 (sadece training sorguları, term_id bazında)")
qrows = []
for tid, grp in adf.groupby("term_id"):
    lbl, prd = grp["label"].values, grp["pred"].values
    f1 = f1_score(lbl, prd, average="macro", zero_division=0)
    qrows.append({
        "term_id": tid, "query": grp["query"].iloc[0], "f1": f1,
        "precision": precision_score(lbl, prd, zero_division=0),
        "recall": recall_score(lbl, prd, zero_division=0),
        "n_pos": int(lbl.sum()), "n_neg": int((lbl == 0).sum()),
        "q_len": len(str(grp["query"].iloc[0]).split()),
        "main_cat": grp["main_category"].mode()[0] if len(grp) else "unknown",
    })
qdf = pd.DataFrame(qrows)
lines.append(f"  Sorgu sayısı: {len(qdf):,} | Ortalama query F1: {qdf['f1'].mean():.4f} | Medyan: {qdf['f1'].median():.4f}")
lines.append(f"  F1=0.0 olan sorgu: {(qdf['f1']==0).sum():,} ({100*(qdf['f1']==0).mean():.1f}%)")
lines.append(f"  F1<0.5 olan sorgu: {(qdf['f1']<0.5).sum():,} ({100*(qdf['f1']<0.5).mean():.1f}%)")

lines.append(f"\n  Sorgu uzunluğuna göre ortalama F1 (Türkçe query karmaşıklığı sinyali):")
for ql, grp in qdf.groupby("q_len"):
    if len(grp) < 20: continue
    lines.append(f"    {ql} kelime: F1={grp['f1'].mean():.3f} (n={len(grp)})")

lines.append(f"\n  Ana kategoriye göre en düşük F1 (n>=30, en kötü 15):")
cat_f1 = qdf.groupby("main_cat").agg(f1=("f1", "mean"), n=("f1", "size")).query("n>=30").sort_values("f1")
for cat, row in cat_f1.head(15).iterrows():
    lines.append(f"    {cat:<35} F1={row['f1']:.3f} (n={int(row['n'])})")

# ── FP / FN feature profilleri ──
lines.append(f"\n[3] FALSE POSITIVE PROFİLİ (n={n_fp:,}) — model neye kanıyor?")
feat_cols = [f for f in FEATS if f not in ("ana_kategori", "query_type")]
diffs = (fp_df[feat_cols].mean() - tn_df[feat_cols].mean()).sort_values(ascending=False)
for feat in diffs.index[:12]:
    lines.append(f"    {feat:<25} FP={fp_df[feat].mean():.3f}  TN={tn_df[feat].mean():.3f}  fark={diffs[feat]:+.3f}")

lines.append(f"\n[4] FALSE NEGATIVE PROFİLİ (n={n_fn:,}) — model neyi kaçırıyor?")
diffs_fn = (fn_df[feat_cols].mean() - tp_df[feat_cols].mean()).sort_values()
for feat in diffs_fn.index[:12]:
    lines.append(f"    {feat:<25} FN={fn_df[feat].mean():.3f}  TP={tp_df[feat].mean():.3f}  fark={diffs_fn[feat]:+.3f}")

# ── En "emin" yanlışlar — Türkçe anlamlandırma hatası örnekleri ──
lines.append(f"\n[5] EN 'EMİN' YANLIŞ POZİTİFLER (model çok emindi ama yanlıştı — 20 örnek)")
top_fp = fp_df.nlargest(20, "oof_proba")
for _, r in top_fp.iterrows():
    lines.append(f"  proba={r['oof_proba']:.3f} bert={r['bert_score']:.3f} tyembed={r['tyembed_cos']:.3f} fuzz={r['fuzz_set']:.2f} head_in_title={r['head_in_title']:.1f}")
    lines.append(f"    SORGU : {r['query']}")
    lines.append(f"    ÜRÜN  : {r['title'][:90]}")
    lines.append(f"    Marka : {r['brand']} | Kategori: {r['main_category']}")

lines.append(f"\n[6] EN 'EMİN' YANLIŞ NEGATİFLER (model çok emin kaçırdı — 20 örnek)")
top_fn = fn_df.nsmallest(20, "oof_proba")
for _, r in top_fn.iterrows():
    lines.append(f"  proba={r['oof_proba']:.3f} bert={r['bert_score']:.3f} tyembed={r['tyembed_cos']:.3f} fuzz={r['fuzz_set']:.2f} head_in_title={r['head_in_title']:.1f}")
    lines.append(f"    SORGU : {r['query']}")
    lines.append(f"    ÜRÜN  : {r['title'][:90]}")
    lines.append(f"    Marka : {r['brand']} | Kategori: {r['main_category']}")

# ── En düşük F1'li sorgular (worst queries) ──
lines.append(f"\n[7] EN DÜŞÜK F1'Lİ 30 SORGU (Türkçe anlamlandırma en çok başarısız olduğu yerler)")
worst_q = qdf[qdf["n_pos"] > 0].nsmallest(30, "f1")
for _, r in worst_q.iterrows():
    lines.append(f"  F1={r['f1']:.2f} prec={r['precision']:.2f} rec={r['recall']:.2f} | \"{r['query']}\" (kat={r['main_cat']}, {r['n_pos']} poz/{r['n_neg']} neg)")

lines.append(f"\n{'='*70}")
lines.append(f"Toplam süre: {(time.time()-t0)/60:.1f} dk (CPU-only, GPU kullanılmadı)")
lines.append(f"{'='*70}")

report = "\n".join(lines)
print(report, flush=True)

out_path = OUT_DIR / "oof_error_analysis_v34base.txt"
with open(str(out_path), "w", encoding="utf-8") as fobj:
    fobj.write(report)
qdf.sort_values("f1").to_csv(str(OUT_DIR / "per_query_f1_v34base.csv"), index=False, encoding="utf-8")

print(f"\nDosyalar kaydedildi: {OUT_DIR}", flush=True)
