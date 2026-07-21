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
BERT_V36   = MODELS_DIR / "bert_v36"          # v37a: bert_v36'yı yeniden kullanıyor, dokunmuyoruz
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
# YENİ v37: spec_code_match — ince varyant ayrımı (beden/model kodu/miktar)
# v34-tabanı OOF hata analizinde (analysis/oof_error_analysis_v34base.txt) en
# "emin" yanlış pozitiflerin çoğu, sorgudaki bir SAYI/KOD'un (beden "6", model
# "a54", miktar "500 gr") üründe eşleşmediği durumlardı. Query'deki rakam içeren
# tokenleri title+attributes'te ara: hepsi bulunuyorsa +1, hiçbiri yoksa -1,
# query'de kod yoksa 0 (nötr).
# ─────────────────────────────────────────────────────────────────────
def extract_codes(s):
    return set(t for t in s.split() if any(c.isdigit() for c in t))

def spec_code_match(q, t, attrs):
    q_codes = extract_codes(q)
    if not q_codes:
        return 0.0
    target_text = t + " " + attrs
    t_codes = extract_codes(target_text)
    if not t_codes:
        return -1.0
    hits = sum(1 for c in q_codes if c in t_codes)
    if hits == len(q_codes):
        return 1.0
    if hits == 0:
        return -1.0
    return 0.0

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
# PHASE A2: v37a — v36b'nin AYNI TABANINI yeniden kur (v34 + v35 train-LLM + v36 test-LLM),
#           sonra YENİ v37 (hedefli) LLM satırlarını ekle
# ─────────────────────────────────────────────────────────────────────
print("\n[A2] v37a: v36b tabanı yeniden kuruluyor (v35 train-side + v36 test-side)...", flush=True)

BASE_LLM_SOURCES = [
    BASE / "claude only" / "51_llm_labels" / "merged_llm_labels.csv",
    BASE / "claude only" / "57_llm_labels_test" / "merged_test_llm_labels.csv",
]

existing_keys = set(train_df["term_id"] + "\t" + train_df["item_id"])
base_parts = []
for llm_labels_path in BASE_LLM_SOURCES:
    llm_labels = pd.read_csv(str(llm_labels_path))
    llm_labels["term_id"] = llm_labels["term_id"].astype(str)
    llm_labels["item_id"] = llm_labels["item_id"].astype(str)
    llm_keys = llm_labels["term_id"] + "\t" + llm_labels["item_id"]
    llm_labels = llm_labels[~llm_keys.isin(existing_keys)].drop_duplicates(subset=["term_id", "item_id"])
    part = llm_labels[["term_id", "item_id", "label", "src"]].copy()
    part["label"] = part["label"].astype(int)
    base_parts.append(part)
    existing_keys |= set(part["term_id"] + "\t" + part["item_id"])

original_train_df = pd.concat([train_df] + base_parts, ignore_index=True)  # = v36b'nin train_df'i, AYNI SIRA
n_original = len(original_train_df)
print(f"  v36b tabanı yeniden kuruldu: {n_original:,} satır (bert_scores_v36_train.npy ile eşleşmeli)", flush=True)

print("\n[A3] v37a: HEDEFLİ (stratified) yeni LLM satırları ekleniyor (v37 train-side + test-side)...", flush=True)
V37_LLM_SOURCES = [
    BASE / "claude only" / "63_llm_labels_v37_train" / "merged_v37_train_labels.csv",
    BASE / "claude only" / "64_llm_labels_v37_test" / "merged_v37_test_labels.csv",
]

llm_parts = []
for llm_labels_path in V37_LLM_SOURCES:
    if not llm_labels_path.exists():
        print(f"  ✗ {llm_labels_path} bulunamadı — atlanıyor", flush=True)
        continue
    llm_labels = pd.read_csv(str(llm_labels_path))
    llm_labels["term_id"] = llm_labels["term_id"].astype(str)
    llm_labels["item_id"] = llm_labels["item_id"].astype(str)

    llm_keys = llm_labels["term_id"] + "\t" + llm_labels["item_id"]
    llm_labels = llm_labels[~llm_keys.isin(existing_keys)].drop_duplicates(subset=["term_id", "item_id"])

    part = llm_labels[["term_id", "item_id", "label", "src"]].copy()
    part["label"] = part["label"].astype(int)
    n_pos = int((part["label"] == 1).sum())
    n_neg = int((part["label"] == 0).sum())
    print(f"  {llm_labels_path.parent.name}/{llm_labels_path.name}: {len(part):,} yeni satır "
          f"(pos={n_pos:,}, neg={n_neg:,})", flush=True)

    llm_parts.append(part)
    existing_keys |= set(part["term_id"] + "\t" + part["item_id"])

if llm_parts:
    llm_df = pd.concat(llm_parts, ignore_index=True)
    n_llm_pos = int((llm_df["label"] == 1).sum())
    n_llm_neg = int((llm_df["label"] == 0).sum())
    print(f"  Toplam YENİ (v37) LLM etiketli satır: {len(llm_df):,}", flush=True)
    print(f"    Toplam pozitif: {n_llm_pos:,} | Toplam negatif: {n_llm_neg:,}", flush=True)

    train_df = pd.concat([original_train_df, llm_df], ignore_index=True)
    print(f"  Train (v37a genişletilmiş): {len(train_df):,} (v36b tabanı {n_original:,} + YENİ v37 {len(llm_df):,})", flush=True)
else:
    print(f"  ✗ Hiçbir v37 LLM etiket dosyası bulunamadı — v36b tabanıyla devam ediliyor", flush=True)
    llm_df = pd.DataFrame(columns=["term_id", "item_id", "label", "src"])
    train_df = original_train_df

# ─────────────────────────────────────────────────────────────────────
# PHASE B: BERT — bert_v36'dan WARM-START, YENİ v37 (hedefli) verilerle 2. ince ayar turu
# ─────────────────────────────────────────────────────────────────────
print("\n[B] BERT v37 (bert_v36'dan warm-start, YENİ v37 hedefli veriyle 2. tur)...", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}", flush=True)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
BERT_V37 = MODELS_DIR / "bert_v37"          # YENİ: bert_v36'ya DOKUNULMUYOR
BERT_V37.mkdir(parents=True, exist_ok=True)
v37_train_bert_path = CACHE_DIR / "bert_scores_v37_train.npy"
v37_test_bert_path  = SUBM / "bert_scores_v37_test.npy"

# v37b: model her zaman yeniden ince ayar yapılır (warm-start) — SADECE bert_v37 klasörü
# zaten doluysa (önceki bir çalıştırma tamamlanmışsa) atlanır.
bert_needs_finetune = not (BERT_V37 / "config.json").exists()
bert_needs_train    = (not v37_train_bert_path.exists() or
                       len(np.load(str(v37_train_bert_path))) != len(train_df))
bert_needs_test     = not v37_test_bert_path.exists()


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


if bert_needs_finetune:
    print("  BERT v37 warm-start fine-tune başlıyor (bert_v36'dan devam)...", flush=True)
    tokenizer  = AutoTokenizer.from_pretrained(str(BERT_V36))
    bert_model = AutoModelForSequenceClassification.from_pretrained(str(BERT_V36)).to(device)

    # ── v37b: YENİ v37 satırları (hedefli) TAM dahil + nadir conflict tipleri TAM
    # dahil, kalanı rastgele doldur. bert_v36 zaten v35+v36 LLM verisini biliyor —
    # bu turun odağı SADECE yeni (v37, hedefli: kısa sorgu + zayıf kategori) veriler.
    RARE_CONFLICT_SRC = {"gender_conflict", "age_conflict", "color_conflict"}
    V37_LLM_SRC = {"llm_recovered_positive_v37_train", "llm_verified_negative_v37_train",
                   "llm_recovered_positive_v37_test", "llm_verified_negative_v37_test"}
    priority_mask = train_df["src"].isin(RARE_CONFLICT_SRC | V37_LLM_SRC)
    priority_df = train_df[priority_mask]
    rest_df     = train_df[~priority_mask]

    BERT_SAMPLE = min(500_000, len(train_df))
    n_priority = len(priority_df)
    if n_priority >= BERT_SAMPLE:
        bert_train = priority_df.sample(BERT_SAMPLE, random_state=42)
    else:
        n_rest = BERT_SAMPLE - n_priority
        rest_sample = rest_df.sample(min(n_rest, len(rest_df)), random_state=42)
        bert_train = pd.concat([priority_df, rest_sample], ignore_index=True)
        bert_train = bert_train.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"  BERT_SAMPLE stratified: {len(bert_train):,} "
          f"(rare-conflict+v37 tam dahil: {n_priority:,}, rastgele tamamlama: {len(bert_train)-n_priority:,})", flush=True)
    print(f"  BERT sample kaynak dağılımı: {dict(bert_train['src'].value_counts())}", flush=True)
    print(f"  BERT train sample: {len(bert_train):,} | pos~{bert_train.label.sum():,}", flush=True)

    print("  Toplu tokenize ediliyor (fast tokenizer, tek seferde)...", flush=True)
    t_tok = time.time()
    tr_queries, tr_items_t = build_query_item_texts(
        bert_train["term_id"].astype(str).tolist(), bert_train["item_id"].astype(str).tolist())
    tr_ids, tr_mask, tr_type = bulk_tokenize(tr_queries, tr_items_t, tokenizer)
    print(f"  → tokenize {len(bert_train):,} örnek | {time.time()-t_tok:.1f}s", flush=True)

    ds_train = TensorPairDataset(tr_ids, tr_mask, tr_type, bert_train["label"].tolist())
    # v37b: warm-start ince ayar — az epoch, düşük LR (5 epoch/2e-5 sıfırdan eğitim DEĞİL)
    EPOCHS, BATCH, LR = 3, 32, 1e-5
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

    bert_model.save_pretrained(str(BERT_V37))
    tokenizer.save_pretrained(str(BERT_V37))
    print(f"  BERT v37 saved → {BERT_V37}", flush=True)

else:
    print(f"  BERT v37 cache'de var → yükleniyor (yeniden eğitim YOK)", flush=True)
    tokenizer  = AutoTokenizer.from_pretrained(str(BERT_V37))
    bert_model = AutoModelForSequenceClassification.from_pretrained(str(BERT_V37)).to(device)

# v37b: model değişti (2. warm-start turu) — TÜM train_df için TAM inference gerekiyor,
# eski cache (bert_scores_v36_*) artık bu modele karşılık gelmiyor.
if bert_needs_train:
    print(f"  Train BERT inference ({len(train_df):,} çift, YENİ v37 modeliyle)...", flush=True)
    t1 = time.time()
    train_bert = bert_inference(
        train_df["term_id"].astype(str).tolist(),
        train_df["item_id"].astype(str).tolist(),
        bert_model, tokenizer, desc="train")
    np.save(str(v37_train_bert_path), train_bert)
    print(f"  → {v37_train_bert_path.name} | {(time.time()-t1)/60:.1f} dk", flush=True)
else:
    train_bert = np.load(str(v37_train_bert_path))
    print(f"  Train BERT cache'den: {len(train_bert):,}", flush=True)

if bert_needs_test:
    print(f"  Test BERT inference ({len(sub_pairs):,} çift, YENİ v37 modeliyle)...", flush=True)
    t1 = time.time()
    test_bert = bert_inference(sub_pairs["term_id"].tolist(), sub_pairs["item_id"].tolist(),
                               bert_model, tokenizer, desc="test")
    np.save(str(v37_test_bert_path), test_bert)
    print(f"  → {v37_test_bert_path.name} | {(time.time()-t1)/60:.1f} dk", flush=True)
else:
    test_bert = np.load(str(v37_test_bert_path))
    print(f"  Test BERT cache'den: {len(test_bert):,}", flush=True)

del bert_model; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

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
    # ── YENİ v37 özelliği
    f["spec_code_match"]     = [spec_code_match(q,t,a) for q,t,a in zip(qs,ts,attrs)]

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
    # v37 (YENİ)
    "spec_code_match",
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
V36B_OOF = 0.97304
if best_f1 > V36B_OOF + 0.02:
    print(f"  ⚠ UYARI: OOF F1 ({best_f1:.4f}) v36b'nin {V36B_OOF}'ından çok yüksek!", flush=True)
    print(f"    v25/v26'da bu durum gerçek Kaggle skorunun 0.84->0.68 çökmesiyle sonuçlanmıştı.", flush=True)
    print(f"    Feature importance tablosuna bak: tek bir feature domine ediyor mu?", flush=True)
else:
    print(f"  ✓ OOF F1 ({best_f1:.4f}) v36b'ye ({V36B_OOF}) yakın/altında — makul bir sinyal.", flush=True)

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
# PHASE F: TEST INFERENCE + SUBMISSION (DOĞRUDAN threshold — v23/v31/v32 yöntemi)
# ─────────────────────────────────────────────────────────────────────
print("\n[F] TEST INFERENCE + SUBMISSION (doğrudan threshold, quantile mapping YOK)...", flush=True)
X_te, _ = build_features(sub_pairs, tfidf, test_bert, test_tyembed_cos)
test_proba = sum(m.predict_proba(X_te[FEATS])[:, 1] for m in models) / len(models)

final     = (test_proba > best_thr).astype(int)
pos_n     = final.sum()

print(f"  thr={best_thr:.4f} (OOF'tan doğrudan) → Pozitif={pos_n:,} ({100*pos_n/len(sub_pairs):.1f}%)", flush=True)

out_path = SUBM / "submission_v37b_warmstart2.csv"
pd.DataFrame({"id": sub_pairs["id"], "prediction": final}).to_csv(str(out_path), index=False)
np.save(str(SUBM / "v37b_test_proba.npy"), test_proba.astype(np.float32))

print(f"\n{'='*65}", flush=True)
print(f"TAMAMLANDI — v37b ELECTRA warm-start turu 2 (bert_v36'dan devam, hedefli v37 veri)", flush=True)
print(f"  OOF F1        : {best_f1:.5f} (v36b: {V36B_OOF}, v37a: 0.97263)", flush=True)
print(f"  OOF threshold : {best_thr:.4f}", flush=True)
print(f"  Pozitif       : {pos_n:,} ({100*pos_n/len(sub_pairs):.1f}%)", flush=True)
print(f"  Train çifti   : {len(train_df):,} (v36b tabanı {n_original:,} + YENİ v37 {len(llm_df):,})", flush=True)
print(f"  Feature sayısı: {len(FEATS)} (v36 ile aynı 41 + spec_code_match)", flush=True)
print(f"  BERT modeli   : bert_v37 (bert_v36'dan warm-start, 3 epoch, lr=1e-5)", flush=True)
print(f"  Toplam süre   : {(time.time()-t0)/60:.1f} dk", flush=True)
print(f"  Dosya         : {out_path}", flush=True)
print(f"{'='*65}", flush=True)
