"""
44_v30_lgbm.py — v30 LGBM: bert_v30 scores + v23 35 features (NO tyembed_cos)
================================================================================
v27'nin tam kopyası ile 2 farklılık:
  1. bert_score = bert_v30 (NN hard neg ile fine-tune edilmis) inference
  2. tyembed_cos feature KALDIRILDI (v27'de 0.83 cikti, zarar veriyor)

Gereksinim:
  - models/bert_v30/              : 43_v30_bert_hardneg.py ciktisi
  - submissions/bert_scores_v30_test.npy : 43_v30_bert_hardneg.py ciktisi

Beklenti: v23 (0.84) >= v30 > v27 (0.83)  -- bert_score iyilesmesi ile gap kapanabilir
"""

import gc, re, sys, os, time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import lightgbm as lgb
from lightgbm import LGBMClassifier
from rapidfuzz import fuzz as rfuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # RTX 5080 sm_100 uyumsuz
sys.stdout.reconfigure(encoding="utf-8")

BASE      = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA      = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
SUBM      = BASE / "claude only" / "submissions"
CACHE_DIR = BASE / "claude only"
MODELS    = BASE / "claude only" / "models"
BERT_V30  = MODELS / "bert_v30"

device = torch.device("cpu")

LOWER = str.maketrans("IiSsGgUuOoCc", "IiSsGgUuOoCc")
def trl(s):
    return str(s).lower().strip().replace("i", "i").replace("İ", "i")\
                 .replace("ş", "s").replace("ğ", "g").replace("ü", "u")\
                 .replace("ö", "o").replace("ç", "c")

UNKNOWN = "unknown"

# ── TÜRKÇE STEM + METRIKLER (v23 ile ayni) ────────────────────────────
TR_SUFFIXES = sorted([
    "lari","leri","lar","ler","nin","nun","nün","in","un","ün",
    "daki","deki","taki","teki","dan","den","tan","ten","da","de","ta","te",
    "ya","ye","yi","yü","la","le","ca","ce",
    "lik","lik","luk","lük","ci","cu","cü",
    "si","su","sü","sal","sel","li","lu","lü","ki",
    "a","e","i","u","ü",
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

TR_STOP = {'ve','ile','için','bu','bir','de','da','den','dan','te','ta',
           'ki','mi','mu','mü','mi','ne','ya','veya','her','tüm','çok','az','en'}

def query_head(q):
    words = [w for w in q.split() if w not in TR_STOP and len(w) > 1]
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
    t_words = set(t.split()); t_stems = {stem(w) for w in t_words}
    if h in t_words: return 1.0
    hs = stem(h)
    if hs in t_stems: return 0.8
    return 0.0

def head_in_cat(q, c):
    h = query_head(q)
    if not h: return 0.0
    return 1.0 if h in c or stem(h) in c else 0.0

def product_type_cover(q, t, brand):
    BR_STOP = set(brand.split()) if brand != UNKNOWN else set()
    GENDER_WORDS = {'kadin','bayan','erkek','çocuk','bebek','kiz','erkek','unisex'}
    COLOR_WORDS = {'kirmizi','mavi','yesil','sari','beyaz','siyah','gri','mor','turuncu','pembe'}
    MATERIAL_WORDS = {'pamuk','deri','plastik','çelik','metal','ahsap','kumas','polyester'}
    skip = BR_STOP | GENDER_WORDS | COLOR_WORDS | MATERIAL_WORDS | TR_STOP | {'unknown'}
    q_type = [w for w in q.split() if w not in skip and len(w) > 1]
    if not q_type: return 0.0
    t_words = set(t.split()); t_stems = {stem(w) for w in t_words}
    hits = sum(1 for w in q_type if _word_match(w, t_words, t_stems))
    return hits / len(q_type)

def weighted_q_cov(q, t):
    if not q.split(): return 0.0
    t_words = set(t.split()); t_stems = {stem(w) for w in t_words}
    hits = sum(1 for w in q.split() if _word_match(w, t_words, t_stems))
    return hits / len(q.split())

RENKLER = {'kirmizi','mavi','yesil','sari','beyaz','siyah','gri','mor',
           'turuncu','pembe','bej','altın','gümüs','lacivert','bordo','füme',
           'ekru','krem','antrasit','haki','kahve','petrol'}
COLOR_NORM = {"gold":"altin","silver":"gümüs","rose":"pembe","krem":"bej"}
COLOR_FAMILY = {"antrasit":"gri","lacivert":"mavi","bordo":"kirmizi","altın":"sari",
                "gümüs":"metalik","krem":"bej","ekru":"bej"}

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
    if not s or str(s) in (UNKNOWN, ""): return {}
    d = {}
    for part in str(s).split(","):
        if ":" in part:
            k, _, v = part.partition(":")
            d[k.strip()] = v.strip()
    return d

def attr_typed_jac(q, a):
    d = parse_attrs(a)
    return jac(q, " ".join(d.values())) if d else jac(q, a)

def brand_tok_set(brand):
    return set(re.sub(r'[^a-z0-9çgiosü\s]', ' ', brand).split()) - {''} if brand != UNKNOWN else set()

def brand_tok_overlap(q, brand):
    b_toks = brand_tok_set(brand)
    q_toks = set(re.sub(r'[^a-z0-9çgiosü\s]', ' ', q).split())
    return len(b_toks & q_toks) / len(b_toks) if b_toks else 0.0

MATERIAL_MAP = {"pamuk":"pamuk","pamuklu":"pamuk","deri":"deri","polyester":"polyester",
                "yün":"yün","keten":"keten","naylon":"naylon","çelik":"çelik",
                "plastik":"plastik","ahsap":"ahsap"}

def material_match(q, attr_mat):
    if not attr_mat: return 0.0
    for tok in q.split():
        mat = MATERIAL_MAP.get(tok)
        if mat and mat in attr_mat: return 1.0
    return 0.0

def gender_cross(q, g):
    q_k = bool(re.search(r'\b(kadin|bayan)\b', q))
    q_e = bool(re.search(r'\berkek\b', q))
    g_k = g in ("kadin","bayan","kiz")
    g_e = g == "erkek"
    if q_k and g_e: return -1.0
    if q_e and g_k: return -1.0
    if q_k and g_k: return  1.0
    if q_e and g_e: return  1.0
    return 0.0

def brand_weight(q):
    # proxy: ne kadar marka tokeni var (brand_tok_overlap burada kullanamiyoruz)
    # v23 ile tutarli: brand_weight = brand_tok_overlap(q, brand) idi
    return 0.0  # fallback (asil hesaplama build_features'da yapiliyor)

# ── PHASE A: VERI + NEGATIFLER (v23 ile ayni sampling) ─────────────────
print("="*65, flush=True)
print("[A] VERI + NEGATIFLER (NEG_PER_POS=5, rng=42)...", flush=True)
t0 = time.time()

items       = pd.read_csv(DATA / "items.csv")
terms       = pd.read_csv(DATA / "terms.csv")
train_pairs = pd.read_csv(DATA / "training_pairs.csv")
sub_pairs   = pd.read_csv(DATA / "submission_pairs.csv")

for df in [items, train_pairs, sub_pairs, terms]:
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].fillna(UNKNOWN)

items["item_id"]       = items["item_id"].astype(str)
train_pairs["term_id"] = train_pairs["term_id"].astype(str)
train_pairs["item_id"] = train_pairs["item_id"].astype(str)
sub_pairs["term_id"]   = sub_pairs["term_id"].astype(str)
sub_pairs["item_id"]   = sub_pairs["item_id"].astype(str)

if "main_category" not in items.columns:
    items["main_category"] = items["category"].apply(
        lambda x: x.split("/")[0] if x and x != UNKNOWN else UNKNOWN)
if "sub_category" not in items.columns:
    items["sub_category"] = items["category"].apply(
        lambda x: x.split("/")[1] if x and "/" in str(x) else UNKNOWN)
if "L1" not in items.columns:
    items["L1"] = items["main_category"]

iid_to_row = {row.item_id: row for row in items.itertuples(index=False)}
tid_to_q   = dict(zip(terms["term_id"].astype(str), terms["query"].astype(str)))

NEG_PER_POS = 5
rng_neg = np.random.default_rng(42)

by_main = {}; by_brand = {}; by_gender = {}; by_age = {}
by_mg = {}; by_ma = {}; by_bm = {}; by_bs = {}

for row in items.itertuples(index=False):
    iid = str(row.item_id)
    mc  = str(row.main_category)
    sc  = str(row.sub_category) if hasattr(row, 'sub_category') else UNKNOWN
    br  = str(row.brand)
    ge  = str(row.gender) if hasattr(row, 'gender') else UNKNOWN
    ag  = str(row.age_group) if hasattr(row, 'age_group') else UNKNOWN
    by_main.setdefault(mc, []).append(iid)
    by_brand.setdefault(br, []).append(iid)
    by_gender.setdefault(ge, []).append(iid)
    by_age.setdefault(ag, []).append(iid)
    by_mg.setdefault((mc, ge), []).append(iid)
    by_ma.setdefault((mc, ag), []).append(iid)
    by_bm.setdefault((br, mc), []).append(iid)
    by_bs.setdefault((br, sc), []).append(iid)

positive_keys = set(train_pairs["term_id"] + "\t" + train_pairs["item_id"])
used_keys = set()

def sample_pool(pool, tid, pos_id):
    if not pool: return None
    idxs = rng_neg.permutation(min(len(pool), 50))
    for i in idxs:
        iid = str(pool[i])
        k = tid + "\t" + iid
        if iid != pos_id and k not in positive_keys and k not in used_keys:
            used_keys.add(k); return iid
    return None

def sample_same_brand_diff_main(brand, main, tid, pos_id):
    pool = [i for mc_k, pool_k in by_bm.items() if mc_k[0] == brand and mc_k[1] != main
            for i in pool_k]
    return sample_pool(pool, tid, pos_id)

def sample_same_brand_diff_sub(brand, sub, tid, pos_id):
    # same brand, same main_cat, different sub_cat
    pool = [i for i in by_brand.get(brand, []) if i not in by_bs.get((brand, sub), [])]
    return sample_pool(pool, tid, pos_id)

def sample_diff_main(main, tid, pos_id):
    others = [k for k in by_main if k != main]
    if not others: return None
    mc = others[rng_neg.integers(len(others))]
    return sample_pool(by_main[mc], tid, pos_id)

neg_tids = []; neg_iids = []; neg_src = []

for row in train_pairs.itertuples(index=False):
    tid = str(row.term_id); pos_id = str(row.item_id)
    item = iid_to_row.get(pos_id)
    if item is None: continue
    main  = str(item.main_category)
    sub   = str(item.sub_category) if hasattr(item, 'sub_category') else UNKNOWN
    brand = str(item.brand)
    gender= str(item.gender) if hasattr(item, 'gender') else UNKNOWN
    age   = str(item.age_group) if hasattr(item, 'age_group') else UNKNOWN
    query = tid_to_q.get(tid, "")

    selected = []

    iid = sample_same_brand_diff_main(brand, main, tid, pos_id)
    if iid: selected.append((iid, "same_brand_diff_main"))

    if len(selected) < NEG_PER_POS:
        iid = sample_same_brand_diff_sub(brand, sub, tid, pos_id)
        if iid: selected.append((iid, "same_brand_diff_sub"))

    if len(selected) < NEG_PER_POS:
        if re.search(r'\berkek\b', query):
            pool = by_mg.get((main, "kadin"))
            if pool is None: pool = by_gender.get("kadin")
            iid = sample_pool(pool, tid, pos_id)
            if iid: selected.append((iid, "gender_conflict"))
        elif re.search(r'\b(kadin|bayan)\b', query):
            pool = by_mg.get((main, "erkek"))
            if pool is None: pool = by_gender.get("erkek")
            iid = sample_pool(pool, tid, pos_id)
            if iid: selected.append((iid, "gender_conflict"))

    if len(selected) < NEG_PER_POS:
        if re.search(r'\b(bebek|çocuk)\b', query):
            pool = by_ma.get((main, "yetiskin"))
            if pool is None: pool = by_age.get("yetiskin")
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
print(f"  Train: {len(train_df):,} | pos={train_df.label.sum():,} | {time.time()-t0:.0f}s", flush=True)

# ── PHASE B: BERT v30 INFERENCE ──────────────────────────────────────────
print("\n[B] BERT v30 INFERENCE...", flush=True)

if not (BERT_V30 / "config.json").exists():
    print(f"  HATA: bert_v30 bulunamadi: {BERT_V30}", flush=True)
    print("  43_v30_bert_hardneg.py'yi once calistir!", flush=True)
    sys.exit(1)

test_bert_path  = SUBM / "bert_scores_v30_test.npy"
train_bert_path = CACHE_DIR / "bert_scores_v30_train.npy"

if not test_bert_path.exists():
    print(f"  HATA: {test_bert_path} bulunamadi", flush=True)
    sys.exit(1)

def parse_attrs_bert(s):
    if not s or str(s) in (UNKNOWN, ""): return {}
    d = {}
    for part in str(s).split(","):
        if ":" in part:
            k, _, v = part.partition(":")
            d[k.strip()] = v.strip()
    return d

def build_bert_item_text(row):
    if row is None: return ""
    parts = []
    t = getattr(row, "title", "") or ""
    b = getattr(row, "brand", "") or ""
    c = getattr(row, "category", "") or ""
    if t and t != UNKNOWN: parts.append(str(t))
    if b and b != UNKNOWN: parts.append(str(b))
    mc = str(c).split("/")[0] if c and c != UNKNOWN else ""
    if mc: parts.append(mc)
    d = parse_attrs_bert(getattr(row, "attributes", "") or "")
    renk = d.get("renk", "")
    if renk and renk != UNKNOWN: parts.append(f"renk:{renk}")
    return " | ".join(parts[:4])

class PairDataset(Dataset):
    def __init__(self, tids, iids, labels, tok, max_len=96):
        self.queries = [tid_to_q.get(str(t), "") for t in tids]
        self.items   = [build_bert_item_text(iid_to_row.get(str(i))) for i in iids]
        self.labels  = list(labels)
        self.tok     = tok; self.max_len = max_len

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tok(self.queries[idx], self.items[idx],
                       max_length=self.max_len, truncation=True,
                       padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(0.0)

def bert_inference(tids, iids, model, tok, batch=512, desc=""):
    ds = PairDataset(tids, iids, [0]*len(tids), tok, max_len=96)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    scores = []
    model.eval()
    n = len(tids)
    with torch.no_grad():
        for i, (enc, _) in enumerate(dl):
            logits = model(**enc).logits.squeeze(-1)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            if (i+1) % 1000 == 0:
                print(f"    {desc}: {(i+1)*batch:,}/{n:,} ({100*(i+1)*batch/n:.0f}%)", flush=True)
    return np.array(scores, dtype=np.float32)

print(f"  bert_v30 yukleniyor: {BERT_V30}", flush=True)
tokenizer  = AutoTokenizer.from_pretrained(str(BERT_V30))
bert_model = AutoModelForSequenceClassification.from_pretrained(str(BERT_V30)).to(device)

if train_bert_path.exists():
    train_bert = np.load(str(train_bert_path))
    if len(train_bert) == len(train_df):
        print(f"  Train BERT cache: {len(train_bert):,} (reuse)", flush=True)
    else:
        print(f"  Train BERT boyut uyusmadi ({len(train_bert)} vs {len(train_df)}), yeniden hesaplaniyor...", flush=True)
        train_bert = None
else:
    train_bert = None

if train_bert is None:
    print(f"  Train BERT inference: {len(train_df):,} cift...", flush=True)
    t_b = time.time()
    train_bert = bert_inference(train_df["term_id"].tolist(), train_df["item_id"].tolist(),
                                bert_model, tokenizer, batch=512, desc="train")
    np.save(str(train_bert_path), train_bert)
    print(f"  Kaydedildi: {train_bert_path} | {(time.time()-t_b)/60:.1f} dk", flush=True)

test_bert = np.load(str(test_bert_path))
print(f"  Test BERT: {len(test_bert):,} | min={test_bert.min():.4f} max={test_bert.max():.4f}", flush=True)
print(f"  Train BERT: min={train_bert.min():.4f} max={train_bert.max():.4f} mean={train_bert.mean():.4f}", flush=True)

del bert_model; gc.collect()

# ── PHASE C: FEATURE ENGINEERING (35 features, NO tyembed_cos) ──────────
print("\n[C] FEATURE ENGINEERING (35 features, no tyembed_cos)...", flush=True)
t1 = time.time()

def build_features(df_in, tfidf_vect, bert_arr, fit=False, all_texts=None):
    df = df_in.copy()
    df["term_id"] = df["term_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    terms_l = terms.copy(); terms_l["term_id"] = terms_l["term_id"].astype(str)
    items_l = items.copy(); items_l["item_id"] = items_l["item_id"].astype(str)

    df = df.merge(terms_l[["term_id","query"]], on="term_id", how="left")
    df = df.merge(items_l[["item_id","title","category","brand","gender",
                            "age_group","attributes","main_category","L1"]],
                  on="item_id", how="left")
    for col in ["query","title","brand","category","gender","age_group","attributes","main_category","L1"]:
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

    Q_mat = tfidf_vect.transform(qs)
    T_mat = tfidf_vect.transform(ts)
    from sklearn.preprocessing import normalize as sk_normalize
    norms_q = sk_normalize(Q_mat, norm='l2')
    norms_t = sk_normalize(T_mat, norm='l2')
    tfidf_scores = np.array((norms_q.multiply(norms_t)).sum(axis=1)).ravel().astype(np.float32)

    f = pd.DataFrame(index=df.index)
    f["fuzz_partial"]  = [rfuzz.partial_ratio(q, t)/100   for q,t in zip(qs,ts)]
    f["fuzz_set"]      = [rfuzz.token_set_ratio(q, t)/100 for q,t in zip(qs,ts)]
    f["fuzz_sort"]     = [rfuzz.token_sort_ratio(q, t)/100 for q,t in zip(qs,ts)]
    f["fuzz_basic"]    = [rfuzz.ratio(q, t)/100            for q,t in zip(qs,ts)]
    f["jaccard"]       = [jac(q,t)       for q,t in zip(qs,ts)]
    f["tfidf_cos"]     = tfidf_scores
    f["q_cov_title"]   = [q_cov(q,t)    for q,t in zip(qs,ts)]
    f["t_cov_query"]   = [q_cov(t,q)    for q,t in zip(qs,ts)]
    f["cat_overlap"]   = [jac(q,c)       for q,c in zip(qs,cats)]
    f["exact_in_title"]= [(q in t)*1.0   for q,t in zip(qs,ts)]
    f["token_overlap"] = [len(set(q.split()) & set(t.split()))/max(len(set(q.split())),1)
                          for q,t in zip(qs,ts)]
    f["age_in_q"]      = [1.0 if re.search(r'\b(bebek|çocuk|yetiskin)\b', q) else 0.0
                          for q in qs]
    f["stem_jaccard"]  = [stem_jac(q,t)    for q,t in zip(qs,ts)]
    f["stem_cat_jac"]  = [stem_jac(q,c)    for q,c in zip(qs,cats)]
    f["gender_cross"]  = [gender_cross(q,g) for q,g in zip(qs,gens)]
    f["first_tok_title"] = [(q.split()[0] in t.split() if q.split() else False)*1.0
                            for q,t in zip(qs,ts)]
    f["first_tok_brand"] = [(q.split()[0] in b.split() if q.split() else False)*1.0
                            for q,b in zip(qs,brs)]
    f["q_len"]         = [len(q.split()) for q in qs]
    f["t_len"]         = [len(t.split()) for t in ts]
    f["bert_score"]    = bert_arr.astype(np.float32)
    f["ana_kategori"]  = [jac(q, mc) for q,mc in zip(qs, df["main_category"].tolist())]
    q_colors = [get_query_color(q) for q in qs]
    item_renkler = [parse_attrs(a).get("renk","") for a in attrs]
    f["color_typed"]   = [color_typed_match(qc, ir) for qc,ir in zip(q_colors, item_renkler)]
    f["has_q_color"]   = [1.0 if qc else 0.0 for qc in q_colors]
    f["attr_has_renk"] = [1.0 if parse_attrs(a).get("renk","") else 0.0 for a in attrs]
    f["brand_tok_ovlp"]= [brand_tok_overlap(q,b)  for q,b in zip(qs,brs)]
    attr_mat_col = [parse_attrs(a).get("materyal bileseni", parse_attrs(a).get("materyal","")) for a in attrs]
    f["material_match"]= [material_match(q,am) for q,am in zip(qs, attr_mat_col)]
    attr_kol_col = [parse_attrs(a).get("kol boyu","") for a in attrs]
    f["kol_boyu_match"]= [(("uzun" in q and "uzun" in ak) or ("kisa" in q and "kisa" in ak) or
                           ("kolsuz" in q and "kolsuz" in ak))*1.0
                          for q,ak in zip(qs, attr_kol_col)]
    f["attr_jac_fixed"]= [attr_typed_jac(q,a) for q,a in zip(qs,attrs)]
    f["attr_q_cov"]    = [q_cov(q," ".join(d.values())) if (d:=parse_attrs(a)) else 0.0
                          for q,a in zip(qs,attrs)]
    f["head_in_title"] = [head_in_title(q,t)  for q,t in zip(qs,ts)]
    f["weighted_q_cov"]= [weighted_q_cov(q,t) for q,t in zip(qs,ts)]
    f["head_in_cat"]   = [head_in_cat(q,c)    for q,c in zip(qs,cats)]
    f["product_type_cover"] = [product_type_cover(q,t,b) for q,t,b in zip(qs,ts,brs)]
    f["all_q_in_title"]= [(all(tok in t.split() for tok in q.split()) if q.split() else False)*1.0
                          for q,t in zip(qs,ts)]
    f["brand_weight"]  = [brand_tok_overlap(q,b) for q,b in zip(qs,brs)]
    return f

tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=60_000, sublinear_tf=True, min_df=2)

FEATS = [
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
]
assert len(FEATS) == 35, f"Beklenmedik feature sayisi: {len(FEATS)}"

X_tr = build_features(
    train_df, tfidf, train_bert, fit=True,
    all_texts=items["title"].tolist() + terms["query"].tolist()
)
y    = train_df["label"].values
tids = train_df["term_id"].values
print(f"  Train features: {X_tr[FEATS].shape} | {(time.time()-t1):.0f}s", flush=True)

# ── PHASE D: LGBM 5-FOLD ────────────────────────────────────────────────
print("\n[D] BINARY LGBM 5-FOLD (iter=8000, early_stop=200)...", flush=True)
oof    = np.zeros(len(train_df), dtype=np.float32)
models = []
gkf    = GroupKFold(n_splits=5)

for fold, (tri, vli) in enumerate(gkf.split(X_tr, y, tids)):
    X_tr_f = X_tr[FEATS].iloc[tri]; y_tr_f = y[tri]
    X_vl_f = X_tr[FEATS].iloc[vli]; y_vl_f = y[vli]

    model = LGBMClassifier(
        n_estimators=8000, learning_rate=0.03, num_leaves=127, max_depth=8,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, class_weight="balanced",
        random_state=42+fold, n_jobs=-1, verbose=-1,
    )
    model.fit(X_tr_f, y_tr_f, eval_set=[(X_vl_f, y_vl_f)],
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])

    val_proba = model.predict_proba(X_vl_f)[:, 1]
    oof[vli]  = val_proba.astype(np.float32)
    models.append(model)

    best_f = max(f1_score(y_vl_f, (val_proba > thr).astype(int), average="macro")
                 for thr in np.arange(0.20, 0.90, 0.02))
    print(f"  Fold {fold+1} | iter={model.best_iteration_} | val F1={best_f:.4f}", flush=True)

best_thr, best_f1 = 0.5, 0.0
for thr in np.arange(0.20, 0.90, 0.005):
    s = f1_score(y, (oof > thr).astype(int), average="macro")
    if s > best_f1: best_f1, best_thr = s, thr
print(f"\n  OOF F1={best_f1:.5f} | OOF thr={best_thr:.4f}", flush=True)

fi = pd.Series(sum(m.feature_importances_ for m in models) / len(models), index=FEATS).sort_values(ascending=False)
print("  Feature Onemleri (top 15):", flush=True)
for feat, imp in fi.head(15).items():
    marker = " <- bert_v30!" if feat == "bert_score" else ""
    print(f"    {feat:25s}: {imp:6.0f}{marker}", flush=True)

# ── PHASE E: TEST INFERENCE + SUBMISSION ────────────────────────────────
print("\n[E] TEST INFERENCE + SUBMISSION...", flush=True)
X_te       = build_features(sub_pairs, tfidf, test_bert)
test_proba = sum(m.predict_proba(X_te[FEATS])[:, 1] for m in models) / len(models)

np.save(str(SUBM / "v30_test_proba.npy"), test_proba.astype(np.float32))
final = (test_proba > best_thr).astype(int)
pos_n = final.sum()
print(f"  thr={best_thr:.4f} -> {pos_n:,} pozitif ({100*pos_n/len(sub_pairs):.1f}%)", flush=True)

out_path = SUBM / "submission_v30_bert_hardneg.csv"
pd.DataFrame({"id": sub_pairs["id"], "prediction": final}).to_csv(str(out_path), index=False)

total_min = (time.time()-t0)/60
print(f"\n{'='*65}", flush=True)
print(f"TAMAMLANDI -- v30 BERT hard neg + 35 features (no tyembed_cos)", flush=True)
print(f"  OOF F1        : {best_f1:.5f}", flush=True)
print(f"  OOF threshold : {best_thr:.4f}", flush=True)
print(f"  Pozitif       : {pos_n:,} ({100*pos_n/len(sub_pairs):.1f}%)", flush=True)
print(f"  Toplam sure   : {total_min:.1f} dk", flush=True)
print(f"  Dosya         : {out_path}", flush=True)
print(f"{'='*65}", flush=True)
