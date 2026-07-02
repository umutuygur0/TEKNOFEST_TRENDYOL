"""
30_oof_analysis.py — Detaylı OOF Hata Analizi
==============================================
Hedef: v22→v23 neden sadece +0.01 verdi? Gerçek bottleneck nedir?

Analiz soruları:
1. Per-query F1: Hangi sorgular çok düşük? Pattern var mı?
2. False Positives: Model 1 dedi ama 0 — neden kandı?
3. False Negatives: Model 0 dedi ama 1 — neden kaçırdı?
4. Feature correlation: Hangi feature'lar error'larla ilişkili?
5. Category/brand/length dağılımı: Hata nerede yoğun?
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
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import normalize

sys.stdout.reconfigure(encoding="utf-8")

BASE      = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA      = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
CACHE_DIR = BASE / "claude only"
OUT_DIR   = BASE / "claude only" / "analysis"
OUT_DIR.mkdir(exist_ok=True)

LOWER = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(s): return str(s).translate(LOWER).lower().strip()
UNKNOWN = "unknown"

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

GENDER_WORDS = {"erkek","kadın","bayan","kız","unisex","erkekler","kadınlar","kızlar"}
AGE_WORDS    = {"bebek","çocuk","yetişkin","çocuklar","bebekler"}
RENKLER = {"kırmızı","mavi","beyaz","siyah","sarı","yeşil","pembe","mor","gri","turuncu",
           "lacivert","bej","kahverengi","altın","gold","gümüş","silver","rose","ekru","krem",
           "bordo","haki","füme","antrasit","indigo","petrol"}
MATERIAL_MAP = {"pamuk","pamuklu","deri","hakiki","polyester","yün","keten","naylon",
                "çelik","plastik","ahşap","akrilik","viskon","modal","bambu","ipek"}
MATERIAL_MAP_FULL = {"pamuk":"pamuk","pamuklu":"pamuk","deri":"deri","hakiki":"deri",
                     "polyester":"polyester","yün":"yün","keten":"keten","naylon":"naylon",
                     "çelik":"çelik","plastik":"plastik","ahşap":"ahşap"}
COLOR_NORM = {"gold":"altın","silver":"gümüş","rose":"pembe","krem":"bej","kiremit":"kırmızı"}
COLOR_FAMILY = {"antrasit":"gri","füme":"gri","platin":"gri","lacivert":"mavi","indigo":"mavi",
                "petrol":"mavi","bordo":"kırmızı","altın":"sarı","gold":"sarı","krem":"bej","ekru":"bej"}

def brand_tok_set(brand):
    if not brand or brand==UNKNOWN: return set()
    return set(t for t in re.sub(r'[^a-z0-9çğıöşü\s]',' ',brand).split() if len(t)>1)

def product_type_cover(q, t, brand, btok_ovlp):
    b_toks = brand_tok_set(brand) if btok_ovlp>0.3 else set()
    q_words = [w for w in q.split()
               if w not in GENDER_WORDS and w not in AGE_WORDS
               and w not in RENKLER and w not in MATERIAL_MAP
               and w not in TR_STOP_FUNC and w not in b_toks and len(w)>1]
    if not q_words: return q_cov(q, t)
    t_words = set(t.split()); t_stems = {stem(w) for w in t_words}
    hits = sum(1 for w in q_words if _word_match(w, t_words, t_stems))
    return hits/len(q_words)

def all_q_in_title(q, t):
    return 1.0 if set(q.split()).issubset(set(t.split())) else 0.0

def brand_weight(q, brand):
    b_toks = brand_tok_set(brand)
    q_toks = q.split()
    return sum(1 for t in q_toks if t in b_toks)/len(q_toks) if q_toks else 0.0

def norm_color(c): return COLOR_NORM.get(c, c)
def get_query_color(q):
    for tok in q.split():
        if tok in RENKLER: return norm_color(tok)
    return None

def color_typed_match(q_color, item_renk):
    if q_color is None: return 0.0
    if not item_renk: return 0.0
    if q_color==item_renk: return 2.0
    if COLOR_FAMILY.get(q_color)==COLOR_FAMILY.get(item_renk): return 1.0
    return -1.0

def parse_attrs(s):
    if not s or s in (UNKNOWN,""): return {}
    d = {}
    for part in s.split(","):
        if ":" in part:
            k,_,v = part.partition(":"); d[k.strip()] = v.strip()
    return d

def attr_typed_jac(q, a):
    d = parse_attrs(a)
    return jac(q," ".join(d.values())) if d else jac(q,a)

def brand_tok_overlap(q, brand):
    b_toks = brand_tok_set(brand)
    q_toks = set(re.sub(r'[^a-z0-9çğıöşü\s]',' ',q).split())
    return len(b_toks & q_toks)/len(b_toks) if b_toks else 0.0

def material_match(q, attr_mat):
    if not attr_mat: return 0.0
    for tok in q.split():
        mat = MATERIAL_MAP_FULL.get(tok)
        if mat and mat in attr_mat: return 1.0
    return 0.0

def gender_cross(q, g):
    q_k = bool(re.search(r'\b(kadın|bayan)\b',q)); q_e = bool(re.search(r'\berkek\b',q))
    g_k = g in ("kadın","bayan","kız"); g_e = g=="erkek"
    if q_k and g_e: return -1.0
    if q_e and g_k: return -1.0
    if q_k and g_k: return  1.0
    if q_e and g_e: return  1.0
    return 0.0

# ─── PHASE A: VERİ ────────────────────────────────────────────────
print("[A] VERİ + NEGATİFLER...", flush=True)
t0 = time.time()

items       = pd.read_csv(DATA/"items.csv")
terms       = pd.read_csv(DATA/"terms.csv")
train_pairs = pd.read_csv(DATA/"training_pairs.csv")
sub_pairs   = pd.read_csv(DATA/"submission_pairs.csv")

for col in ["title","category","brand","gender","age_group","attributes"]:
    items[col] = items[col].fillna(UNKNOWN).apply(trl)
terms["query"] = terms["query"].fillna("").apply(trl)
items["main_category"] = items["category"].str.split("/").str[0].fillna(UNKNOWN).apply(trl)
items["sub_category"]  = items["category"].str.split("/").str[1].fillna(UNKNOWN).apply(trl)

item_ids_arr   = items["item_id"].values
item_mains_arr = items["main_category"].values
item_subs_arr  = items["sub_category"].values

assert len(set(train_pairs["term_id"].unique()) & set(sub_pairs["term_id"].unique())) == 0

def build_group_idx(df, cols):
    df_r = df.reset_index(drop=True)
    if len(cols)==1: return {k: g.index.values for k,g in df_r.groupby(cols[0], sort=False)}
    return {k: g.index.values for k,g in df_r.groupby(cols, sort=False)}

by_main  = build_group_idx(items, ["main_category"])
by_gender= build_group_idx(items, ["gender"])
by_mg    = build_group_idx(items, ["main_category","gender"])
by_age   = build_group_idx(items, ["age_group"])
by_ma    = build_group_idx(items, ["main_category","age_group"])
by_brand = build_group_idx(items, ["brand"])

positive_keys = set(train_pairs["term_id"].astype(str)+"\t"+train_pairs["item_id"].astype(str))
used_keys: set = set()
rng = np.random.default_rng(42)

def sample_pool(pool, tid, pos_iid, max_tries=40):
    if pool is None or len(pool)==0: return None
    for _ in range(max_tries):
        idx = int(pool[rng.integers(0,len(pool))]); iid = str(item_ids_arr[idx])
        k = tid+"\t"+iid
        if iid!=pos_iid and k not in positive_keys and k not in used_keys: return iid
    return None

def sample_diff_main(cur_main, tid, pos_iid, max_tries=80):
    n = len(item_ids_arr)
    for _ in range(max_tries):
        idx = int(rng.integers(0,n))
        if item_mains_arr[idx]==cur_main: continue
        iid = str(item_ids_arr[idx]); k = tid+"\t"+iid
        if iid!=pos_iid and k not in positive_keys and k not in used_keys: return iid
    return None

def sample_same_brand_diff_main(brand, main, tid, pos_iid, max_tries=60):
    if not brand or brand==UNKNOWN: return None
    pool = by_brand.get(brand)
    if pool is None: return None
    for _ in range(max_tries):
        idx = int(pool[rng.integers(0,len(pool))])
        if item_mains_arr[idx]==main: continue
        iid = str(item_ids_arr[idx]); k = tid+"\t"+iid
        if iid!=pos_iid and k not in positive_keys and k not in used_keys: return iid
    return None

def sample_same_brand_diff_sub(brand, main, sub, tid, pos_iid, max_tries=80):
    if not brand or brand==UNKNOWN: return None
    pool = by_brand.get(brand)
    if pool is None: return None
    for _ in range(max_tries):
        idx = int(pool[rng.integers(0,len(pool))])
        if item_mains_arr[idx]!=main: continue
        if item_subs_arr[idx]==sub: continue
        iid = str(item_ids_arr[idx]); k = tid+"\t"+iid
        if iid!=pos_iid and k not in positive_keys and k not in used_keys: return iid
    return None

pos_with_info = train_pairs.merge(terms, on="term_id", how="left")
pos_with_info = pos_with_info.merge(
    items[["item_id","main_category","sub_category","gender","age_group","brand"]],
    on="item_id", how="left"
)
for col in ["main_category","sub_category","gender","age_group","brand","query"]:
    pos_with_info[col] = pos_with_info[col].fillna(UNKNOWN).apply(trl)

neg_tids, neg_iids = [], []
NEG_PER_POS = 5

for row in pos_with_info.itertuples(index=False):
    tid=str(row.term_id); pos_id=str(row.item_id)
    main=str(row.main_category); sub=str(row.sub_category)
    query=str(row.query); brand=str(row.brand)
    selected=[]
    if len(selected)<NEG_PER_POS:
        iid=sample_same_brand_diff_main(brand,main,tid,pos_id)
        if iid: selected.append(iid)
    if len(selected)<NEG_PER_POS:
        iid=sample_same_brand_diff_sub(brand,main,sub,tid,pos_id)
        if iid: selected.append(iid)
    if len(selected)<NEG_PER_POS:
        if re.search(r'\berkek\b',query):
            pool=by_mg.get((main,"kadın"))
            if pool is None: pool=by_gender.get("kadın")
            iid=sample_pool(pool,tid,pos_id)
            if iid: selected.append(iid)
        elif re.search(r'\b(kadın|bayan)\b',query):
            pool=by_mg.get((main,"erkek"))
            if pool is None: pool=by_gender.get("erkek")
            iid=sample_pool(pool,tid,pos_id)
            if iid: selected.append(iid)
    if len(selected)<NEG_PER_POS:
        if re.search(r'\b(bebek|çocuk)\b',query):
            pool=by_ma.get((main,"yetişkin"))
            if pool is None: pool=by_age.get("yetişkin")
            iid=sample_pool(pool,tid,pos_id)
            if iid: selected.append(iid)
    if len(selected)<NEG_PER_POS:
        iid=sample_pool(by_main.get(main),tid,pos_id)
        if iid: selected.append(iid)
    while len(selected)<NEG_PER_POS:
        iid=sample_diff_main(main,tid,pos_id)
        if iid: selected.append(iid)
        else: break
    for iid in selected[:NEG_PER_POS]:
        k=tid+"\t"+iid
        if k in used_keys or k in positive_keys: continue
        used_keys.add(k); neg_tids.append(tid); neg_iids.append(iid)

negatives = pd.DataFrame({"term_id":neg_tids,"item_id":neg_iids,"label":0})
train_pairs["label"] = 1
train_df = pd.concat([train_pairs[["term_id","item_id","label"]],
                      negatives[["term_id","item_id","label"]]], ignore_index=True)
train_df = train_df.sort_values("term_id").reset_index(drop=True)
train_df["term_id"] = train_df["term_id"].astype(str)
train_df["item_id"] = train_df["item_id"].astype(str)
print(f"  Train: {len(train_df):,} | pos={train_df.label.sum():,}", flush=True)

# ─── PHASE B: BERT (cache) ────────────────────────────────────────
print("[B] BERT cache...", flush=True)
train_bert = np.load(str(CACHE_DIR/"bert_scores_v23_train.npy"))
assert len(train_bert)==len(train_df)
print(f"  Train BERT: {len(train_bert):,} ✓", flush=True)

# ─── PHASE C: FEATURES ───────────────────────────────────────────
print("[C] FEATURES...", flush=True)

def build_features(df_in, tfidf_vect, bert_arr, fit=False, all_texts=None):
    df = df_in.copy()
    df["term_id"] = df["term_id"].astype(str); df["item_id"] = df["item_id"].astype(str)
    terms_l = terms.copy(); terms_l["term_id"] = terms_l["term_id"].astype(str)
    items_l = items.copy(); items_l["item_id"] = items_l["item_id"].astype(str)
    df = df.merge(terms_l, on="term_id", how="left").merge(items_l, on="item_id", how="left")
    for col in ["query","title","brand","category","gender","age_group","attributes","main_category"]:
        df[col] = df[col].fillna(UNKNOWN).apply(trl)
    if fit:
        corpus = list(df["title"])+list(df["query"])+(all_texts or [])
        tfidf_vect.fit(corpus)
    qs=df["query"].tolist(); ts=df["title"].tolist(); cats=df["category"].tolist()
    brs=df["brand"].tolist(); gens=df["gender"].tolist(); ages=df["age_group"].tolist()
    attrs=df["attributes"].tolist()

    def tfidf_cos(ql, tl, chunk=50_000):
        n=len(ql); out=np.zeros(n,dtype=np.float32)
        for i in range(0,n,chunk):
            qm=normalize(tfidf_vect.transform(ql[i:i+chunk]),"l2")
            tm=normalize(tfidf_vect.transform(tl[i:i+chunk]),"l2")
            out[i:i+chunk]=np.array(qm.multiply(tm).sum(axis=1)).flatten()
        return out

    parsed=[parse_attrs(a) for a in attrs]
    attr_renk=[d.get("renk","") for d in parsed]
    attr_mat=[d.get("materyal bileşeni",d.get("materyal","")) for d in parsed]
    attr_kol=[d.get("kol boyu",d.get("kol tipi","")) for d in parsed]
    q_colors=[get_query_color(q) for q in qs]
    btok_ovlp=[brand_tok_overlap(q,b) for q,b in zip(qs,brs)]

    f=pd.DataFrame()
    f["fuzz_partial"]   =[rfuzz.partial_ratio(q,t)/100    for q,t in zip(qs,ts)]
    f["fuzz_set"]       =[rfuzz.token_set_ratio(q,t)/100  for q,t in zip(qs,ts)]
    f["fuzz_sort"]      =[rfuzz.token_sort_ratio(q,t)/100 for q,t in zip(qs,ts)]
    f["fuzz_basic"]     =[rfuzz.ratio(q,t)/100            for q,t in zip(qs,ts)]
    f["jaccard"]        =[jac(q,t)                        for q,t in zip(qs,ts)]
    f["tfidf_cos"]      =tfidf_cos(qs,ts)
    f["q_cov_title"]    =[q_cov(q,t)                     for q,t in zip(qs,ts)]
    f["t_cov_query"]    =[q_cov(t,q)                     for q,t in zip(qs,ts)]
    f["cat_overlap"]    =[jac(q,c.replace("/"," "))       for q,c in zip(qs,cats)]
    f["exact_in_title"] =[(q in t)*1.0                    for q,t in zip(qs,ts)]
    f["token_overlap"]  =[len(set(q.split())&set(t.split())) for q,t in zip(qs,ts)]
    f["age_in_q"]       =[(a not in (UNKNOWN,"") and a in q)*1.0 for a,q in zip(ages,qs)]
    f["stem_jaccard"]   =[stem_jac(q,t)                   for q,t in zip(qs,ts)]
    f["stem_cat_jac"]   =[stem_jac(q,c.replace("/"," "))  for q,c in zip(qs,cats)]
    f["gender_cross"]   =[gender_cross(q,g)               for q,g in zip(qs,gens)]
    f["first_tok_title"]=[(q.split()[0] in t if q.split() else False)*1.0 for q,t in zip(qs,ts)]
    f["first_tok_brand"]=[(q.split()[0] in b if q.split() and b!=UNKNOWN else False)*1.0 for q,b in zip(qs,brs)]
    f["q_len"]          =[len(q.split()) for q in qs]
    f["t_len"]          =[len(t.split()) for t in ts]
    f["bert_score"]     =bert_arr
    f["ana_kategori"]   =pd.Categorical([c.split("/")[0] for c in cats])
    f["color_typed"]    =[color_typed_match(qc,ir) for qc,ir in zip(q_colors,attr_renk)]
    f["has_q_color"]    =[(qc is not None)*1.0 for qc in q_colors]
    f["attr_has_renk"]  =[(ar!="")*1.0 for ar in attr_renk]
    f["brand_tok_ovlp"] =btok_ovlp
    f["material_match"] =[material_match(q,am)  for q,am in zip(qs,attr_mat)]
    f["kol_boyu_match"] =[( ("uzun" in q and "uzun" in ak) or
                            ("kısa" in q and "kısa" in ak) or
                            ("kolsuz" in q and "kolsuz" in ak) )*1.0
                          for q,ak in zip(qs,attr_kol)]
    f["attr_jac_fixed"] =[attr_typed_jac(q,a) for q,a in zip(qs,attrs)]
    f["attr_q_cov"]     =[q_cov(q," ".join(d.values())) if (d:=parse_attrs(a)) else 0.0
                          for q,a in zip(qs,attrs)]
    f["head_in_title"]  =[head_in_title(q,t) for q,t in zip(qs,ts)]
    f["weighted_q_cov"] =[weighted_q_cov(q,t) for q,t in zip(qs,ts)]
    f["head_in_cat"]    =[head_in_cat(q,c)    for q,c in zip(qs,cats)]
    f["product_type_cover"]=[product_type_cover(q,t,b,bo) for q,t,b,bo in zip(qs,ts,brs,btok_ovlp)]
    f["all_q_in_title"] =[all_q_in_title(q,t)  for q,t in zip(qs,ts)]
    f["brand_weight"]   =[brand_weight(q,b)     for q,b in zip(qs,brs)]
    return f, df

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

X_tr, merged_df = build_features(train_df, tfidf, train_bert, fit=True,
                                  all_texts=items["title"].tolist()+terms["query"].tolist())
y    = train_df["label"].values
tids = train_df["term_id"].values
print(f"  {X_tr[FEATS].shape} ✓", flush=True)

# ─── PHASE D: LGBM + OOF ─────────────────────────────────────────
print("[D] LGBM 5-FOLD...", flush=True)
oof    = np.zeros(len(train_df), dtype=np.float32)
gkf    = GroupKFold(n_splits=5)

for fold, (tri, vli) in enumerate(gkf.split(X_tr, y, tids)):
    model = LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=127,
        max_depth=8, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, class_weight="balanced",
        random_state=42+fold, n_jobs=-1, verbose=-1,
    )
    model.fit(X_tr[FEATS].iloc[tri], y[tri],
              eval_set=[(X_tr[FEATS].iloc[vli], y[vli])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    oof[vli] = model.predict_proba(X_tr[FEATS].iloc[vli])[:,1].astype(np.float32)
    print(f"  Fold {fold+1} done | iter={model.best_iteration_}", flush=True)

best_thr, best_f1 = 0.5, 0.0
for thr in np.arange(0.20, 0.90, 0.005):
    s = f1_score(y, (oof>thr).astype(int), average="macro")
    if s>best_f1: best_f1, best_thr = s, thr
print(f"  OOF F1={best_f1:.5f} | thr={best_thr:.4f}", flush=True)

# ─── PHASE E: ANALİZ ─────────────────────────────────────────────
print("\n[E] DETAYLI ANALİZ BAŞLIYOR...", flush=True)

# Analysis dataframe
adf = merged_df.copy()
adf["label"]     = y
adf["oof_proba"] = oof
adf["pred"]      = (oof > best_thr).astype(int)
adf["fp"] = ((adf["pred"]==1) & (adf["label"]==0)).astype(int)  # False Positive
adf["fn"] = ((adf["pred"]==0) & (adf["label"]==1)).astype(int)  # False Negative
adf["tp"] = ((adf["pred"]==1) & (adf["label"]==1)).astype(int)  # True Positive
adf["tn"] = ((adf["pred"]==0) & (adf["label"]==0)).astype(int)  # True Negative

# Feature sütunları ekle
for feat in FEATS:
    if feat == "ana_kategori": continue
    adf[feat] = X_tr[feat].values

fp_df = adf[adf["fp"]==1]
fn_df = adf[adf["fn"]==1]
tp_df = adf[adf["tp"]==1]
tn_df = adf[adf["tn"]==1]

lines = []
lines.append("="*70)
lines.append("DETAYLI OOF HATA ANALİZİ — v23 Calibrated")
lines.append("="*70)

# ── 1. GENEL ÖZET
total = len(adf)
n_fp = len(fp_df); n_fn = len(fn_df); n_tp = len(tp_df); n_tn = len(tn_df)
lines.append(f"\n[1] GENEL ÖZET")
lines.append(f"  Toplam OOF çift    : {total:>10,}")
lines.append(f"  True Positive (TP) : {n_tp:>10,} ({100*n_tp/total:.1f}%)")
lines.append(f"  True Negative (TN) : {n_tn:>10,} ({100*n_tn/total:.1f}%)")
lines.append(f"  False Positive (FP): {n_fp:>10,} ({100*n_fp/total:.1f}%) ← model 1 dedi, 0 olan")
lines.append(f"  False Negative (FN): {n_fn:>10,} ({100*n_fn/total:.1f}%) ← model 0 dedi, 1 olan")
precision = n_tp/(n_tp+n_fp) if (n_tp+n_fp)>0 else 0
recall    = n_tp/(n_tp+n_fn) if (n_tp+n_fn)>0 else 0
lines.append(f"  Precision          : {precision:.4f}")
lines.append(f"  Recall             : {recall:.4f}")
lines.append(f"  OOF F1 (macro)     : {best_f1:.5f}")

# ── 2. PER-QUERY F1
lines.append(f"\n[2] PER-QUERY F1 ANALİZİ")
query_stats = []
for tid, grp in adf.groupby("term_id"):
    q = grp["query"].iloc[0]
    lbl = grp["label"].values
    prd = grp["pred"].values
    f1 = f1_score(lbl, prd, average="macro", zero_division=0)
    prec = precision_score(lbl, prd, zero_division=0)
    rec  = recall_score(lbl, prd, zero_division=0)
    n_pos = lbl.sum()
    n_neg = (lbl==0).sum()
    q_len = len(q.split())
    has_brand = any(b!=UNKNOWN and b in q for b in grp["brand"].values)
    query_stats.append({
        "term_id": tid, "query": q, "f1": f1,
        "precision": prec, "recall": rec,
        "n_pos": n_pos, "n_neg": n_neg,
        "q_len": q_len, "has_brand": has_brand,
        "main_cat": grp["main_category"].mode()[0] if len(grp)>0 else UNKNOWN,
    })
qdf = pd.DataFrame(query_stats)

lines.append(f"  Sorgu sayısı          : {len(qdf):,}")
lines.append(f"  Ortalama query F1     : {qdf['f1'].mean():.4f}")
lines.append(f"  Medyan query F1       : {qdf['f1'].median():.4f}")
lines.append(f"  F1=0.0 olan sorgu     : {(qdf['f1']==0).sum():,} ({100*(qdf['f1']==0).mean():.1f}%)")
lines.append(f"  F1<0.5 olan sorgu     : {(qdf['f1']<0.5).sum():,} ({100*(qdf['f1']<0.5).mean():.1f}%)")
lines.append(f"  F1=1.0 olan sorgu     : {(qdf['f1']==1.0).sum():,} ({100*(qdf['f1']==1.0).mean():.1f}%)")

# F1 dağılımı
bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
labels = ["0.0","0.1","0.3","0.5","0.7","0.9"]
qdf["f1_bin"] = pd.cut(qdf["f1"], bins=bins, labels=labels, right=False)
dist = qdf["f1_bin"].value_counts().sort_index()
lines.append(f"\n  F1 dağılımı (per-query):")
for lb, cnt in dist.items():
    bar = "█" * int(cnt/max(dist)*30)
    lines.append(f"  [{lb:>4}+]: {cnt:>5,} sorgu  {bar}")

# Sorgu uzunluğuna göre
lines.append(f"\n  Sorgu uzunluğuna göre ortalama F1:")
for ql, grp in qdf.groupby("q_len"):
    lines.append(f"  {ql} kelime: {grp['f1'].mean():.3f} (n={len(grp)})")

# Marka içeren sorgular
lines.append(f"\n  Brand içeren sorgular  : F1={qdf[qdf['has_brand']]['f1'].mean():.4f} (n={qdf['has_brand'].sum()})")
lines.append(f"  Brand içermeyen sorgu  : F1={qdf[~qdf['has_brand']]['f1'].mean():.4f} (n={(~qdf['has_brand']).sum()})")

# ── 3. FALSE POSITIVE ANALİZİ
lines.append(f"\n[3] FALSE POSITIVE ANALİZİ (n={n_fp:,})")
lines.append("    ← Bunlar: model 1 dedi ama 0 olan (negatif örnekler, yanlış pozitif)")

# FP feature ortalamaları vs TN
feat_cols = [f for f in FEATS if f not in ("ana_kategori",)]
lines.append(f"\n  Feature ortalamaları (FP vs TN):")
lines.append(f"  {'Feature':<22} {'FP ort':>8} {'TN ort':>8} {'fark':>8}")
lines.append(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")
fp_means = fp_df[feat_cols].mean()
tn_means = tn_df[feat_cols].mean()
diffs = (fp_means - tn_means).sort_values(ascending=False)
for feat in diffs.index[:15]:
    diff = diffs[feat]
    lines.append(f"  {feat:<22} {fp_means[feat]:>8.3f} {tn_means[feat]:>8.3f} {diff:>+8.3f}")

# FP kategori dağılımı
lines.append(f"\n  FP'lerin ana kategori dağılımı (top 10):")
fp_cat = fp_df["main_category"].value_counts().head(10)
total_fp = len(fp_df)
for cat, cnt in fp_cat.items():
    pct = 100*cnt/total_fp
    lines.append(f"  {cat:<30} {cnt:>6,} ({pct:.1f}%)")

# FP'lerde brand_tok_ovlp dağılımı
lines.append(f"\n  FP'lerde brand_tok_ovlp > 0.5: {(fp_df['brand_tok_ovlp']>0.5).mean():.1%}")
lines.append(f"  FP'lerde product_type_cover   : {fp_df['product_type_cover'].mean():.3f}")
lines.append(f"  FP'lerde head_in_title        : {fp_df['head_in_title'].mean():.3f}")
lines.append(f"  FP'lerde bert_score           : {fp_df['bert_score'].mean():.3f}")

# FP'ler arasında yüksek bert_score olanlar (BERT kandı)
bert_fp = fp_df[fp_df["bert_score"] > 0.7]
lines.append(f"\n  BERT'in kandığı FP (bert_score>0.7): {len(bert_fp):,} ({100*len(bert_fp)/n_fp:.1f}%)")
lines.append(f"    Bu FP'lerde avg fuzz_set={bert_fp['fuzz_set'].mean():.3f}, head_in_title={bert_fp['head_in_title'].mean():.3f}")

# Örnek FP'ler
lines.append(f"\n  En yüksek OOF proba'lı FP'ler (en 'emin' yanlışlar):")
top_fp = fp_df.nlargest(10, "oof_proba")[["query","title","brand","main_category","oof_proba","bert_score","fuzz_set","head_in_title","product_type_cover"]]
for _, row in top_fp.iterrows():
    lines.append(f"  proba={row['oof_proba']:.3f} bert={row['bert_score']:.3f} fuzz={row['fuzz_set']:.2f} hit={row['head_in_title']:.1f} ptc={row['product_type_cover']:.2f}")
    lines.append(f"    Q: {row['query'][:60]}")
    lines.append(f"    T: {row['title'][:60]}")
    lines.append(f"    Brand: {row['brand']} | Cat: {row['main_category']}")

# ── 4. FALSE NEGATIVE ANALİZİ
lines.append(f"\n[4] FALSE NEGATIVE ANALİZİ (n={n_fn:,})")
lines.append("    ← Bunlar: model 0 dedi ama 1 olan (pozitif örnekler, model kaçırdı)")

lines.append(f"\n  Feature ortalamaları (FN vs TP):")
lines.append(f"  {'Feature':<22} {'FN ort':>8} {'TP ort':>8} {'fark':>8}")
lines.append(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")
fn_means = fn_df[feat_cols].mean()
tp_means = tp_df[feat_cols].mean()
diffs_fn = (fn_means - tp_means).sort_values()
for feat in diffs_fn.index[:15]:
    diff = diffs_fn[feat]
    lines.append(f"  {feat:<22} {fn_means[feat]:>8.3f} {tp_means[feat]:>8.3f} {diff:>+8.3f}")

lines.append(f"\n  FN'lerin ana kategori dağılımı (top 10):")
fn_cat = fn_df["main_category"].value_counts().head(10)
for cat, cnt in fn_cat.items():
    pct = 100*cnt/len(fn_df)
    lines.append(f"  {cat:<30} {cnt:>6,} ({pct:.1f}%)")

lines.append(f"\n  FN'lerde bert_score < 0.3 (BERT kaçırdı): {(fn_df['bert_score']<0.3).mean():.1%}")
lines.append(f"  FN'lerde head_in_title=0 (ürün tipi yok) : {(fn_df['head_in_title']==0).mean():.1%}")
lines.append(f"  FN'lerde fuzz_set < 0.3 (text farklı)    : {(fn_df['fuzz_set']<0.3).mean():.1%}")
lines.append(f"  FN'lerde brand_tok_ovlp=0 (brand yok)    : {(fn_df['brand_tok_ovlp']==0).mean():.1%}")

lines.append(f"\n  En düşük OOF proba'lı FN'ler (model en emin kaçırdığı pozitifler):")
top_fn = fn_df.nsmallest(10, "oof_proba")[["query","title","brand","main_category","oof_proba","bert_score","fuzz_set","head_in_title","product_type_cover"]]
for _, row in top_fn.iterrows():
    lines.append(f"  proba={row['oof_proba']:.3f} bert={row['bert_score']:.3f} fuzz={row['fuzz_set']:.2f} hit={row['head_in_title']:.1f} ptc={row['product_type_cover']:.2f}")
    lines.append(f"    Q: {row['query'][:60]}")
    lines.append(f"    T: {row['title'][:60]}")
    lines.append(f"    Brand: {row['brand']} | Cat: {row['main_category']}")

# ── 5. BOTTLENECK TEŞHİSİ
lines.append(f"\n[5] BOTTLENECK TEŞHİSİ")

# FP'lerde hangi mismatch var?
fp_high_bert = (fp_df["bert_score"] > 0.5).mean()
fp_high_text = (fp_df["fuzz_set"] > 0.5).mean()
fp_high_ptc  = (fp_df["product_type_cover"] > 0.5).mean()
fp_zero_hit  = (fp_df["head_in_title"] == 0).mean()
lines.append(f"\n  FP'lerin profili:")
lines.append(f"  bert_score>0.5 : {fp_high_bert:.1%}  ← BERT kandı mı?")
lines.append(f"  fuzz_set>0.5   : {fp_high_text:.1%}  ← Text benzerliği kandırdı mı?")
lines.append(f"  ptc>0.5        : {fp_high_ptc:.1%}  ← Ürün tipi eşleşti mi?")
lines.append(f"  head_in_title=0: {fp_zero_hit:.1%}  ← Baş kelime yoktu ama model kandı?")

fn_low_bert  = (fn_df["bert_score"] < 0.3).mean()
fn_low_text  = (fn_df["fuzz_set"] < 0.3).mean()
fn_zero_hit  = (fn_df["head_in_title"] == 0).mean()
lines.append(f"\n  FN'lerin profili:")
lines.append(f"  bert_score<0.3 : {fn_low_bert:.1%}  ← BERT pozitifi kaçırdı mı?")
lines.append(f"  fuzz_set<0.3   : {fn_low_text:.1%}  ← Text farklıydı, model doğru mu kaçırdı?")
lines.append(f"  head_in_title=0: {fn_zero_hit:.1%}  ← Ürün tipi title'da yok ama pozitif?")

# Dominant FP tipi
lines.append(f"\n  FP TIPI ANALİZİ:")
fp_same_brand  = (fp_df["brand_tok_ovlp"]>0.5).mean()
fp_same_cat_q  = (fp_df["cat_overlap"]>0.3).mean()
fp_gender_ok   = (fp_df["gender_cross"]>=0).mean()
lines.append(f"  Aynı brand    : {fp_same_brand:.1%}")
lines.append(f"  Benzer cat    : {fp_same_cat_q:.1%}")
lines.append(f"  Gender uyumlu : {fp_gender_ok:.1%}")

# ── 6. OOF-TEST GAP ANALİZİ
lines.append(f"\n[6] OOF-TEST GAP ANALİZİ")
lines.append(f"  OOF F1  = {best_f1:.4f}")
lines.append(f"  Test F1 = 0.8400 (v23_calibrated Kaggle skoru)")
lines.append(f"  Gap     = {best_f1-0.84:.4f} ← Bu dev gap neden oluşuyor?")
lines.append(f"\n  Olası sebepler:")
lines.append(f"  A) Domain shift: test query'leri tamamen farklı dağılım")
lines.append(f"  B) Sahte/geçersiz çiftler: yarışmacı anti-probe mekanizması")
lines.append(f"  C) Eğitim negativleri yapay: gerçek 'zor' vakalar öğrenilmedi")
lines.append(f"  D) OOF negatifler bizim dağılımımızdan: test negativleri farklı")
lines.append(f"\n  Kanıt:")
lines.append(f"  - v22 (OOF 0.9704, test 0.83) gap=0.140")
lines.append(f"  - v23 (OOF 0.9686, test 0.84) gap=0.129")
lines.append(f"  - Daha düşük OOF = daha az overfit = daha iyi test ← dikkat!")
lines.append(f"  - Çözüm: overfit'i azalt, genellemeyi artır")

# ── 7. ÖNERİLER
lines.append(f"\n[7] TAVSİYELER (analiz sonuçlarına göre)")
# Karar ağacı
if fp_high_bert > 0.4:
    lines.append(f"  ✗ BERT yüksek skorlu FP'ler çok ({fp_high_bert:.0%}) → BERT text benzerliğine kanıyor")
    lines.append(f"    → BM25 hard negatives EKLEME (text benzer ama yanlış ürün tipi)")
if fn_low_bert > 0.5:
    lines.append(f"  ✗ BERT düşük skorlu FN'ler çok ({fn_low_bert:.0%}) → BERT pozitifi kaçırıyor")
    lines.append(f"    → BERT daha büyük model veya daha uzun eğitim")
if fp_high_text > 0.5:
    lines.append(f"  ✗ Yüksek text benzerlikli FP'ler var ({fp_high_text:.0%})")
    lines.append(f"    → Ürün tipi farklılaştıran feature'lar ekle")
if fn_zero_hit > 0.5:
    lines.append(f"  ✗ FN'lerin {fn_zero_hit:.0%}'ünde head_in_title=0 ama pozitif")
    lines.append(f"    → Türkçe morfoloji daha agresif: stem matching güçlendir")
lines.append(f"\n  Genel önerim:")
lines.append(f"  Eğer FP>FN: precisions sorunu → hard negative mining (BM25)")
lines.append(f"  Eğer FN>FP: recall sorunu → daha az negative, daha yumuşak threshold")
fp_ratio = n_fp / (n_fp + n_fn) if (n_fp+n_fn)>0 else 0
lines.append(f"  Mevcut: FP={n_fp:,} FN={n_fn:,} FP/(FP+FN)={fp_ratio:.1%}")
if fp_ratio > 0.6:
    lines.append(f"  → Precision sorunu dominant: hard negatives + threshold yükselt")
elif fp_ratio < 0.4:
    lines.append(f"  → Recall sorunu dominant: daha yumuşak model, threshold düşür")
else:
    lines.append(f"  → Dengeli hata: genel genelleme sorunu")

lines.append(f"\n{'='*70}")
lines.append(f"Toplam süre: {(time.time()-t0)/60:.1f} dk")
lines.append(f"{'='*70}")

report = "\n".join(lines)
print(report, flush=True)

out_path = OUT_DIR / "oof_analysis_v23.txt"
with open(str(out_path), "w", encoding="utf-8") as f:
    f.write(report)

# CSV kaydet
qdf.to_csv(str(OUT_DIR / "per_query_f1.csv"), index=False, encoding="utf-8")
fp_df[["query","title","brand","main_category","oof_proba","bert_score",
       "fuzz_set","head_in_title","product_type_cover"]].head(500).to_csv(
    str(OUT_DIR / "false_positives_top500.csv"), index=False, encoding="utf-8")
fn_df[["query","title","brand","main_category","oof_proba","bert_score",
       "fuzz_set","head_in_title","product_type_cover"]].head(500).to_csv(
    str(OUT_DIR / "false_negatives_top500.csv"), index=False, encoding="utf-8")

print(f"\nDosyalar kaydedildi: {OUT_DIR}", flush=True)
