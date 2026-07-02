"""
31_v24_bm25_hardneg.py — v24: BM25 Hard Negatives + BM25 Feature
=================================================================
OOF analizi bulgusu (30_oof_analysis.py):
  - FP'lerin %80'inde bert_score > 0.5 → model text-benzer çiftlere kanıyor
  - Çözüm 1: BM25 hard negatives — query için yüksek BM25 ama label=0 çiftleri eğitimde göster
  - Çözüm 2: bm25_score feature — LGBM "yüksek BM25 = zorunlu pozitif değil" öğrensin

Leak kontrolü:
  ✓ BM25 neg: SADECE item catalog'dan, submission_pairs'ten ASLA
  ✓ positive_keys: bilinen tüm pozitifler negatif olamaz
  ✓ used_keys: çift tekrarı yok (hem eski hem yeni negatifler için)
  ✓ BM25 index: item title'ları üzerinde (label yok, test query yok)
  ✓ GroupKFold(term_id): query-level sızıntı yok
  ✓ assert train_term_ids ∩ test_term_ids = ∅

Farklar (v23_calibrate.py'ye göre):
  + BM25Okapi index: item title'ları üzerinde
  + TF-IDF hard negatives: N_HARDNEG=10 per unique query (~180K yeni negatif, scipy sparse — hızlı)
  + BERT inference: yeni negatifler için bert_v23 modeli kullanılır (CUDA varsa hızlı)
  + Aynı 35 feature (tfidf_cos zaten text-benzerliği kapsar)
  Tahmini süre: ~1.5-2 saat
"""

import gc, re, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from rapidfuzz import fuzz as rfuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

# rank_bm25 KULLANILMIYOR — çok yavaş (pure Python loop, 962K doc için ~1 saat)
# Bunun yerine scipy sparse matrix tabanlı TF-IDF cosine similarity kullanılıyor

import torch
from torch import nn
import torch.amp as torch_amp
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout.reconfigure(encoding="utf-8")

BASE       = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA       = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
SUBM       = BASE / "claude only" / "submissions"
MODELS_DIR = BASE / "claude only" / "models"
CACHE_DIR  = BASE / "claude only"
SUBM.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

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

# ── BERT için yardımcı fonksiyonlar ─────────────────────────────────
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
        self.queries = [tid_to_q_str.get(str(t), "") for t in tids]
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
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, pin_memory=(device=="cuda"))
    scores = []
    model.eval()
    n = len(tids)
    with torch.no_grad():
        for i, (batch_enc, _) in enumerate(dl):
            batch_enc = {k: v.to(device) for k, v in batch_enc.items()}
            with torch_amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                out = model(**batch_enc).logits.squeeze(-1)
            scores.extend(torch.sigmoid(out).float().cpu().tolist())
            if (i+1) % 200 == 0:
                print(f"    {desc} {(i+1)*batch:,}/{n:,}", flush=True)
    return np.array(scores, dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────
# PHASE A: VERİ + v23 İLE AYNI NEGATİFLER
# ─────────────────────────────────────────────────────────────────────
print("="*65, flush=True)
print("[A] VERİ + NEGATİFLER (v23 ile aynı, rng=42)...", flush=True)
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
items = items.reset_index(drop=True)  # BM25 index alignment için

item_ids_arr   = items["item_id"].values
item_mains_arr = items["main_category"].values
item_subs_arr  = items["sub_category"].values
iid_to_str     = {str(row.item_id): row for row in items.itertuples()}
tid_to_q       = {row.term_id: row.query for row in terms.itertuples()}
tid_to_q_str   = {str(k): v for k, v in tid_to_q.items()}  # string key versiyonu

print(f"  {len(items):,} ürün | {len(sub_pairs):,} test", flush=True)

# Leak kontrolü
train_tids = set(train_pairs["term_id"].unique())
test_tids  = set(sub_pairs["term_id"].unique())
assert len(train_tids & test_tids) == 0, f"LEAK! Ortak term_id: {len(train_tids & test_tids)}"
print(f"  ✓ train/test term_id disjoint ({len(train_tids):,} train, {len(test_tids):,} test)", flush=True)

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
rng = np.random.default_rng(42)  # v23 ile AYNI seed

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
    if pool is None or len(pool)==0: return None
    for _ in range(max_tries):
        idx = int(pool[rng.integers(0,len(pool))])
        if item_mains_arr[idx]==main: continue
        iid = str(item_ids_arr[idx]); k = tid+"\t"+iid
        if iid!=pos_iid and k not in positive_keys and k not in used_keys: return iid
    return None

def sample_same_brand_diff_sub(brand, main, sub, tid, pos_iid, max_tries=80):
    if not brand or brand==UNKNOWN: return None
    pool = by_brand.get(brand)
    if pool is None or len(pool)==0: return None
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
NEG_PER_POS = 5  # v23 ile aynı

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
print(f"  Train (v23 neg): {len(train_df):,} | pos={train_df.label.sum():,}", flush=True)
print(f"  Süre: {(time.time()-t0):.0f}s", flush=True)

# ─────────────────────────────────────────────────────────────────────
# PHASE B: TFIDF-TABANLI HARD NEGATIVES (scipy sparse — hızlı)
# ─────────────────────────────────────────────────────────────────────
print("\n[B] TFIDF HARD NEGATIVES (scipy sparse)...", flush=True)
t1 = time.time()

HARDNEG_CACHE = CACHE_DIR / "tfidf_hardneg_pairs_v24.csv"
N_HARDNEG = 10  # her unique train query için kaç hard negative

if HARDNEG_CACHE.exists():
    print("  Cache'den yükleniyor...", flush=True)
    bm25_hardneg_df = pd.read_csv(str(HARDNEG_CACHE))
    bm25_hardneg_df["term_id"] = bm25_hardneg_df["term_id"].astype(str)
    bm25_hardneg_df["item_id"] = bm25_hardneg_df["item_id"].astype(str)
    print(f"  Hard neg: {len(bm25_hardneg_df):,}", flush=True)
else:
    print("  TF-IDF item matrix inşa ediliyor...", flush=True)
    from sklearn.feature_extraction.text import TfidfVectorizer as TfIdfV2

    item_tfidf = TfIdfV2(max_features=80000, sublinear_tf=True, min_df=2, analyzer='word')
    item_matrix = item_tfidf.fit_transform(items["title"].tolist())
    item_matrix_norm = normalize(item_matrix, norm='l2')
    print(f"  Item TF-IDF: {item_matrix.shape} | {(time.time()-t1):.0f}s", flush=True)

    # item_id → satır indexi (items.reset_index edildi)
    item_id_to_idx = {str(row.item_id): i for i, row in enumerate(items.itertuples())}

    # train_df'yi term_id'ye göre grupla
    train_by_tid = defaultdict(list)
    for idx, row in train_df.iterrows():
        train_by_tid[row["term_id"]].append((idx, str(row["item_id"])))

    hardneg_rows = []
    hn_used_keys = set(used_keys)
    unique_train_tids = sorted(train_by_tid.keys())
    n_unique = len(unique_train_tids)
    print(f"  Unique train query: {n_unique:,}", flush=True)
    t_b2 = time.time()

    for qi, tid_str in enumerate(unique_train_tids):
        query = tid_to_q_str.get(tid_str, "")
        if not query:
            continue

        # Bu query için pozitif item'ları bul
        pos_iids_for_tid = set()
        for df_idx, iid in train_by_tid[tid_str]:
            if train_df.loc[df_idx, "label"] == 1:
                pos_iids_for_tid.add(iid)

        # TF-IDF cosine similarity — scipy sparse, hızlı
        try:
            q_vec = normalize(item_tfidf.transform([query]), norm='l2')
            sims = (item_matrix_norm @ q_vec.T).toarray().ravel()
        except Exception:
            continue

        # Top-300 benzer item → hard neg olarak al
        top_indices = np.argpartition(sims, -min(300, len(sims)))[-min(300, len(sims)):]
        top_sorted = top_indices[np.argsort(sims[top_indices])[::-1]]

        added = 0
        for item_idx in top_sorted:
            if added >= N_HARDNEG:
                break
            iid = str(items.iloc[item_idx]["item_id"])
            key = f"{tid_str}\t{iid}"
            if iid in pos_iids_for_tid:
                continue
            if key in positive_keys:
                continue
            if key in hn_used_keys:
                continue
            hardneg_rows.append({"term_id": tid_str, "item_id": iid, "label": 0})
            hn_used_keys.add(key)
            added += 1

        if (qi+1) % 3000 == 0:
            elapsed = time.time() - t_b2
            print(f"    {qi+1}/{n_unique} | hard neg: {len(hardneg_rows):,} | {elapsed:.0f}s", flush=True)

    bm25_hardneg_df = pd.DataFrame(hardneg_rows)
    bm25_hardneg_df["term_id"] = bm25_hardneg_df["term_id"].astype(str)
    bm25_hardneg_df["item_id"] = bm25_hardneg_df["item_id"].astype(str)
    bm25_hardneg_df.to_csv(str(HARDNEG_CACHE), index=False)
    print(f"  Hard neg: {len(bm25_hardneg_df):,} çift | {(time.time()-t1):.0f}s", flush=True)

# Leak kontrolü: hard neg içinde pozitif var mı?
hardneg_keys = set(bm25_hardneg_df["term_id"].astype(str) + "\t" + bm25_hardneg_df["item_id"].astype(str))
overlap_check = hardneg_keys & positive_keys
assert len(overlap_check) == 0, f"LEAK! Hard neg içinde {len(overlap_check)} pozitif var!"
print(f"  ✓ Hard neg leak kontrolü: 0 pozitif sızıntı", flush=True)

# ─────────────────────────────────────────────────────────────────────
# PHASE C: BERT v23 INFERENCE (yeni BM25 negatifler için)
# ─────────────────────────────────────────────────────────────────────
print("\n[C] BERT INFERENCE (BM25 hard neg + v23 cache)...", flush=True)
t2 = time.time()

v23_train_path   = CACHE_DIR / "bert_scores_v23_train.npy"
v23_test_path    = SUBM / "bert_scores_v23_test.npy"
bm25neg_bert_path = CACHE_DIR / "bert_scores_v24_bm25neg.npy"

BERT_V23 = MODELS_DIR / "bert_v23"
assert BERT_V23.exists(), "bert_v23 modeli bulunamadı!"
assert v23_train_path.exists(), "bert_scores_v23_train.npy bulunamadı!"
assert v23_test_path.exists(), "bert_scores_v23_test.npy bulunamadı!"

train_bert_v23 = np.load(str(v23_train_path))
test_bert      = np.load(str(v23_test_path))
print(f"  v23 train BERT: {len(train_bert_v23):,} ✓", flush=True)
print(f"  v23 test BERT : {len(test_bert):,} ✓", flush=True)
assert len(train_bert_v23) == len(train_df), \
    f"v23 train BERT boyutu uyuşmuyor: {len(train_bert_v23)} vs {len(train_df)}"

if bm25neg_bert_path.exists() and len(np.load(str(bm25neg_bert_path))) == len(bm25_hardneg_df):
    print(f"  BM25 neg BERT cache'den: {bm25neg_bert_path.name}", flush=True)
    bert_bm25neg = np.load(str(bm25neg_bert_path))
else:
    print(f"  BERT inference: {len(bm25_hardneg_df):,} BM25 hard neg çifti...", flush=True)
    tokenizer  = AutoTokenizer.from_pretrained(str(BERT_V23))
    bert_model = AutoModelForSequenceClassification.from_pretrained(str(BERT_V23)).to(device)
    bert_model.eval()
    bert_bm25neg = bert_inference(
        bm25_hardneg_df["term_id"].tolist(),
        bm25_hardneg_df["item_id"].tolist(),
        bert_model, tokenizer, batch=256, desc="bm25neg"
    )
    np.save(str(bm25neg_bert_path), bert_bm25neg)
    print(f"  → {bm25neg_bert_path.name} kaydedildi | {(time.time()-t2)/60:.1f} dk", flush=True)
    del bert_model, tokenizer; gc.collect()
    if device == "cuda": torch.cuda.empty_cache()

print(f"  BM25 neg BERT: {len(bert_bm25neg):,} ✓", flush=True)

# Phase D artık gerekmiyor: bm25_score feature kaldırıldı (tfidf_cos zaten bunu kapsar)

# ─────────────────────────────────────────────────────────────────────
# PHASE E: EĞİTİM VERİSİNİ BİRLEŞTİR
# ─────────────────────────────────────────────────────────────────────
print("\n[E] EĞİTİM VERİSİ BİRLEŞTİRME...", flush=True)

# Hard neg çiftleri mevcut train_df'ye ekle
hardneg_for_concat = bm25_hardneg_df[["term_id","item_id","label"]].copy()

# Orijinal index'i takip et (BERT score dizisini hizalamak için)
train_df["_orig_idx"] = range(len(train_df))
hardneg_for_concat = hardneg_for_concat.copy()
hardneg_for_concat["_orig_idx"] = range(len(train_df), len(train_df)+len(bm25_hardneg_df))

combined_with_idx = pd.concat([
    train_df[["term_id","item_id","label","_orig_idx"]],
    hardneg_for_concat
], ignore_index=True).sort_values("term_id").reset_index(drop=True)

orig_indices = combined_with_idx["_orig_idx"].values

train_df_full = combined_with_idx[["term_id","item_id","label"]].copy()
train_df_full["term_id"] = train_df_full["term_id"].astype(str)
train_df_full["item_id"] = train_df_full["item_id"].astype(str)

# BERT score'ları yeniden sırala
train_bert_full_unsorted = np.concatenate([train_bert_v23, bert_bm25neg])
train_bert_full = train_bert_full_unsorted[orig_indices]

assert len(train_df_full) == len(train_bert_full), \
    f"Boyut uyuşmazlığı: {len(train_df_full)} vs {len(train_bert_full)}"

print(f"  Toplam train: {len(train_df_full):,} | pos={train_df_full.label.sum():,}", flush=True)
print(f"  Hard neg eklendi: +{len(bm25_hardneg_df):,}", flush=True)

del train_df["_orig_idx"]
del train_bert_v23, bert_bm25neg
gc.collect()

# ─────────────────────────────────────────────────────────────────────
# PHASE F: FEATURE ENGINEERING (35 özellik — v23 ile aynı)
# ─────────────────────────────────────────────────────────────────────
print("\n[F] FEATURE ENGINEERING (36 özellik)...", flush=True)
t4 = time.time()

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

X_tr, _ = build_features(train_df_full, tfidf, train_bert_full,
                          fit=True,
                          all_texts=items["title"].tolist()+terms["query"].tolist())
y    = train_df_full["label"].values
tids = train_df_full["term_id"].values
print(f"  Train features: {X_tr[FEATS].shape} | {(time.time()-t4):.0f}s", flush=True)

# ─────────────────────────────────────────────────────────────────────
# PHASE G: LGBM 5-FOLD (GroupKFold by term_id)
# ─────────────────────────────────────────────────────────────────────
print("\n[G] BINARY LGBM 5-FOLD...", flush=True)
t5 = time.time()
oof    = np.zeros(len(train_df_full), dtype=np.float32)
models = []
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
    val_proba = model.predict_proba(X_tr[FEATS].iloc[vli])[:,1]
    oof[vli]  = val_proba.astype(np.float32)
    models.append(model)

    best_f = max(f1_score(y[vli],(val_proba>thr).astype(int),average="macro")
                 for thr in np.arange(0.20,0.90,0.02))
    print(f"  Fold {fold+1} | iter={model.best_iteration_} | val F1≈{best_f:.4f}", flush=True)

# OOF optimal threshold — DOĞRUDAN uygula (quantile değil!)
best_thr, best_f1 = 0.5, 0.0
for thr in np.arange(0.20, 0.90, 0.005):
    s = f1_score(y, (oof>thr).astype(int), average="macro")
    if s>best_f1: best_f1, best_thr = s, thr
print(f"\n  OOF F1={best_f1:.5f} | OOF thr={best_thr:.4f}", flush=True)
print(f"  Süre: {(time.time()-t5)/60:.1f} dk", flush=True)

# ─────────────────────────────────────────────────────────────────────
# PHASE H: TEST INFERENCE + SUBMISSION
# ─────────────────────────────────────────────────────────────────────
print("\n[H] TEST INFERENCE + SUBMISSION...", flush=True)

sub_pairs_local = sub_pairs.copy()
sub_pairs_local["term_id"] = sub_pairs_local["term_id"].astype(str)
sub_pairs_local["item_id"] = sub_pairs_local["item_id"].astype(str)

X_te, _ = build_features(sub_pairs_local, tfidf, test_bert)
test_proba = sum(m.predict_proba(X_te[FEATS])[:,1] for m in models) / len(models)

# Raw test proba kaydet
proba_path = SUBM / "v24_test_proba.npy"
np.save(str(proba_path), test_proba)

# DIRECT THRESHOLD — sabit oran hedeflenmez!
final = (test_proba > best_thr).astype(int)
pos_n = final.sum()
print(f"  Direct thr={best_thr:.4f} → Pozitif={pos_n:,} ({100*pos_n/len(sub_pairs):.1f}%)", flush=True)

out_path = SUBM / "submission_v24_bm25neg.csv"
pd.DataFrame({"id": sub_pairs["id"], "prediction": final}).to_csv(str(out_path), index=False)

toplam_sure = (time.time()-t0)/60
print(f"\n{'='*65}", flush=True)
print(f"TAMAMLANDI — v24 BM25 Hard Negatives", flush=True)
print(f"  OOF F1        : {best_f1:.5f}", flush=True)
print(f"  OOF threshold : {best_thr:.4f} (doğrudan test'e uygulandı)", flush=True)
print(f"  Pozitif       : {pos_n:,} ({100*pos_n/len(sub_pairs):.1f}%)", flush=True)
print(f"  Train çifti   : {len(train_df_full):,} (+{len(bm25_hardneg_df):,} TF-IDF hard neg)", flush=True)
print(f"  Toplam süre   : {toplam_sure:.1f} dk", flush=True)
print(f"  Dosya         : {out_path}", flush=True)
print(f"{'='*65}", flush=True)
