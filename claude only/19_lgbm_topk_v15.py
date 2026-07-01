"""
19_lgbm_topk_v15.py — Phase: Per-Query Top-K + Zengin Özellikler
=================================================================
TEŞHIS: 1.49M pozitif (%44) tahmin ediyoruz ama gerçek oran ~%13-14
NEDEN: Global threshold (0.44) calibration sorunu
FIX:
  1. Per-query Top-K inference (her sorgu için en iyi K item = pozitif)
     K = train setindeki ortalama pozitif sayısı (~14)
  2. Yeni zengin özellikler:
     - query_word_cov_in_title: query kelimelerinin % kaçı title'da?
     - title_word_cov_in_query: title kelimelerinin % kaçı query'de?
     - query_exact_in_title: query tam olarak title içinde var mı?
     - l2_cat_overlap: L2 kategori ile query benzerliği
     - query_is_brand: query büyük ihtimalle marka mı?
     - token_overlap_count: kaç kelime ortak?

Beklenen: 0.68 → 0.74+ (pos rate %44 → %13 düzeltmesi büyük etki yapabilir)
"""

import sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz import fuzz as rfuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

sys.stdout.reconfigure(encoding="utf-8")

BASE  = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA  = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
SUBM  = BASE / "claude only" / "submissions"
CACHE = BASE / "claude only" / "emb_cache_v13"
SUBM.mkdir(parents=True, exist_ok=True)

LOWER_MAP = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(t): return str(t).translate(LOWER_MAP).lower().strip()

# ─── 1. Veri ─────────────────────────────────────────────────────────────────
print("[1] Veri yükleniyor...", flush=True)
t0 = time.time()
items       = pd.read_csv(DATA / "items.csv")
terms       = pd.read_csv(DATA / "terms.csv")
train_pairs = pd.read_csv(DATA / "training_pairs.csv")
sub_pairs   = pd.read_csv(DATA / "submission_pairs.csv")

for col in ["title","category","brand","gender","age_group","attributes"]:
    items[col] = items[col].fillna("unknown").apply(trl)
terms["query"] = terms["query"].fillna("").apply(trl)

# Eğitimdeki K dağılımını öğren
pos_per_query = train_pairs.groupby("term_id").size()
K_MEAN   = pos_per_query.mean()
K_MEDIAN = pos_per_query.median()
K_P75    = pos_per_query.quantile(0.75)
print(f"  Train: K_mean={K_MEAN:.1f} | K_median={K_MEDIAN:.0f} | K_p75={K_P75:.0f}", flush=True)

# Bilinen marka listesi
brand_set = set(items["brand"].str.lower().unique()) - {"unknown","nan",""}
known_brands_sorted = sorted(brand_set, key=len, reverse=True)

# ─── 2. Hard Negatives (transductive + ayni-kategori) ────────────────────────
print("\n[2] Negatifler üretiliyor...", flush=True)
t1 = time.time()

# (a) Gender mismatch — transductive
pool = sub_pairs.merge(terms, on="term_id", how="left").merge(items, on="item_id", how="left")
for col in ["query","title","gender","category"]:
    pool[col] = pool[col].fillna("unknown").apply(trl)

mask_gender = (
    (pool["query"].str.contains("erkek") & (pool["gender"] == "kadın")) |
    (pool["query"].str.contains("kadın") & (pool["gender"] == "erkek"))
)
mask_zero = [
    len(set(q.split()) & set((t+" "+c.replace("/", " ")).split())) == 0
    for q, t, c in zip(pool["query"], pool["title"], pool["category"])
]
pool["mask_zero"] = mask_zero
hard_neg_trans = pool[mask_gender | pool["mask_zero"]][["id","term_id","item_id"]].copy()
hard_neg_trans["label"] = 0

# (b) Aynı L1 kategoriden rasgele (modele gerçek hard case göster)
pos_items_per_tid = defaultdict(set)
for tid, iid in zip(train_pairs["term_id"], train_pairs["item_id"]):
    pos_items_per_tid[tid].add(iid)

tid_to_query = dict(zip(terms["term_id"], terms["query"]))
iid_to_cat   = dict(zip(items["item_id"], items["category"]))
iid_to_l1    = {iid: cat.split("/")[0] for iid, cat in iid_to_cat.items()}

# Pozitif item'ların L1 kategorilerini bul → aynı L1'den negatif seç
cat_to_items = defaultdict(list)
for iid, l1 in iid_to_l1.items():
    cat_to_items[l1].append(iid)

same_cat_neg_rows = []
rng = np.random.default_rng(42)
for tid in list(pos_items_per_tid.keys()):
    pos_iids = pos_items_per_tid[tid]
    l1_cats = {iid_to_l1.get(iid,"") for iid in pos_iids if iid_to_l1.get(iid,"")}
    for l1 in l1_cats:
        candidates = cat_to_items.get(l1, [])
        neg_cands  = [i for i in candidates if i not in pos_iids]
        if not neg_cands: continue
        chosen = rng.choice(neg_cands, size=min(5, len(neg_cands)), replace=False)
        for iid in chosen:
            same_cat_neg_rows.append({"term_id": tid, "item_id": iid, "label": 0})

same_cat_df = pd.DataFrame(same_cat_neg_rows)
print(f"  Gender/zero neg: {len(hard_neg_trans):,} | Same-cat neg: {len(same_cat_df):,}", flush=True)

# Karıştır ve örnekle
TARGET = 250_000
neg_trans = hard_neg_trans.sample(n=min(100_000, len(hard_neg_trans)), random_state=42)
neg_scat  = same_cat_df.sample(n=min(150_000, len(same_cat_df)), random_state=42)

train_pairs["label"] = 1
train_ready = pd.concat([
    train_pairs[["term_id","item_id","label"]],
    neg_trans[["term_id","item_id","label"]],
    neg_scat[["term_id","item_id","label"]]
], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  Toplam: {len(train_ready):,} | pos={train_ready['label'].sum():,} | "
      f"neg={(train_ready['label']==0).sum():,}", flush=True)

# ─── 3. Özellikler ────────────────────────────────────────────────────────────
print("\n[3] Metin hazırlama...", flush=True)
df = train_ready.merge(terms, on="term_id", how="left").merge(items, on="item_id", how="left")
for col in ["query","title","brand","category","gender","age_group","attributes"]:
    df[col] = df[col].fillna("unknown").apply(trl)

print("[4] TF-IDF...", flush=True)
tfidf_vect = TfidfVectorizer(ngram_range=(1,2), max_features=50_000, sublinear_tf=True, min_df=2)
tfidf_vect.fit(pd.concat([df["title"], df["query"],
               pd.Series(items["title"].tolist() + terms["query"].tolist())]))

def tfidf_cos(qs, ts, vect, chunk=50_000):
    n = len(qs); out = np.zeros(n, dtype=np.float32)
    for i in range(0, n, chunk):
        qm = normalize(vect.transform(qs[i:i+chunk]), "l2")
        tm = normalize(vect.transform(ts[i:i+chunk]), "l2")
        out[i:i+chunk] = np.array(qm.multiply(tm).sum(axis=1)).flatten()
    return out

def jaccard(a, b):
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def query_cov_in_title(q, t):
    """query kelimelerinin kaçı title'da? (directional)"""
    qw = set(q.split())
    if not qw: return 0.0
    tw = set(t.split())
    return len(qw & tw) / len(qw)

def title_cov_in_query(q, t):
    """title kelimelerinin kaçı query'de?"""
    tw = set(t.split())
    if not tw: return 0.0
    qw = set(q.split())
    return len(tw & qw) / len(tw)

def is_brand_query(q, brand_set, known_brands):
    """Query büyük ihtimalle marka içeriyor mu?"""
    for b in known_brands:
        if len(b) >= 3 and b in q:
            return 1
    return 0

print("[5] Özellikler hesaplanıyor...", flush=True)
t1 = time.time()
qs = df["query"].tolist(); ts = df["title"].tolist()
cats = df["category"].tolist(); brands = df["brand"].tolist()
genders = df["gender"].tolist(); ages = df["age_group"].tolist()
attrs = df["attributes"].tolist()

df["tfidf_cos_sim"]      = tfidf_cos(qs, ts, tfidf_vect)
df["jaccard_sim"]        = [jaccard(q, t) for q, t in zip(qs, ts)]
df["fuzz_token_set"]     = [rfuzz.token_set_ratio(q, t)/100.0 for q, t in zip(qs, ts)]
df["fuzz_partial"]       = [rfuzz.partial_ratio(q, t)/100.0 for q, t in zip(qs, ts)]
df["fuzz_token_sort"]    = [rfuzz.token_sort_ratio(q, t)/100.0 for q, t in zip(qs, ts)]
df["fuzz_basic"]         = [rfuzz.ratio(q, t)/100.0 for q, t in zip(qs, ts)]
df["category_overlap"]   = [jaccard(q, c.replace("/", " ")) for q, c in zip(qs, cats)]
df["l2_cat_overlap"]     = [jaccard(q, "/".join(c.split("/")[:2]).replace("/", " "))
                             for q, c in zip(qs, cats)]
df["is_brand_in_query"]  = [1 if b != "unknown" and b in q else 0 for q, b in zip(qs, brands)]
df["is_gender_in_query"] = [1 if g in q else 0 for q, g in zip(qs, genders)]
df["is_age_in_query"]    = [1 if a not in ("unknown","") and a in q else 0 for q, a in zip(qs, ages)]
df["attr_match_score"]   = [jaccard(q, a) for q, a in zip(qs, attrs)]
df["len_diff"]           = [abs(len(q) - len(t)) for q, t in zip(qs, ts)]
# YENİ özellikler
df["query_cov_in_title"] = [query_cov_in_title(q, t) for q, t in zip(qs, ts)]
df["title_cov_in_query"] = [title_cov_in_query(q, t) for q, t in zip(qs, ts)]
df["query_exact_in_title"] = [1 if q in t else 0 for q, t in zip(qs, ts)]
df["token_overlap_count"]  = [len(set(q.split()) & set(t.split())) for q, t in zip(qs, ts)]
df["query_len"]            = [len(q.split()) for q in qs]
df["ana_kategori"]         = pd.Categorical(df["category"].apply(lambda x: x.split("/")[0]))

print(f"  {time.time()-t1:.1f}s", flush=True)

# ─── 6. LightGBM ─────────────────────────────────────────────────────────────
FEATURES = [
    "tfidf_cos_sim", "jaccard_sim", "fuzz_token_set", "fuzz_partial",
    "fuzz_token_sort", "fuzz_basic", "category_overlap", "l2_cat_overlap",
    "is_brand_in_query", "is_gender_in_query", "is_age_in_query",
    "attr_match_score", "len_diff",
    "query_cov_in_title", "title_cov_in_query",
    "query_exact_in_title", "token_overlap_count", "query_len",
    "ana_kategori"
]

print(f"\n[6] LightGBM ({len(df):,} çift, {len(FEATURES)} özellik)...", flush=True)
t1 = time.time()
X, y, groups = df[FEATURES], df["label"], df["term_id"]
oof_preds = np.zeros(len(df))
models = []
gkf = GroupKFold(n_splits=5)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, max_depth=7,
                           num_leaves=128, random_state=42, class_weight="balanced",
                           n_jobs=-1, verbose=-1)
    m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
          eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
          categorical_feature=["ana_kategori"],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    oof_preds[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]
    models.append(m)
    f = f1_score(y.iloc[val_idx], (oof_preds[val_idx] > 0.5).astype(int), average="macro")
    print(f"  Fold {fold+1} | best_iter={m.best_iteration_:4d} | val F1={f:.4f}", flush=True)

# Threshold (global, referans için)
best_thresh, best_f1 = 0.5, 0.0
for thr in np.arange(0.3, 0.71, 0.01):
    s = f1_score(y, (oof_preds > thr).astype(int), average="macro")
    if s > best_f1: best_f1, best_thresh = s, thr
print(f"  Global threshold: {best_thresh:.2f} | OOF F1: {best_f1:.5f}", flush=True)
print(f"  {time.time()-t1:.1f}s", flush=True)

imp = pd.DataFrame({"Feature": FEATURES, "Importance": models[0].feature_importances_}) \
        .sort_values("Importance", ascending=False)
print("\n  Özellik Önemleri:")
for _, row in imp.iterrows():
    print(f"    {row['Feature']:25s}: {int(row['Importance']):5d}", flush=True)

# ─── 7. Test Inference — PER-QUERY TOP-K ────────────────────────────────────
print(f"\n[7] Test inference — PER-QUERY Top-K ({len(sub_pairs):,} çift)...", flush=True)
t1 = time.time()
test_df = sub_pairs.merge(terms, on="term_id", how="left").merge(items, on="item_id", how="left")
for col in ["query","title","brand","category","gender","age_group","attributes"]:
    test_df[col] = test_df[col].fillna("unknown").apply(trl)

tq = test_df["query"].tolist(); tt = test_df["title"].tolist()
tc = test_df["category"].tolist()

test_df["tfidf_cos_sim"]      = tfidf_cos(tq, tt, tfidf_vect)
test_df["jaccard_sim"]        = [jaccard(q, t) for q, t in zip(tq, tt)]
test_df["fuzz_token_set"]     = [rfuzz.token_set_ratio(q, t)/100.0 for q, t in zip(tq, tt)]
test_df["fuzz_partial"]       = [rfuzz.partial_ratio(q, t)/100.0 for q, t in zip(tq, tt)]
test_df["fuzz_token_sort"]    = [rfuzz.token_sort_ratio(q, t)/100.0 for q, t in zip(tq, tt)]
test_df["fuzz_basic"]         = [rfuzz.ratio(q, t)/100.0 for q, t in zip(tq, tt)]
test_df["category_overlap"]   = [jaccard(q, c.replace("/", " ")) for q, c in zip(tq, tc)]
test_df["l2_cat_overlap"]     = [jaccard(q, "/".join(c.split("/")[:2]).replace("/", " ")) for q, c in zip(tq, tc)]
test_df["is_brand_in_query"]  = [1 if b != "unknown" and b in q else 0 for q, b in zip(tq, test_df["brand"].tolist())]
test_df["is_gender_in_query"] = [1 if g in q else 0 for q, g in zip(tq, test_df["gender"].tolist())]
test_df["is_age_in_query"]    = [1 if a not in ("unknown","") and a in q else 0 for q, a in zip(tq, test_df["age_group"].tolist())]
test_df["attr_match_score"]   = [jaccard(q, a) for q, a in zip(tq, test_df["attributes"].tolist())]
test_df["len_diff"]           = [abs(len(q)-len(t)) for q, t in zip(tq, tt)]
test_df["query_cov_in_title"] = [query_cov_in_title(q, t) for q, t in zip(tq, tt)]
test_df["title_cov_in_query"] = [title_cov_in_query(q, t) for q, t in zip(tq, tt)]
test_df["query_exact_in_title"] = [1 if q in t else 0 for q, t in zip(tq, tt)]
test_df["token_overlap_count"]  = [len(set(q.split()) & set(t.split())) for q, t in zip(tq, tt)]
test_df["query_len"]            = [len(q.split()) for q in tq]
test_df["ana_kategori"]         = pd.Categorical(test_df["category"].apply(lambda x: x.split("/")[0]))

X_test = test_df[FEATURES]
test_scores = sum(m.predict_proba(X_test)[:, 1] for m in models) / len(models)
test_df["score"] = test_scores

print(f"  Fuzz+özellik+ensemble: {time.time()-t1:.1f}s", flush=True)

# ── Per-query Top-K inference ──
# K = training'deki ortalama pozitif sayısı per query
K_DEFAULT = int(round(K_MEAN))  # ~14
print(f"\n  Per-query Top-K inference (K={K_DEFAULT})...", flush=True)

test_df["pred_topk"] = 0
for tid, grp in test_df.groupby("term_id"):
    k = K_DEFAULT
    top_idx = grp["score"].nlargest(k).index
    test_df.loc[top_idx, "pred_topk"] = 1

# Global threshold versiyonu da kaydet (karşılaştırma için)
test_df["pred_global"] = (test_scores > best_thresh).astype(int)

pos_topk   = test_df["pred_topk"].sum()
pos_global = test_df["pred_global"].sum()

print(f"\n  Top-K  pos: {pos_topk:,} ({100*pos_topk/len(test_df):.1f}%)", flush=True)
print(f"  Global pos: {pos_global:,} ({100*pos_global/len(test_df):.1f}%)", flush=True)

# İki submission kaydet
out_topk   = SUBM / "submission_v15_topk.csv"
out_global = SUBM / "submission_v15_global.csv"
test_df[["id","pred_topk"]].rename(columns={"pred_topk":"prediction"}).to_csv(str(out_topk), index=False)
test_df[["id","pred_global"]].rename(columns={"pred_global":"prediction"}).to_csv(str(out_global), index=False)

print(f"\n{'='*60}", flush=True)
print(f"TAMAMLANDI — v15 Per-Query Top-K", flush=True)
print(f"  OOF F1 (global thr) : {best_f1:.5f}", flush=True)
print(f"  Top-K={K_DEFAULT} pozitif  : {pos_topk:,} ({100*pos_topk/len(test_df):.1f}%)", flush=True)
print(f"  Global pozitif       : {pos_global:,} ({100*pos_global/len(test_df):.1f}%)", flush=True)
print(f"  Toplam süre          : {(time.time()-t0)/60:.1f} dk", flush=True)
print(f"  Top-K dosya          : {out_topk}", flush=True)
print(f"  Global dosya         : {out_global}", flush=True)
print(f"{'='*60}", flush=True)
