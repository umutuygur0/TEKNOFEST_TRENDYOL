"""
17_lgbm_embedding_v13.py — Phase 2: LightGBM + Embedding Features
===================================================================
Phase 1 (v12, 13 özellik) üzerine eklenenler:
  + embed_cos_sim: multilingual-MiniLM-L12-v2 cosine similarity
  + query_embed_norm, title_embed_norm: embedding büyüklüğü

Neden embedding:
  - TF-IDF kelime tabanlı, embedding anlamsal
  - "spor ayakkabı" ≈ "koşu ayakkabısı" → TF-IDF düşük, embedding yüksek
  - Türkçeyi destekliyor (multilingual model)

Beklenen: ~0.77 (Phase 1: 0.72)
Süre: ~45-60 dk (encode 50K query + 900K title, sonra LightGBM)
"""

import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz import fuzz as rfuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
SUBM = BASE / "claude only" / "submissions"
CACHE = BASE / "claude only" / "emb_cache_v13"
CACHE.mkdir(parents=True, exist_ok=True)
SUBM.mkdir(parents=True, exist_ok=True)

LOWER_MAP = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(t): return str(t).translate(LOWER_MAP).lower().strip()

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_BATCH = 512

# ─── 1. Veri ─────────────────────────────────────────────────────────────────
print("[1] Veri yükleniyor...", flush=True)
t0 = time.time()
items       = pd.read_csv(DATA / "items.csv")
terms       = pd.read_csv(DATA / "terms.csv")
train_pairs = pd.read_csv(DATA / "training_pairs.csv")
sub_pairs   = pd.read_csv(DATA / "submission_pairs.csv")
sample      = pd.read_csv(DATA / "sample_submission.csv")

for col in ["title","category","brand","gender","age_group","attributes"]:
    items[col] = items[col].fillna("unknown").apply(trl)
terms["query"] = terms["query"].fillna("").apply(trl)
print(f"  {time.time()-t0:.1f}s", flush=True)

# ─── 2. Transductive Hard Negatives ──────────────────────────────────────────
print("\n[2] Hard negatives...", flush=True)
t1 = time.time()
pool = sub_pairs.merge(terms, on="term_id", how="left").merge(items, on="item_id", how="left")
for col in ["query","title","gender","category"]:
    pool[col] = pool[col].fillna("unknown").apply(trl)

mask_gender = (
    (pool["query"].str.contains("erkek") & (pool["gender"] == "kadın")) |
    (pool["query"].str.contains("kadın") & (pool["gender"] == "erkek"))
)
pool["is_unrelated"] = [
    len(set(q.split()) & set((t+" "+c.replace("/", " ")).split())) == 0
    for q, t, c in zip(pool["query"], pool["title"], pool["category"])
]
hard_neg = pool[mask_gender | pool["is_unrelated"]].sample(n=250_000, random_state=42)[["id","term_id","item_id"]].copy()
hard_neg["label"] = 0
train_pairs["label"] = 1
train_ready = pd.concat([train_pairs[["id","term_id","item_id","label"]], hard_neg],
                         ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"  {len(train_ready):,} çift | {time.time()-t1:.1f}s", flush=True)

# ─── 3. Metin Hazırlama ───────────────────────────────────────────────────────
print("\n[3] Metin alanları ekleniyor...", flush=True)
df = train_ready.merge(terms, on="term_id", how="left").merge(items, on="item_id", how="left")
for col in ["query","title","brand","category","gender","age_group","attributes"]:
    df[col] = df[col].fillna("unknown").apply(trl)

# ─── 4. Embedding Encode (Cache'li) ──────────────────────────────────────────
print("\n[4] Embedding modeli yükleniyor...", flush=True)
t1 = time.time()
embedder = SentenceTransformer(EMBED_MODEL)
print(f"  Model yüklendi: {time.time()-t1:.1f}s", flush=True)

def get_embeddings(texts, cache_file, batch=EMBED_BATCH):
    if cache_file.exists():
        print(f"  Cache yükleniyor: {cache_file.name}", flush=True)
        return np.load(str(cache_file))
    print(f"  Encode ediliyor: {len(texts):,} metin...", flush=True)
    t = time.time()
    embs = embedder.encode(texts, batch_size=batch, show_progress_bar=True,
                           normalize_embeddings=True, convert_to_numpy=True)
    np.save(str(cache_file), embs)
    print(f"  {time.time()-t:.1f}s | {cache_file.name}", flush=True)
    return embs

# Unique query ve title'ları encode et (her birini bir kez)
unique_queries = terms["query"].tolist()
unique_titles  = items["title"].tolist()
print(f"\n  Unique query: {len(unique_queries):,} | Unique title: {len(unique_titles):,}", flush=True)

q_embs = get_embeddings(unique_queries, CACHE / "query_embs.npy")
t_embs = get_embeddings(unique_titles,  CACHE / "title_embs.npy")

# Lookup dict
tid_to_qidx = {tid: i for i, tid in enumerate(terms["term_id"])}
iid_to_tidx = {iid: i for i, iid in enumerate(items["item_id"])}

def embed_cos_batch(term_ids, item_ids):
    q_idx = np.array([tid_to_qidx.get(tid, 0) for tid in term_ids])
    t_idx = np.array([iid_to_tidx.get(iid, 0) for iid in item_ids])
    q = q_embs[q_idx]
    t = t_embs[t_idx]
    return (q * t).sum(axis=1).astype(np.float32)

print("\n  Eğitim embedding cosine hesaplanıyor...", flush=True)
df["embed_cos_sim"] = embed_cos_batch(df["term_id"].tolist(), df["item_id"].tolist())
print(f"  embed_cos_sim train: {df['embed_cos_sim'].describe().to_dict()}", flush=True)

# ─── 5. Diğer Özellikler ─────────────────────────────────────────────────────
print("\n[5] Özellik Mühendisliği...", flush=True)
t1 = time.time()

def tfidf_cos_batch(queries, titles, vectorizer, chunk=50_000):
    n = len(queries)
    sims = np.zeros(n, dtype=np.float32)
    for i in range(0, n, chunk):
        q_mat = vectorizer.transform(queries[i:i+chunk])
        t_mat = vectorizer.transform(titles[i:i+chunk])
        q_n = normalize(q_mat, norm="l2")
        t_n = normalize(t_mat, norm="l2")
        sims[i:i+chunk] = np.array(q_n.multiply(t_n).sum(axis=1)).flatten()
    return sims

def jaccard(a, b):
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

tfidf_vect = TfidfVectorizer(ngram_range=(1,2), max_features=50_000, sublinear_tf=True, min_df=2)
all_texts = pd.concat([df["title"], df["query"],
                       pd.Series(items["title"].tolist() + terms["query"].tolist())])
tfidf_vect.fit(all_texts)

qs = df["query"].tolist(); ts = df["title"].tolist()
cats = df["category"].tolist(); brands = df["brand"].tolist()
genders = df["gender"].tolist(); ages = df["age_group"].tolist()
attrs = df["attributes"].tolist()

df["tfidf_cos_sim"]      = tfidf_cos_batch(qs, ts, tfidf_vect)
df["jaccard_sim"]        = [jaccard(q, t) for q, t in zip(qs, ts)]
df["fuzz_token_set"]     = [rfuzz.token_set_ratio(q, t) / 100.0 for q, t in zip(qs, ts)]
df["fuzz_partial"]       = [rfuzz.partial_ratio(q, t) / 100.0 for q, t in zip(qs, ts)]
df["fuzz_token_sort"]    = [rfuzz.token_sort_ratio(q, t) / 100.0 for q, t in zip(qs, ts)]
df["fuzz_basic"]         = [rfuzz.ratio(q, t) / 100.0 for q, t in zip(qs, ts)]
df["category_overlap"]   = [jaccard(q, c.replace("/", " ")) for q, c in zip(qs, cats)]
df["is_brand_in_query"]  = [1 if b != "unknown" and b in q else 0 for q, b in zip(qs, brands)]
df["is_gender_in_query"] = [1 if g in q else 0 for q, g in zip(qs, genders)]
df["is_age_in_query"]    = [1 if a not in ("unknown","") and a in q else 0 for q, a in zip(qs, ages)]
df["attr_match_score"]   = [jaccard(q, a) for q, a in zip(qs, attrs)]
df["len_diff"]           = [abs(len(q) - len(t)) for q, t in zip(qs, ts)]
df["ana_kategori"]       = pd.Categorical(df["category"].apply(lambda x: x.split("/")[0]))

print(f"  {time.time()-t1:.1f}s", flush=True)

# ─── 6. LightGBM ─────────────────────────────────────────────────────────────
FEATURES = [
    "embed_cos_sim", "tfidf_cos_sim",
    "jaccard_sim", "fuzz_token_set", "fuzz_partial", "fuzz_token_sort", "fuzz_basic",
    "category_overlap", "is_brand_in_query", "is_gender_in_query",
    "is_age_in_query", "attr_match_score", "len_diff", "ana_kategori"
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

print(f"  {time.time()-t1:.1f}s", flush=True)

print("\n[7] Threshold optimizasyonu...", flush=True)
best_thresh, best_f1 = 0.5, 0.0
for thresh in np.arange(0.3, 0.71, 0.01):
    s = f1_score(y, (oof_preds > thresh).astype(int), average="macro")
    if s > best_f1: best_f1, best_thresh = s, thresh
print(f"  Threshold: {best_thresh:.2f} | OOF F1: {best_f1:.5f}", flush=True)

# ─── 7. Test Inference ───────────────────────────────────────────────────────
print(f"\n[8] Test inference ({len(sub_pairs):,} çift)...", flush=True)
t1 = time.time()
test_df = sub_pairs.merge(terms, on="term_id", how="left").merge(items, on="item_id", how="left")
for col in ["query","title","brand","category","gender","age_group","attributes"]:
    test_df[col] = test_df[col].fillna("unknown").apply(trl)

test_df["embed_cos_sim"] = embed_cos_batch(test_df["term_id"].tolist(), test_df["item_id"].tolist())
test_df["tfidf_cos_sim"] = tfidf_cos_batch(test_df["query"].tolist(), test_df["title"].tolist(), tfidf_vect)

tq = test_df["query"].tolist(); tt = test_df["title"].tolist()
test_df["jaccard_sim"]       = [jaccard(q, t) for q, t in zip(tq, tt)]
test_df["fuzz_token_set"]    = [rfuzz.token_set_ratio(q, t)/100.0 for q, t in zip(tq, tt)]
test_df["fuzz_partial"]      = [rfuzz.partial_ratio(q, t)/100.0 for q, t in zip(tq, tt)]
test_df["fuzz_token_sort"]   = [rfuzz.token_sort_ratio(q, t)/100.0 for q, t in zip(tq, tt)]
test_df["fuzz_basic"]        = [rfuzz.ratio(q, t)/100.0 for q, t in zip(tq, tt)]
test_df["category_overlap"]  = [jaccard(q, c.replace("/", " ")) for q, c in zip(tq, test_df["category"].tolist())]
test_df["is_brand_in_query"] = [1 if b != "unknown" and b in q else 0 for q, b in zip(tq, test_df["brand"].tolist())]
test_df["is_gender_in_query"]= [1 if g in q else 0 for q, g in zip(tq, test_df["gender"].tolist())]
test_df["is_age_in_query"]   = [1 if a not in ("unknown","") and a in q else 0 for q, a in zip(tq, test_df["age_group"].tolist())]
test_df["attr_match_score"]  = [jaccard(q, a) for q, a in zip(tq, test_df["attributes"].tolist())]
test_df["len_diff"]          = [abs(len(q)-len(t)) for q, t in zip(tq, tt)]
test_df["ana_kategori"]      = pd.Categorical(test_df["category"].apply(lambda x: x.split("/")[0]))

X_test = test_df[FEATURES]
test_preds = sum(m.predict_proba(X_test)[:, 1] for m in models) / len(models)
final = (test_preds > best_thresh).astype(int)

out = SUBM / "submission_v13_lgbm_embedding.csv"
pd.DataFrame({"id": test_df["id"], "prediction": final}).to_csv(str(out), index=False)

pos = final.sum()
print(f"\n{'='*60}", flush=True)
print(f"TAMAMLANDI — v13 LightGBM + Embedding", flush=True)
print(f"  OOF Macro F1  : {best_f1:.5f}", flush=True)
print(f"  Threshold     : {best_thresh:.2f}", flush=True)
print(f"  Pozitif       : {pos:,} ({100*pos/len(final):.1f}%)", flush=True)
print(f"  Toplam süre   : {(time.time()-t0)/60:.1f} dk", flush=True)
print(f"  Dosya         : {out}", flush=True)
print(f"{'='*60}", flush=True)
