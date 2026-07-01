"""
18_lgbm_hardneg_v14.py — Phase 3: BM25 Hard Negative Mining
=============================================================
TEMEL PROBLEM: OOF=0.95 ama test=0.68 → 0.27 uçurum
NEDEN: Eğitim negativleri çok kolay (gender mismatch, sıfır overlap)
       → Model hard case'leri görmüyor → test'te başarısız

ÇÖZüM: BM25 Hard Negatives
  Her pozitif çift (query, rel_item) için:
  1. Query'e TF-IDF ile en benzer 20 item bul
  2. training_pairs'te pozitif olmayan → "hard negative"
  3. Bunlar gerçekten zor: query'e benzer ama alakasız

Beklenen: OOF düşer (0.95→0.80) ama TEST SCORE ARTAR (0.68→0.78+)
Çünkü artık model gerçek hard case'leri görüyor.
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

iid_list  = items["item_id"].tolist()
iid_index = {iid: i for i, iid in enumerate(iid_list)}
tid_to_q  = dict(zip(terms["term_id"], terms["query"]))

# Pozitif küme: term_id → set of item_ids
from collections import defaultdict
pos_set = defaultdict(set)
for _, row in train_pairs.iterrows():
    pos_set[row["term_id"]].add(row["item_id"])

print(f"  {time.time()-t0:.1f}s | {len(train_pairs):,} pozitif | {len(pos_set):,} unique query", flush=True)

# ─── 2. BM25 Hard Negative Mining ────────────────────────────────────────────
print("\n[2] BM25 Hard Negative Mining...", flush=True)
print("    TF-IDF indeksi oluşturuluyor (tüm item title+cat)...", flush=True)
t1 = time.time()

item_texts = [trl(t + " " + c.replace("/", " "))
              for t, c in zip(items["title"], items["category"])]

tfidf_item = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4),
                              max_features=150_000, sublinear_tf=True, min_df=2)
item_mat = tfidf_item.fit_transform(item_texts)
print(f"  İndeks hazır: {item_mat.shape} | {time.time()-t1:.1f}s", flush=True)

print("  Her query için top-20 hard negative seçiliyor...", flush=True)
t1 = time.time()

all_tids = list(pos_set.keys())
hard_neg_rows = []
TOPK = 20

for i, tid in enumerate(all_tids):
    q_text   = tid_to_q.get(tid, "")
    if not q_text: continue
    pos_iids = pos_set[tid]

    q_vec = tfidf_item.transform([q_text])
    sims  = (q_vec * item_mat.T).toarray()[0]

    # En yüksek benzerlikli ama POZİTİF OLMAYAN item'lar
    top_idx = np.argpartition(sims, -TOPK*3)[-TOPK*3:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    count = 0
    for idx in top_idx:
        iid = iid_list[idx]
        if iid not in pos_iids and sims[idx] > 0.01:
            hard_neg_rows.append({
                "term_id": tid,
                "item_id": iid,
                "label":   0,
                "neg_sim": float(sims[idx])
            })
            count += 1
            if count >= TOPK: break

    if i % 2000 == 0 and i > 0:
        print(f"  {i:5d}/{len(all_tids)} | {len(hard_neg_rows):,} hard neg | {time.time()-t1:.0f}s", flush=True)

hard_neg_df = pd.DataFrame(hard_neg_rows)
print(f"  Toplam hard neg: {len(hard_neg_df):,} | {time.time()-t1:.1f}s", flush=True)

# Transductive easy negatives (ek veri noktaları için)
print("\n  + Transductive negatives (tamamlayıcı)...", flush=True)
pool = sub_pairs.merge(terms, on="term_id", how="left").merge(items, on="item_id", how="left")
for col in ["query","title","gender","category"]:
    pool[col] = pool[col].fillna("unknown").apply(trl)
mask_gender = (
    (pool["query"].str.contains("erkek") & (pool["gender"] == "kadın")) |
    (pool["query"].str.contains("kadın") & (pool["gender"] == "erkek"))
)
trans_neg = pool[mask_gender][["term_id","item_id"]].copy()
trans_neg["label"] = 0
trans_neg["neg_sim"] = 0.0
print(f"  Transductive gender neg: {len(trans_neg):,}", flush=True)

# Eğitim seti: 250K pos + hard neg + tamamlayıcı easy neg
TARGET_HARD = min(len(hard_neg_df), 350_000)
TARGET_EASY = 150_000

neg_hard   = hard_neg_df.sample(n=TARGET_HARD, random_state=42) if len(hard_neg_df) > TARGET_HARD else hard_neg_df
neg_easy   = trans_neg.sample(n=min(TARGET_EASY, len(trans_neg)), random_state=42)

train_pairs["label"] = 1
train_pairs["neg_sim"] = 1.0
all_pos = train_pairs[["term_id","item_id","label","neg_sim"]].copy()

train_ready = pd.concat([
    all_pos,
    neg_hard[["term_id","item_id","label","neg_sim"]],
    neg_easy[["term_id","item_id","label","neg_sim"]]
], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n  Eğitim seti: {len(train_ready):,} | pos={train_ready['label'].sum():,} "
      f"| hard_neg={TARGET_HARD:,} | easy_neg={len(neg_easy):,}", flush=True)

# ─── 3. Metin + Özellikler ────────────────────────────────────────────────────
print("\n[3] Metin alanları ekleniyor...", flush=True)
df = train_ready.merge(terms, on="term_id", how="left").merge(items, on="item_id", how="left")
for col in ["query","title","brand","category","gender","age_group","attributes"]:
    df[col] = df[col].fillna("unknown").apply(trl)

# Embedding (cache kullan)
print("\n[4] Embedding features...", flush=True)
emb_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def get_embs(texts, cache_file):
    if cache_file.exists():
        return np.load(str(cache_file))
    embs = emb_model.encode(texts, batch_size=512, normalize_embeddings=True,
                            show_progress_bar=True, convert_to_numpy=True)
    np.save(str(cache_file), embs)
    return embs

q_embs = get_embs(terms["query"].tolist(), CACHE / "query_embs.npy")
t_embs = get_embs(items["title"].tolist(), CACHE / "title_embs.npy")
tid_to_qi = {tid: i for i, tid in enumerate(terms["term_id"])}
iid_to_ti = {iid: i for i, iid in enumerate(items["item_id"])}

def embed_cos(term_ids, item_ids):
    qi = np.array([tid_to_qi.get(t, 0) for t in term_ids])
    ti = np.array([iid_to_ti.get(i, 0) for i in item_ids])
    return (q_embs[qi] * t_embs[ti]).sum(axis=1).astype(np.float32)

df["embed_cos_sim"] = embed_cos(df["term_id"].tolist(), df["item_id"].tolist())

# TF-IDF cosine
print("\n[5] TF-IDF + fuzz...", flush=True)
t1 = time.time()

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

qs = df["query"].tolist(); ts = df["title"].tolist()

df["tfidf_cos_sim"]      = tfidf_cos(qs, ts, tfidf_vect)
df["jaccard_sim"]        = [jaccard(q, t) for q, t in zip(qs, ts)]
df["fuzz_token_set"]     = [rfuzz.token_set_ratio(q, t)/100.0 for q, t in zip(qs, ts)]
df["fuzz_partial"]       = [rfuzz.partial_ratio(q, t)/100.0 for q, t in zip(qs, ts)]
df["fuzz_token_sort"]    = [rfuzz.token_sort_ratio(q, t)/100.0 for q, t in zip(qs, ts)]
df["fuzz_basic"]         = [rfuzz.ratio(q, t)/100.0 for q, t in zip(qs, ts)]
df["category_overlap"]   = [jaccard(q, c.replace("/", " ")) for q, c in zip(qs, df["category"].tolist())]
df["is_brand_in_query"]  = [1 if b != "unknown" and b in q else 0 for q, b in zip(qs, df["brand"].tolist())]
df["is_gender_in_query"] = [1 if g in q else 0 for q, g in zip(qs, df["gender"].tolist())]
df["is_age_in_query"]    = [1 if a not in ("unknown","") and a in q else 0 for q, a in zip(qs, df["age_group"].tolist())]
df["attr_match_score"]   = [jaccard(q, a) for q, a in zip(qs, df["attributes"].tolist())]
df["len_diff"]           = [abs(len(q)-len(t)) for q, t in zip(qs, ts)]
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
print("  ! OOF düştüyse NORMAL — hard neg var, model daha zorlu örnekler görüyor", flush=True)

imp = pd.DataFrame({"Feature": FEATURES, "Importance": models[0].feature_importances_}) \
        .sort_values("Importance", ascending=False)
print("\n  Özellik Önemleri:")
for _, row in imp.iterrows():
    print(f"    {row['Feature']:25s}: {int(row['Importance']):5d}", flush=True)

# ─── 7. Test Inference ───────────────────────────────────────────────────────
print(f"\n[8] Test inference ({len(sub_pairs):,} çift)...", flush=True)
t1 = time.time()
test_df = sub_pairs.merge(terms, on="term_id", how="left").merge(items, on="item_id", how="left")
for col in ["query","title","brand","category","gender","age_group","attributes"]:
    test_df[col] = test_df[col].fillna("unknown").apply(trl)

test_df["embed_cos_sim"] = embed_cos(test_df["term_id"].tolist(), test_df["item_id"].tolist())
tq = test_df["query"].tolist(); tt = test_df["title"].tolist()
test_df["tfidf_cos_sim"]     = tfidf_cos(tq, tt, tfidf_vect)
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

out = SUBM / "submission_v14_lgbm_hardneg.csv"
pd.DataFrame({"id": test_df["id"], "prediction": final}).to_csv(str(out), index=False)
pos = final.sum()

print(f"\n{'='*60}", flush=True)
print(f"TAMAMLANDI — v14 LightGBM Hard Negatives", flush=True)
print(f"  OOF Macro F1  : {best_f1:.5f}", flush=True)
print(f"  Threshold     : {best_thresh:.2f}", flush=True)
print(f"  Pozitif       : {pos:,} ({100*pos/len(final):.1f}%)", flush=True)
print(f"  Toplam süre   : {(time.time()-t0)/60:.1f} dk", flush=True)
print(f"  Dosya         : {out}", flush=True)
print(f"{'='*60}", flush=True)
