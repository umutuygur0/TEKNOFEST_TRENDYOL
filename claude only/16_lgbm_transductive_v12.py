"""
16_lgbm_transductive_v12.py
============================
0.68 notebook'un tam portu + 6 eksik özellik eklendi:
  Notebook (7 özellik) → 0.68
  Bu script (13 özellik) → hedef 0.72+

Yeni özellikler: tfidf_cos_sim, fuzz_partial, fuzz_token_sort,
                 fuzz_basic, attr_match_score, len_diff

Yaklaşım:
  1. Test setinden transductive hard negatives (gender mismatch + sıfır overlap)
  2. 13 özellikli LightGBM
  3. 5-fold GroupKFold (term_id gruplama)
  4. Threshold optimizasyonu (0.3-0.7 sweep)
"""

import sys, time, re
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

BASE = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
SUBM = BASE / "claude only" / "submissions"
SUBM.mkdir(parents=True, exist_ok=True)

LOWER_MAP = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(t): return str(t).translate(LOWER_MAP).lower().strip()

# ─── 1. Veri Yükle ────────────────────────────────────────────────────────────
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

print(f"  {time.time()-t0:.1f}s | items={len(items):,} | test={len(sub_pairs):,}", flush=True)

# ─── 2. Hard Negative Mining (Transductive) ───────────────────────────────────
print("\n[2] Test setinden hard negatives üretiliyor...", flush=True)
t1 = time.time()

pool = sub_pairs.merge(terms, on="term_id", how="left") \
                .merge(items, on="item_id", how="left")

# Cinsiyet çatışması
mask_gender = (
    (pool["query"].str.contains("erkek") & (pool["gender"] == "kadın")) |
    (pool["query"].str.contains("kadın") & (pool["gender"] == "erkek"))
)

# Sıfır kelime kesişimi (query kelimeleri title+category'de hiç yok)
def is_unrelated(q, title, cat):
    q_words = set(q.split())
    if not q_words: return False
    target = set((title + " " + cat.replace("/", " ")).split())
    return len(q_words & target) == 0

print("  Sıfır kelime filtresi hesaplanıyor (~2-4 dk)...", flush=True)
pool["is_unrelated"] = [is_unrelated(q, t, c)
                        for q, t, c in zip(pool["query"], pool["title"], pool["category"])]

hard_neg = pool[mask_gender | pool["is_unrelated"]].copy()
print(f"  Bulunan hard negative: {len(hard_neg):,}", flush=True)

TARGET_NEG = 250_000
if len(hard_neg) >= TARGET_NEG:
    final_neg = hard_neg.sample(n=TARGET_NEG, random_state=42)
else:
    final_neg = hard_neg
    print(f"  ! Yeterli yok, {len(final_neg):,} alındı", flush=True)

final_neg = final_neg[["id","term_id","item_id"]].copy()
final_neg["label"] = 0
train_pairs["label"] = 1

train_ready = pd.concat([train_pairs[["id","term_id","item_id","label"]], final_neg],
                         ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  Eğitim seti: {len(train_ready):,} çift | pos={train_ready['label'].sum():,} | "
      f"neg={(train_ready['label']==0).sum():,}", flush=True)
print(f"  {time.time()-t1:.1f}s", flush=True)

# ─── 3. Metin Hazırlama ────────────────────────────────────────────────────────
print("\n[3] Metin alanları ekleniyor...", flush=True)
df = train_ready.merge(terms, on="term_id", how="left") \
                .merge(items, on="item_id", how="left")

for col in ["query","title","brand","category","gender","age_group","attributes"]:
    df[col] = df[col].fillna("unknown").apply(trl)

# ─── 4. TF-IDF Cosine Similarity (En Güçlü Özellik) ─────────────────────────
print("\n[4] TF-IDF vektörleri hazırlanıyor...", flush=True)
t1 = time.time()

tfidf_vect = TfidfVectorizer(ngram_range=(1,2), max_features=50_000,
                              sublinear_tf=True, min_df=2)
all_texts = pd.concat([df["title"], df["query"],
                       pd.Series(items["title"].tolist() + terms["query"].tolist())])
tfidf_vect.fit(all_texts)

def tfidf_cos_batch(queries, titles, vectorizer, chunk=50_000):
    """Batch cosine similarity: yüksek hız, düşük RAM"""
    n = len(queries)
    sims = np.zeros(n, dtype=np.float32)
    for i in range(0, n, chunk):
        q_mat = vectorizer.transform(queries[i:i+chunk])
        t_mat = vectorizer.transform(titles[i:i+chunk])
        q_norm = normalize(q_mat, norm="l2")
        t_norm = normalize(t_mat, norm="l2")
        sims[i:i+chunk] = np.array(q_norm.multiply(t_norm).sum(axis=1)).flatten()
        if i % 500_000 == 0 and i > 0:
            print(f"  TF-IDF {i:,}/{n:,}", flush=True)
    return sims

df["tfidf_cos_sim"] = tfidf_cos_batch(df["query"].tolist(), df["title"].tolist(), tfidf_vect)
print(f"  TF-IDF eğitim: {time.time()-t1:.1f}s", flush=True)

# ─── 5. Diğer Özellikler ──────────────────────────────────────────────────────
print("\n[5] Özellik Mühendisliği (rapidfuzz)...", flush=True)
t1 = time.time()

def jaccard(a, b):
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

qs = df["query"].tolist()
ts = df["title"].tolist()
cats = df["category"].tolist()
brands = df["brand"].tolist()
genders = df["gender"].tolist()
ages = df["age_group"].tolist()
attrs = df["attributes"].tolist()

print("  jaccard + fuzz...", flush=True)
df["jaccard_sim"]     = [jaccard(q, t) for q, t in zip(qs, ts)]
df["fuzz_token_set"]  = [rfuzz.token_set_ratio(q, t) / 100.0 for q, t in zip(qs, ts)]
df["fuzz_partial"]    = [rfuzz.partial_ratio(q, t)   / 100.0 for q, t in zip(qs, ts)]
df["fuzz_token_sort"] = [rfuzz.token_sort_ratio(q, t)/ 100.0 for q, t in zip(qs, ts)]
df["fuzz_basic"]      = [rfuzz.ratio(q, t)            / 100.0 for q, t in zip(qs, ts)]

print("  kategorik + diğer...", flush=True)
df["category_overlap"]   = [jaccard(q, c.replace("/", " ")) for q, c in zip(qs, cats)]
df["is_brand_in_query"]  = [1 if b != "unknown" and b in q else 0 for q, b in zip(qs, brands)]
df["is_gender_in_query"] = [1 if g in q else 0 for q, g in zip(qs, genders)]
df["is_age_in_query"]    = [1 if a not in ("unknown","") and a in q else 0 for q, a in zip(qs, ages)]
df["attr_match_score"]   = [jaccard(q, a) for q, a in zip(qs, attrs)]
df["len_diff"]           = [abs(len(q) - len(t)) for q, t in zip(qs, ts)]
df["ana_kategori"]       = pd.Categorical(df["category"].apply(lambda x: x.split("/")[0]))

print(f"  {time.time()-t1:.1f}s", flush=True)

# ─── 6. LightGBM Training ─────────────────────────────────────────────────────
FEATURES = [
    "jaccard_sim", "fuzz_token_set", "fuzz_partial", "fuzz_token_sort", "fuzz_basic",
    "tfidf_cos_sim", "category_overlap", "is_brand_in_query", "is_gender_in_query",
    "is_age_in_query", "attr_match_score", "len_diff", "ana_kategori"
]

print(f"\n[6] LightGBM 5-Fold GroupKFold ({len(df):,} çift, {len(FEATURES)} özellik)...", flush=True)
t1 = time.time()

X = df[FEATURES]
y = df["label"]
groups = df["term_id"]

oof_preds = np.zeros(len(df))
models = []
gkf = GroupKFold(n_splits=5)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=7,
        num_leaves=128, random_state=42, class_weight="balanced",
        n_jobs=-1, verbose=-1
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        categorical_feature=["ana_kategori"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)]
    )
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    models.append(model)
    fold_f1 = f1_score(y_val, (oof_preds[val_idx] > 0.5).astype(int), average="macro")
    print(f"  Fold {fold+1} | best_iter={model.best_iteration_:4d} | val F1={fold_f1:.4f}", flush=True)

print(f"  LightGBM training: {time.time()-t1:.1f}s", flush=True)

print("\n[7] Optimum threshold aranıyor...", flush=True)
best_thresh, best_f1 = 0.5, 0.0
for thresh in np.arange(0.3, 0.71, 0.01):
    preds = (oof_preds > thresh).astype(int)
    score = f1_score(y, preds, average="macro")
    if score > best_f1:
        best_f1, best_thresh = score, thresh

print(f"  En İyi Threshold: {best_thresh:.2f} | OOF Macro F1: {best_f1:.5f}", flush=True)

imp = pd.DataFrame({"Feature": FEATURES,
                    "Importance": models[0].feature_importances_}) \
        .sort_values("Importance", ascending=False)
print("\n  Özellik Önemleri:")
for _, row in imp.iterrows():
    print(f"    {row['Feature']:25s}: {int(row['Importance']):5d}", flush=True)

# ─── 7. Test Inference ────────────────────────────────────────────────────────
print(f"\n[8] Test inference ({len(sub_pairs):,} çift)...", flush=True)
t1 = time.time()

test_df = sub_pairs.merge(terms, on="term_id", how="left") \
                   .merge(items, on="item_id", how="left")

for col in ["query","title","brand","category","gender","age_group","attributes"]:
    test_df[col] = test_df[col].fillna("unknown").apply(trl)

print("  TF-IDF cos sim (test)...", flush=True)
test_df["tfidf_cos_sim"] = tfidf_cos_batch(
    test_df["query"].tolist(), test_df["title"].tolist(), tfidf_vect)
print(f"  TF-IDF test: {time.time()-t1:.1f}s", flush=True)

print("  Fuzz features (test)...", flush=True)
tq = test_df["query"].tolist()
tt = test_df["title"].tolist()
tc = test_df["category"].tolist()
tb = test_df["brand"].tolist()
tg = test_df["gender"].tolist()
ta = test_df["age_group"].tolist()
tat = test_df["attributes"].tolist()

test_df["jaccard_sim"]     = [jaccard(q, t) for q, t in zip(tq, tt)]
test_df["fuzz_token_set"]  = [rfuzz.token_set_ratio(q, t) / 100.0 for q, t in zip(tq, tt)]
test_df["fuzz_partial"]    = [rfuzz.partial_ratio(q, t)   / 100.0 for q, t in zip(tq, tt)]
test_df["fuzz_token_sort"] = [rfuzz.token_sort_ratio(q, t)/ 100.0 for q, t in zip(tq, tt)]
test_df["fuzz_basic"]      = [rfuzz.ratio(q, t)            / 100.0 for q, t in zip(tq, tt)]
test_df["category_overlap"]   = [jaccard(q, c.replace("/", " ")) for q, c in zip(tq, tc)]
test_df["is_brand_in_query"]  = [1 if b != "unknown" and b in q else 0 for q, b in zip(tq, tb)]
test_df["is_gender_in_query"] = [1 if g in q else 0 for q, g in zip(tq, tg)]
test_df["is_age_in_query"]    = [1 if a not in ("unknown","") and a in q else 0 for q, a in zip(tq, ta)]
test_df["attr_match_score"]   = [jaccard(q, a) for q, a in zip(tq, tat)]
test_df["len_diff"]           = [abs(len(q) - len(t)) for q, t in zip(tq, tt)]
test_df["ana_kategori"]       = pd.Categorical(test_df["category"].apply(lambda x: x.split("/")[0]))

print(f"  Fuzz test bitti: {time.time()-t1:.1f}s", flush=True)

print("  Ensemble tahmin...", flush=True)
X_test = test_df[FEATURES]
test_preds = np.zeros(len(test_df))
for m in models:
    test_preds += m.predict_proba(X_test)[:, 1] / len(models)

final_preds = (test_preds > best_thresh).astype(int)
sub = pd.DataFrame({"id": test_df["id"], "prediction": final_preds})

out = SUBM / "submission_v12_lgbm_transductive.csv"
sub.to_csv(str(out), index=False)

pos = final_preds.sum()
print(f"\n{'='*60}", flush=True)
print(f"TAMAMLANDI — v12 LightGBM Transductive", flush=True)
print(f"  OOF Macro F1 : {best_f1:.5f}", flush=True)
print(f"  Threshold    : {best_thresh:.2f}", flush=True)
print(f"  Pozitif tahmin: {pos:,} ({100*pos/len(sub):.1f}%)", flush=True)
print(f"  Toplam süre  : {(time.time()-t0)/60:.1f} dk", flush=True)
print(f"  Dosya        : {out}", flush=True)
print(f"{'='*60}", flush=True)
