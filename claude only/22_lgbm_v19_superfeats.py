"""
22_lgbm_v19_superfeats.py — LightGBM + BERT score + Türkçe features
====================================================================
Analiz bulgularına dayanan iyileştirmeler:
  + bert_score         : Turkish BERT semantik skoru (en güçlü tek ekleme)
  + gender_cross       : -1 (mismatch), 0 (nötr), +1 (match)
  + stem_jaccard       : Türkçe suffix atarak kök benzerliği
  + first_token_title  : Marka query'leri için ilk kelime title'da var mı?
  + query_rank_group   : Query içindeki göreli LightGBM sırası
  - len_diff           : KALDIRILDI (AUC=0.50, gürültü)
  - l1_match           : KALDIRILDI (AUC=0.50, gürültü)

Beklenti: 0.70 → 0.73+
"""

import sys, time, json, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
BERT_MODEL = BASE / "claude only" / "models" / "bert_pseudolabel_v16"

LOWER = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(t): return str(t).translate(LOWER).lower().strip()

# Türkçe stem (basit suffix atma)
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

def stem_jac(a, b):
    sa = {stem(w) for w in a.split()}
    sb = {stem(w) for w in b.split()}
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0

def jac(a, b):
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0

def q_cov(q, t):
    qw = set(q.split())
    return len(qw & set(t.split())) / len(qw) if qw else 0.0

RENKLER = {"kırmızı","mavi","beyaz","siyah","sarı","yeşil","pembe","mor","gri","turuncu",
            "lacivert","bej","kahverengi","altın","gold","gümüş","silver","rose","ekru","krem","bordo","haki","füme"}

# ──────────────────────────────────────────────────────────────
# VERİ
# ──────────────────────────────────────────────────────────────
print("=" * 60, flush=True)
print("[1] Veri yükleniyor...", flush=True)
t0 = time.time()

items       = pd.read_csv(DATA / "items.csv")
terms       = pd.read_csv(DATA / "terms.csv")
train_pairs = pd.read_csv(DATA / "training_pairs.csv")
sub_pairs   = pd.read_csv(DATA / "submission_pairs.csv")

for col in ["title","category","brand","gender","age_group","attributes"]:
    items[col] = items[col].fillna("unknown").apply(trl)
terms["query"] = terms["query"].fillna("").apply(trl)

iid_to = {row.item_id: row for row in items.itertuples()}
tid_to_q = dict(zip(terms["term_id"], terms["query"]))

print(f"  {len(items):,} ürün | {len(train_pairs):,} pozitif | {len(sub_pairs):,} test çifti | {time.time()-t0:.1f}s", flush=True)

# ──────────────────────────────────────────────────────────────
# NEGATİFLER — test setinden gender/zero-overlap
# ──────────────────────────────────────────────────────────────
print("\n[2] Negatifler üretiliyor...", flush=True)
pool = sub_pairs.merge(terms, on="term_id").merge(items, on="item_id")
for col in ["query","title","gender","category"]:
    pool[col] = pool[col].fillna("unknown").apply(trl)

mask_g = (
    (pool["query"].str.contains("kadın|bayan", regex=True) & (pool["gender"] == "erkek")) |
    (pool["query"].str.contains(r"\berkek\b", regex=True) & (pool["gender"] == "kadın"))
)
mask_z = [len(set(q.split()) & set((t+" "+c.replace("/", " ")).split())) == 0
          for q, t, c in zip(pool["query"], pool["title"], pool["category"])]
pool["mask_z"] = mask_z
neg_pool = pool[mask_g | pool["mask_z"]][["term_id","item_id"]]
neg_pool = neg_pool.sample(n=min(350_000, len(neg_pool)), random_state=42)
neg_pool["label"] = 0
train_pairs["label"] = 1
train_df = pd.concat([train_pairs[["term_id","item_id","label"]], neg_pool], ignore_index=True)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"  Train: {len(train_df):,} çift | pos={train_pairs.label.sum():,} neg={len(neg_pool):,}", flush=True)

# ──────────────────────────────────────────────────────────────
# BERT INFERENCE (train + test için)
# ──────────────────────────────────────────────────────────────
print("\n[3] BERT inference (train + test)...", flush=True)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(str(BERT_MODEL))
bert_mdl  = AutoModelForSequenceClassification.from_pretrained(str(BERT_MODEL)).to(device).eval()
print(f"  Device: {device}", flush=True)

def bert_scores_for(tids, iids, batch=512):
    scores = []
    for i in range(0, len(tids), batch):
        bt, bi = tids[i:i+batch], iids[i:i+batch]
        qs = [tid_to_q.get(t, "") for t in bt]
        row_items = [iid_to.get(ii) for ii in bi]
        ps = [" | ".join(p for p in [
                  getattr(r, "title", ""), getattr(r, "brand", ""), getattr(r, "category", "").split("/")[0]
              ] if p and p != "unknown") if r else "" for r in row_items]
        enc = tokenizer(qs, ps, max_length=128, truncation=True, padding=True, return_tensors="pt").to(device)
        with torch.no_grad(), autocast("cuda"):
            logits = bert_mdl(**enc).logits.squeeze(-1)
        scores.extend(torch.sigmoid(logits).float().cpu().tolist())
        if i % 100_000 == 0 and i > 0:
            print(f"  bert {i:,}/{len(tids)}", flush=True)
    return np.array(scores)

t1 = time.time()
# Test BERT scores (zaten var → yükle)
test_bert_path = SUBM / "bert_scores_v16.npy"
if test_bert_path.exists():
    test_bert = np.load(str(test_bert_path))
    print(f"  Test BERT yüklendi: {len(test_bert):,}", flush=True)
else:
    test_bert = bert_scores_for(sub_pairs["term_id"].tolist(), sub_pairs["item_id"].tolist())
    np.save(str(test_bert_path), test_bert)

# Train BERT scores (yeni inference)
train_bert_path = BASE / "claude only" / "train_bert_scores_v19.npy"
if train_bert_path.exists():
    train_bert = np.load(str(train_bert_path))
    print(f"  Train BERT yüklendi: {len(train_bert):,}", flush=True)
else:
    train_bert = bert_scores_for(train_df["term_id"].tolist(), train_df["item_id"].tolist())
    np.save(str(train_bert_path), train_bert)
print(f"  BERT inference: {(time.time()-t1)/60:.1f} dk", flush=True)

# ──────────────────────────────────────────────────────────────
# FEATURE BUILDER
# ──────────────────────────────────────────────────────────────
print("\n[4] Özellikler hesaplanıyor...", flush=True)

def attr_jac(q, a):
    if a in ("unknown",""): return 0.0
    try:
        vals = " ".join(str(v) for v in json.loads(a).values() if v)
        return jac(q, vals)
    except: return jac(q, a)

def gender_cross(q, g):
    q_kadin  = bool(re.search(r'\b(kadın|bayan)\b', q))
    q_erkek  = bool(re.search(r'\berkek\b', q))
    g_kadin  = g in ("kadın","bayan","kız")
    g_erkek  = g == "erkek"
    if q_kadin and g_erkek: return -1.0
    if q_erkek and g_kadin: return -1.0
    if q_kadin and g_kadin: return  1.0
    if q_erkek and g_erkek: return  1.0
    return 0.0

def build_features(df_in, tfidf_vect, bert_arr, fit=False, all_texts=None):
    df = df_in.merge(terms, on="term_id", how="left").merge(items, on="item_id", how="left")
    for col in ["query","title","brand","category","gender","age_group","attributes"]:
        df[col] = df[col].fillna("unknown").apply(trl)

    if fit:
        corpus = list(df["title"]) + list(df["query"]) + (all_texts or [])
        tfidf_vect.fit(corpus)

    qs = df["query"].tolist()
    ts = df["title"].tolist()
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

    f = pd.DataFrame()
    # Temel metin özellikleri (kanıtlı)
    f["fuzz_partial"]     = [rfuzz.partial_ratio(q,t)/100    for q,t in zip(qs,ts)]
    f["fuzz_set"]         = [rfuzz.token_set_ratio(q,t)/100  for q,t in zip(qs,ts)]
    f["fuzz_sort"]        = [rfuzz.token_sort_ratio(q,t)/100 for q,t in zip(qs,ts)]
    f["fuzz_basic"]       = [rfuzz.ratio(q,t)/100            for q,t in zip(qs,ts)]
    f["jaccard"]          = [jac(q,t)                         for q,t in zip(qs,ts)]
    f["tfidf_cos"]        = tfidf_cos(qs, ts)
    f["q_cov_title"]      = [q_cov(q,t)                      for q,t in zip(qs,ts)]
    f["t_cov_query"]      = [q_cov(t,q)                      for q,t in zip(qs,ts)]
    f["cat_overlap"]      = [jac(q, c.replace("/", " "))      for q,c in zip(qs,cats)]
    f["exact_in_title"]   = [(q in t)*1.0                     for q,t in zip(qs,ts)]
    f["token_overlap"]    = [len(set(q.split())&set(t.split())) for q,t in zip(qs,ts)]
    f["attr_match"]       = [attr_jac(q,a)                    for q,a in zip(qs,attrs)]
    f["brand_in_q"]       = [(b!="unknown" and b in q)*1.0    for b,q in zip(brs,qs)]
    f["age_in_q"]         = [(a not in ("unknown","") and a in q)*1.0 for a,q in zip(ages,qs)]

    # YENİ özellikler
    f["stem_jaccard"]     = [stem_jac(q,t)                    for q,t in zip(qs,ts)]
    f["stem_cat_jac"]     = [stem_jac(q, c.replace("/", " ")) for q,c in zip(qs,cats)]
    f["gender_cross"]     = [gender_cross(q,g)                for q,g in zip(qs,gens)]
    f["first_tok_title"]  = [(q.split()[0] in t if q.split() else False)*1.0 for q,t in zip(qs,ts)]
    f["first_tok_brand"]  = [(q.split()[0] in b if q.split() and b!="unknown" else False)*1.0 for q,b in zip(qs,brs)]
    f["renk_q_in_t"]      = [any(r in q.split() and r in t.split() for r in RENKLER)*1.0 for q,t in zip(qs,ts)]
    f["renk_mismatch"]    = [(any(r in q.split() for r in RENKLER) and not any(r in t.split() for r in RENKLER))*1.0
                              for q,t in zip(qs,ts)]
    f["q_len"]            = [len(q.split()) for q in qs]
    f["t_len"]            = [len(t.split()) for t in ts]
    f["bert_score"]       = bert_arr
    f["ana_kategori"]     = pd.Categorical([c.split("/")[0] for c in cats])
    return f, df

t1 = time.time()
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=60_000, sublinear_tf=True, min_df=2)
FEATS = [c for c in [
    "fuzz_partial","fuzz_set","fuzz_sort","fuzz_basic","jaccard","tfidf_cos",
    "q_cov_title","t_cov_query","cat_overlap","exact_in_title","token_overlap",
    "attr_match","brand_in_q","age_in_q",
    "stem_jaccard","stem_cat_jac","gender_cross","first_tok_title","first_tok_brand",
    "renk_q_in_t","renk_mismatch","q_len","t_len","bert_score","ana_kategori"
] if True]

X_tr, df_tr = build_features(train_df, tfidf, train_bert, fit=True,
                               all_texts=items["title"].tolist()+terms["query"].tolist())
y = train_df["label"].values
groups = df_tr["term_id"].values
print(f"  Train features: {X_tr.shape} | {(time.time()-t1):.1f}s", flush=True)

# ──────────────────────────────────────────────────────────────
# LightGBM 5-FOLD
# ──────────────────────────────────────────────────────────────
print("\n[5] LightGBM 5-fold...", flush=True)
oof = np.zeros(len(train_df))
models_lgbm = []
gkf = GroupKFold(n_splits=5)
for fold, (tri, vli) in enumerate(gkf.split(X_tr, y, groups)):
    m = lgb.LGBMClassifier(
        n_estimators=1200, learning_rate=0.04, max_depth=8,
        num_leaves=127, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
    )
    m.fit(X_tr[FEATS].iloc[tri], y[tri],
          eval_set=[(X_tr[FEATS].iloc[vli], y[vli])],
          categorical_feature=["ana_kategori"],
          callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
    oof[vli] = m.predict_proba(X_tr[FEATS].iloc[vli])[:, 1]
    f = f1_score(y[vli], (oof[vli] > 0.5).astype(int), average="macro")
    print(f"  Fold {fold+1} | best_iter={m.best_iteration_} | val F1={f:.4f}", flush=True)
    models_lgbm.append(m)

# Threshold optimizasyonu
best_thr, best_f1 = 0.5, 0.0
for thr in np.arange(0.30, 0.71, 0.01):
    s = f1_score(y, (oof > thr).astype(int), average="macro")
    if s > best_f1: best_f1, best_thr = s, thr
print(f"\n  OOF F1={best_f1:.5f} | threshold={best_thr:.2f}", flush=True)

# Feature önemleri
fi = pd.Series(
    sum(m.feature_importances_ for m in models_lgbm) / len(models_lgbm),
    index=FEATS
).sort_values(ascending=False)
print("\n  Feature Önemleri (top 15):", flush=True)
for feat, imp in fi.head(15).items():
    print(f"    {feat:25s}: {imp:6.0f}", flush=True)

# ──────────────────────────────────────────────────────────────
# TEST INFERENCE
# ──────────────────────────────────────────────────────────────
print("\n[6] Test inference...", flush=True)
X_te, df_te = build_features(sub_pairs, tfidf, test_bert)
test_scores = sum(m.predict_proba(X_te[FEATS])[:,1] for m in models_lgbm) / len(models_lgbm)

# Query içi rank feature (post-hoc, re-inference değil — sadece submission için)
# Her query için test_scores'ın percentile rank'ını hesapla
df_te2 = sub_pairs.copy()
df_te2["score"] = test_scores
df_te2["rank_in_q"] = df_te2.groupby("term_id")["score"].rank(pct=True)

# Global threshold
final_global = (test_scores > best_thr).astype(int)
pos_g = final_global.sum()

print(f"  Global threshold={best_thr:.2f} | pozitif={pos_g:,} ({100*pos_g/len(sub_pairs):.1f}%)", flush=True)

# Submission kaydet
out_path = SUBM / "submission_v19_superfeats.csv"
pd.DataFrame({"id": sub_pairs["id"], "prediction": final_global}).to_csv(str(out_path), index=False)

print(f"\n{'='*60}", flush=True)
print(f"TAMAMLANDI — v19 SuperFeats", flush=True)
print(f"  OOF F1      : {best_f1:.5f}", flush=True)
print(f"  Threshold   : {best_thr:.2f}", flush=True)
print(f"  Pozitif     : {pos_g:,} ({100*pos_g/len(sub_pairs):.1f}%)", flush=True)
print(f"  Toplam süre : {(time.time()-t0)/60:.1f} dk", flush=True)
print(f"  Dosya       : {out_path}", flush=True)
print(f"{'='*60}", flush=True)
