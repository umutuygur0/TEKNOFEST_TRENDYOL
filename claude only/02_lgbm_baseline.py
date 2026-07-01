# -*- coding: utf-8 -*-
"""
Submission v2 — LightGBM Eğitimli Baseline
===========================================
v1 neden 0.48 aldı: Fixed top-K, öğrenme yok, eşik yok.
  - Hepsini 0 söylemek macro F1 ~0.465 verir (pozitif oran %13 olunca)
  - Bizim top-K bunu zar zor geçti

v2 yaklaşımı:
  1. Negatif örnekleme: %50 aynı kategori (hard) + %50 random
  2. IDF-ağırlıklı token overlap + metadata özellikler
  3. LightGBM binary classifier (scale_pos_weight ile)
  4. Validation üzerinde macro F1 threshold optimizasyonu
  5. Test'e uygula → submissions/ klasörüne kaydet
"""

import pandas as pd
import numpy as np
import re, math, time, csv
from pathlib import Path
from collections import Counter, defaultdict
import lightgbm as lgb
from sklearn.metrics import f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ─── Yollar ───────────────────────────────────────────────────────────────────
DATA_DIR = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL\trendyol-e-ticaret-yarismasi-2026-kaggle")
SUBM_DIR = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL\claude only\submissions")
SUBM_DIR.mkdir(exist_ok=True)

t_global = time.time()

# ─── Türkçe Tokenizer ─────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"[^0-9a-zçğıöşü]+")
COLORS  = {"kırmızı","mavi","yeşil","sarı","siyah","beyaz","gri","mor","turuncu",
           "pembe","kahverengi","bej","lacivert","bordo","krem","altın","gümüş"}
GENDERS = {"erkek","kadın","kız","bay","bayan"}

def norm_tr(s):
    if not isinstance(s, str): return ""
    return s.replace("İ","i").replace("I","ı").replace("i̇","i").lower()

def tokenize(s, min_len=2):
    return [t for t in _TOKEN_RE.split(norm_tr(s)) if len(t) >= min_len]

def tok_set(s): return frozenset(tokenize(s))

# ─── Veri Yükle ───────────────────────────────────────────────────────────────
print("=" * 60)
print("Veri yükleniyor...")
t = time.time()
items  = pd.read_csv(DATA_DIR / "items.csv")
terms  = pd.read_csv(DATA_DIR / "terms.csv")
train  = pd.read_csv(DATA_DIR / "training_pairs.csv")
test   = pd.read_csv(DATA_DIR / "submission_pairs.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
print(f"  items:{len(items):,} terms:{len(terms):,} train:{len(train):,} test:{len(test):,}  [{time.time()-t:.1f}s]")

# ─── Item Özellikleri ─────────────────────────────────────────────────────────
print("\nItem öznitelikleri hazırlanıyor...")
t = time.time()

items["full_text"] = (items["title"].fillna("") + " " +
                      items["category"].fillna("") + " " +
                      items["brand"].fillna("") + " " +
                      items["gender"].fillna("") + " " +
                      items["age_group"].fillna("") + " " +
                      items["attributes"].fillna(""))

items["cat_l1"] = items["category"].fillna("").apply(
    lambda x: x.split("/")[0].strip() if x else "")

# Tokenize (vektörize için önce list, sonra dict'e alacağız)
print("  Tokenize ediliyor...")
item_full_toks  = [tok_set(s) for s in tqdm(items["full_text"], ncols=70, desc="  full_text")]
item_title_toks = [tok_set(s) for s in items["title"].fillna("")]
item_cat_toks   = [tok_set(s) for s in items["category"].fillna("")]
item_brand_norm = [norm_tr(s) for s in items["brand"].fillna("")]
item_gender_str = [norm_tr(s) for s in items["gender"].fillna("")]

# Cat L1 encoding
cat_l1_vals  = items["cat_l1"].unique()
cat_l1_enc_m = {c: i for i, c in enumerate(cat_l1_vals)}
item_cat_enc = [cat_l1_enc_m.get(c, 0) for c in items["cat_l1"]]

# item_id → index
item_id_arr = items["item_id"].values
iid_to_idx  = {iid: i for i, iid in enumerate(item_id_arr)}

# cat_l1 → [item_idx list]
cat_to_idxs = defaultdict(list)
for idx, cat in enumerate(items["cat_l1"].values):
    cat_to_idxs[cat].append(idx)

print(f"  Tamamlandı  [{time.time()-t:.1f}s]")

# ─── IDF Hesapla ──────────────────────────────────────────────────────────────
print("\nIDF hesaplanıyor (tüm item full_text)...")
t = time.time()
df_cnt = Counter()
N = len(item_full_toks)
for ftoks in item_full_toks:
    df_cnt.update(ftoks)
idf = {tok: math.log((N - cnt + 0.5) / (cnt + 0.5) + 1)
       for tok, cnt in df_cnt.items()}
print(f"  Vocab: {len(idf):,}  [{time.time()-t:.1f}s]")

# ─── Query Hazırlık ───────────────────────────────────────────────────────────
q_toks_map = dict(zip(terms["term_id"], terms["query"].apply(tokenize)))
q_set_map  = dict(zip(terms["term_id"], terms["query"].apply(tok_set)))

# ─── Feature Fonksiyonu ───────────────────────────────────────────────────────
# 16 özellik — hepsi dict/list lookup, .iloc yok (hızlı)
FEATURE_NAMES = [
    "idf_score_full",     # 0  IDF-ağırlıklı overlap vs full_text
    "idf_score_title",    # 1  IDF-ağırlıklı overlap vs title
    "overlap_full_ratio", # 2  raw overlap / query_len
    "overlap_title_ratio",# 3
    "overlap_cat_ratio",  # 4
    "jaccard_title",      # 5  |∩| / |∪| title
    "inter_full_count",   # 6  absolüt token sayısı
    "inter_title_count",  # 7
    "exact_sub_in_title", # 8  query tam alt-dize mi?
    "brand_in_query",     # 9  item markası query'de mi?
    "gender_match",       # 10 cinsiyet tutarlılığı
    "color_in_query",     # 11 renk adı geçiyor mu?
    "color_match",        # 12 geçiyorsa item full_text'te var mı?
    "query_len",          # 13 query token uzunluğu
    "title_len",          # 14 item title token uzunluğu
    "cat_l1_enc",         # 15 kategori (sayısal)
]

def compute_features_batch(term_ids, item_ids_or_idxs):
    """Batch feature hesaplama — liste döndürür."""
    results = []
    for tid, iid in zip(term_ids, item_ids_or_idxs):
        idx = iid_to_idx.get(iid) if isinstance(iid, (str, np.int64, np.int32, int)) else iid
        if idx is None:
            results.append([0.0] * 16)
            continue

        q_t = q_toks_map.get(tid, [])
        q_s = q_set_map.get(tid, frozenset())
        q_len = len(q_t)

        if q_len == 0:
            results.append([0.0] * 16)
            continue

        ft = item_full_toks[idx]
        tt = item_title_toks[idx]
        ct = item_cat_toks[idx]

        # IDF overlap
        idf_full  = sum(idf.get(tok, 0) for tok in q_t if tok in ft)
        idf_title = sum(idf.get(tok, 0) for tok in q_t if tok in tt)

        # Raw overlap
        inter_full  = len(q_s & ft)
        inter_title = len(q_s & tt)
        inter_cat   = len(q_s & ct)

        full_r  = inter_full  / q_len
        title_r = inter_title / q_len
        cat_r   = inter_cat   / q_len

        # Jaccard title
        u_tt = len(q_s | tt)
        jacc = inter_title / u_tt if u_tt > 0 else 0.0

        # Exact substring
        q_joined   = " ".join(q_t)
        title_norm = " ".join(tt)  # already normalized
        exact = int(q_joined in title_norm)

        # Brand match
        brand = item_brand_norm[idx]
        b_in_q = int(bool(brand) and brand != "unknown" and
                     any(brand in tok or tok in brand for tok in q_t if len(tok) >= 3))

        # Gender
        q_gen = q_s & GENDERS
        if not q_gen:
            gender_m = 1  # belirsiz → uyumlu say
        else:
            ig = tok_set(item_gender_str[idx])
            gender_m = int(bool(q_gen & ig))

        # Renk
        q_col = q_s & COLORS
        col_in_q  = int(bool(q_col))
        col_match = int(bool(q_col & ft)) if col_in_q else 0

        results.append([
            idf_full, idf_title,
            full_r, title_r, cat_r,
            jacc,
            float(inter_full), float(inter_title),
            float(exact), float(b_in_q), float(gender_m),
            float(col_in_q), float(col_match),
            float(q_len), float(len(tt)),
            float(item_cat_enc[idx]),
        ])
    return results

# ─── Negatif Örnekleme (Vektörize) ───────────────────────────────────────────
# HIZLI: term döngüsü yok, kategori bazlı numpy batch sampling
print("\nNegatif örnekleme yapılıyor (vektörize)...")
t = time.time()
np.random.seed(42)

item_cat_arr = items["cat_l1"].values          # tüm itemlar için kategori dizisi
item_cat_map = dict(zip(item_id_arr, item_cat_arr))
cat_unique   = np.unique(item_cat_arr)
# kategori → item indeks dizisi
cat_idx_map  = {cat: np.where(item_cat_arr == cat)[0] for cat in cat_unique}

# Her eğitim pozitifinin kategorisi
pos_item_cat = np.array([
    item_cat_map.get(iid, "") for iid in train["item_id"].values
], dtype=object)

NEG_RATIO = 4
HARD_N    = 2   # her pozitif için aynı kategoriden 2 hard negatif
EASY_N    = NEG_RATIO - HARD_N  # 2 kolay negatif

pos_set = set(zip(train["term_id"].values.tolist(), train["item_id"].values.tolist()))

# 1) Hard negatives — aynı kategori (kategoriye göre toplu)
hard_term_ids_list = []
hard_item_ids_list = []

for cat in cat_unique:
    cat_mask   = pos_item_cat == cat
    cat_tids   = train["term_id"].values[cat_mask]       # bu kategori pozitiflerinin term_id'leri
    pool_idxs  = cat_idx_map[cat]                         # bu kategorideki tüm item indeksleri
    if len(pool_idxs) == 0 or len(cat_tids) == 0:
        continue
    n_sample = len(cat_tids) * HARD_N
    sampled_idxs = np.random.choice(pool_idxs, n_sample, replace=True)  # replace=True → hızlı
    sampled_iids = item_id_arr[sampled_idxs]
    # term_id'leri tekrarla (her pozitife HARD_N negatif)
    repeated_tids = np.repeat(cat_tids, HARD_N)
    hard_term_ids_list.append(repeated_tids)
    hard_item_ids_list.append(sampled_iids)

hard_term_ids = np.concatenate(hard_term_ids_list)
hard_item_ids = np.concatenate(hard_item_ids_list)

# 2) Easy negatives — tamamen rastgele (replace=True, çok hızlı)
n_train = len(train)
easy_rand_idxs = np.random.randint(0, len(item_id_arr), n_train * EASY_N)
easy_item_ids  = item_id_arr[easy_rand_idxs]
easy_term_ids  = np.repeat(train["term_id"].values, EASY_N)

# 3) Birleştir
neg_term_ids = np.concatenate([hard_term_ids, easy_term_ids])
neg_item_ids = np.concatenate([hard_item_ids, easy_item_ids])

# 4) Eğer (term, item) çifti pozitif setinde varsa çıkar (nadir ama olabilir)
neg_mask = np.array([
    (t, i) not in pos_set
    for t, i in zip(neg_term_ids, neg_item_ids)
], dtype=bool)
neg_term_ids = neg_term_ids[neg_mask]
neg_item_ids = neg_item_ids[neg_mask]

print(f"  Pozitif: {len(train):,}  Negatif: {len(neg_term_ids):,}  [{time.time()-t:.1f}s]")

# ─── Eğitim Feature Matrix ────────────────────────────────────────────────────
print("\nEğitim öznitelikleri hesaplanıyor...")
t = time.time()

BATCH = 50_000

# Pozitif çiftler
pos_tids = train["term_id"].values.tolist()
pos_iids = train["item_id"].values.tolist()

X_pos_list = []
for i in tqdm(range(0, len(pos_tids), BATCH), ncols=70, desc="  pozitif"):
    X_pos_list.extend(compute_features_batch(pos_tids[i:i+BATCH], pos_iids[i:i+BATCH]))

# Negatif çiftler
X_neg_list = []
for i in tqdm(range(0, len(neg_term_ids), BATCH), ncols=70, desc="  negatif"):
    X_neg_list.extend(compute_features_batch(neg_term_ids[i:i+BATCH], neg_item_ids[i:i+BATCH]))

X = np.array(X_pos_list + X_neg_list, dtype=np.float32)
y = np.array([1]*len(X_pos_list) + [0]*len(X_neg_list), dtype=np.int8)
print(f"  Matrix: {X.shape}  [{time.time()-t:.1f}s]")

# ─── Train / Validation Split ─────────────────────────────────────────────────
perm = np.random.permutation(len(y))
X, y = X[perm], y[perm]

n_val = int(0.2 * len(y))
X_val, y_val = X[:n_val], y[:n_val]
X_tr,  y_tr  = X[n_val:], y[n_val:]
print(f"  Train: {len(y_tr):,}  Val: {len(y_val):,}")
print(f"  Train pos rate: {y_tr.mean():.3f}")

# ─── LightGBM Eğitimi ─────────────────────────────────────────────────────────
print("\nLightGBM eğitiliyor...")
t = time.time()

# scale_pos_weight: sınıf dengesizliğini telafi et
spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
print(f"  scale_pos_weight = {spw:.2f}")

params = {
    "objective":        "binary",
    "metric":           "auc",
    "learning_rate":    0.05,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_child_samples": 30,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq":     5,
    "scale_pos_weight": spw,
    "lambda_l1":        0.1,
    "lambda_l2":        0.1,
    "verbose":         -1,
    "n_jobs":          -1,
}

dtrain = lgb.Dataset(X_tr,  label=y_tr,  feature_name=FEATURE_NAMES)
dval   = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=dtrain)

cbs = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
model = lgb.train(params, dtrain, num_boost_round=600, valid_sets=[dval], callbacks=cbs)
print(f"  Best iter: {model.best_iteration}  [{time.time()-t:.1f}s]")

# ─── Threshold Optimizasyonu ──────────────────────────────────────────────────
print("\nEşik optimizasyonu...")
val_prob = model.predict(X_val)
best_f1, best_thr = 0.0, 0.5
for thr in np.arange(0.05, 0.95, 0.01):
    preds = (val_prob >= thr).astype(int)
    f1 = f1_score(y_val, preds, average="macro", zero_division=0)
    if f1 > best_f1:
        best_f1, best_thr = f1, float(thr)

print(f"  Validation Macro F1: {best_f1:.4f}  |  Threshold: {best_thr:.2f}")

# F1 per class
val_preds = (val_prob >= best_thr).astype(int)
f1_per = f1_score(y_val, val_preds, average=None, zero_division=0)
print(f"  F1 per class: neg={f1_per[0]:.4f}  pos={f1_per[1]:.4f}")

# Feature önemleri
fi = sorted(zip(FEATURE_NAMES, model.feature_importance("gain")), key=lambda x: -x[1])
print("\nFeature Önemleri (gain):")
max_imp = max(f[1] for f in fi) if fi else 1
for name, imp in fi:
    bar = "#" * int(imp / max_imp * 30)
    print(f"  {name:<25} {imp:>12,.0f}  {bar}")

# ─── Test Inference ───────────────────────────────────────────────────────────
print("\nTest çiftleri skorlanıyor...")
t = time.time()

test_tids = test["term_id"].values.tolist()
test_iids = test["item_id"].values.tolist()

X_test_batches = []
for i in tqdm(range(0, len(test_tids), BATCH), ncols=70, desc="  batches"):
    batch_feats = compute_features_batch(test_tids[i:i+BATCH], test_iids[i:i+BATCH])
    X_test_batches.append(np.array(batch_feats, dtype=np.float32))

X_test = np.vstack(X_test_batches)
test_prob  = model.predict(X_test)
test_preds = (test_prob >= best_thr).astype(np.int8)

print(f"  Inference tamamlandı  [{time.time()-t:.1f}s]")
print(f"  Pozitif: {test_preds.sum():,} ({test_preds.mean()*100:.1f}%)")

# ─── Submission ───────────────────────────────────────────────────────────────
print("\nSubmission oluşturuluyor...")
sub = pd.DataFrame({"id": test["id"].values, "prediction": test_preds.astype(int)})
sub = sample[["id"]].merge(sub, on="id", how="left")
sub["prediction"] = sub["prediction"].fillna(0).astype(int)

sub_path = SUBM_DIR / "submission_v2_lgbm_baseline.csv"
sub.to_csv(sub_path, index=False)
print(f"  Kaydedildi: {sub_path}")

# ─── Log ──────────────────────────────────────────────────────────────────────
log_path = SUBM_DIR / "submissions_log.csv"

# v1'i de ekle (zaten biliyoruz sonucunu)
v1_row = {
    "version": "v1", "model": "BM25 top-K fixed",
    "val_macro_f1": "N/A", "threshold": "top-14",
    "pos_rate_train": "N/A", "pos_rate_test": "13.0%",
    "public_score": "0.48",
    "neg_sampling": "none", "n_features": 0,
    "notes": "Fixed K=14, no learning, all-zero-ish predictions",
    "file": "submission_v1_bm25_topk.csv",
    "runtime_s": "64",
}
v2_row = {
    "version": "v2", "model": "LightGBM",
    "val_macro_f1": f"{best_f1:.4f}", "threshold": f"{best_thr:.2f}",
    "pos_rate_train": f"{y.mean()*100:.1f}%", "pos_rate_test": f"{test_preds.mean()*100:.1f}%",
    "public_score": "TBD",
    "neg_sampling": "50% same-cat + 50% random, ratio 1:4",
    "n_features": len(FEATURE_NAMES),
    "notes": "IDF overlap + metadata features, scale_pos_weight, threshold opt",
    "file": "submission_v2_lgbm_baseline.csv",
    "runtime_s": f"{time.time()-t_global:.0f}",
}

if log_path.exists():
    log_df = pd.read_csv(log_path)
    # v1 yoksa ekle
    if "v1" not in log_df["version"].values:
        log_df = pd.concat([pd.DataFrame([v1_row]), log_df], ignore_index=True)
    # v2 varsa güncelle, yoksa ekle
    if "v2" in log_df["version"].values:
        log_df.loc[log_df["version"] == "v2"] = list(v2_row.values())
    else:
        log_df = pd.concat([log_df, pd.DataFrame([v2_row])], ignore_index=True)
else:
    log_df = pd.DataFrame([v1_row, v2_row])

log_df.to_csv(log_path, index=False)
print(f"  Log güncellendi: {log_path}")

# ─── Özet ─────────────────────────────────────────────────────────────────────
total_t = time.time() - t_global
print("\n" + "=" * 60)
print("ÖZET")
print("=" * 60)
print(f"  Toplam süre          : {total_t:.0f}s ({total_t/60:.1f} dk)")
print(f"  Eğitim verisi        : {len(y_tr):,} çift (pos={y_tr.mean():.1%})")
print(f"  Validation Macro F1  : {best_f1:.4f}  (threshold={best_thr:.2f})")
print(f"  Test pozitif tahmin  : {test_preds.mean():.1%}")
print(f"  Submission           : {sub_path.name}")
print()
print("  Kaggle'a yükle ve public score'u submissions_log.csv'ye gir.")
print("=" * 60)
