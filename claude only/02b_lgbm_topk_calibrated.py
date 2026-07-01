# -*- coding: utf-8 -*-
"""
Submission v2b — LightGBM per-query Top-K (Kalibre edilmiş)
============================================================
v2 sorunu: Test'te 100 aday BM25 ile seçilmiş (zor negatifler).
Model bunları pozitif sanıyor → %33.7 pozitif (beklenen %13).

Çözüm: LightGBM'i ranker olarak kullan.
Her query için top-K = 14 (training'den kalibre).
Submission format: submissions/submission_v2b_lgbm_topk.csv
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path

DATA_DIR = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL\trendyol-e-ticaret-yarismasi-2026-kaggle")
SUBM_DIR = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL\claude only\submissions")

t0 = time.time()
print("Veriler yükleniyor...")
test   = pd.read_csv(DATA_DIR / "submission_pairs.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
train  = pd.read_csv(DATA_DIR / "training_pairs.csv")

# v2'nin ürettiği submission'dan skoru çek (scores yok, sadece 0/1 var)
# Yeniden çalıştırmak yerine 02_lgbm_baseline.py'den model çıktısını kullanıyoruz.
# Bu script, 02_lgbm_baseline.py ile birlikte çalışmak için tasarlanmıştır.
# LightGBM modelini yeniden eğitmek yerine sadece threshold değiştireceğiz.

# ─── LightGBM + feature pipeline'ı import et ──────────────────────────────────
# Aşağıdaki kısım 02_lgbm_baseline.py ile aynı pipeline'ı çalıştırır,
# sadece son adımda top-K kalibrasyonu yapar.

import re, math
from collections import Counter, defaultdict
import lightgbm as lgb
from sklearn.metrics import f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

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

print("Items yükleniyor...")
items  = pd.read_csv(DATA_DIR / "items.csv")
terms  = pd.read_csv(DATA_DIR / "terms.csv")

items["full_text"] = (items["title"].fillna("") + " " +
                      items["category"].fillna("") + " " +
                      items["brand"].fillna("") + " " +
                      items["gender"].fillna("") + " " +
                      items["age_group"].fillna("") + " " +
                      items["attributes"].fillna(""))
items["cat_l1"] = items["category"].fillna("").apply(
    lambda x: x.split("/")[0].strip() if x else "")

print("  Tokenize ediliyor...")
item_full_toks  = [tok_set(s) for s in tqdm(items["full_text"], ncols=70)]
item_title_toks = [tok_set(s) for s in items["title"].fillna("")]
item_cat_toks   = [tok_set(s) for s in items["category"].fillna("")]
item_brand_norm = [norm_tr(s) for s in items["brand"].fillna("")]
item_gender_str = [norm_tr(s) for s in items["gender"].fillna("")]
cat_l1_vals  = items["cat_l1"].unique()
cat_l1_enc_m = {c: i for i, c in enumerate(cat_l1_vals)}
item_cat_enc = [cat_l1_enc_m.get(c, 0) for c in items["cat_l1"]]
item_id_arr  = items["item_id"].values
iid_to_idx   = {iid: i for i, iid in enumerate(item_id_arr)}
item_cat_arr = items["cat_l1"].values
item_cat_map = dict(zip(item_id_arr, item_cat_arr))

print("  IDF hesaplanıyor...")
df_cnt = Counter()
N = len(item_full_toks)
for ftoks in item_full_toks:
    df_cnt.update(ftoks)
idf = {tok: math.log((N - cnt + 0.5) / (cnt + 0.5) + 1) for tok, cnt in df_cnt.items()}

q_toks_map = dict(zip(terms["term_id"], terms["query"].apply(tokenize)))
q_set_map  = dict(zip(terms["term_id"], terms["query"].apply(tok_set)))

FEATURE_NAMES = [
    "idf_score_full","idf_score_title","overlap_full_ratio","overlap_title_ratio",
    "overlap_cat_ratio","jaccard_title","inter_full_count","inter_title_count",
    "exact_sub_in_title","brand_in_query","gender_match","color_in_query",
    "color_match","query_len","title_len","cat_l1_enc"
]

def compute_features_batch(term_ids, item_ids):
    results = []
    for tid, iid in zip(term_ids, item_ids):
        idx = iid_to_idx.get(iid)
        if idx is None:
            results.append([0.0]*16); continue
        q_t = q_toks_map.get(tid, []); q_s = q_set_map.get(tid, frozenset())
        q_len = len(q_t)
        if q_len == 0:
            results.append([0.0]*16); continue
        ft = item_full_toks[idx]; tt = item_title_toks[idx]; ct = item_cat_toks[idx]
        idf_full  = sum(idf.get(t,0) for t in q_t if t in ft)
        idf_title = sum(idf.get(t,0) for t in q_t if t in tt)
        inter_full=len(q_s&ft); inter_title=len(q_s&tt); inter_cat=len(q_s&ct)
        full_r=inter_full/q_len; title_r=inter_title/q_len; cat_r=inter_cat/q_len
        u_tt=len(q_s|tt); jacc=inter_title/u_tt if u_tt>0 else 0.0
        q_joined=" ".join(q_t); title_norm=" ".join(tt)
        exact=int(q_joined in title_norm)
        brand=item_brand_norm[idx]
        b_in_q=int(bool(brand) and brand!="unknown" and
                   any(brand in tok or tok in brand for tok in q_t if len(tok)>=3))
        q_gen=q_s&GENDERS
        if not q_gen: gender_m=1
        else: ig=tok_set(item_gender_str[idx]); gender_m=int(bool(q_gen&ig))
        q_col=q_s&COLORS; col_in_q=int(bool(q_col))
        col_match=int(bool(q_col&ft)) if col_in_q else 0
        results.append([idf_full,idf_title,full_r,title_r,cat_r,jacc,
                        float(inter_full),float(inter_title),float(exact),float(b_in_q),
                        float(gender_m),float(col_in_q),float(col_match),
                        float(q_len),float(len(tt)),float(item_cat_enc[idx])])
    return results

# ─── Negatif Örnekleme ────────────────────────────────────────────────────────
print("\nNegatif örnekleme (vektörize)...")
np.random.seed(42)
cat_unique = np.unique(item_cat_arr)
cat_idx_map_local = {cat: np.where(item_cat_arr == cat)[0] for cat in cat_unique}
pos_item_cat = np.array([item_cat_map.get(iid,"") for iid in train["item_id"].values], dtype=object)
pos_set = set(zip(train["term_id"].values.tolist(), train["item_id"].values.tolist()))

HARD_N=2; EASY_N=2
hard_tl,hard_il=[],[]
for cat in cat_unique:
    cm=pos_item_cat==cat; ct=train["term_id"].values[cm]; pi=cat_idx_map_local[cat]
    if len(pi)==0 or len(ct)==0: continue
    si=np.random.choice(pi,len(ct)*HARD_N,replace=True)
    hard_tl.append(np.repeat(ct,HARD_N)); hard_il.append(item_id_arr[si])

easy_idx=np.random.randint(0,len(item_id_arr),len(train)*EASY_N)
easy_il=item_id_arr[easy_idx]; easy_tl=np.repeat(train["term_id"].values,EASY_N)

neg_term_ids=np.concatenate(hard_tl+[easy_tl])
neg_item_ids=np.concatenate(hard_il+[easy_il])
neg_mask=np.array([(t,i) not in pos_set for t,i in zip(neg_term_ids,neg_item_ids)],dtype=bool)
neg_term_ids=neg_term_ids[neg_mask]; neg_item_ids=neg_item_ids[neg_mask]

# ─── Feature Matrix ───────────────────────────────────────────────────────────
print("Özellikler hesaplanıyor...")
BATCH=50_000
pos_tids=train["term_id"].values.tolist(); pos_iids=train["item_id"].values.tolist()
X_pos=[]
for i in tqdm(range(0,len(pos_tids),BATCH),desc="  pos",ncols=70):
    X_pos.extend(compute_features_batch(pos_tids[i:i+BATCH],pos_iids[i:i+BATCH]))
X_neg=[]
for i in tqdm(range(0,len(neg_term_ids),BATCH),desc="  neg",ncols=70):
    X_neg.extend(compute_features_batch(neg_term_ids[i:i+BATCH].tolist(),neg_item_ids[i:i+BATCH].tolist()))
X=np.array(X_pos+X_neg,dtype=np.float32)
y=np.array([1]*len(X_pos)+[0]*len(X_neg),dtype=np.int8)
perm=np.random.permutation(len(y)); X,y=X[perm],y[perm]
n_val=int(0.2*len(y)); X_val,y_val=X[:n_val],y[:n_val]; X_tr,y_tr=X[n_val:],y[n_val:]
print(f"  Matrix: {X.shape}  pos={y.mean():.2f}")

# ─── LightGBM ─────────────────────────────────────────────────────────────────
print("\nLightGBM eğitimi...")
spw=(y_tr==0).sum()/max((y_tr==1).sum(),1)
params={"objective":"binary","metric":"auc","learning_rate":0.05,"num_leaves":63,
        "min_child_samples":30,"feature_fraction":0.85,"bagging_fraction":0.85,
        "bagging_freq":5,"scale_pos_weight":spw,"verbose":-1,"n_jobs":-1}
dtrain=lgb.Dataset(X_tr,label=y_tr,feature_name=FEATURE_NAMES)
dval=lgb.Dataset(X_val,label=y_val,feature_name=FEATURE_NAMES,reference=dtrain)
cbs=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(100)]
model=lgb.train(params,dtrain,num_boost_round=600,valid_sets=[dval],callbacks=cbs)

val_prob=model.predict(X_val)
best_f1,best_thr=0.0,0.5
for thr in np.arange(0.05,0.95,0.01):
    preds=(val_prob>=thr).astype(int)
    f1=f1_score(y_val,preds,average="macro",zero_division=0)
    if f1>best_f1: best_f1,best_thr=f1,float(thr)
print(f"  Val Macro F1: {best_f1:.4f}  threshold={best_thr:.2f}")

# ─── Test Inference + Per-Query Top-K ─────────────────────────────────────────
print("\nTest skoru hesaplanıyor...")
test_tids=test["term_id"].values.tolist(); test_iids=test["item_id"].values.tolist()
X_test_b=[]
for i in tqdm(range(0,len(test_tids),BATCH),ncols=70):
    X_test_b.append(np.array(compute_features_batch(test_tids[i:i+BATCH],test_iids[i:i+BATCH]),dtype=np.float32))
X_test=np.vstack(X_test_b)
test_prob=model.predict(X_test)

# K calibration from training
K_FIXED = max(1, round(train.groupby("term_id")["item_id"].count().mean()))
print(f"\nPer-query Top-K kalibrasyonu: K={K_FIXED}")

# Attach scores to test dataframe
test_copy = test.copy()
test_copy["lgbm_score"] = test_prob

# Per-query top-K
test_copy["prediction"] = 0
for tid, grp in tqdm(test_copy.groupby("term_id"), desc="  calibrating", ncols=70):
    top_k_ids = grp.nlargest(K_FIXED, "lgbm_score").index
    test_copy.loc[top_k_ids, "prediction"] = 1

# Global threshold submission as well (for comparison)
test_copy["pred_global_thr"] = (test_copy["lgbm_score"] >= best_thr).astype(int)

# ─── Save submissions ─────────────────────────────────────────────────────────
# v2b: Per-query top-K
sub_topk = sample[["id"]].merge(test_copy[["id","prediction"]], on="id", how="left")
sub_topk["prediction"] = sub_topk["prediction"].fillna(0).astype(int)
path_topk = SUBM_DIR / "submission_v2b_lgbm_topk.csv"
sub_topk.to_csv(path_topk, index=False)
print(f"\n  [v2b top-K] Pozitif: {sub_topk['prediction'].sum():,} ({sub_topk['prediction'].mean()*100:.1f}%)")

# v2c: Global threshold
sub_gthr = sample[["id"]].merge(test_copy[["id","pred_global_thr"]].rename(
    columns={"pred_global_thr":"prediction"}), on="id", how="left")
sub_gthr["prediction"] = sub_gthr["prediction"].fillna(0).astype(int)
path_gthr = SUBM_DIR / "submission_v2c_lgbm_global_thr.csv"
sub_gthr.to_csv(path_gthr, index=False)
print(f"  [v2c global thr] Pozitif: {sub_gthr['prediction'].sum():,} ({sub_gthr['prediction'].mean()*100:.1f}%)")

# ─── Log ──────────────────────────────────────────────────────────────────────
log_path = SUBM_DIR / "submissions_log.csv"
if log_path.exists():
    log_df = pd.read_csv(log_path)
else:
    log_df = pd.DataFrame()

new_rows = [
    {"version":"v2b","model":"LightGBM per-query top-K",
     "val_macro_f1":f"{best_f1:.4f}","threshold":f"top-{K_FIXED}",
     "pos_rate_train":f"{y.mean()*100:.1f}%","pos_rate_test":f"{sub_topk['prediction'].mean()*100:.1f}%",
     "public_score":"TBD","neg_sampling":"50% same-cat + 50% random, 1:4",
     "n_features":16,"notes":"LightGBM ranker, per-query top-14 calibration",
     "file":"submission_v2b_lgbm_topk.csv","runtime_s":f"{time.time()-t0:.0f}"},
    {"version":"v2c","model":"LightGBM global threshold",
     "val_macro_f1":f"{best_f1:.4f}","threshold":f"{best_thr:.2f}",
     "pos_rate_train":f"{y.mean()*100:.1f}%","pos_rate_test":f"{sub_gthr['prediction'].mean()*100:.1f}%",
     "public_score":"TBD","neg_sampling":"50% same-cat + 50% random, 1:4",
     "n_features":16,"notes":"LightGBM, global threshold from val F1 opt",
     "file":"submission_v2c_lgbm_global_thr.csv","runtime_s":f"{time.time()-t0:.0f}"},
]
log_df = pd.concat([log_df, pd.DataFrame(new_rows)], ignore_index=True)
log_df.to_csv(log_path, index=False)

print(f"\n  Toplam süre: {time.time()-t0:.0f}s")
print("\nYükleyecek dosyalar:")
print(f"  1) {path_topk.name}  → ~%13.3 pozitif (ÖNERILEN)")
print(f"  2) {path_gthr.name}  → ~%33 pozitif (karşılaştırma)")
