"""
v6 Submission — Category-Aware KNN + Token Overlap
====================================================
Hata analizinden dersler:
  - K=8 (yüzde tabanlı) → K=14 MUTLAK
  - Fine-tuning embedding collapse → fine-tuning yok
  - Kategori filtresi yok → kategori bonusu ekliyoruz
  - 3680 adaylı sorgular için 294 tahmin → sabit K=14

Strateji:
  1. KNN: eğitim query benzerliği (E5 fine-tuned query emb - sorgu-sorgu similar OK)
  2. Kategori tahmini: eğitim datası token→kategori
  3. Token overlap: query kelimeleri item title'da arama
  4. Combined score → top-14 per query (MUTLAK)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
import torch
import time, sys

sys.stdout.reconfigure(encoding="utf-8")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE  = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA  = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
CACHE = BASE / "claude only" / "emb_cache"
SUBM  = BASE / "claude only" / "submissions"
SUBM.mkdir(parents=True, exist_ok=True)

K_PREDICT = 14       # MUTLAK SAYI — yüzde DEĞİL
KNN_TOP   = 50       # Benzer eğitim sorgusu sayısı

LOWER_MAP = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")

def tr_lower(text: str) -> str:
    return str(text).translate(LOWER_MAP).lower().strip()

# ─── 1. Veri Yükle ───────────────────────────────────────────────────────────
print("[1] Veri yükleniyor...")
t0 = time.time()
items = pd.read_csv(DATA / "items.csv")
terms = pd.read_csv(DATA / "terms.csv")
train = pd.read_csv(DATA / "training_pairs.csv")
test  = pd.read_csv(DATA / "submission_pairs.csv")
sample = pd.read_csv(DATA / "sample_submission.csv")

iid_to_title = dict(zip(items["item_id"], items["title"].fillna("")))
iid_to_brand = dict(zip(items["item_id"], items["brand"].fillna("")))
iid_to_cat   = dict(zip(items["item_id"], items["category"].fillna("")))
iid_to_catl1 = {iid: str(cat).split("/")[0].strip() for iid, cat in iid_to_cat.items()}
tid_to_query = dict(zip(terms["term_id"], terms["query"]))

train_q_tids = train["term_id"].unique()
test_q_tids  = test["term_id"].unique()
print(f"  Egitim query: {len(train_q_tids):,}  Test query: {len(test_q_tids):,}")
print(f"  Veri yükleme: {time.time()-t0:.1f}s")

# ─── 2. Egitim Pozitif Haritası ───────────────────────────────────────────────
print("[2] Eğitim pozitif haritası oluşturuluyor...")
train_pos = defaultdict(set)       # term_id → {item_id}
train_item_freq = Counter()        # item_id → kaç training query'de pozitif
for tid, iid in zip(train["term_id"].values, train["item_id"].values):
    train_pos[tid].add(iid)
    train_item_freq[iid] += 1

# ─── 3. Kategori Tahmin Sistemi ──────────────────────────────────────────────
print("[3] Kategori tahmin sistemi kuruluyor...")
t0 = time.time()

# a) Eğitim querysi → baskın kategori
train_q_cat = {}                   # term_id → most_common_category
for tid, group in train.merge(items[["item_id","category"]], on="item_id").groupby("term_id"):
    cats = group["category"].str.split("/", expand=False).str[0].value_counts()
    train_q_cat[tid] = cats.idxmax()

# b) Token → kategori vote sözlüğü
# Kısa tokenları (≤2 harf) ve yaygın stopwords atla
STOPWORDS = {"ve","ile","bir","bu","da","de","mi","mı","mu","mü","ben","sen",
             "o","biz","siz","için","ama","veya","gibi","göre","kadar","sonra",
             "the","a","an","of","in","to","for","on","at","by","or","and",
             "i","ii","iii","iv","v","vi","vii","1","2","3","4","5","6","7",
             "0","10","100","200","ml","gr","kg","cm","mm","lt","adet","paket",
             "xl","xs","xxl","s","m","l"}

token_to_cat = defaultdict(Counter)   # token → {kategori: count}
for tid, cat in train_q_cat.items():
    q = tr_lower(tid_to_query.get(tid, ""))
    for tok in q.split():
        if len(tok) > 2 and tok not in STOPWORDS:
            token_to_cat[tok][cat] += 1

print(f"  Kategori vocab: {len(token_to_cat):,} token   Süre: {time.time()-t0:.1f}s")

def predict_category(query_text: str):
    """Sorgu metni → (predicted_category, confidence) döndür."""
    votes = Counter()
    for tok in tr_lower(query_text).split():
        if tok in token_to_cat:
            total = sum(token_to_cat[tok].values())
            for cat, cnt in token_to_cat[tok].items():
                votes[cat] += cnt / total   # normalize by token popularity
    if not votes:
        return None, 0.0
    top_cat, top_v = votes.most_common(1)[0]
    return top_cat, top_v

# ─── 4. KNN Collaborative (GPU) ──────────────────────────────────────────────
print("[4] KNN: query embeddings yükleniyor...")
t0 = time.time()

train_q_emb_file = CACHE / "train_q_embs_e5_v5.npy"
test_q_emb_file  = CACHE / "test_q_embs_e5_v5.npy"

use_knn = train_q_emb_file.exists() and test_q_emb_file.exists()

if use_knn:
    train_q_embs = np.load(str(train_q_emb_file))    # (17968, 768)
    test_q_embs  = np.load(str(test_q_emb_file))     # (32185, 768)

    # Sıra index haritaları
    train_tid_arr = np.load(str(CACHE / "train_q_ids_e5_v5.npy"), allow_pickle=True)
    test_tid_arr  = np.load(str(CACHE / "test_q_ids_e5_v5.npy"),  allow_pickle=True)

    train_tid_to_idx = {tid: i for i, tid in enumerate(train_tid_arr)}
    test_tid_to_idx  = {tid: i for i, tid in enumerate(test_tid_arr)}

    # GPU'ya taşı
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    TQ = torch.tensor(train_q_embs, dtype=torch.float32, device=device)  # (N_train, D)
    TEQ = torch.tensor(test_q_embs, dtype=torch.float32, device=device)  # (N_test, D)

    # Normalize (cosine sim için)
    TQ  = TQ  / TQ.norm(dim=1, keepdim=True).clamp(min=1e-8)
    TEQ = TEQ / TEQ.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Batch matmul → top-50 benzer eğitim query per test query
    BATCH = 512
    n_test = TEQ.shape[0]
    knn_top_idxs  = torch.zeros(n_test, KNN_TOP, dtype=torch.long, device=device)
    knn_top_sims  = torch.zeros(n_test, KNN_TOP, dtype=torch.float32, device=device)

    for start in range(0, n_test, BATCH):
        end  = min(start + BATCH, n_test)
        sims = TEQ[start:end] @ TQ.T     # (batch, N_train)
        top_vals, top_idx = sims.topk(KNN_TOP, dim=1)
        knn_top_idxs[start:end] = top_idx
        knn_top_sims[start:end] = top_vals

    knn_top_idxs = knn_top_idxs.cpu().numpy()
    knn_top_sims = knn_top_sims.cpu().numpy()
    print(f"  KNN hesaplama: {time.time()-t0:.1f}s  ({n_test} test query)")

    # KNN score sözlüğü: test_tid → {item_id: weighted_score}
    print("[4b] KNN item skorları hesaplanıyor...")
    t0 = time.time()
    knn_scores = {}      # test_tid → {item_id: score}

    for ti, tid in enumerate(test_tid_arr):
        top_idxs = knn_top_idxs[ti]
        top_sims = knn_top_sims[ti]
        item_scores = Counter()
        for rank, (train_idx, sim) in enumerate(zip(top_idxs, top_sims)):
            if sim < 0.3:
                break
            train_tid = train_tid_arr[train_idx]
            for item_id in train_pos.get(train_tid, set()):
                # Sim ağırlıklı sayım
                item_scores[item_id] += float(sim)
        knn_scores[tid] = item_scores

    knn_coverage = sum(1 for sc in knn_scores.values() if sc) / len(knn_scores)
    print(f"  KNN coverage: {knn_coverage:.1%}  Süre: {time.time()-t0:.1f}s")
else:
    print("  UYARI: KNN embeddings bulunamadı, KNN devre dışı.")
    knn_scores = {}

# ─── 5. Test Adaylarına Skor Ver ─────────────────────────────────────────────
print("[5] Skorlama + tahmin...")
t0 = time.time()

# Test adaylarını grupla
test_copy = test.copy()
test_copy["prediction"] = 0

# Token overlap helper — hızlı
def query_tokens(q_text: str) -> set:
    return set(tr_lower(q_text).split()) - STOPWORDS

def token_overlap(q_toks: set, title: str, brand: str) -> float:
    item_toks = set(tr_lower(title + " " + brand).split())
    if not q_toks or not item_toks:
        return 0.0
    inter = q_toks & item_toks
    return len(inter) / len(q_toks)  # query coverage

# Önce kategori tahminlerini hazırla (tüm test queryleri için)
test_cat_pred = {}
for tid in test_q_tids:
    q = tid_to_query.get(tid, "")
    cat, conf = predict_category(q)
    test_cat_pred[tid] = (cat, conf)

# Grup bazlı skorlama
groups = test_copy.groupby("term_id")
total_groups = len(groups)
interval = max(1, total_groups // 20)

predictions = {}   # id → 0/1

for gi, (tid, grp) in enumerate(groups):
    if gi % interval == 0:
        elapsed = time.time() - t0
        print(f"  {gi}/{total_groups} ({100*gi/total_groups:.0f}%)  {elapsed:.0f}s geçti")

    q_text  = tid_to_query.get(tid, "")
    q_toks  = query_tokens(q_text)
    pred_cat, cat_conf = test_cat_pred.get(tid, (None, 0.0))

    # KNN skorları (item_id → score)
    knn_sc = knn_scores.get(tid, {})
    max_knn = max(knn_sc.values()) if knn_sc else 0.0

    # Her aday için skor hesapla
    scores = []
    for row_id, item_id in zip(grp["id"].values, grp["item_id"].values):
        # a) KNN skoru (normalize)
        knn_val = knn_sc.get(item_id, 0.0)
        knn_norm = knn_val / (max_knn + 1e-8)

        # b) Kategori bonusu
        item_cat = iid_to_catl1.get(item_id, "")
        if pred_cat and cat_conf > 0.5:
            cat_bonus = 1.0 if item_cat == pred_cat else 0.05
        else:
            cat_bonus = 0.3   # Belirsiz kategori → daha az penaltı

        # c) Token overlap
        title = iid_to_title.get(item_id, "")
        brand = iid_to_brand.get(item_id, "")
        tok_score = token_overlap(q_toks, title, brand)

        # d) Combined — ağırlıklar
        if max_knn > 0:
            # KNN sinyali var: KNN dominant
            final = 0.50 * knn_norm + 0.35 * cat_bonus + 0.15 * tok_score
        else:
            # KNN sinyali yok: kategori + token dominant
            final = 0.00 * knn_norm + 0.55 * cat_bonus + 0.45 * tok_score

        scores.append((row_id, final))

    # MUTLAK K=14 — yüzde değil
    k = K_PREDICT
    scores.sort(key=lambda x: x[1], reverse=True)
    for i, (row_id, sc) in enumerate(scores):
        predictions[row_id] = 1 if i < k else 0

print(f"  Skorlama tamamlandı: {time.time()-t0:.1f}s")

# ─── 6. Submission Oluştur ────────────────────────────────────────────────────
print("[6] Submission oluşturuluyor...")
pred_df = pd.DataFrame({"id": list(predictions.keys()), "prediction": list(predictions.values())})
sub = sample[["id"]].merge(pred_df, on="id", how="left")
sub["prediction"] = sub["prediction"].fillna(0).astype(int)

pos_count = sub["prediction"].sum()
print(f"  Toplam tahmin: {len(sub):,}")
print(f"  Pozitif: {pos_count:,} ({100*pos_count/len(sub):.1f}%)")
print(f"  Query başına ort pozitif: {pos_count/test['term_id'].nunique():.1f}")

# Kategori dağılımı kontrol
sub_with_test = test.merge(sub, on="id")
pos_items = sub_with_test[sub_with_test["prediction"]==1]
pos_items_cats = [iid_to_catl1.get(iid,"?") for iid in pos_items["item_id"]]
print(f"\nPozitif tahminlerin L1 kategori dağılımı:")
print(pd.Series(pos_items_cats).value_counts().head(10).to_string())

out_path = SUBM / "submission_v6_cat_knn_k14.csv"
sub.to_csv(str(out_path), index=False)
print(f"\n  Kaydedildi: {out_path}")

# Log güncelle
log_path = SUBM / "submissions_log.csv"
log = pd.read_csv(str(log_path))
new_row = pd.DataFrame([{
    "version": "v6",
    "description": "Category-aware KNN + token overlap, K=14 absolute",
    "positive_rate": f"{100*pos_count/len(sub):.1f}%",
    "public_score": "TBD",
    "notes": "Category prediction from training tokens + KNN collab + token overlap"
}])
log = pd.concat([log, new_row], ignore_index=True)
log.to_csv(str(log_path), index=False)

print("\n=== TAMAMLANDI ===")
