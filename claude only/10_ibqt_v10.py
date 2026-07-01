"""
v10: Query-Level Transfer + Cross-Validation Analizi
=====================================================
1. Training sorgularından test sorgularına TF-IDF transfer
2. Submit ETMEDEN önce training hold-out ile F1 tahmini
3. Kalite kontrol: "luck", "pandora", "samsung" örnekleri

Fikir:
  "luck" test sorgusu → "avon luck parfüm edp 50ml" eğitim sorgusuna benzer
  → O sorgunun pozitiflerini (avon luck itemları) test için yüksek puan ver
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import time, sys

sys.stdout.reconfigure(encoding="utf-8")

BASE  = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA  = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
SUBM  = BASE / "claude only" / "submissions"
SUBM.mkdir(parents=True, exist_ok=True)

K_PREDICT  = 14
TOP_TRAIN_Q = 10   # Her test sorgusu için en benzer kaç eğitim sorgusu kullanılsın
LOWER_MAP  = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(t): return str(t).translate(LOWER_MAP).lower().strip()
STOPWORDS = {"ve","ile","bir","bu","da","de","mi","mı","mu","mü","için","ama",
             "gibi","olan","her","ne","ki","çok","az","en","set","adet","ml","gr","kg"}

# ─── 1. Veri ─────────────────────────────────────────────────────────────────
print("[1] Veri yükleniyor...")
t0 = time.time()
items  = pd.read_csv(DATA / "items.csv")
terms  = pd.read_csv(DATA / "terms.csv")
train  = pd.read_csv(DATA / "training_pairs.csv")
test   = pd.read_csv(DATA / "submission_pairs.csv")
sample = pd.read_csv(DATA / "sample_submission.csv")

iid_to_title = dict(zip(items["item_id"], items["title"].fillna("")))
iid_to_brand = dict(zip(items["item_id"], items["brand"].fillna("")))
iid_to_catl1 = {iid: str(c).split("/")[0] for iid, c in zip(items["item_id"], items["category"].fillna(""))}
tid_to_query  = dict(zip(terms["term_id"], terms["query"]))

# item → {train_tid} mapping
item_to_tids = defaultdict(set)
train_pos    = defaultdict(set)
for tid, iid in zip(train["term_id"].values, train["item_id"].values):
    train_pos[tid].add(iid)
    item_to_tids[iid].add(tid)

print(f"  {time.time()-t0:.1f}s | Train queries: {len(train_pos):,} | Items in train: {len(item_to_tids):,}")

def itext(iid):
    t = trl(iid_to_title.get(iid, ""))
    b = trl(iid_to_brand.get(iid, ""))
    return f"{t} {b}".strip()

# ─── 2. TF-IDF Vektörleri ────────────────────────────────────────────────────
print("[2] TF-IDF vektörleri...")
t0 = time.time()

train_tids   = list(train_pos.keys())
train_texts  = [trl(tid_to_query.get(t,"")) for t in train_tids]
test_tids_u  = list(test["term_id"].unique())
test_texts   = [trl(tid_to_query.get(t,"")) for t in test_tids_u]

# Char 2-4 gram — Türkçe kısa sorgu eşleştirme için ideal
vect = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4),
                       min_df=2, max_features=300000, sublinear_tf=True)
vect.fit(train_texts + test_texts)

train_vecs = vect.transform(train_texts)   # (N_train, V) sparse
test_vecs  = vect.transform(test_texts)    # (N_test, V)  sparse

train_idx = {tid: i for i, tid in enumerate(train_tids)}
test_idx  = {tid: i for i, tid in enumerate(test_tids_u)}
print(f"  Vocab: {len(vect.vocabulary_):,} | Süre: {time.time()-t0:.1f}s")

# ─── 3. Brand Detection ───────────────────────────────────────────────────────
print("[3] Brand detection mapping...")
from collections import Counter
q_tok_to_brand = defaultdict(Counter)
for tid, pos_iids in train_pos.items():
    q = trl(tid_to_query.get(tid, ""))
    brands = {trl(iid_to_brand.get(i,"")).split()[0]
              for i in pos_iids if trl(iid_to_brand.get(i,""))}
    brands -= {"", "nan"}
    for tok in q.split():
        if len(tok) >= 3 and tok not in STOPWORDS:
            for b in brands:
                q_tok_to_brand[tok][b] += 1

def detect_brand(q):
    votes = Counter()
    for tok in q.split():
        if tok in q_tok_to_brand:
            for b, c in q_tok_to_brand[tok].items():
                votes[b] += c
    if not votes:
        return None, 0.0
    total = sum(votes.values())
    best, cnt = votes.most_common(1)[0]
    return best, cnt/total

print(f"  Brand token map: {len(q_tok_to_brand):,} token")

# ─── 4. Scoring Fonksiyonu ───────────────────────────────────────────────────
def score_candidates(query_tid, candidate_iids,
                     ref_tids, ref_vecs, ref_train_pos,
                     topq=TOP_TRAIN_Q):
    """
    Verilen query için candidate item'larını puanla.
    ref_tids / ref_vecs: hangi eğitim sorguları referans alınacak
    """
    q = trl(tid_to_query.get(query_tid, ""))
    brand, brand_conf = detect_brand(q)
    has_brand = brand is not None and brand_conf > 0.35

    # Query TF-IDF vektörü
    q_vec = vect.transform([q])   # (1, V)

    # Tüm referans sorgularla benzerlik
    sims = (q_vec * ref_vecs.T).toarray()[0]   # (N_ref,)

    # En benzer TOP_TRAIN_Q eğitim sorgusunun pozitif item setini bul
    top_indices = np.argpartition(sims, -min(topq, len(sims)))[-min(topq, len(sims)):]
    top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

    # Transfer skoru: her item için en iyi eşleşen sorgunun benzerliği
    transfer = {}
    for ri in top_indices:
        sim = sims[ri]
        if sim < 0.05:
            break
        tid_ref = ref_tids[ri]
        for iid in ref_train_pos.get(tid_ref, set()):
            if iid not in transfer or transfer[iid] < sim:
                transfer[iid] = sim

    # Her adayı puan
    scores = {}
    for iid in candidate_iids:
        # SINYAL 1: Query transfer
        t_score = transfer.get(iid, 0.0)

        # SINYAL 2: Token overlap (test query ∩ item text)
        q_toks = set(t for t in q.split() if len(t) >= 3 and t not in STOPWORDS)
        it_txt = trl(iid_to_title.get(iid,"") + " " + iid_to_brand.get(iid,""))
        overlap = sum(1 for tok in q_toks if tok in it_txt) / max(len(q_toks), 1)

        # SINYAL 3: Brand bonus
        b_score = 0.0
        if has_brand:
            item_brand = trl(iid_to_brand.get(iid,"")).split()
            item_brand_str = " ".join(item_brand[:2])
            if brand and brand[:4] in item_brand_str:
                b_score = 1.0

        # Final: transfer dominant ise kullan, yoksa fallback
        if t_score > 0.1:
            final = 3.0*t_score + 0.5*overlap + 2.0*b_score
        elif has_brand and b_score > 0.5:
            final = 2.5*b_score + 1.0*overlap
        else:
            final = overlap + 0.1*t_score

        scores[iid] = final

    return scores

# ─── 5. Cross-Validation (Training Hold-Out) ─────────────────────────────────
print("\n[4] CROSS-VALIDATION — Training hold-out ile F1 tahmini...")
print("    (Submit etmeden önce gerçek performans tahmini)")

import random
random.seed(42)
all_train_tids = list(train_pos.keys())
random.shuffle(all_train_tids)
HOLDOUT_N = 2000
holdout_tids = all_train_tids[:HOLDOUT_N]
ref_tids_cv  = all_train_tids[HOLDOUT_N:]   # Geri kalan referans

ref_vecs_cv      = train_vecs[[train_idx[t] for t in ref_tids_cv]]
ref_train_pos_cv = {t: train_pos[t] for t in ref_tids_cv}

tp_total = 0; fp_total = 0; fn_total = 0
tn_total = 0; n_queries = 0

# Her holdout query için: eğitim positiflerini gizle, K=14 tahmin et
for tid in holdout_tids:
    true_pos = train_pos[tid]
    if not true_pos:
        continue

    # Bu query'nin test "adayları" = True positifs + sahteler
    # Sahteler: aynı kategoriden rastgele itemlar + diğer random itemlar
    true_pos_list = list(true_pos)
    cat = iid_to_catl1.get(true_pos_list[0], "")

    # Aynı kategoriden sahte adaylar
    same_cat_items = [i for i, c in iid_to_catl1.items() if c == cat and i not in true_pos]
    random.shuffle(same_cat_items)
    decoys = same_cat_items[:min(50, 100 - len(true_pos_list))]

    candidates = true_pos_list + decoys
    random.shuffle(candidates)
    if not candidates:
        continue

    # Puanla
    scores = score_candidates(tid, candidates, ref_tids_cv, ref_vecs_cv, ref_train_pos_cv)
    ranked = sorted(candidates, key=lambda i: scores.get(i,0), reverse=True)
    predicted_pos = set(ranked[:K_PREDICT])

    tp = len(predicted_pos & true_pos)
    fp = len(predicted_pos - true_pos)
    fn = len(true_pos - predicted_pos)
    tn = len(candidates) - len(true_pos) - fp

    tp_total += tp; fp_total += fp; fn_total += fn; tn_total += tn
    n_queries += 1

precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
recall    = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
f1_1      = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0

total_neg = tn_total + fp_total
prec_0 = tn_total / (tn_total + fn_total) if (tn_total + fn_total) > 0 else 0
rec_0  = tn_total / (tn_total + fp_total) if (tn_total + fp_total) > 0 else 0
f1_0   = 2*prec_0*rec_0/(prec_0+rec_0) if (prec_0+rec_0) > 0 else 0

macro_f1 = (f1_0 + f1_1) / 2

print(f"\n  === CROSS-VALIDATION SONUÇLARI ({n_queries} sorgu) ===")
print(f"  Precision (class 1): {precision:.3f}")
print(f"  Recall    (class 1): {recall:.3f}")
print(f"  F1        (class 1): {f1_1:.3f}")
print(f"  F1        (class 0): {f1_0:.3f}")
print(f"  MACRO F1 (tahmini) : {macro_f1:.3f}  ← Bu değer test skorumuzu tahmin ediyor")
print(f"  TP={tp_total}, FP={fp_total}, FN={fn_total}")

# ─── 6. Gerçek Test Submission ────────────────────────────────────────────────
print("\n[5] Test prediction oluşturuluyor...")
t0 = time.time()

test_grp_items = test.groupby("term_id")["item_id"].apply(list).to_dict()
test_grp_ids   = test.groupby("term_id")["id"].apply(list).to_dict()

predictions = {}
for i, tid in enumerate(test_tids_u):
    candidates = test_grp_items.get(tid, [])
    row_ids    = test_grp_ids.get(tid, [])
    scores = score_candidates(tid, candidates, train_tids, train_vecs, train_pos)
    ranked = sorted(candidates, key=lambda x: scores.get(x, 0), reverse=True)
    for j, iid in enumerate(ranked):
        pid = row_ids[candidates.index(iid)]
        predictions[pid] = 1 if j < K_PREDICT else 0

    if i % 5000 == 0:
        print(f"  {i}/{len(test_tids_u)}  {(time.time()-t0)/60:.1f}dk")

sub = sample[["id"]].copy()
sub["prediction"] = sub["id"].map(predictions).fillna(0).astype(int)
pos_count = sub["prediction"].sum()
print(f"  Pozitif: {pos_count:,} ({100*pos_count/len(sub):.1f}%)")

out_path = SUBM / "submission_v10_ibqt.csv"
sub.to_csv(str(out_path), index=False)
print(f"  Kaydedildi: {out_path}")

# ─── 7. Kalite Kontrol ───────────────────────────────────────────────────────
print("\n[6] KALİTE KONTROL — Örnek sorgular")
test_v10 = test.merge(sub, on="id", how="left")

cases = [
    ("TERM_eacb5395",  "luck",                "avon"),
    ("TERM_85856557",  "pandora kadın takı",  "pandora"),
    ("TERM_68034abd",  "samsung galaxy s24 fe","samsung"),
]
for tid, label, brand in cases:
    q = tid_to_query.get(tid, "?")
    grp = test_v10[test_v10["term_id"]==tid]
    pos = grp[grp["prediction"]==1]["item_id"].tolist()
    brand_cnt = sum(1 for iid in pos if brand in trl(iid_to_brand.get(iid,"")))
    print(f"\n  '{q}' → {brand_cnt}/14 doğru marka ({brand})")
    for iid in pos[:5]:
        t = iid_to_title.get(iid,"?")[:52]
        b = iid_to_brand.get(iid,"?")[:14]
        mark = " ✓" if brand in trl(b) else " ✗"
        print(f"    [{b}] {t}{mark}")

# Ek: "luck" için skora göre tüm sıralamanın başı
print("\n  --- 'luck' score sıralaması (top-5, ham puanlar) ---")
luck_tid = "TERM_eacb5395"
luck_cands = test_grp_items.get(luck_tid, [])
luck_scores = score_candidates(luck_tid, luck_cands, train_tids, train_vecs, train_pos)
luck_ranked = sorted(luck_cands, key=lambda x: luck_scores.get(x,0), reverse=True)
for iid in luck_ranked[:7]:
    t = iid_to_title.get(iid,"?")[:52]
    b = iid_to_brand.get(iid,"?")[:14]
    sc = luck_scores.get(iid, 0)
    mark = " ✓AVON" if "avon" in trl(b) else ""
    print(f"    [{b}] {t}  score={sc:.3f}{mark}")

print(f"\n=== BİTTİ | CV Macro F1 tahmini: {macro_f1:.3f} ===")
