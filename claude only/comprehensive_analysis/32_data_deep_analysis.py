"""
32_data_deep_analysis.py — Derin Veri Analizi
================================================
Çalıştırınca:
  - Veri boyutu, kapsam, label density
  - Negatif dağılım analizi (v22/v23 negatives)
  - Test pair yapısı (query başına kaç aday)
  - OOF proba dağılımları
  - v23 vs v24 proba karşılaştırması
  - Unlabeled positive riski tahmini (v24 hard neg içinde)
  - Feature importance dağılımı

Çıktı: comprehensive_analysis/data_analysis_output.txt
"""

import sys, time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.stdout.reconfigure(encoding="utf-8")

BASE   = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA   = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
CACHE  = BASE / "claude only"
OUT    = BASE / "claude only" / "comprehensive_analysis"
OUT.mkdir(exist_ok=True)

LOG = open(str(OUT / "data_analysis_output.txt"), "w", encoding="utf-8")

def pr(*args, **kw):
    print(*args, **kw)
    print(*args, **kw, file=LOG)
    LOG.flush()

t0 = time.time()
pr("=" * 70)
pr("DERİN VERİ ANALİZİ — Trendyol 2026")
pr("=" * 70)

# ─────────────────────────────────────────────────────────────────────
# 1. TEMEL VERİ YÜKLEMESİ
# ─────────────────────────────────────────────────────────────────────
pr("\n[1] VERİ YÜKLENİYOR...")
LOWER = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(s): return str(s).translate(LOWER).lower().strip()

items       = pd.read_csv(DATA / "items.csv")
terms       = pd.read_csv(DATA / "terms.csv")
train_pairs = pd.read_csv(DATA / "training_pairs.csv")
sub_pairs   = pd.read_csv(DATA / "submission_pairs.csv")

for col in ["title","category","brand","gender","age_group","attributes"]:
    items[col] = items[col].fillna("unknown").apply(trl)
terms["query"] = terms["query"].fillna("").apply(trl)

train_tids = set(train_pairs["term_id"].unique())
test_tids  = set(sub_pairs["term_id"].unique())
overlap    = train_tids & test_tids

pr(f"  items.csv       : {len(items):,} ürün")
pr(f"  terms.csv       : {len(terms):,} sorgu")
pr(f"  training_pairs  : {len(train_pairs):,} çift (hepsi pozitif)")
pr(f"  submission_pairs: {len(sub_pairs):,} çift")
pr(f"  Train term_id   : {len(train_tids):,}")
pr(f"  Test term_id    : {len(test_tids):,}")
pr(f"  Train-test OVERLAP: {len(overlap)} ← {'✓ sıfır (iyi)' if len(overlap)==0 else '⚠ VAR (kötü!)'}")

# ─────────────────────────────────────────────────────────────────────
# 2. QUERY ANALİZİ
# ─────────────────────────────────────────────────────────────────────
pr("\n[2] QUERY YAPISI ANALİZİ")

terms_train = terms[terms["term_id"].isin(train_tids)].copy()
terms_test  = terms[terms["term_id"].isin(test_tids)].copy()

def qlens(df):
    return df["query"].apply(lambda q: len(str(q).split()))

pr("  [Train queries]")
trl_lens = qlens(terms_train)
pr(f"    Ortalama uzunluk: {trl_lens.mean():.2f} token")
pr(f"    Medyan          : {trl_lens.median():.0f} token")
pr(f"    1-kelime        : {(trl_lens==1).sum()} ({100*(trl_lens==1).mean():.1f}%)")
pr(f"    2-kelime        : {(trl_lens==2).sum()} ({100*(trl_lens==2).mean():.1f}%)")
pr(f"    3-kelime        : {(trl_lens==3).sum()} ({100*(trl_lens==3).mean():.1f}%)")
pr(f"    4+kelime        : {(trl_lens>=4).sum()} ({100*(trl_lens>=4).mean():.1f}%)")

pr("  [Test queries]")
test_lens = qlens(terms_test)
pr(f"    Ortalama uzunluk: {test_lens.mean():.2f} token")
pr(f"    1-kelime        : {(test_lens==1).sum()} ({100*(test_lens==1).mean():.1f}%)")
pr(f"    2-kelime        : {(test_lens==2).sum()} ({100*(test_lens==2).mean():.1f}%)")

# ─────────────────────────────────────────────────────────────────────
# 3. LABEL DENSITY ANALİZİ
# ─────────────────────────────────────────────────────────────────────
pr("\n[3] LABEL DENSITY (train pozitif başına)")

pos_per_term = train_pairs.groupby("term_id").size()
pr(f"  Train: term başına ortalama pozitif: {pos_per_term.mean():.2f}")
pr(f"  Train: term başına medyan pozitif  : {pos_per_term.median():.1f}")
pr(f"  Train: tek pozitifli term          : {(pos_per_term==1).sum()} ({100*(pos_per_term==1).mean():.1f}%)")
pr(f"  Train: 5+ pozitifli term           : {(pos_per_term>=5).sum()} ({100*(pos_per_term>=5).mean():.1f}%)")
pr(f"  Train: 10+ pozitifli term          : {(pos_per_term>=10).sum()}")

# Test başına aday sayısı
cands_per_term = sub_pairs.groupby("term_id").size()
pr(f"\n  Test: term başına ortalama aday    : {cands_per_term.mean():.1f}")
pr(f"  Test: term başına medyan aday      : {cands_per_term.median():.0f}")
pr(f"  Test: min aday                     : {cands_per_term.min()}")
pr(f"  Test: max aday                     : {cands_per_term.max()}")
pr(f"  Test: 10+ aday olan term           : {(cands_per_term>=10).sum()} ({100*(cands_per_term>=10).mean():.1f}%)")
pr(f"  Test: 100+ aday olan term          : {(cands_per_term>=100).sum()}")

# ─────────────────────────────────────────────────────────────────────
# 4. NEGATİF KALİTE: UNLABELED POSITIVE RİSKİ
# ─────────────────────────────────────────────────────────────────────
pr("\n[4] UNLABELED POSITIVE RİSKİ (v24 hard neg sorusu)")

# TF-IDF hard neg pairs
hard_neg_path = CACHE / "tfidf_hardneg_pairs_v24.csv"
if hard_neg_path.exists():
    hard_neg = pd.read_csv(str(hard_neg_path))
    pr(f"  v24 hard neg pairs: {len(hard_neg):,}")

    # Bu çiftlerde query ile item tokenı örtüşme oranını hesapla
    hard_neg["term_id"] = hard_neg["term_id"].astype(str)
    hard_neg["item_id"] = hard_neg["item_id"].astype(str)

    tid_to_q = dict(zip(terms["term_id"].astype(str), terms["query"]))
    iid_to_title = dict(zip(items["item_id"].astype(str), items["title"]))

    queries = hard_neg["term_id"].map(tid_to_q).fillna("")
    titles  = hard_neg["item_id"].map(iid_to_title).fillna("")

    def q_cov(q, t):
        qw = set(q.split())
        return len(qw & set(t.split()))/len(qw) if qw else 0.0

    sample_size = min(20000, len(hard_neg))
    idx = np.random.choice(len(hard_neg), sample_size, replace=False)
    covs = [q_cov(queries.iloc[i], titles.iloc[i]) for i in idx]
    covs = np.array(covs)

    pr(f"  Hard neg (sample={sample_size:,}) token örtüşme:")
    pr(f"    Ortalama      : {covs.mean():.3f}")
    pr(f"    Medyan        : {np.median(covs):.3f}")
    pr(f"    >0.5 (riskli) : {(covs>0.5).sum()} ({100*(covs>0.5).mean():.1f}%)")
    pr(f"    >0.8 (çok riskli): {(covs>0.8).sum()} ({100*(covs>0.8).mean():.1f}%)")
    pr(f"    == 1.0 (tam eşleşme): {(covs==1.0).sum()} ({100*(covs==1.0).mean():.1f}%)")

    # Tahmin: bu hard negatifler arasında kaçı gerçek pozitif olabilir?
    pr(f"\n  TAHMİN: Hard neg içinde unlabeled positive oranı")
    pr(f"    Token örtüşme>0.8 çiftlerin %'si yüksek riskli")
    pr(f"    Riskli çift sayısı: ~{int((covs>0.5).mean() * len(hard_neg)):,}")
    pr(f"    Eğer %5'i gerçek pozitif olsa: {int(0.05 * len(hard_neg)):,} gürültü")
    pr(f"    Bu durum OOF'u ~0.048 düşürebilir (gözlemlenen: 0.9685→0.9210)")
else:
    pr(f"  ⚠ hard neg dosyası bulunamadı: {hard_neg_path}")

# ─────────────────────────────────────────────────────────────────────
# 5. OOF PROBA DAĞILIMI (v23 vs v24)
# ─────────────────────────────────────────────────────────────────────
pr("\n[5] TEST PROBA DAĞILIMI (v23 vs v24)")

v23_proba_path = CACHE / "submissions" / "v23_test_proba.npy"
v24_proba_path = CACHE / "submissions" / "v24_test_proba.npy"

if v23_proba_path.exists():
    v23_proba = np.load(str(v23_proba_path))
    pr(f"  v23 test proba (n={len(v23_proba):,}):")
    pr(f"    Ortalama         : {v23_proba.mean():.4f}")
    pr(f"    Medyan           : {np.median(v23_proba):.4f}")
    pr(f"    >0.775 (thr=v23) : {(v23_proba>0.775).sum():,} ({100*(v23_proba>0.775).mean():.1f}%)")
    pr(f"    0-0.2 band       : {(v23_proba<0.2).sum():,} ({100*(v23_proba<0.2).mean():.1f}%)")
    pr(f"    0.2-0.8 belirsiz : {((v23_proba>=0.2)&(v23_proba<=0.8)).sum():,} ({100*((v23_proba>=0.2)&(v23_proba<=0.8)).mean():.1f}%)")
    pr(f"    0.8-1.0 band     : {(v23_proba>0.8).sum():,} ({100*(v23_proba>0.8).mean():.1f}%)")

    pct = [0,1,5,10,25,50,75,90,95,99,100]
    vals = np.percentile(v23_proba, pct)
    pr(f"    Percentiles: {dict(zip(pct, [f'{v:.3f}' for v in vals]))}")

if v24_proba_path.exists():
    v24_proba = np.load(str(v24_proba_path))
    pr(f"\n  v24 test proba (n={len(v24_proba):,}):")
    pr(f"    Ortalama         : {v24_proba.mean():.4f}")
    pr(f"    Medyan           : {np.median(v24_proba):.4f}")
    pr(f"    >0.795 (thr=v24) : {(v24_proba>0.795).sum():,} ({100*(v24_proba>0.795).mean():.1f}%)")
    pr(f"    0-0.2 band       : {(v24_proba<0.2).sum():,} ({100*(v24_proba<0.2).mean():.1f}%)")
    pr(f"    0.2-0.8 belirsiz : {((v24_proba>=0.2)&(v24_proba<=0.8)).sum():,} ({100*((v24_proba>=0.2)&(v24_proba<=0.8)).mean():.1f}%)")
    pr(f"    0.8-1.0 band     : {(v24_proba>0.8).sum():,} ({100*(v24_proba>0.8).mean():.1f}%)")

if v23_proba_path.exists() and v24_proba_path.exists():
    pr(f"\n  v23 vs v24 KARŞILAŞTIRMA:")
    pr(f"    v23 pozitif (thr=0.775): {(v23_proba>0.775).sum():,} (27.9%)")
    pr(f"    v24 pozitif (thr=0.795): {(v24_proba>0.795).sum():,} (24.4%)")

    # Aynı çiftlerde tahmin değişimi
    n_flip_01 = ((v23_proba<=0.775) & (v24_proba>0.795)).sum()
    n_flip_10 = ((v23_proba>0.775) & (v24_proba<=0.795)).sum()
    pr(f"    v23=0 → v24=1 flip: {n_flip_01:,}")
    pr(f"    v23=1 → v24=0 flip: {n_flip_10:,}")
    pr(f"    Net flip: {n_flip_01-n_flip_10:+,}")

# ─────────────────────────────────────────────────────────────────────
# 6. ITEM METADATA KALİTESİ
# ─────────────────────────────────────────────────────────────────────
pr("\n[6] ITEM METADATA KALİTESİ")

pr(f"  Toplam ürün           : {len(items):,}")
pr(f"  brand='unknown'       : {(items['brand']=='unknown').sum():,} ({100*(items['brand']=='unknown').mean():.1f}%)")
pr(f"  brand dolu            : {(items['brand']!='unknown').sum():,}")
pr(f"  gender='unknown'      : {(items['gender']=='unknown').sum():,} ({100*(items['gender']=='unknown').mean():.1f}%)")
pr(f"  age_group='unknown'   : {(items['age_group']=='unknown').sum():,} ({100*(items['age_group']=='unknown').mean():.1f}%)")
pr(f"  attributes dolu       : {(items['attributes']!='unknown').sum():,} ({100*(items['attributes']!='unknown').mean():.1f}%)")
pr(f"  category derinlik (/) : {items['category'].str.count('/').mean():.2f} ortalama seviye")

pr("\n  Gender dağılımı:")
for g, cnt in items["gender"].value_counts().head(6).items():
    pr(f"    {g:15s}: {cnt:,} ({100*cnt/len(items):.1f}%)")

pr("\n  Age group dağılımı:")
for a, cnt in items["age_group"].value_counts().head(6).items():
    pr(f"    {a:20s}: {cnt:,} ({100*cnt/len(items):.1f}%)")

pr("\n  Ana kategori (L1) dağılımı (top 10):")
items["L1"] = items["category"].str.split("/").str[0]
for cat, cnt in items["L1"].value_counts().head(10).items():
    pr(f"    {cat:35s}: {cnt:,}")

# ─────────────────────────────────────────────────────────────────────
# 7. POSITIVE PAIR ÖRTÜŞME ANALİZİ
# ─────────────────────────────────────────────────────────────────────
pr("\n[7] POZİTİF ÇİFT ÖRTÜŞME ANALİZİ")

tid_to_q = dict(zip(terms["term_id"].astype(str), terms["query"]))
iid_to_title = dict(zip(items["item_id"].astype(str), items["title"]))

tp = train_pairs.copy()
tp["query"] = tp["term_id"].astype(str).map(tid_to_q).fillna("")
tp["title"] = tp["item_id"].astype(str).map(iid_to_title).fillna("")

def token_overlap(q, t):
    qw = set(q.split()); tw = set(t.split())
    return len(qw & tw)

def q_cov_fn(q, t):
    qw = set(q.split())
    return len(qw & set(t.split()))/len(qw) if qw else 0.0

sample_n = min(5000, len(tp))
idx = np.random.choice(len(tp), sample_n, replace=False)
overlaps = [token_overlap(tp.iloc[i]["query"], tp.iloc[i]["title"]) for i in idx]
covs     = [q_cov_fn(tp.iloc[i]["query"], tp.iloc[i]["title"]) for i in idx]
overlaps = np.array(overlaps); covs = np.array(covs)

pr(f"  Pozitif örneklem (n={sample_n:,}):")
pr(f"    Token overlap=0  : {(overlaps==0).sum()} ({100*(overlaps==0).mean():.1f}%) ← sıfır örtüşme pozitif!")
pr(f"    Token overlap=1  : {(overlaps==1).sum()} ({100*(overlaps==1).mean():.1f}%)")
pr(f"    Token overlap>=2 : {(overlaps>=2).sum()} ({100*(overlaps>=2).mean():.1f}%)")
pr(f"    Query coverage   : ortalama {covs.mean():.3f}, medyan {np.median(covs):.3f}")
pr(f"    qcov < 0.5       : {(covs<0.5).sum()} ({100*(covs<0.5).mean():.1f}%) ← zor vakalar")
pr(f"    qcov = 0         : {(covs==0).sum()} ({100*(covs==0).mean():.1f}%) ← semantik gap")

# FN örnek tahmin: sıfır text örtüşmeli pozitifler
zero_ovlp = [(tp.iloc[j]["query"], tp.iloc[j]["title"]) for j, i in enumerate(idx) if overlaps[j]==0]
pr(f"\n  Sıfır overlap pozitif örnekler (top 10):")
for q, t in zero_ovlp[:10]:
    pr(f"    Q: {q[:40]:40s} | T: {t[:50]}")

# ─────────────────────────────────────────────────────────────────────
# 8. GAP ANALİZİ ÖZET
# ─────────────────────────────────────────────────────────────────────
pr("\n[8] OOF-TEST GAP ÖZET TABLOSU")
pr("  " + "-"*60)
pr(f"  {'Versiyon':10} {'OOF F1':10} {'Test F1':10} {'Gap':10} {'Not'}")
pr("  " + "-"*60)
versions = [
    ("v22",  0.9704, 0.83, "same_brand_diff_main + head features"),
    ("v23",  0.9686, 0.84, "bert_v23 retrain + 5 neg/pos + direct thr"),
    ("v24",  0.9210, 0.79, "TF-IDF hard neg 180K — REGRESSION"),
]
for vname, oof, test, note in versions:
    gap = oof - test
    pr(f"  {vname:10} {oof:10.4f} {test:10.2f} {gap:10.4f}  {note}")
pr("  " + "-"*60)
pr(f"\n  Gap v22→v23: 0.140 → 0.129 (kapandı +0.011)")
pr(f"  Gap v23→v24: 0.129 → 0.131 (AÇILDI -0.002) — hard neg işe yaramadı")

# ─────────────────────────────────────────────────────────────────────
# 9. V24 BAŞARISIZLIK KÖKÜ
# ─────────────────────────────────────────────────────────────────────
pr("\n[9] V24 BAŞARISIZLIK KÖK NEDENİ")
pr("""
  v24 yaklaşımı: TF-IDF cosine ile text-benzer ama label=0 item seç → hard neg

  NEDEN BAŞARISIZ OLDU:

  A) Gürültülü hard negatives
     - TF-IDF text-benzer ama label=0 → bu çiftlerin bir kısmı GERÇEKTE POZİTİF
     - training_pairs.csv seyrek etiketlidir (250K çift / 17,968 query = ~14 pos/query)
     - Katalogda çok daha fazla unlabeled positive var
     - Model yanlış etiket öğrendi → konfüzyon → OOF 0.048 düştü

  B) Underfitting
     - 1.5M → 1.68M çift (+12%) ama iter=2000 SABİT kaldı
     - 5 foldun 5'i de iter=2000'e erişti (early stopping HİÇ tetiklenmedi)
     - Model yeterince öğrenemedi → test performansı düştü

  C) Gap kapanmadı
     - Gap: 0.129 → 0.131 (kötüleşti)
     - Hard negatives test dağılımını yansıtmıyor
     - Test'teki gerçek hard negatives: brand-specific, semantic-shift, synonym
     - TF-IDF bunları tam yakalayamıyor

  D) Pozitif oranı düştü: %16.7 → %14.9
     - Model az pozitifle öğreniyor → recall azalıyor
     - Macro F1'de her iki sınıf eşit önem → recall düşmesi ağır bedel
""")

pr(f"\n[Toplam süre: {(time.time()-t0)/60:.1f} dk]")
pr("=" * 70)
LOG.close()
print(f"\nRapor kaydedildi: {OUT}/data_analysis_output.txt")
