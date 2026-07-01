"""
23_lgbm_v20_typed.py — LightGBM + Typed Attribute Matching
===========================================================
Analiz bulgularına dayanan kritik düzeltmeler (DATA_DETAYLI_INCELEME.md'den):

DÜZELTMELER (v19 bug'ları):
  FIX attr_jac    : Attributes JSON DEĞİL comma-sep "key: value" → parse_attrs() ile düzeldi
  FIX renk feature: typed attribute field kontrolü (title text değil)
  FIX brand_in_q  : Noktalama normalize et (us polo ≡ u.s. polo assn.)

YENİ FEATURE'LAR:
  + color_typed   : query rengi vs item["renk"] attr (2=exact, 1=aile, 0=yok, -1=farklı)
  + color_family  : gri~antrasit~platin, mavi~lacivert~indigo
  + brand_tok_overlap : brand token overlap ratio
  + attr_renk     : attributes'tan "renk" değeri var mı?
  + material_match: query malzeme kelimesi vs "materyal bileşeni"
  + kol_boyu_match: "uzun kollu" query → item kol boyu = uzun

KALDIRILANLAR:
  - renk_q_in_t   (color_typed daha iyi)
  - renk_mismatch (color_typed içinde)
  - len_diff / l1_match (v19'da da yoktu, AUC=0.50)

Beklenti: 0.70 → 0.74+
"""

import sys, time, re
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

BASE       = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA       = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
SUBM       = BASE / "claude only" / "submissions"
BERT_MODEL = BASE / "claude only" / "models" / "bert_pseudolabel_v16"

LOWER = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(t): return str(t).translate(LOWER).lower().strip()

# ──────────────────────────────────────────────────────────────
# TÜRKÇE STEM
# ──────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────
# RENK SİSTEMİ
# ──────────────────────────────────────────────────────────────
# Query'de aranacak renk kelimeleri (Türkçe + İngilizce varyantlar)
RENKLER = {
    "kırmızı","mavi","beyaz","siyah","sarı","yeşil","pembe","mor","gri","turuncu",
    "lacivert","bej","kahverengi","altın","gold","gümüş","silver","rose","ekru","krem",
    "bordo","haki","füme","antrasit","indigo","petrol","kiremit",
}

# Query rengi → normalize (İngilizce → Türkçe, eşanlamlılar)
COLOR_NORM = {
    "gold":    "altın",
    "silver":  "gümüş",
    "rose":    "pembe",
    "krem":    "bej",
    "kiremit": "kırmızı",
}

# Renk → aile (item rengi → aile grubu)
COLOR_FAMILY = {
    "antrasit": "gri", "füme": "gri", "platin": "gri",
    "metalik gri": "gri", "koyu gri": "gri", "açık gri": "gri",
    "kurşun": "gri", "kurşun gri": "gri", "gri melanj": "gri",
    "lacivert": "mavi", "indigo": "mavi", "petrol": "mavi",
    "saks mavi": "mavi", "bebe mavisi": "mavi", "açık mavi": "mavi",
    "bordo": "kırmızı", "kiremit": "kırmızı",
    "altın": "sarı", "gold": "sarı",
    "gümüş": "metalik", "silver": "metalik",
    "krem": "bej", "kırık beyaz": "bej", "ekru": "bej",
}

def norm_color(c):
    return COLOR_NORM.get(c, c)

def color_family(c):
    return COLOR_FAMILY.get(c, c)

def get_query_color(q):
    for tok in q.split():
        if tok in RENKLER:
            return norm_color(tok)
    return None

# ──────────────────────────────────────────────────────────────
# ATTRIBUTE PARSER (DÜZELTILDI: JSON değil comma-sep)
# ──────────────────────────────────────────────────────────────
def parse_attrs(s):
    """attributes alanını parse et: 'renk: siyah, materyal: pamuklu' → dict"""
    if not s or s in ("unknown", ""): return {}
    d = {}
    for part in s.split(","):
        part = part.strip()
        if ":" in part:
            k, _, v = part.partition(":")
            d[k.strip()] = v.strip()
    return d

def attr_typed_jac(q, a):
    """Attribute string ile query'yi karşılaştır — tüm değerleri birleştir"""
    d = parse_attrs(a)
    if not d: return jac(q, a)  # fallback
    vals = " ".join(v for v in d.values() if v)
    return jac(q, vals)

def get_attr_color(a):
    """Attributes'tan ana rengi çıkar"""
    d = parse_attrs(a)
    return d.get("renk", d.get("color", ""))

def get_attr_material(a):
    """Attributes'tan materyal bilgisini çıkar"""
    d = parse_attrs(a)
    # "materyal bileşeni" tercih et, yoksa "materyal"
    return d.get("materyal bileşeni", d.get("materyal", ""))

def get_attr_kol(a):
    """kol boyu bilgisini çıkar"""
    d = parse_attrs(a)
    return d.get("kol boyu", d.get("kol tipi", ""))

# ──────────────────────────────────────────────────────────────
# TYPED FEATURE'LAR
# ──────────────────────────────────────────────────────────────
# Malzeme eşleştirme: query kelimesi → attribute'da aranacak string
MATERIAL_MAP = {
    "pamuk":     ["pamuk"],
    "pamuklu":   ["pamuk"],
    "deri":      ["deri"],
    "hakiki":    ["deri"],
    "polyester": ["polyester"],
    "yün":       ["yün"],
    "keten":     ["keten"],
    "naylon":    ["naylon", "nylon"],
    "çelik":     ["çelik"],
    "plastik":   ["plastik"],
    "ahşap":     ["ahşap", "ahsap"],
}

def color_typed_match(q_color, item_renk):
    """
    Renk eşleştirme skoru:
      2.0 = exact eşleşme (siyah → siyah)
      1.5 = synonym eşleşme (gold → altın)
      1.0 = aynı aile (antrasit → gri ailesi, lacivert → mavi ailesi)
      0.0 = query'de renk yok veya item renk boş
     -1.0 = farklı renk (siyah query ↔ beyaz item)
    """
    if q_color is None: return 0.0
    if not item_renk: return 0.0

    # Exact
    if q_color == item_renk: return 2.0

    # Aile karşılaştırması
    q_fam = color_family(q_color)
    i_fam = color_family(item_renk)
    if q_fam == i_fam: return 1.0

    return -1.0

def brand_tok_overlap(q, brand):
    """
    Noktalama kaldırarak marka token overlap oranı.
    'us polo tişört' vs 'u.s. polo assn.' → {us,polo} / {us,polo,assn} = 0.67
    """
    if not brand or brand == "unknown": return 0.0
    b_clean = re.sub(r'[^a-z0-9çğıöşü\s]', ' ', brand).split()
    q_clean = set(re.sub(r'[^a-z0-9çğıöşü\s]', ' ', q).split())
    if not b_clean: return 0.0
    b_toks = set(t for t in b_clean if len(t) > 1)  # 1 karakter tokenleri at
    if not b_toks: return 0.0
    return len(b_toks & q_clean) / len(b_toks)

def material_match(q, attr_mat):
    """Query'de malzeme kelimesi varsa attribute'da buluyor mu?"""
    if not attr_mat: return 0.0
    for q_tok in q.split():
        if q_tok in MATERIAL_MAP:
            for mat_term in MATERIAL_MAP[q_tok]:
                if mat_term in attr_mat:
                    return 1.0
    return 0.0

def kol_boyu_match(q, attr_kol):
    """'uzun kollu', 'kısa kollu' query → kol boyu attribute eşleşmesi"""
    if not attr_kol: return 0.0
    if "uzun" in q and "uzun" in attr_kol: return 1.0
    if "kısa" in q and "kısa" in attr_kol: return 1.0
    if "kolsuz" in q and "kolsuz" in attr_kol: return 1.0
    return 0.0

def gender_cross(q, g):
    q_kadin = bool(re.search(r'\b(kadın|bayan)\b', q))
    q_erkek = bool(re.search(r'\berkek\b', q))
    g_kadin = g in ("kadın", "bayan", "kız")
    g_erkek = g == "erkek"
    if q_kadin and g_erkek: return -1.0
    if q_erkek and g_kadin: return -1.0
    if q_kadin and g_kadin: return  1.0
    if q_erkek and g_erkek: return  1.0
    return 0.0

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

for col in ["title", "category", "brand", "gender", "age_group", "attributes"]:
    items[col] = items[col].fillna("unknown").apply(trl)
terms["query"] = terms["query"].fillna("").apply(trl)

iid_to = {row.item_id: row for row in items.itertuples()}
tid_to_q = dict(zip(terms["term_id"], terms["query"]))

print(f"  {len(items):,} ürün | {len(train_pairs):,} pozitif | {len(sub_pairs):,} test çifti | {time.time()-t0:.1f}s", flush=True)

# ──────────────────────────────────────────────────────────────
# NEGATİFLER (v19 ile aynı — BERT skorlarını yeniden kullanmak için)
# ──────────────────────────────────────────────────────────────
print("\n[2] Negatifler üretiliyor...", flush=True)
pool = sub_pairs.merge(terms, on="term_id").merge(items, on="item_id")
for col in ["query", "title", "gender", "category"]:
    pool[col] = pool[col].fillna("unknown").apply(trl)

mask_g = (
    (pool["query"].str.contains("kadın|bayan", regex=True) & (pool["gender"] == "erkek")) |
    (pool["query"].str.contains(r"\berkek\b", regex=True) & (pool["gender"] == "kadın"))
)
mask_z = [len(set(q.split()) & set((t + " " + c.replace("/", " ")).split())) == 0
          for q, t, c in zip(pool["query"], pool["title"], pool["category"])]
pool["mask_z"] = mask_z
neg_pool = pool[mask_g | pool["mask_z"]][["term_id", "item_id"]]
neg_pool = neg_pool.sample(n=min(350_000, len(neg_pool)), random_state=42)
neg_pool["label"] = 0
train_pairs["label"] = 1
train_df = pd.concat([train_pairs[["term_id", "item_id", "label"]], neg_pool], ignore_index=True)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"  Train: {len(train_df):,} | pos={train_pairs.label.sum():,} neg={len(neg_pool):,}", flush=True)

# ──────────────────────────────────────────────────────────────
# BERT SCORES (cache'den yükle — v19 ile aynı negatifler)
# ──────────────────────────────────────────────────────────────
print("\n[3] BERT scores yükleniyor...", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_bert_path  = SUBM / "bert_scores_v16.npy"
train_bert_path = BASE / "claude only" / "train_bert_scores_v19.npy"

def bert_scores_for(tids, iids, batch=512):
    tokenizer = AutoTokenizer.from_pretrained(str(BERT_MODEL))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(BERT_MODEL)).to(device).eval()
    scores = []
    for i in range(0, len(tids), batch):
        bt, bi = tids[i:i+batch], iids[i:i+batch]
        qs = [tid_to_q.get(t, "") for t in bt]
        row_items = [iid_to.get(ii) for ii in bi]
        ps = [" | ".join(p for p in [
                  getattr(r, "title", ""), getattr(r, "brand", ""),
                  getattr(r, "category", "").split("/")[0]
              ] if p and p != "unknown") if r else "" for r in row_items]
        enc = tokenizer(qs, ps, max_length=128, truncation=True, padding=True,
                        return_tensors="pt").to(device)
        with torch.no_grad(), autocast("cuda"):
            logits = mdl(**enc).logits.squeeze(-1)
        scores.extend(torch.sigmoid(logits).float().cpu().tolist())
        if i % 100_000 == 0 and i > 0:
            print(f"  bert {i:,}/{len(tids)}", flush=True)
    return np.array(scores)

if test_bert_path.exists():
    test_bert = np.load(str(test_bert_path))
    print(f"  Test BERT yüklendi (cache): {len(test_bert):,}", flush=True)
else:
    print("  Test BERT hesaplanıyor...", flush=True)
    test_bert = bert_scores_for(sub_pairs["term_id"].tolist(), sub_pairs["item_id"].tolist())
    np.save(str(test_bert_path), test_bert)

if train_bert_path.exists():
    train_bert = np.load(str(train_bert_path))
    print(f"  Train BERT yüklendi (cache): {len(train_bert):,}", flush=True)
else:
    print(f"  Train BERT hesaplanıyor... (device={device})", flush=True)
    train_bert = bert_scores_for(train_df["term_id"].tolist(), train_df["item_id"].tolist())
    np.save(str(train_bert_path), train_bert)

# ──────────────────────────────────────────────────────────────
# FEATURE BUILDER
# ──────────────────────────────────────────────────────────────
print("\n[4] Özellikler hesaplanıyor...", flush=True)

def build_features(df_in, tfidf_vect, bert_arr, fit=False, all_texts=None):
    df = df_in.merge(terms, on="term_id", how="left").merge(items, on="item_id", how="left")
    for col in ["query", "title", "brand", "category", "gender", "age_group", "attributes"]:
        df[col] = df[col].fillna("unknown").apply(trl)

    if fit:
        corpus = list(df["title"]) + list(df["query"]) + (all_texts or [])
        tfidf_vect.fit(corpus)

    qs    = df["query"].tolist()
    ts    = df["title"].tolist()
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

    # Attribute pre-parse (bir kez yap)
    parsed_attrs = [parse_attrs(a) for a in attrs]
    attr_renk_vals   = [d.get("renk", "") for d in parsed_attrs]
    attr_mat_vals    = [d.get("materyal bileşeni", d.get("materyal", "")) for d in parsed_attrs]
    attr_kol_vals    = [d.get("kol boyu", d.get("kol tipi", "")) for d in parsed_attrs]

    # Query renk çıkar
    q_colors = [get_query_color(q) for q in qs]

    f = pd.DataFrame()

    # ── Temel metin özellikleri ──
    f["fuzz_partial"]    = [rfuzz.partial_ratio(q, t) / 100   for q, t in zip(qs, ts)]
    f["fuzz_set"]        = [rfuzz.token_set_ratio(q, t) / 100 for q, t in zip(qs, ts)]
    f["fuzz_sort"]       = [rfuzz.token_sort_ratio(q, t) / 100 for q, t in zip(qs, ts)]
    f["fuzz_basic"]      = [rfuzz.ratio(q, t) / 100           for q, t in zip(qs, ts)]
    f["jaccard"]         = [jac(q, t)                         for q, t in zip(qs, ts)]
    f["tfidf_cos"]       = tfidf_cos(qs, ts)
    f["q_cov_title"]     = [q_cov(q, t)                       for q, t in zip(qs, ts)]
    f["t_cov_query"]     = [q_cov(t, q)                       for q, t in zip(qs, ts)]
    f["cat_overlap"]     = [jac(q, c.replace("/", " "))        for q, c in zip(qs, cats)]
    f["exact_in_title"]  = [(q in t) * 1.0                    for q, t in zip(qs, ts)]
    f["token_overlap"]   = [len(set(q.split()) & set(t.split())) for q, t in zip(qs, ts)]
    f["age_in_q"]        = [(a not in ("unknown", "") and a in q) * 1.0 for a, q in zip(ages, qs)]

    # ── Türkçe stem ──
    f["stem_jaccard"]    = [stem_jac(q, t)                    for q, t in zip(qs, ts)]
    f["stem_cat_jac"]    = [stem_jac(q, c.replace("/", " "))  for q, c in zip(qs, cats)]

    # ── Gender ──
    f["gender_cross"]    = [gender_cross(q, g)                for q, g in zip(qs, gens)]

    # ── İlk token ──
    f["first_tok_title"] = [(q.split()[0] in t if q.split() else False) * 1.0
                             for q, t in zip(qs, ts)]
    f["first_tok_brand"] = [(q.split()[0] in b if q.split() and b != "unknown" else False) * 1.0
                             for q, b in zip(qs, brs)]

    # ── Q uzunluk ──
    f["q_len"]           = [len(q.split()) for q in qs]
    f["t_len"]           = [len(t.split()) for t in ts]

    # ── BERT ──
    f["bert_score"]      = bert_arr

    # ── Kategori ──
    f["ana_kategori"]    = pd.Categorical([c.split("/")[0] for c in cats])

    # ── YENİ: Typed color feature (v20) ──
    f["color_typed"]     = [color_typed_match(qc, ir)
                             for qc, ir in zip(q_colors, attr_renk_vals)]
    f["has_q_color"]     = [(qc is not None) * 1.0 for qc in q_colors]
    f["attr_has_renk"]   = [(ar != "") * 1.0 for ar in attr_renk_vals]

    # ── YENİ: Attribute renk title'da da var mı (title text ile) ──
    f["attr_renk_in_title"] = [(ar != "" and ar in t) * 1.0
                                for ar, t in zip(attr_renk_vals, ts)]

    # ── YENİ: Brand token overlap (normalizasyonlu) ──
    f["brand_tok_ovlp"]  = [brand_tok_overlap(q, b) for q, b in zip(qs, brs)]

    # ── YENİ: Material match ──
    f["material_match"]  = [material_match(q, am) for q, am in zip(qs, attr_mat_vals)]

    # ── YENİ: Kol boyu match ──
    f["kol_boyu_match"]  = [kol_boyu_match(q, ak) for q, ak in zip(qs, attr_kol_vals)]

    # ── YENİ: attr_jac DÜZELTILDI (comma-sep parser) ──
    f["attr_jac_fixed"]  = [attr_typed_jac(q, a) for q, a in zip(qs, attrs)]

    # ── YENİ: Attribute değerlerinin query'yi ne kadar kapsadığı ──
    f["attr_q_cov"]      = [q_cov(q, " ".join(d.values())) if (d := parse_attrs(a)) else 0.0
                             for q, a in zip(qs, attrs)]

    return f, df

t1 = time.time()
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=60_000,
                        sublinear_tf=True, min_df=2)

FEATS = [
    # Temel
    "fuzz_partial", "fuzz_set", "fuzz_sort", "fuzz_basic",
    "jaccard", "tfidf_cos", "q_cov_title", "t_cov_query",
    "cat_overlap", "exact_in_title", "token_overlap", "age_in_q",
    # Türkçe
    "stem_jaccard", "stem_cat_jac",
    # Demografik
    "gender_cross",
    # İlk token
    "first_tok_title", "first_tok_brand",
    # Uzunluk
    "q_len", "t_len",
    # BERT
    "bert_score",
    # Kategori (categorical)
    "ana_kategori",
    # YENİ v20
    "color_typed", "has_q_color", "attr_has_renk", "attr_renk_in_title",
    "brand_tok_ovlp",
    "material_match",
    "kol_boyu_match",
    "attr_jac_fixed",
    "attr_q_cov",
]

X_tr, df_tr = build_features(
    train_df, tfidf, train_bert, fit=True,
    all_texts=items["title"].tolist() + terms["query"].tolist()
)
y = train_df["label"].values
groups = df_tr["term_id"].values
print(f"  Train features: {X_tr[FEATS].shape} | {(time.time()-t1):.1f}s", flush=True)

# ── Yeni feature dağılımları (quick check) ──
print("\n  [Yeni feature istatistikleri]", flush=True)
for col in ["color_typed", "brand_tok_ovlp", "material_match", "attr_jac_fixed"]:
    pos_mask = y == 1
    neg_mask = y == 0
    print(f"    {col:20s}: pos_ort={X_tr.loc[pos_mask, col].mean():.3f} "
          f"neg_ort={X_tr.loc[neg_mask, col].mean():.3f}", flush=True)

# ──────────────────────────────────────────────────────────────
# LightGBM 5-FOLD
# ──────────────────────────────────────────────────────────────
print("\n[5] LightGBM 5-fold...", flush=True)
oof = np.zeros(len(train_df))
models_lgbm = []
gkf = GroupKFold(n_splits=5)
for fold, (tri, vli) in enumerate(gkf.split(X_tr, y, groups)):
    m = lgb.LGBMClassifier(
        n_estimators=1500, learning_rate=0.04, max_depth=8,
        num_leaves=127, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, class_weight="balanced",
        random_state=42, n_jobs=-1, verbose=-1
    )
    m.fit(
        X_tr[FEATS].iloc[tri], y[tri],
        eval_set=[(X_tr[FEATS].iloc[vli], y[vli])],
        categorical_feature=["ana_kategori"],
        callbacks=[lgb.early_stopping(70, verbose=False), lgb.log_evaluation(0)]
    )
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
print("\n  Feature Önemleri (top 20):", flush=True)
for feat, imp in fi.head(20).items():
    print(f"    {feat:25s}: {imp:6.0f}", flush=True)

# ──────────────────────────────────────────────────────────────
# TEST INFERENCE
# ──────────────────────────────────────────────────────────────
print("\n[6] Test inference...", flush=True)
X_te, _ = build_features(sub_pairs, tfidf, test_bert)
test_scores = sum(m.predict_proba(X_te[FEATS])[:, 1] for m in models_lgbm) / len(models_lgbm)

# Global threshold
final = (test_scores > best_thr).astype(int)
pos_n = final.sum()
print(f"  Threshold={best_thr:.2f} | pozitif={pos_n:,} ({100*pos_n/len(sub_pairs):.1f}%)", flush=True)

# Submission
out_path = SUBM / "submission_v20_typed.csv"
pd.DataFrame({"id": sub_pairs["id"], "prediction": final}).to_csv(str(out_path), index=False)

print(f"\n{'='*60}", flush=True)
print(f"TAMAMLANDI — v20 Typed Attribute Matching", flush=True)
print(f"  OOF F1      : {best_f1:.5f}", flush=True)
print(f"  Threshold   : {best_thr:.2f}", flush=True)
print(f"  Pozitif     : {pos_n:,} ({100*pos_n/len(sub_pairs):.1f}%)", flush=True)
print(f"  Süre        : {(time.time()-t0)/60:.1f} dk", flush=True)
print(f"  Dosya       : {out_path}", flush=True)
print(f"{'='*60}", flush=True)
