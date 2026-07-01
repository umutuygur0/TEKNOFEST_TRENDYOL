"""
14_structured_neg_gen.py — Yapısal Hard Negatif Üretici
=========================================================
Önceki negatif stratejilerinin sorunları:
  v9 cross-query: "avon luck" + "granül kahve" → çok kolay
  v7 TF-IDF:      kontaminasyon riski

Bu script — garantili temiz negativler:
  BRAND_SWAP  : aynı L1 kategori, farklı marka      → %100 negatif
  GENDER_SWAP : aynı L1 kategori, yanlış cinsiyet   → %100 negatif
  AGE_SWAP    : bebek↔yetişkin, aynı kategori       → %100 negatif
  SAME_CAT    : aynı L1 kategori, farklı sorgu+marka → güçlü negatif

Çıktı:
  parsed_queries/training_pairs_structured.csv
    → (query_text, item_text, label, neg_type) 4 sütun
    → label=1 pozitif, label=0 negatif
    → ~1M çift (250K pos + ~750K neg)

Çalıştır:
  python "claude only/14_structured_neg_gen.py"
"""

import random
import time
import sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
random.seed(42)

BASE   = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA   = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
PARSED = BASE / "claude only" / "parsed_queries"
OUT    = PARSED

LOWER_MAP = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(t): return str(t).translate(LOWER_MAP).lower().strip()

NEG_PER_POS = 4   # her pozitif için kaç negatif üret
MAX_BRAND_SWAP   = 2   # en fazla kaç brand_swap
MAX_GENDER_SWAP  = 1   # en fazla kaç gender_swap
MAX_AGE_SWAP     = 1   # en fazla kaç age_swap
MAX_SAME_CAT     = 1   # en fazla kaç same_cat (fallback)

# ─── 1. Veri Yükle ──────────────────────────────────────────────────────────
print("[1] Veri yükleniyor...")
t0 = time.time()
items  = pd.read_csv(DATA / "items.csv")
terms  = pd.read_csv(DATA / "terms.csv")
train  = pd.read_csv(DATA / "training_pairs.csv")
qparsed = pd.read_csv(PARSED / "train_queries_parsed.csv")

# item özellikleri
iid_to_brand    = dict(zip(items["item_id"], items["brand"].fillna("").apply(trl)))
iid_to_cat      = dict(zip(items["item_id"], items["category"].fillna("")))
iid_to_catl1    = {iid: c.split("/")[0] for iid, c in iid_to_cat.items()}
iid_to_title    = dict(zip(items["item_id"], items["title"].fillna("").apply(trl)))
tid_to_query    = dict(zip(terms["term_id"], terms["query"]))

# query parsed map
qp_map = qparsed.set_index("term_id").to_dict("index")

# training positives
train_pos = defaultdict(set)
for tid, iid in zip(train["term_id"].values, train["item_id"].values):
    train_pos[tid].add(iid)

all_iids = items["item_id"].tolist()
print(f"  {time.time()-t0:.1f}s | {len(train_pos):,} sorgu | {len(all_iids):,} item")

# ─── 2. İndeksler: Hızlı Arama İçin ─────────────────────────────────────────
print("\n[2] İndeksler oluşturuluyor...")
t0 = time.time()

# L1 kategori → item listesi
catl1_to_iids = defaultdict(list)
for iid in all_iids:
    catl1_to_iids[iid_to_catl1[iid]].append(iid)

# brand → item listesi (hızlı brand_swap için)
brand_to_iids = defaultdict(list)
for iid in all_iids:
    b = iid_to_brand[iid].split()[0] if iid_to_brand[iid] else ""
    if b:
        brand_to_iids[b].append(iid)

print(f"  L1 kategoriler: {len(catl1_to_iids)} | Brands: {len(brand_to_iids)}")

# ─── 3. Item Cinsiyet/Yaş Tespiti ────────────────────────────────────────────
print("\n[3] Item cinsiyet/yaş etiketleri çıkarılıyor...")
# Item başlıklarından cinsiyet ve yaş aralığı çıkar
# (Daha sonra gender_swap ve age_swap için)

GENDER_KEYWORDS = {
    "kadın": "kadın", "bayan": "kadın", "kız": "kadın", "women": "kadın",
    "erkek": "erkek", "men": "erkek", "boys": "erkek",
    "unisex": "unisex",
}
AGE_KEYWORDS = {
    "bebek": "bebek", "infant": "bebek",
    "çocuk": "çocuk", "kids": "çocuk", "junior": "çocuk",
}

iid_to_gender = {}
iid_to_age    = {}
for iid in all_iids:
    t = iid_to_title[iid]
    for kw, label in GENDER_KEYWORDS.items():
        if kw in t:
            iid_to_gender[iid] = label
            break
    for kw, label in AGE_KEYWORDS.items():
        if kw in t:
            iid_to_age[iid] = label
            break

gender_labeled = len(iid_to_gender)
age_labeled    = len(iid_to_age)
print(f"  Cinsiyet etiketli item: {gender_labeled:,} ({100*gender_labeled/len(all_iids):.1f}%)")
print(f"  Yaş etiketli item     : {age_labeled:,} ({100*age_labeled/len(all_iids):.1f}%)")

# L1 kategori + cinsiyet → item listesi (gender_swap için)
catl1_gender_to_iids = defaultdict(list)
for iid, g in iid_to_gender.items():
    catl1_gender_to_iids[(iid_to_catl1[iid], g)].append(iid)

# L1 kategori + yaş → item listesi (age_swap için)
catl1_age_to_iids = defaultdict(list)
for iid, a in iid_to_age.items():
    catl1_age_to_iids[(iid_to_catl1[iid], a)].append(iid)

# ─── 4. Aksesuar Sorgusu Tespiti ─────────────────────────────────────────────
# "galaxy tab kılıf" gibi sorgularda brand=samsung doğru ama
# pozitif ürünler samsung değil, kılıf üreticileri → brand_swap yanlış olur
ACCESSORY_KEYWORDS = {
    "kılıf", "kapak", "koruyucu", "ekran", "aksesuar", "şarj", "kablo",
    "tutucu", "stand", "uyumlu", "için", "compatible", "cover", "case",
    "yedek", "parça", "filtre", "toner", "mürekkep", "kartuş",
}

def is_accessory_query(q_normalized):
    tokens = set(q_normalized.split())
    return bool(tokens & ACCESSORY_KEYWORDS)

# ─── 5. Item Metni Oluşturma ─────────────────────────────────────────────────
def item_text(iid):
    t = iid_to_title.get(iid, "")
    b = iid_to_brand.get(iid, "")
    c = iid_to_catl1.get(iid, "").split("/")[0]
    parts = [p for p in [t, b, c] if p and p != "nan"]
    return " | ".join(parts)

# ─── 6. Yapısal Negatif Üretim Fonksiyonları ─────────────────────────────────

def brand_swap_negs(tid, pos_iids, q_brand, catl1s, n=2):
    """Aynı L1 kategori, farklı marka → garantili negatif"""
    negs = []
    for catl1 in catl1s:
        candidates = catl1_to_iids.get(catl1, [])
        # Farklı marka, pozitiflerden değil
        pool = [
            iid for iid in candidates
            if (iid_to_brand.get(iid, "") or "").split()
            and (iid_to_brand.get(iid, "") or "").split()[0] != q_brand
            and iid not in pos_iids
        ]
        if pool:
            negs.extend(random.sample(pool, min(n, len(pool))))
        if len(negs) >= n:
            break
    return negs[:n]


def gender_swap_negs(pos_iids, query_gender, catl1s, n=1):
    """Yanlış cinsiyet, aynı L1 kategori → garantili negatif"""
    opposite = {"kadın": "erkek", "erkek": "kadın"}.get(query_gender)
    if not opposite:
        return []
    negs = []
    for catl1 in catl1s:
        pool = [
            iid for iid in catl1_gender_to_iids.get((catl1, opposite), [])
            if iid not in pos_iids
        ]
        if pool:
            negs.extend(random.sample(pool, min(n, len(pool))))
        if len(negs) >= n:
            break
    return negs[:n]


def age_swap_negs(pos_iids, query_age, catl1s, n=1):
    """Yanlış yaş grubu → garantili negatif"""
    opposite_map = {"bebek": "yetişkin", "çocuk": "yetişkin"}
    opposite = opposite_map.get(query_age)
    if not opposite:
        return []
    # Yetişkin = bebek/çocuk olmayan aynı kategorideki ürünler
    negs = []
    for catl1 in catl1s:
        all_cat = catl1_to_iids.get(catl1, [])
        pool = [
            iid for iid in all_cat
            if iid not in pos_iids
            and iid not in iid_to_age  # yaş etiketi yok = yetişkin varsayım
        ]
        if pool:
            negs.extend(random.sample(pool, min(n, len(pool))))
        if len(negs) >= n:
            break
    return negs[:n]


def same_cat_negs(pos_iids, catl1s, n=2):
    """Aynı L1 kategori, pozitif olmayan ürün → zorlu ama belirsiz negatif"""
    negs = []
    for catl1 in catl1s:
        pool = [
            iid for iid in catl1_to_iids.get(catl1, [])
            if iid not in pos_iids
        ]
        if pool:
            negs.extend(random.sample(pool, min(n, len(pool))))
        if len(negs) >= n:
            break
    return negs[:n]


# ─── 7. Tüm Training Sorguları İçin Negatif Üret ────────────────────────────
print("\n[4] Yapısal negatifler üretiliyor...")
t0 = time.time()

rows = []
neg_type_counter = Counter()
skip_count = 0

train_tids = list(train_pos.keys())

for qi, tid in enumerate(train_tids):
    pos_iids = train_pos[tid]
    if not pos_iids:
        continue

    q_raw  = tid_to_query.get(tid, "")
    q_norm = trl(q_raw)
    qp     = qp_map.get(tid, {})

    q_brand  = qp.get("brand")   if qp else None
    q_gender = qp.get("gender")  if qp else None
    q_age    = qp.get("age_range") if qp else None

    # NaN kontrolü
    if q_brand  == "nan" or pd.isna(q_brand):  q_brand  = None
    if q_gender == "nan" or pd.isna(q_gender): q_gender = None
    if q_age    == "nan" or pd.isna(q_age):    q_age    = None

    # Pozitif item'ların L1 kategorileri
    catl1s = list({iid_to_catl1[iid] for iid in pos_iids if iid in iid_to_catl1})

    # Pozitif çiftleri ekle
    for iid in pos_iids:
        rows.append({
            "query_text": q_raw,
            "item_text":  item_text(iid),
            "label":      1,
            "neg_type":   "positive",
        })

    # Negatifler toplanacak
    neg_iids_typed = []  # [(iid, type)]

    # 1. BRAND_SWAP — en güvenilir negatif
    is_acc = is_accessory_query(q_norm)
    if q_brand and not is_acc:
        bn = brand_swap_negs(tid, pos_iids, q_brand, catl1s, n=MAX_BRAND_SWAP)
        neg_iids_typed.extend([(iid, "brand_swap") for iid in bn])

    # 2. GENDER_SWAP
    if q_gender:
        gn = gender_swap_negs(pos_iids, q_gender, catl1s, n=MAX_GENDER_SWAP)
        neg_iids_typed.extend([(iid, "gender_swap") for iid in gn])

    # 3. AGE_SWAP
    if q_age:
        an = age_swap_negs(pos_iids, q_age, catl1s, n=MAX_AGE_SWAP)
        neg_iids_typed.extend([(iid, "age_swap") for iid in an])

    # 4. SAME_CAT — yeterli negatif yoksa fallback
    current_neg_count = len(neg_iids_typed)
    remaining = (NEG_PER_POS * len(pos_iids)) - current_neg_count
    if remaining > 0:
        sc = same_cat_negs(pos_iids, catl1s, n=min(remaining, MAX_SAME_CAT * len(pos_iids)))
        neg_iids_typed.extend([(iid, "same_cat") for iid in sc])

    if not neg_iids_typed:
        skip_count += 1

    # Negatif satırları ekle (deduplicate)
    seen_neg_iids = set()
    for iid, ntype in neg_iids_typed:
        if iid not in seen_neg_iids:
            rows.append({
                "query_text": q_raw,
                "item_text":  item_text(iid),
                "label":      0,
                "neg_type":   ntype,
            })
            neg_type_counter[ntype] += 1
            seen_neg_iids.add(iid)

    if (qi + 1) % 2000 == 0:
        elapsed = time.time() - t0
        print(f"  {qi+1:6d}/{len(train_tids)} | {len(rows):,} çift | {elapsed:.0f}s")

# ─── 8. Kaydet ve Rapor ─────────────────────────────────────────────────────
df = pd.DataFrame(rows)
out_path = OUT / "training_pairs_structured.csv"
df.to_csv(str(out_path), index=False)

pos_count = (df["label"] == 1).sum()
neg_count = (df["label"] == 0).sum()

print(f"\n{'='*60}")
print(f"YAPILSAL NEGATİF ÜRETİM RAPORU")
print(f"{'='*60}")
print(f"\n  Toplam çift : {len(df):,}")
print(f"  Pozitif     : {pos_count:,}")
print(f"  Negatif     : {neg_count:,}")
print(f"  Oran (neg/pos): {neg_count/pos_count:.1f}×")
print(f"  Skip (neg yok): {skip_count} sorgu")

print(f"\n  Negatif tipi dağılımı:")
for ntype, cnt in neg_type_counter.most_common():
    pct = 100 * cnt / neg_count
    bar = "█" * (cnt // 5000)
    print(f"    {ntype:15s}: {cnt:7,} ({pct:.1f}%) {bar}")

print(f"\n  Kaydedildi: {out_path}")

# ─── 9. Validation — Negatifler Gerçekten Yanlış Marka mı? ──────────────────
print(f"\n  VALİDASYON (brand_swap doğruluğu):")
brand_swaps = df[(df["neg_type"] == "brand_swap")].head(500)
# item_text'teki marka alanını kontrol et (| ayracından sonraki 2. kısım)
correct_brand_swap = 0
for _, row in brand_swaps.iterrows():
    q_text   = trl(row["query_text"])
    item_txt = row["item_text"]
    # Query'de marka var mı?
    for tok in q_text.split():
        if tok in [r.split()[0] for r in brand_to_iids.keys()]:
            # Item'da bu token var mı? Varsa brand_swap başarısız
            if tok not in item_txt:
                correct_brand_swap += 1
            break
print(f"    Brand_swap negatif kalitesi: ~{100*correct_brand_swap/max(len(brand_swaps),1):.0f}% temiz")

# ─── 10. Örnek Çiftler ───────────────────────────────────────────────────────
print(f"\n  ÖRNEK ÇİFTLER:")
for ntype in ["positive", "brand_swap", "gender_swap", "age_swap", "same_cat"]:
    sample = df[df["neg_type"] == ntype].head(2)
    if len(sample) == 0:
        continue
    print(f"\n  [{ntype.upper()}]")
    for _, row in sample.iterrows():
        lbl = "✓ POS" if row["label"] == 1 else "✗ NEG"
        print(f"    {lbl}  Q: '{row['query_text'][:50]}'")
        print(f"           I: '{row['item_text'][:60]}'")

print(f"\n{'='*60}")
print(f"TAMAMLANDI")
print(f"  Sonraki adım: 15_bge_structured_v11.py")
print(f"  Bu veri → BGE fine-tune → submission v11")
print(f"{'='*60}")
