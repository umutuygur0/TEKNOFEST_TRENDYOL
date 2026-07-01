"""
13_query_parser.py — Query Özellik Çıkarıcı
==============================================
Tüm training (50K) + test (32K) queryleri ayrıştır:
  brand     : avon, pandora, samsung, huggies...
  product   : luck, ring, galaxy, bezi...
  gender    : kadın / erkek / unisex / bebek / çocuk
  color     : beyaz, siyah, kırmızı, gold, gümüş...
  size      : 50ml, 128gb, 3 numara, xl, l, m...
  age_range : bebek (0-2) / çocuk (3-12) / yetişkin
  material  : deri, gümüş, altın, pamuk, polyester...

Yöntem (3 katmanlı):
  1. Brand dictionary → doğrudan marka eşleştirme
  2. Regex kuralları  → boyut/renk/numara
  3. Training verisinden istatistik → query token + item brand korelasyonu

Çıktı:
  claude only/parsed_queries/train_queries_parsed.csv
  claude only/parsed_queries/test_queries_parsed.csv
  claude only/parsed_queries/brand_dictionary.csv  (validation için)

Çalıştır:
  python "claude only/13_query_parser.py"
"""

import re
import json
import time
import sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE  = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA  = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
OUT   = BASE / "claude only" / "parsed_queries"
OUT.mkdir(parents=True, exist_ok=True)

LOWER_MAP = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(t): return str(t).translate(LOWER_MAP).lower().strip()

# ─── 1. Veri Yükle ──────────────────────────────────────────────────────────
print("[1] Veri yükleniyor...")
items  = pd.read_csv(DATA / "items.csv")
terms  = pd.read_csv(DATA / "terms.csv")
train  = pd.read_csv(DATA / "training_pairs.csv")
test   = pd.read_csv(DATA / "submission_pairs.csv")

iid_to_brand    = dict(zip(items["item_id"], items["brand"].fillna("")))
iid_to_category = dict(zip(items["item_id"], items["category"].fillna("")))
iid_to_title    = dict(zip(items["item_id"], items["title"].fillna("")))
tid_to_query    = dict(zip(terms["term_id"], terms["query"]))

train_pos = defaultdict(set)
for tid, iid in zip(train["term_id"].values, train["item_id"].values):
    train_pos[tid].add(iid)

train_tids = list(set(train["term_id"]))
test_tids  = list(set(test["term_id"]))

print(f"  Train: {len(train_tids):,} sorgu | Test: {len(test_tids):,} sorgu | Items: {len(items):,}")

# ─── 2. Brand Dictionary (Training Verisinden İstatistiksel) ─────────────────
print("\n[2] Brand dictionary oluşturuluyor...")

# Her training query için: positive item'ların markalarını bak
# Query token → pozitif item brandlarında kaç kez geçiyor?
token_brand_count = defaultdict(Counter)   # token → {brand: count}
token_query_count = Counter()              # token → kaç sorguda geçiyor

stopwords = {
    "ve", "ile", "bir", "bu", "da", "de", "mi", "mı", "mu", "mü", "için",
    "ama", "gibi", "olan", "her", "ne", "ki", "çok", "az", "en", "set",
    "adet", "ml", "gr", "kg", "cm", "mm", "lt", "ltr", "li", "lı", "lu",
    "lü", "li", "lik", "lık", "luk", "lük", "no", "no.", "numara",
    "erkek", "kadın", "unisex", "bebek", "çocuk", "kız", "oğlan",
    "büyük", "küçük", "orta", "kısa", "uzun",
}

for tid in train_tids:
    q = trl(tid_to_query.get(tid, ""))
    pos_iids = train_pos[tid]
    pos_brands = {
        trl(iid_to_brand.get(i, "")).split()[0]
        for i in pos_iids
        if trl(iid_to_brand.get(i, "")) not in ("", "nan")
    }
    pos_brands -= {"", "nan"}

    toks = [t for t in q.split() if len(t) >= 3 and t not in stopwords]
    for tok in toks:
        token_query_count[tok] += 1
        for b in pos_brands:
            token_brand_count[tok][b] += 1

# Her token için "bu token = bu marka" güven skoru
# güven = (bu tokenla gelen sorgularda brand / toplam sorgu sayısı)
brand_dict = {}  # token → (brand, confidence)
for tok, brand_counts in token_brand_count.items():
    total_q = token_query_count[tok]
    if total_q < 2:  # Çok nadir tokenları atla
        continue
    best_brand, best_cnt = brand_counts.most_common(1)[0]
    confidence = best_cnt / total_q
    if confidence >= 0.7 and total_q >= 3:  # Güçlü brand sinyal
        brand_dict[tok] = (best_brand, confidence, total_q)

# Brand dictionary'yi kaydet (validation için)
brand_df = pd.DataFrame([
    {"token": tok, "brand": b, "confidence": round(c, 3), "n_queries": n}
    for tok, (b, c, n) in sorted(brand_dict.items(), key=lambda x: -x[1][2])
])
brand_df.to_csv(OUT / "brand_dictionary.csv", index=False)
print(f"  Brand token sayısı: {len(brand_dict):,}")
print(f"  Örnek (top 20):")
for _, row in brand_df.head(20).iterrows():
    print(f"    '{row['token']}' → {row['brand']} (conf={row['confidence']:.2f}, n={row['n_queries']})")

# ─── 3. Regex Kuralları ─────────────────────────────────────────────────────
print("\n[3] Regex desenleri tanımlanıyor...")

# Renk listesi
COLORS = {
    "beyaz", "siyah", "kırmızı", "mavi", "yeşil", "sarı", "turuncu",
    "mor", "pembe", "gri", "kahverengi", "bej", "lacivert", "krem",
    "altın", "gold", "gümüş", "silver", "rose", "ekru", "bordo",
    "antrasit", "füme", "haki", "kiremit", "şampanya", "mint",
}

# Malzeme listesi
MATERIALS = {
    "deri", "süet", "metal", "plastik", "ahşap", "bambu", "pamuk",
    "polyester", "nylon", "keten", "ipek", "yün", "kaşmir", "kadife",
    "gümüş", "altın", "titanyum", "paslanmaz", "seramik", "cam",
}

# Cinsiyet anahtar kelimeleri
GENDER_MAP = {
    "kadın": "kadın", "bayan": "kadın", "kız": "kadın",
    "erkek": "erkek", "bay": "erkek",
    "unisex": "unisex",
    "bebek": "bebek", "bebeği": "bebek", "bebeklere": "bebek",
    "çocuk": "çocuk", "çocuğu": "çocuk", "kids": "çocuk",
}

# Boyut desenleri
SIZE_PATTERNS = [
    (r"\b(\d+(?:\.\d+)?)\s*ml\b",    "ml"),
    (r"\b(\d+(?:\.\d+)?)\s*lt\b",    "lt"),
    (r"\b(\d+(?:\.\d+)?)\s*ltr\b",   "lt"),
    (r"\b(\d+(?:\.\d+)?)\s*gr\b",    "gr"),
    (r"\b(\d+(?:\.\d+)?)\s*kg\b",    "kg"),
    (r"\b(\d+(?:\.\d+)?)\s*gb\b",    "gb"),
    (r"\b(\d+(?:\.\d+)?)\s*tb\b",    "tb"),
    (r"\b(\d+(?:\.\d+)?)\s*cm\b",    "cm"),
    (r"\b(\d+(?:\.\d+)?)\s*mm\b",    "mm"),
    (r"\b(\d+(?:\.\d+)?)\s*inç\b",   "inç"),
    (r"\b(\d+(?:\.\d+)?)\s*inch\b",  "inç"),
    (r"\b(\d+)\s*numara\b",           "numara"),
    (r"\bno[\s.]?(\d+)\b",            "numara"),
    (r"\b(xs|s|m|l|xl|xxl|xxxl|2xl|3xl)\b",  "tekstil_beden"),
]

# ─── 4. Ana Parser Fonksiyonu ────────────────────────────────────────────────
def parse_query(q_raw):
    q = trl(q_raw)
    tokens = q.split()
    result = {
        "raw": q_raw,
        "normalized": q,
        "brand": None,
        "brand_confidence": 0.0,
        "product_tokens": [],      # brand çıkarıldıktan sonra kalan
        "gender": None,
        "colors": [],
        "sizes": [],
        "materials": [],
        "age_range": None,
    }

    # --- Marka tespiti ---
    best_brand = None
    best_conf  = 0.0
    best_tok   = None
    for tok in tokens:
        if tok in brand_dict:
            b, c, _ = brand_dict[tok]
            if c > best_conf:
                best_conf  = c
                best_brand = b
                best_tok   = tok

    if best_brand and best_conf >= 0.7:
        result["brand"]             = best_brand
        result["brand_confidence"]  = round(best_conf, 3)
        # Marka tokenını ürün tokenlarından çıkar
        product_toks = [t for t in tokens if t != best_tok]
    else:
        product_toks = tokens

    # --- Cinsiyet / Yaş tespiti ---
    gender_found = None
    age_found    = None
    for tok in tokens:
        if tok in GENDER_MAP:
            g = GENDER_MAP[tok]
            if g == "bebek":
                age_found = "bebek"
            elif g == "çocuk":
                age_found = "çocuk"
            else:
                gender_found = g

    result["gender"]    = gender_found
    result["age_range"] = age_found

    # --- Renk tespiti ---
    result["colors"] = [t for t in tokens if t in COLORS]

    # --- Malzeme tespiti ---
    result["materials"] = [t for t in tokens if t in MATERIALS]

    # --- Boyut tespiti ---
    sizes = []
    for pattern, unit in SIZE_PATTERNS:
        for m in re.finditer(pattern, q):
            sizes.append(f"{m.group(1)}{unit}")
    result["sizes"] = sizes

    # --- Ürün tokenları (brand/gender/renk/boyut çıkarıldıktan sonra) ---
    skip = set(result["colors"]) | set(result["materials"])
    for tok in list(GENDER_MAP.keys()):
        skip.add(tok)
    skip |= stopwords
    result["product_tokens"] = [
        t for t in product_toks
        if t not in skip and len(t) >= 3 and not re.match(r"^\d", t)
    ]

    return result


# ─── 5. Tüm Query'leri Parse Et ─────────────────────────────────────────────
def parse_all(tids, label):
    print(f"\n[4] {label} ({len(tids):,} sorgu) ayrıştırılıyor...")
    rows = []
    t0   = time.time()
    for i, tid in enumerate(tids):
        q_raw  = tid_to_query.get(tid, "")
        parsed = parse_query(q_raw)
        rows.append({
            "term_id":          tid,
            "query":            q_raw,
            "normalized":       parsed["normalized"],
            "brand":            parsed["brand"],
            "brand_conf":       parsed["brand_confidence"],
            "gender":           parsed["gender"],
            "age_range":        parsed["age_range"],
            "colors":           "|".join(parsed["colors"]),
            "sizes":            "|".join(parsed["sizes"]),
            "materials":        "|".join(parsed["materials"]),
            "product_tokens":   " ".join(parsed["product_tokens"]),
        })
        if (i + 1) % 5000 == 0:
            print(f"  {i+1:6d}/{len(tids)} ({(time.time()-t0):.0f}s)")
    return pd.DataFrame(rows)

train_parsed = parse_all(train_tids, "TRAIN")
test_parsed  = parse_all(test_tids,  "TEST")

# ─── 6. Kaydet ──────────────────────────────────────────────────────────────
train_parsed.to_csv(OUT / "train_queries_parsed.csv", index=False)
test_parsed.to_csv(OUT / "test_queries_parsed.csv",   index=False)
print(f"\n  Kaydedildi → {OUT}/")

# ─── 7. Validation Raporu ───────────────────────────────────────────────────
print("\n" + "="*60)
print("VALİDASYON RAPORU")
print("="*60)

# Marka tespiti coverage
brand_found_train = (train_parsed["brand"].notna()).sum()
brand_found_test  = (test_parsed["brand"].notna()).sum()
print(f"\n  Marka tespit edilen (train): {brand_found_train}/{len(train_parsed)} "
      f"({100*brand_found_train/len(train_parsed):.1f}%)")
print(f"  Marka tespit edilen (test) : {brand_found_test}/{len(test_parsed)} "
      f"({100*brand_found_test/len(test_parsed):.1f}%)")

# Cinsiyet coverage
gen_train = train_parsed["gender"].notna().sum()
gen_test  = test_parsed["gender"].notna().sum()
age_train = train_parsed["age_range"].notna().sum()
print(f"\n  Cinsiyet tespit (train): {gen_train} ({100*gen_train/len(train_parsed):.1f}%)")
print(f"  Cinsiyet tespit (test) : {gen_test} ({100*gen_test/len(test_parsed):.1f}%)")
print(f"  Yaş grubu tespit (train): {age_train} ({100*age_train/len(train_parsed):.1f}%)")

# Renk coverage
color_train = (train_parsed["colors"] != "").sum()
print(f"\n  Renk tespit (train): {color_train} ({100*color_train/len(train_parsed):.1f}%)")

# Boyut coverage
size_train = (train_parsed["sizes"] != "").sum()
print(f"  Boyut tespit (train): {size_train} ({100*size_train/len(train_parsed):.1f}%)")

# ─── 8. Doğruluk Kontrolü — Brand vs Gerçek Pozitif ─────────────────────────
print("\n  MARKA DOĞRULUK KONTROLÜ:")
print("  (Parse edilen marka = pozitif ürünlerin markası mı?)")

correct = 0
total   = 0
wrong_examples = []

for _, row in train_parsed[train_parsed["brand"].notna()].head(200).iterrows():
    tid         = row["term_id"]
    parsed_brand= row["brand"]
    pos_iids    = train_pos.get(tid, set())
    if not pos_iids:
        continue

    # Pozitif ürünlerin brand'larını al
    pos_brands = {
        trl(iid_to_brand.get(i, "")).split()[0]
        for i in pos_iids
        if trl(iid_to_brand.get(i, "")) not in ("", "nan")
    }
    pos_brands -= {"", "nan"}

    total += 1
    if parsed_brand in pos_brands:
        correct += 1
    else:
        wrong_examples.append({
            "query": row["query"],
            "parsed_brand": parsed_brand,
            "actual_brands": list(pos_brands)[:3]
        })

if total > 0:
    acc = 100 * correct / total
    print(f"  Doğruluk: {correct}/{total} = {acc:.1f}%")

    if wrong_examples:
        print(f"\n  Yanlış örnekler (ilk 10):")
        for ex in wrong_examples[:10]:
            print(f"    Query: '{ex['query']}'")
            print(f"      → Parse: '{ex['parsed_brand']}'  Gerçek: {ex['actual_brands']}")

# ─── 9. Örnek Çıktılar ──────────────────────────────────────────────────────
print("\n  ÖRNEK ÇIKTILAR (train):")
examples = train_parsed[train_parsed["brand"].notna()].head(15)
for _, row in examples.iterrows():
    parts = []
    if row["brand"]:   parts.append(f"brand={row['brand']}({row['brand_conf']:.2f})")
    if row["gender"]:  parts.append(f"gender={row['gender']}")
    if row["age_range"]: parts.append(f"age={row['age_range']}")
    if row["colors"]:  parts.append(f"color={row['colors']}")
    if row["sizes"]:   parts.append(f"size={row['sizes']}")
    print(f"  '{row['query']}'")
    print(f"    → {' | '.join(parts) if parts else '(hiç özellik yok)'}")

print("\n  ÖRNEK ÇIKTILAR (markasız sorgular):")
no_brand = train_parsed[train_parsed["brand"].isna()].head(10)
for _, row in no_brand.iterrows():
    parts = []
    if row["gender"]:  parts.append(f"gender={row['gender']}")
    if row["age_range"]: parts.append(f"age={row['age_range']}")
    if row["colors"]:  parts.append(f"color={row['colors']}")
    if row["sizes"]:   parts.append(f"size={row['sizes']}")
    if row["product_tokens"]: parts.append(f"product={row['product_tokens']}")
    print(f"  '{row['query']}'")
    print(f"    → {' | '.join(parts) if parts else '(özellik çıkarılamadı)'}")

print("\n" + "="*60)
print("TAMAMLANDI")
print(f"  Çıktılar: {OUT}/")
print("  Sonraki adım: 14_structured_neg_gen.py")
print("="*60)
