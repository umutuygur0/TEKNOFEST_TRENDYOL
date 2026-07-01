"""
02_query_parser_analiz.py — Katman Katman Query Anlama Sistemi
==============================================================
Query'leri 5 katmana ayırıp her katmanı analiz eder:
  L1: Marka          — hangi marka, kaç kelime, hangi pozisyon
  L2: Demografik     — cinsiyet + yaş grubu
  L3: Ürün Tipi      — ne arıyor (spesifik/kategori/soyut)
  L4: Nitelik        — renk, beden, malzeme, stil
  L5: Türkçe yapısı  — morphological complexity

Çıktı: RAPOR_QUERY_PARSER.md + her katmana ait istatistikler
"""

import re, sys, json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
OUT  = BASE / "data incelemesi"

LOWER = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(t): return str(t).translate(LOWER).lower().strip()

print("Veri yükleniyor...", flush=True)
items       = pd.read_csv(DATA / "items.csv")
terms       = pd.read_csv(DATA / "terms.csv")
train_pairs = pd.read_csv(DATA / "training_pairs.csv")
sub_pairs   = pd.read_csv(DATA / "submission_pairs.csv")

for col in ["title","category","brand","gender","age_group","attributes"]:
    items[col] = items[col].fillna("").apply(trl)
terms["query"] = terms["query"].fillna("").apply(trl)

# ═══════════════════════════════════════════════════════════════
# KATMAN 1: MARKA TESPİTİ
# ═══════════════════════════════════════════════════════════════
print("\n═══ L1: MARKA TESPİTİ ═══", flush=True)

# Marka listesi: ürünlerin brand alanından, hem tam hem token bazlı
all_brands = set(items["brand"].unique()) - {"", "unknown"}
# Marka tokenleri (tek kelimelik marka parçaları, 2+ karakter)
brand_single_tokens = Counter()
for b in all_brands:
    for tok in b.split():
        if len(tok) >= 3:
            brand_single_tokens[tok] += 1
# Sık geçen marka tokenleri (en az 5 markada geçiyor)
frequent_brand_tokens = {t for t, c in brand_single_tokens.items() if c >= 5}

def detect_brand(query, all_brands, brand_tokens):
    """
    Query'de marka tespiti.
    Döner: (brand_found: str|None, brand_pos: int|None, brand_len: int, method: str)
    """
    q_tokens = query.split()
    n = len(q_tokens)

    # 1. Tam marka eşleşmesi (uzundan kısaya)
    for length in range(min(n, 5), 0, -1):
        for start in range(n - length + 1):
            candidate = " ".join(q_tokens[start:start+length])
            if candidate in all_brands:
                return candidate, start, length, "exact"

    # 2. Token bazlı marka tespiti (herhangi bir marka tokeni query'de var mı)
    for i, tok in enumerate(q_tokens):
        if tok in brand_tokens and len(tok) >= 4:  # kısa ortak kelimeler hariç
            return tok, i, 1, "token"

    return None, None, 0, "none"

# Tüm query'lere marka tespiti uygula
print("  Marka tespiti uygulanıyor...", flush=True)
brand_results = [detect_brand(q, all_brands, frequent_brand_tokens)
                 for q in terms["query"].tolist()]
terms["brand_found"]  = [r[0] for r in brand_results]
terms["brand_pos"]    = [r[1] for r in brand_results]
terms["brand_len"]    = [r[2] for r in brand_results]
terms["brand_method"] = [r[3] for r in brand_results]

# İstatistikler
method_dist = terms["brand_method"].value_counts()
print(f"\n  Marka tespit yöntemi dağılımı:")
for m, c in method_dist.items():
    print(f"    {m:10s}: {c:6,} query ({100*c/len(terms):.1f}%)")

# Marka pozisyon analizi (sadece exact match için)
exact = terms[terms["brand_method"] == "exact"]
pos_dist = exact["brand_pos"].value_counts().sort_index()
print(f"\n  Marka pozisyonu (exact match, 0=başta):")
for pos, cnt in pos_dist.head(5).items():
    print(f"    pozisyon {pos}: {cnt:,} query ({100*cnt/len(exact):.1f}%)")

brand_len_dist = exact["brand_len"].value_counts().sort_index()
print(f"\n  Marka uzunluğu (kelime sayısı):")
for bl, cnt in brand_len_dist.head(5).items():
    print(f"    {bl} kelime: {cnt:,} ({100*cnt/len(exact):.1f}%)")

# Marka sonrası kalan kelimeler (ürün tipi)
def get_product_part(query, brand_found, brand_pos, brand_len):
    if brand_found is None:
        return query
    tokens = query.split()
    rest = [t for i, t in enumerate(tokens)
            if i < brand_pos or i >= brand_pos + brand_len]
    return " ".join(rest).strip()

terms["product_part"] = [
    get_product_part(q, bf, bp, bl)
    for q, bf, bp, bl in zip(terms["query"], terms["brand_found"], terms["brand_pos"], terms["brand_len"])
]

print(f"\n  Örnek query ayrıştırmaları:")
sample_queries = [
    "nike erkek spor ayakkabı", "us polo tişört", "fono yayınları",
    "pandora gold", "kadın topuklu ayakkabı", "iphone 15 pro",
    "bebek battaniyesi", "karaca çaydanlık", "mavi kot pantolon"
]
for q in sample_queries:
    q_trl = trl(q)
    bf, bp, bl, bm = detect_brand(q_trl, all_brands, frequent_brand_tokens)
    pp = get_product_part(q_trl, bf, bp, bl)
    print(f"    [{q_trl}] → marka=[{bf}]({bm}) | ürün=[{pp}]")

# ═══════════════════════════════════════════════════════════════
# KATMAN 2: DEMOGRAFİK (CİNSİYET + YAŞ)
# ═══════════════════════════════════════════════════════════════
print("\n═══ L2: DEMOGRAFİK ═══", flush=True)

CINSIYET_MAP = {
    "kadın": "kadın", "bayan": "kadın", "women": "kadın", "woman": "kadın",
    "kız": "kız_çocuk",
    "erkek": "erkek", "men": "erkek", "man": "erkek",
    "unisex": "unisex", "nötr": "unisex",
    "erkek çocuk": "erkek_çocuk",
}
YAS_MAP = {
    "bebek": "bebek", "infant": "bebek",
    "çocuk": "çocuk", "kids": "çocuk", "junior": "çocuk",
    "genç": "genç",
    "yetişkin": "yetişkin",
}

def detect_demographics(query):
    cinsiyet, yas = None, None
    # Çok kelimeli önce (sıra önemli)
    if "erkek çocuk" in query or "erkekçocuk" in query:
        cinsiyet = "erkek_çocuk"
    elif "kız çocuk" in query or "kızçocuk" in query:
        cinsiyet = "kız_çocuk"
    else:
        for kw, val in [("kadın","kadın"),("bayan","kadın"),("kız","kız_çocuk"),
                        ("erkek","erkek"),("unisex","unisex")]:
            if re.search(rf"\b{kw}\b", query):
                cinsiyet = val
                break
    for kw, val in YAS_MAP.items():
        if re.search(rf"\b{kw}\b", query):
            yas = val
            break
    return cinsiyet, yas

terms[["cinsiyet","yas"]] = pd.DataFrame(
    [detect_demographics(q) for q in terms["query"]], index=terms.index
)
print(f"  Cinsiyet tespiti:")
for v, c in terms["cinsiyet"].value_counts(dropna=False).items():
    print(f"    {str(v):15s}: {c:6,} ({100*c/len(terms):.1f}%)")
print(f"  Yaş tespiti:")
for v, c in terms["yas"].value_counts(dropna=False).items():
    print(f"    {str(v):12s}: {c:6,} ({100*c/len(terms):.1f}%)")

# ═══════════════════════════════════════════════════════════════
# KATMAN 3: ÜRÜN TİPİ
# ═══════════════════════════════════════════════════════════════
print("\n═══ L3: ÜRÜN TİPİ ═══", flush=True)

# Kategorilerden ürün tipi kelimesi havuzu
cat_tokens = Counter()
for cat in items["category"]:
    for tok in cat.replace("/", " ").split():
        if len(tok) >= 3:
            cat_tokens[tok] += 1
product_type_vocab = {t for t, c in cat_tokens.items() if c >= 50}

# Türkçe kök çıkarma (basit suffix)
TR_SUFFIX = sorted([
    "ları","leri","lar","ler","nın","nin","nun","nün","ın","in","un","ün",
    "daki","deki","taki","teki","dan","den","tan","ten","da","de","ta","te",
    "ya","ye","yı","yi","yu","yü","la","le","ça","çe","ca","ce",
    "lik","lık","luk","lük","cı","ci","cu","cü","çı","çi","çu","çü",
    "sı","si","su","sü","sal","sel","li","lı","lu","lü","ki","ları","leri",
    "a","e","ı","i","u","ü",
], key=len, reverse=True)

def stem(w):
    for s in TR_SUFFIX:
        if w.endswith(s) and len(w) - len(s) >= 3:
            return w[:-len(s)]
    return w

product_type_stems = {stem(t) for t in product_type_vocab}

def detect_product_type(product_part):
    """Ürün tipi kelimelerini tespit et (kategori kelime dağarcığından)"""
    if not product_part:
        return [], "soyut"
    tokens = product_part.split()
    matched = []
    for tok in tokens:
        if tok in product_type_vocab or stem(tok) in product_type_stems:
            matched.append(tok)
    if not matched:
        return [], "belirsiz"
    if len(matched) >= 2:
        return matched, "spesifik"
    return matched, "kategori"

terms[["product_tokens", "product_specificity"]] = pd.DataFrame(
    [detect_product_type(pp) for pp in terms["product_part"]], index=terms.index
)
print(f"  Ürün tipi spesifiklik dağılımı:")
for v, c in terms["product_specificity"].value_counts().items():
    print(f"    {v:12s}: {c:6,} ({100*c/len(terms):.1f}%)")

# ═══════════════════════════════════════════════════════════════
# KATMAN 4: NİTELİKLER
# ═══════════════════════════════════════════════════════════════
print("\n═══ L4: NİTELİKLER ═══", flush=True)

RENKLER = {
    "kırmızı","mavi","beyaz","siyah","sarı","yeşil","pembe","mor","gri","turuncu",
    "lacivert","bej","kahverengi","altın","gold","gümüş","silver","rose","ekru","krem",
    "bordo","haki","füme","karışık","renkli","şeffaf","lila","pudra","antrasit","vizon"
}
BEDENLER = {
    "xs","s","m","l","xl","xxl","2xl","3xl","4xl","5xl",
    "36","37","38","39","40","41","42","43","44","45","46","47","48",
    "small","medium","large","onesize","standart"
}
MALZEME = {
    "deri","hakiki","süet","kumaş","pamuk","keten","polyester","viskon","ipek",
    "kadife","kaşmir","yün","akrilik","naylon","polar","şifon","saten","dantel","örgü"
}
STİL = {
    "uzun","kısa","kollu","kolsuz","yüksek","alçak","beli","dar","bol","slim",
    "klasik","spor","casual","elegant","vintage","retro","modern","basic","oversize"
}

def detect_attributes(query):
    tokens = set(query.split())
    renk     = [t for t in tokens if t in RENKLER]
    beden    = [t for t in tokens if t in BEDENLER]
    malzeme  = [t for t in tokens if t in MALZEME]
    stil     = [t for t in tokens if t in STİL]
    return renk, beden, malzeme, stil

attr_results = [detect_attributes(q) for q in terms["query"]]
terms["attr_renk"]    = [r[0] for r in attr_results]
terms["attr_beden"]   = [r[1] for r in attr_results]
terms["attr_malzeme"] = [r[2] for r in attr_results]
terms["attr_stil"]    = [r[3] for r in attr_results]

for attr_name, col in [("Renk","attr_renk"),("Beden","attr_beden"),
                        ("Malzeme","attr_malzeme"),("Stil","attr_stil")]:
    count = terms[col].apply(bool).sum()
    print(f"  {attr_name:10s} içeren query: {count:,} ({100*count/len(terms):.1f}%)")

# ═══════════════════════════════════════════════════════════════
# KATMAN 5: TÜRKÇE ZORLUK ANALİZİ
# ═══════════════════════════════════════════════════════════════
print("\n═══ L5: TÜRKÇE ZORLUK ANALİZİ ═══", flush=True)

def turkish_complexity(query):
    tokens = query.split()
    suffixed = sum(1 for t in tokens
                   if any(t.endswith(s) and len(t)-len(s)>=3 for s in TR_SUFFIX))
    has_compound = any(len(t) > 10 for t in tokens)  # uzun bileşik
    return {
        "n_tokens": len(tokens),
        "suffixed_ratio": suffixed / max(1, len(tokens)),
        "has_compound": has_compound,
    }

complexity = pd.DataFrame([turkish_complexity(q) for q in terms["query"]])
terms = pd.concat([terms, complexity], axis=1)

print(f"  Suffix içeren token oranı: ort={complexity['suffixed_ratio'].mean():.3f}")
print(f"  Bileşik kelime içeren query: {complexity['has_compound'].sum():,} ({100*complexity['has_compound'].mean():.1f}%)")

# ═══════════════════════════════════════════════════════════════
# KATMAN KOMBINASYONU: QUERY TİP MATRİSİ
# ═══════════════════════════════════════════════════════════════
print("\n═══ QUERY TİP MATRİSİ ═══", flush=True)

def full_query_type(row):
    parts = []
    if row["brand_method"] != "none": parts.append("MARKA")
    if row["cinsiyet"] is not None:   parts.append(f"CİNS({row['cinsiyet']})")
    if row["yas"] is not None:        parts.append(f"YAŞ({row['yas']})")
    if row["attr_renk"]:              parts.append("RENK")
    if row["attr_beden"]:             parts.append("BEDEN")
    if row["product_specificity"] == "spesifik": parts.append("ÜRÜN↑")
    elif row["product_specificity"] == "kategori": parts.append("ÜRÜN")
    if not parts:                     return "BELİRSİZ"
    return " + ".join(parts)

terms["query_type_full"] = terms.apply(full_query_type, axis=1)
type_dist = terms["query_type_full"].value_counts().head(20)
print(f"  En sık 20 query tipi kombinasyonu:")
for qtype, cnt in type_dist.items():
    print(f"    {qtype:45s}: {cnt:5,} ({100*cnt/len(terms):.1f}%)")

# ═══════════════════════════════════════════════════════════════
# DOĞRULAMA: PARSER DOĞRULUĞU VE EŞLEŞMESİ
# ═══════════════════════════════════════════════════════════════
print("\n═══ DOĞRULAMA: PARSER vs POZİTİF ÇİFTLER ═══", flush=True)

pos = train_pairs.merge(terms, on="term_id").merge(items, on="item_id")
pos["item_brand"] = pos["brand_x"].fillna("").apply(trl) if "brand_x" in pos.columns \
                    else pos["brand"].fillna("").apply(trl)
pos["item_gender"] = pos["gender"].fillna("").apply(trl)
pos["item_l1"] = pos["category"].apply(lambda x: x.split("/")[0] if "/" in x else x)

# Marka tespiti doğruluğu
# detected_brand içinde item_brand kelimesi var mı? (query'de marka var, ürünün markası örtüşüyor mu)
def brand_overlap(detected, item_brand):
    if detected is None or item_brand in ("","unknown"):
        return "tespit_yok_veya_marka_yok"
    d_toks = set(detected.split())
    i_toks = set(item_brand.split())
    return "eşleşiyor" if d_toks & i_toks else "eşleşmiyor"

pos["brand_overlap"] = [
    brand_overlap(bf, ib)
    for bf, ib in zip(pos["brand_found"], pos["item_brand"])
]
bo_dist = pos["brand_overlap"].value_counts()
print(f"\n  Tespit edilen marka & ürün markası örtüşmesi (pozitif çiftlerde):")
for k, v in bo_dist.items():
    print(f"    {k:30s}: {v:,} ({100*v/len(pos):.1f}%)")

# Cinsiyet doğruluğu
def gender_accuracy(detected, item_gender):
    if detected is None:
        return "tespit_yok"
    d = detected.replace("_çocuk","").replace("kız","kadın")
    i = item_gender
    if d == i: return "doğru"
    if i in ("","unknown","unisex"): return "ürün_nötr"
    return "yanlış"

pos["gender_acc"] = [
    gender_accuracy(cs, ig)
    for cs, ig in zip(pos["cinsiyet"], pos["item_gender"])
]
ga_dist = pos["gender_acc"].value_counts()
print(f"\n  Cinsiyet tespiti doğruluğu (pozitif çiftlerde):")
for k, v in ga_dist.items():
    print(f"    {k:20s}: {v:,} ({100*v/len(pos):.1f}%)")

# ═══════════════════════════════════════════════════════════════
# SPESIFIK ÖRNEKLER: her tip için iyi/kötü çiftler
# ═══════════════════════════════════════════════════════════════
print("\n═══ ÖRNEK ANALİZ ═══", flush=True)

examples = {
    "MARKA + ÜRÜN↑": ["phantom krampon", "hoover kurutma makinesi", "us polo tişört"],
    "MARKA (soyut)":  ["fono yayınları", "pandora gold", "samsung galaxy"],
    "CİNS + ÜRÜN":    ["kadın çanta", "erkek gömlek", "bebek yatağı"],
    "RENK + ÜRÜN":    ["kırmızı elbise", "siyah bot", "beyaz spor ayakkabı"],
    "BELİRSİZ":       ["tiftik", "panjur", "avize"],
}

for tip, sample_queries in examples.items():
    print(f"\n  [{tip}]")
    for sq in sample_queries:
        q = trl(sq)
        bf, bp, bl, bm = detect_brand(q, all_brands, frequent_brand_tokens)
        cs, ys = detect_demographics(q)
        pp = get_product_part(q, bf, bp, bl)
        pt, ps = detect_product_type(pp)
        ar, ab, am, ast = detect_attributes(q)
        print(f"    [{q}]")
        print(f"      marka={bf}({bm},pos={bp}) | cinsiyet={cs} | yaş={ys}")
        print(f"      ürün-part=[{pp}] | tip={ps} | renk={ar} | beden={ab}")

# ═══════════════════════════════════════════════════════════════
# ZAYIF NOKTALAR: NE KAÇIRIYORUZ?
# ═══════════════════════════════════════════════════════════════
print("\n═══ ZAYIF NOKTALAR ═══", flush=True)

# 1. Eşleşmeyen marka query'leri
brand_none = terms[terms["brand_method"] == "none"]
# Training'de bu queryler için ortalama pozitif-title jaccard
brand_none_pos = pos[pos["term_id"].isin(brand_none["term_id"])]
if len(brand_none_pos):
    # Jaccard hesapla
    def jac(a, b):
        sa, sb = set(a.split()), set(b.split())
        return len(sa & sb) / len(sa | sb) if sa | sb else 0.0
    brand_none_pos["jac"] = [jac(q, t) for q, t in zip(brand_none_pos["query"], brand_none_pos["title"])]
    print(f"\n  Marka tespit EDİLEMEYEN querylerde pozitif-title jaccard: {brand_none_pos['jac'].mean():.3f}")
    print(f"  (Marka TESpıt edilenlerde karşılaştırmak için de hesaplanıyor...)")

brand_found_pos = pos[~pos["term_id"].isin(brand_none["term_id"])]
if len(brand_found_pos):
    brand_found_pos = brand_found_pos.copy()
    brand_found_pos["jac"] = [jac(q, t) for q, t in zip(brand_found_pos["query"], brand_found_pos["title"])]
    print(f"  Marka tespit EDİLEN querylerde pozitif-title jaccard   : {brand_found_pos['jac'].mean():.3f}")

# 2. Morfoloji kayıpları
print(f"\n  Türkçe morfoloji zorlukları:")
morph_cases = [
    ("ayakkabı","ayakkabıları"), ("gömlek","gömlekler"), ("çanta","çantası"),
    ("oyuncak","oyuncakları"), ("çocuk","çocuklar"), ("kazak","kazağı"),
    ("kitap","kitaplar"), ("mont","montu"),
]
for stem_word, inflected in morph_cases:
    has_overlap = stem_word[:4] == inflected[:4]  # basit ön ek kontrolü
    s = stem(trl(inflected))
    print(f"    {inflected:15s} → stem: {s:12s} | eşleşiyor: {s == trl(stem_word)}")

# ═══════════════════════════════════════════════════════════════
# ÖZET İSTATİSTİK
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60, flush=True)
print("ÖZET", flush=True)
print("="*60, flush=True)
n = len(terms)
print(f"Toplam query: {n:,}")
print(f"  Marka tespiti - exact  : {(terms['brand_method']=='exact').sum():,} ({100*(terms['brand_method']=='exact').mean():.1f}%)")
print(f"  Marka tespiti - token  : {(terms['brand_method']=='token').sum():,} ({100*(terms['brand_method']=='token').mean():.1f}%)")
print(f"  Marka tespiti - yok    : {(terms['brand_method']=='none').sum():,}  ({100*(terms['brand_method']=='none').mean():.1f}%)")
print(f"  Cinsiyet var           : {terms['cinsiyet'].notna().sum():,}  ({100*terms['cinsiyet'].notna().mean():.1f}%)")
print(f"  Yaş grubu var          : {terms['yas'].notna().sum():,}   ({100*terms['yas'].notna().mean():.1f}%)")
print(f"  Renk var               : {terms['attr_renk'].apply(bool).sum():,}   ({100*terms['attr_renk'].apply(bool).mean():.1f}%)")
print(f"  Beden var              : {terms['attr_beden'].apply(bool).sum():,}    ({100*terms['attr_beden'].apply(bool).mean():.1f}%)")
print(f"  Ürün tipi - spesifik   : {(terms['product_specificity']=='spesifik').sum():,} ({100*(terms['product_specificity']=='spesifik').mean():.1f}%)")
print(f"  Ürün tipi - kategori   : {(terms['product_specificity']=='kategori').sum():,}  ({100*(terms['product_specificity']=='kategori').mean():.1f}%)")
print(f"  Ürün tipi - belirsiz   : {(terms['product_specificity']=='belirsiz').sum():,}  ({100*(terms['product_specificity']=='belirsiz').mean():.1f}%)")

# Kaydet
terms.to_csv(str(OUT / "terms_parsed.csv"), index=False, encoding="utf-8")
print(f"\nAyrıştırılmış query tablosu kaydedildi: data incelemesi/terms_parsed.csv")
print("Analiz tamamlandı.", flush=True)
