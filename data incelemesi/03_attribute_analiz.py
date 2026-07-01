import sys, re
from collections import Counter
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

DATA  = r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL\trendyol-e-ticaret-yarismasi-2026-kaggle"
LOWER = str.maketrans("IİŞĞÜÖÇ", "iışğüöç")
def trl(t): return str(t).translate(LOWER).lower().strip()

items = pd.read_csv(DATA + "/items.csv")
items["attributes"] = items["attributes"].fillna("").apply(trl)

def parse_attrs(s):
    if not s or s == "unknown": return {}
    d = {}
    for part in s.split(","):
        part = part.strip()
        if ":" in part:
            k, _, v = part.partition(":")
            d[k.strip()] = v.strip()
    return d

print("=== ATTRIBUTE KEY ANALİZİ (100K sample) ===")
key_counter = Counter()
val_counter = {}
sample = items["attributes"].dropna().sample(100_000, random_state=42)

for attr_str in sample:
    d = parse_attrs(attr_str)
    for k, v in d.items():
        key_counter[k] += 1
        if k not in val_counter:
            val_counter[k] = Counter()
        val_counter[k][v] += 1

print(f"Unique key sayisi: {len(key_counter):,}")
print("\nEn sik 25 attribute key:")
for k, cnt in key_counter.most_common(25):
    pct = 100 * cnt / 100_000
    ornek = list(val_counter[k].most_common(3))
    ornek_str = " | ".join(f"{v}({c})" for v, c in ornek)
    print(f"  {k:35s}: %{pct:4.1f} | {ornek_str[:65]}")

print("\n\n=== RENK ANALİZİ ===")
renk_vals   = val_counter.get("renk", Counter())
renk_detail = val_counter.get("color detail", Counter())
print(f"Unique 'renk' degeri   : {len(renk_vals):,}")
print(f"Unique 'color detail'  : {len(renk_detail):,}")
print("\nEn sik renk degerleri:")
for r, c in renk_vals.most_common(20):
    print(f"  {r:25s}: {c:6,}")

print("\nGri ailesi (color detail):")
gri = [(v, c) for v, c in renk_detail.most_common(300)
       if any(g in v for g in ["gri", "fume", "antrasit", "kursun", "celik", "platin"])]
for v, c in gri[:15]:
    print(f"  {v:35s}: {c:5,}")

print("\nMavi ailesi (color detail):")
mavi = [(v, c) for v, c in renk_detail.most_common(300)
        if any(m in v for m in ["mavi", "lacivert", "indigo", "petrol", "teal"])]
for v, c in mavi[:15]:
    print(f"  {v:35s}: {c:5,}")

print("\n\n=== MALZEME ===")
for mk in ["materyal", "kumas", "kumas tipi", "materyal bileseni"]:
    found = {k: v for k, v in val_counter.items() if mk in k}
    for k2, vc in found.items():
        print(f"  [{k2}] top 8:")
        for v, c in vc.most_common(8):
            print(f"    {v:40s}: {c:5,}")
        break

print("\n\n=== QUERY ile ATTRIBUTE KORELASYONU ===")
terms = pd.read_csv(DATA + "/terms.csv")
train = pd.read_csv(DATA + "/training_pairs.csv")
terms["query"] = terms["query"].fillna("").apply(trl)

# Pozitif ciftlerde: query'de renk varsa item'in renk attribute'u nedir?
RENKLER = {"kirmizi","mavi","beyaz","siyah","sari","yesil","pembe","mor","gri","turuncu",
            "lacivert","bej","kahverengi","gold","gumus","rose","ekru","krem","bordo","haki"}

pos = train.merge(terms, on="term_id").merge(items[["item_id","attributes"]], on="item_id")
pos["q_color"] = pos["query"].apply(
    lambda q: next((c for c in q.split() if c in RENKLER), None)
)
has_color = pos[pos["q_color"].notna()].copy()
print(f"Query'de renk olan pozitif cift: {len(has_color):,}")

# Her renk icin: item'in renk attribute'u ne?
def get_item_color(attr_str):
    d = parse_attrs(trl(str(attr_str)))
    return d.get("renk", d.get("color", d.get("renk grubu", "")))

has_color["item_renk"] = has_color["attributes"].apply(get_item_color)
has_color["renk_match"] = has_color["q_color"] == has_color["item_renk"]
has_color["renk_partial"] = has_color.apply(
    lambda r: r["q_color"] in r["item_renk"] if r["item_renk"] else False, axis=1
)

print(f"Renk tam eslesme   : {has_color['renk_match'].sum():,} ({100*has_color['renk_match'].mean():.1f}%)")
print(f"Renk partial match : {has_color['renk_partial'].sum():,} ({100*has_color['renk_partial'].mean():.1f}%)")
print(f"Renk hic eslesmiyor: {(~has_color['renk_partial']).sum():,} ({100*(~has_color['renk_partial']).mean():.1f}%)")

print("\nEslesmeyen ornekler (query rengi != item rengi):")
mismatch = has_color[~has_color["renk_partial"]].head(10)
for _, r in mismatch.iterrows():
    print(f"  Query=[{r['query']}] q_color={r['q_color']} | item_renk=[{r['item_renk']}]")

print("\nAnlik en sik renk query -> item renk dagilimi:")
color_cross = has_color.groupby(["q_color","item_renk"]).size().reset_index(name="cnt")
color_cross = color_cross.sort_values("cnt", ascending=False).head(20)
for _, r in color_cross.iterrows():
    print(f"  query:{r['q_color']:10s} -> item:{r['item_renk']:20s}: {r['cnt']:5,}")
