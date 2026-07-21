"""
48_tabibert_smoketest.py — TabiBERT (boun-tabilab/TabiBERT) duman testi
=========================================================================
Amaç: v34'ün ana kaldıracı olarak düşünülen TabiBERT'in (ModernBERT mimarili,
Türkçe, Apache 2.0) bizim query+item ikili sınıflandırma görevimizde
GERÇEKTEN çalışıp çalışmadığını UCUZ bir şekilde doğrulamak — 9-16 saatlik
tam koşuyu göze almadan önce.

Kontrol edilenler:
  1. AutoModelForSequenceClassification ile yükleniyor mu (trust_remote_code
     gerekiyor mu, classifier head düzgün ekleniyor mu)?
  2. Küçük bir örneklemde (80K, v33'ün aynı negatif tarifinden) 2-3 epoch
     fine-tune ederken loss düzgün düşüyor mu?
  3. İnference hızı — dbmdz/bert-base-turkish-cased (zaten diskte, models/bert_v23)
     ile AYNI donanımda, AYNI 50K satırlık veri üzerinde doğrudan kıyaslanıyor.

Karar: Bu script sağlıklı sonuç verirse v34 ana koşusu TabiBERT ile yapılır.
Hata/çökme/anlamsız loss varsa dbmdz/electra-base-turkish-cased-discriminator'a
geçilir (kanıtlanmış yedek).

NOT: Bu script SADECE duman testi — hiçbir model kalıcı olarak kaydedilmez,
hiçbir submission üretilmez.
"""

import gc, re, sys, time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout.reconfigure(encoding="utf-8")

BASE       = Path(__file__).resolve().parents[1]
DATA       = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
MODELS_DIR = BASE / "claude only" / "models"
BERT_V23   = MODELS_DIR / "bert_v23"  # kıyaslama için mevcut model

UNKNOWN = "unknown"
LOWER = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(s): return str(s).translate(LOWER).lower().strip()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# ─────────────────────────────────────────────────────────────────────
# HIZLI VERİ + NEGATİF ÜRETİMİ (v33 ile birebir aynı tarif, seed=42)
# ─────────────────────────────────────────────────────────────────────
print("\n[A] Veri yükleniyor + negatifler üretiliyor (v33 tarifi)...", flush=True)
t0 = time.time()

items       = pd.read_csv(DATA / "items.csv")
terms       = pd.read_csv(DATA / "terms.csv")
train_pairs = pd.read_csv(DATA / "training_pairs.csv")

for col in ["title","category","brand","gender","age_group","attributes"]:
    items[col] = items[col].fillna(UNKNOWN).apply(trl)
terms["query"] = terms["query"].fillna("").apply(trl)
items["main_category"] = items["category"].str.split("/").str[0].fillna(UNKNOWN).apply(trl)
items["sub_category"]  = items["category"].str.split("/").str[1].fillna(UNKNOWN).apply(trl)

RENKLER = {
    "kırmızı","mavi","beyaz","siyah","sarı","yeşil","pembe","mor","gri","turuncu",
    "lacivert","bej","kahverengi","altın","gold","gümüş","silver","rose","ekru","krem",
    "bordo","haki","füme","antrasit","indigo","petrol",
}
COLOR_NORM = {"gold":"altın","silver":"gümüş","rose":"pembe","krem":"bej","kiremit":"kırmızı"}
COLOR_FAMILY = {
    "antrasit":"gri","füme":"gri","platin":"gri","lacivert":"mavi","indigo":"mavi",
    "petrol":"mavi","bordo":"kırmızı","altın":"sarı","gold":"sarı","krem":"bej","ekru":"bej",
}
def norm_color(c): return COLOR_NORM.get(c, c)
def color_family(c): return COLOR_FAMILY.get(c, c)
def get_query_color(q):
    for tok in q.split():
        if tok in RENKLER: return norm_color(tok)
    return None
def parse_attrs(s):
    if not s or s in (UNKNOWN, ""): return {}
    d = {}
    for part in s.split(","):
        if ":" in part:
            k, _, v = part.partition(":")
            d[k.strip()] = v.strip()
    return d
def extract_item_color_family(attrs_str):
    d = parse_attrs(attrs_str)
    renk = d.get("renk", "")
    if not renk: return ""
    return color_family(norm_color(renk))

items["color_family"] = items["attributes"].apply(extract_item_color_family)

item_ids_arr    = items["item_id"].values
item_mains_arr  = items["main_category"].values
item_subs_arr   = items["sub_category"].values
item_colors_arr = items["color_family"].values
iid_to_str = {str(row.item_id): row for row in items.itertuples()}
tid_to_q   = {row.term_id: row.query for row in terms.itertuples()}


def build_group_idx(df, cols):
    df_reset = df.reset_index(drop=True)
    if len(cols) == 1:
        return {k: g.index.values for k, g in df_reset.groupby(cols[0], sort=False)}
    return {k: g.index.values for k, g in df_reset.groupby(cols, sort=False)}

by_main   = build_group_idx(items, ["main_category"])
by_gender = build_group_idx(items, ["gender"])
by_mg     = build_group_idx(items, ["main_category","gender"])
by_age    = build_group_idx(items, ["age_group"])
by_ma     = build_group_idx(items, ["main_category","age_group"])
by_brand  = build_group_idx(items, ["brand"])

positive_keys = set(train_pairs["term_id"].astype(str) + "\t" + train_pairs["item_id"].astype(str))
used_keys: set = set()
rng = np.random.default_rng(42)

def sample_pool(pool, term_id, pos_iid, max_tries=40):
    if pool is None or len(pool) == 0: return None
    for _ in range(max_tries):
        idx = int(pool[rng.integers(0, len(pool))])
        iid = str(item_ids_arr[idx])
        k = term_id + "\t" + iid
        if iid != pos_iid and k not in positive_keys and k not in used_keys: return iid
    return None

def sample_diff_main(cur_main, term_id, pos_iid, max_tries=80):
    n = len(item_ids_arr)
    for _ in range(max_tries):
        idx = int(rng.integers(0, n))
        if item_mains_arr[idx] == cur_main: continue
        iid = str(item_ids_arr[idx])
        k = term_id + "\t" + iid
        if iid != pos_iid and k not in positive_keys and k not in used_keys: return iid
    return None

def sample_same_brand_diff_main(brand, main, term_id, pos_iid, max_tries=60):
    if not brand or brand == UNKNOWN: return None
    pool = by_brand.get(brand)
    if pool is None or len(pool) == 0: return None
    for _ in range(max_tries):
        idx = int(pool[rng.integers(0, len(pool))])
        if item_mains_arr[idx] == main: continue
        iid = str(item_ids_arr[idx])
        k = term_id + "\t" + iid
        if iid != pos_iid and k not in positive_keys and k not in used_keys: return iid
    return None

def sample_same_brand_diff_sub(brand, main, sub, term_id, pos_iid, max_tries=80):
    if not brand or brand == UNKNOWN: return None
    pool = by_brand.get(brand)
    if pool is None or len(pool) == 0: return None
    for _ in range(max_tries):
        idx = int(pool[rng.integers(0, len(pool))])
        if item_mains_arr[idx] != main: continue
        if item_subs_arr[idx] == sub: continue
        iid = str(item_ids_arr[idx])
        k = term_id + "\t" + iid
        if iid != pos_iid and k not in positive_keys and k not in used_keys: return iid
    return None

def sample_color_conflict(q_color, main, term_id, pos_iid, max_tries=60):
    if q_color is None: return None
    target_family = color_family(q_color)
    pool = by_main.get(main)
    if pool is None or len(pool) == 0: return None
    for _ in range(max_tries):
        idx = int(pool[rng.integers(0, len(pool))])
        item_cf = item_colors_arr[idx]
        if not item_cf or item_cf == target_family: continue
        iid = str(item_ids_arr[idx])
        k = term_id + "\t" + iid
        if iid != pos_iid and k not in positive_keys and k not in used_keys: return iid
    return None

pos_with_info = train_pairs.merge(terms, on="term_id", how="left")
pos_with_info = pos_with_info.merge(
    items[["item_id","main_category","sub_category","gender","age_group","brand"]],
    on="item_id", how="left")
for col in ["main_category","sub_category","gender","age_group","brand","query"]:
    pos_with_info[col] = pos_with_info[col].fillna(UNKNOWN).apply(trl)

neg_tids, neg_iids, neg_src = [], [], []
NEG_PER_POS = 5
for row in pos_with_info.itertuples(index=False):
    tid, pos_id = str(row.term_id), str(row.item_id)
    main, sub, query, brand = str(row.main_category), str(row.sub_category), str(row.query), str(row.brand)
    selected = []
    if len(selected) < NEG_PER_POS:
        iid = sample_same_brand_diff_main(brand, main, tid, pos_id)
        if iid: selected.append((iid, "same_brand_diff_main"))
    if len(selected) < NEG_PER_POS:
        iid = sample_same_brand_diff_sub(brand, main, sub, tid, pos_id)
        if iid: selected.append((iid, "same_brand_diff_sub"))
    if len(selected) < NEG_PER_POS:
        if re.search(r'\berkek\b', query):
            pool = by_mg.get((main, "kadın"))
            if pool is None: pool = by_gender.get("kadın")
            iid = sample_pool(pool, tid, pos_id)
            if iid: selected.append((iid, "gender_conflict"))
        elif re.search(r'\b(kadın|bayan)\b', query):
            pool = by_mg.get((main, "erkek"))
            if pool is None: pool = by_gender.get("erkek")
            iid = sample_pool(pool, tid, pos_id)
            if iid: selected.append((iid, "gender_conflict"))
    if len(selected) < NEG_PER_POS:
        if re.search(r'\b(bebek|çocuk)\b', query):
            pool = by_ma.get((main, "yetişkin"))
            if pool is None: pool = by_age.get("yetişkin")
            iid = sample_pool(pool, tid, pos_id)
            if iid: selected.append((iid, "age_conflict"))
    if len(selected) < NEG_PER_POS:
        q_color = get_query_color(query)
        if q_color is not None:
            iid = sample_color_conflict(q_color, main, tid, pos_id)
            if iid: selected.append((iid, "color_conflict"))
    if len(selected) < NEG_PER_POS:
        iid = sample_pool(by_main.get(main), tid, pos_id)
        if iid: selected.append((iid, "same_main_category"))
    while len(selected) < NEG_PER_POS:
        iid = sample_diff_main(main, tid, pos_id)
        if iid: selected.append((iid, "different_main_category"))
        else: break
    for iid, src in selected[:NEG_PER_POS]:
        k = tid + "\t" + iid
        if k in used_keys or k in positive_keys: continue
        used_keys.add(k)
        neg_tids.append(tid); neg_iids.append(iid); neg_src.append(src)

negatives = pd.DataFrame({"term_id": neg_tids, "item_id": neg_iids, "label": 0, "src": neg_src})
train_pairs["label"] = 1
train_df = pd.concat([
    train_pairs[["term_id","item_id","label"]].assign(src="positive"),
    negatives[["term_id","item_id","label","src"]]
], ignore_index=True)
train_df["term_id"] = train_df["term_id"].astype(str)
train_df["item_id"] = train_df["item_id"].astype(str)
print(f"  Train: {len(train_df):,} | {time.time()-t0:.1f}s", flush=True)

# Küçük duman-testi örneklemi: 80K, stratified (rare-conflict önce)
RARE_SRC = {"gender_conflict","age_conflict","color_conflict"}
rare_df  = train_df[train_df["src"].isin(RARE_SRC)]
rest_df  = train_df[~train_df["src"].isin(RARE_SRC)]
SMOKE_SAMPLE = 80_000
n_rare = min(len(rare_df), SMOKE_SAMPLE // 3)
rare_sample = rare_df.sample(n_rare, random_state=42)
rest_sample = rest_df.sample(SMOKE_SAMPLE - n_rare, random_state=42)
smoke_df = pd.concat([rare_sample, rest_sample], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"  Duman testi örneklemi: {len(smoke_df):,} (rare dahil: {n_rare:,})", flush=True)


def build_bert_item_text(item_row):
    if item_row is None: return ""
    d = parse_attrs(getattr(item_row, "attributes", "") or "")
    renk = d.get("renk", "")
    mat = d.get("materyal bileşeni", d.get("materyal", ""))
    parts = []
    t = getattr(item_row, "title", "") or ""
    b = getattr(item_row, "brand", "") or ""
    c = getattr(item_row, "category", "") or ""
    if t and t != UNKNOWN: parts.append(t)
    if b and b != UNKNOWN: parts.append(b)
    cat_main = c.split("/")[0] if c else ""
    if cat_main and cat_main != UNKNOWN: parts.append(cat_main)
    if renk and renk not in (UNKNOWN, ""): parts.append(f"renk:{renk}")
    if mat and mat not in (UNKNOWN, ""): parts.append(f"mat:{mat[:25]}")
    return " | ".join(parts[:5])

def build_query_item_texts(tids, iids):
    queries = [tid_to_q.get(int(t) if str(t).isdigit() else t, "") for t in tids]
    items_t = [build_bert_item_text(iid_to_str.get(str(i))) for i in iids]
    return queries, items_t

def bulk_tokenize(queries, items_t, tokenizer, max_len=128, chunk=20_000):
    all_ids, all_mask, all_type = [], [], []
    n = len(queries)
    for i in range(0, n, chunk):
        enc = tokenizer(queries[i:i+chunk], items_t[i:i+chunk],
                        max_length=max_len, truncation=True, padding="max_length", return_tensors="pt")
        all_ids.append(enc["input_ids"])
        all_mask.append(enc["attention_mask"])
        all_type.append(enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])))
    return (torch.cat(all_ids), torch.cat(all_mask), torch.cat(all_type))

class TensorPairDataset(Dataset):
    def __init__(self, input_ids, attention_mask, token_type_ids, labels):
        self.input_ids, self.attention_mask, self.token_type_ids = input_ids, attention_mask, token_type_ids
        self.labels = torch.as_tensor(labels, dtype=torch.float)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        enc = {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx],
               "token_type_ids": self.token_type_ids[idx]}
        return enc, self.labels[idx]


smoke_queries, smoke_items_t = build_query_item_texts(smoke_df["term_id"].tolist(), smoke_df["item_id"].tolist())

# ─────────────────────────────────────────────────────────────────────
# [B] TabiBERT YÜKLEME + UYUMLULUK KONTROLÜ
# ─────────────────────────────────────────────────────────────────────
print("\n[B] TabiBERT yükleniyor (boun-tabilab/TabiBERT)...", flush=True)
TABIBERT = "boun-tabilab/TabiBERT"
tabi_load_ok = False
try:
    t_load = time.time()
    tabi_tok = AutoTokenizer.from_pretrained(TABIBERT)
    tabi_model = AutoModelForSequenceClassification.from_pretrained(TABIBERT, num_labels=1).to(device)
    print(f"  ✓ Yüklendi ({time.time()-t_load:.1f}s) — model_type: {tabi_model.config.model_type}", flush=True)
    print(f"  Parametre sayısı: {sum(p.numel() for p in tabi_model.parameters()):,}", flush=True)
    tabi_load_ok = True
except Exception as e:
    import traceback
    print(f"  ✗ TabiBERT YÜKLENEMEDİ: {e.__class__.__name__}: {e}", flush=True)
    traceback.print_exc()

if not tabi_load_ok:
    print("\n✗✗✗ TabiBERT yüklenemedi — v34'te dbmdz/electra-base-turkish-cased-discriminator'a geçilmeli.", flush=True)
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────
# [C] KÜÇÜK FINE-TUNE (2 epoch, 80K örnek) — loss eğrisi kontrolü
# ─────────────────────────────────────────────────────────────────────
print("\n[C] Küçük fine-tune (2 epoch, 80K örnek)...", flush=True)
t1 = time.time()
tr_ids, tr_mask, tr_type = bulk_tokenize(smoke_queries, smoke_items_t, tabi_tok)
print(f"  Tokenize: {time.time()-t1:.1f}s", flush=True)

ds = TensorPairDataset(tr_ids, tr_mask, tr_type, smoke_df["label"].tolist())
BATCH = 32
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)

optimizer = torch.optim.AdamW(tabi_model.parameters(), lr=2e-5, weight_decay=0.01)
scaler = GradScaler() if torch.cuda.is_available() else None
criterion = nn.BCEWithLogitsLoss()

EPOCHS_SMOKE = 2
t_train = time.time()
try:
    for epoch in range(EPOCHS_SMOKE):
        tabi_model.train()
        total_loss = 0
        for step, (enc, lbl) in enumerate(dl):
            enc = {k: v.to(device) for k, v in enc.items()}
            lbl = lbl.to(device)
            optimizer.zero_grad()
            if scaler:
                with autocast("cuda"):
                    loss = criterion(tabi_model(**enc).logits.squeeze(-1), lbl)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(tabi_model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss = criterion(tabi_model(**enc).logits.squeeze(-1), lbl)
                loss.backward()
                nn.utils.clip_grad_norm_(tabi_model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            if step % 100 == 0:
                print(f"  Epoch {epoch+1} step {step}/{len(dl)} loss={total_loss/(step+1):.4f}", flush=True)
    train_time = time.time() - t_train
    n_steps_total = EPOCHS_SMOKE * len(dl)
    steps_per_sec = n_steps_total / train_time
    print(f"\n  ✓ Fine-tune tamamlandı: {train_time:.1f}s | {steps_per_sec:.2f} adım/sn", flush=True)

    # 500K x 5 epoch tam koşu tahmini
    full_steps = 5 * (500_000 // BATCH)
    est_full_train_min = full_steps / steps_per_sec / 60
    print(f"  → 500K/5-epoch tam koşu tahmini: {est_full_train_min:.0f} dk (~{est_full_train_min/60:.1f} saat)", flush=True)
except Exception as e:
    import traceback
    print(f"  ✗ Fine-tune sırasında HATA: {e.__class__.__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────
# [D] INFERENCE HIZ KIYASLAMASI: TabiBERT vs mevcut dbmdz (bert_v23)
# ─────────────────────────────────────────────────────────────────────
print("\n[D] Inference hız kıyaslaması (50K satır, aynı donanım)...", flush=True)

def bench_inference(model, tokenizer, queries, items_t, batch=256, label=""):
    model.eval()
    n = len(queries)
    ids, mask, ttype = bulk_tokenize(queries, items_t, tokenizer)
    t_inf = time.time()
    with torch.no_grad():
        for b in range(0, n, batch):
            be = min(b + batch, n)
            batch_enc = {"input_ids": ids[b:be].to(device), "attention_mask": mask[b:be].to(device),
                         "token_type_ids": ttype[b:be].to(device)}
            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                _ = model(**batch_enc).logits
    dt = time.time() - t_inf
    speed = n / dt
    print(f"  {label}: {n:,} satır | {dt:.1f}s | {speed:.0f} satır/sn", flush=True)
    return speed

bench_queries = smoke_queries[:50_000] if len(smoke_queries) >= 50_000 else smoke_queries
bench_items_t = smoke_items_t[:len(bench_queries)]

tabi_speed = bench_inference(tabi_model, tabi_tok, bench_queries, bench_items_t, label="TabiBERT")

del tabi_model; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

if (BERT_V23 / "config.json").exists():
    print("\n  Mevcut dbmdz/bert-base-turkish-cased (models/bert_v23) yükleniyor kıyaslama için...", flush=True)
    dbmdz_tok = AutoTokenizer.from_pretrained(str(BERT_V23))
    dbmdz_model = AutoModelForSequenceClassification.from_pretrained(str(BERT_V23)).to(device)
    dbmdz_speed = bench_inference(dbmdz_model, dbmdz_tok, bench_queries, bench_items_t, label="dbmdz (mevcut)")
    print(f"\n  HIZ ORANI: TabiBERT / dbmdz = {tabi_speed/dbmdz_speed:.2f}x", flush=True)
    del dbmdz_model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
else:
    print("  ⚠ models/bert_v23 bulunamadı, kıyaslama atlanıyor.", flush=True)

print(f"\n{'='*65}", flush=True)
print("DUMAN TESTİ TAMAMLANDI", flush=True)
print(f"{'='*65}", flush=True)
print("Karar kriterleri:", flush=True)
print("  1. Loss düzenli düştü mü? (yukarıdaki loglara bak)", flush=True)
print("  2. Hız oranı > 1.0 mu? (TabiBERT daha hızlıysa v34'te bütçe avantajı var)", flush=True)
print("  3. Hata/çökme olmadı mı? (bu satıra ulaşıldıysa hayır)", flush=True)
print("→ Sağlıklıysa: v34 TabiBERT ile tam koşu.", flush=True)
print("→ Sorunluysa: v34 dbmdz/electra-base-turkish-cased-discriminator ile.", flush=True)
