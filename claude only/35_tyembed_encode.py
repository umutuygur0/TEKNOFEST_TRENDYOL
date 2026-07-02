"""
35_tyembed_encode.py — TY-ecomm-embed Embedding Cache Builder
=============================================================
Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0 modelini kullanarak
tüm item ve query embeddinglerini hesapla, cache'e kaydet.

Strateji (sırayla dener):
  1. GPU + trust_remote_code=False  (en hızlı, custom kod bypass)
  2. GPU + trust_remote_code=True   (custom kod, Blackwell'de fail edebilir)
  3. CPU + trust_remote_code=True   (yavaş ama güvenilir, ~2 saat)

Başarılı strateji bulununca item + query embeddinglerini hesapla.

Çıktı (claude only/emb_cache/):
  item_embs_tyembed.npy     : (962873, 768) float32
  item_ids_tyembed.npy      : (962873,) int
  train_q_embs_tyembed.npy  : (17968, 768) float32
  train_q_ids_tyembed.npy   : (17968,) term_id
  test_q_embs_tyembed.npy   : (32185, 768) float32
  test_q_ids_tyembed.npy    : (32185,) term_id
"""

import gc, sys, time, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")

BASE      = Path(r"C:\Users\Asus\Desktop\projeler egit\TEKNOFEST_TRENDYOL")
DATA      = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
EMB_CACHE = BASE / "claude only" / "emb_cache"
EMB_CACHE.mkdir(exist_ok=True)

MODEL_NAME = "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0"
UNKNOWN = "unknown"
LOWER = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(s): return str(s).translate(LOWER).lower().strip()

ITEM_EMB  = EMB_CACHE / "item_embs_tyembed.npy"
ITEM_IDS  = EMB_CACHE / "item_ids_tyembed.npy"
TRAIN_Q_EMB = EMB_CACHE / "train_q_embs_tyembed.npy"
TRAIN_Q_IDS = EMB_CACHE / "train_q_ids_tyembed.npy"
TEST_Q_EMB  = EMB_CACHE / "test_q_embs_tyembed.npy"
TEST_Q_IDS  = EMB_CACHE / "test_q_ids_tyembed.npy"

# ─── VERİ YÜKLEMESİ ──────────────────────────────────────────────────
print("Veri yükleniyor...", flush=True)
items  = pd.read_csv(DATA / "items.csv")
terms  = pd.read_csv(DATA / "terms.csv")
train  = pd.read_csv(DATA / "training_pairs.csv")
sub    = pd.read_csv(DATA / "submission_pairs.csv")

for col in ["title","category","brand"]:
    items[col] = items[col].fillna(UNKNOWN).apply(trl)
terms["query"] = terms["query"].fillna("").apply(trl)
items["L1"] = items["category"].str.split("/").str[0].fillna(UNKNOWN).apply(trl)
tid_to_q = dict(zip(terms["term_id"].astype(str), terms["query"]))

train_tids = sorted(train["term_id"].astype(str).unique())
test_tids  = sorted(sub["term_id"].astype(str).unique())
print(f"  {len(items):,} item | {len(train_tids):,} train query | {len(test_tids):,} test query", flush=True)

# ─── METİN HAZIRLIĞI ─────────────────────────────────────────────────
_ctrl = re.compile(r'[\x00-\x1f\x7f-\x9f]')

def clean(s, n=200):
    s = _ctrl.sub(' ', str(s))
    return s[:n].strip()

def item_text(row):
    t = clean(row.title, 100) if row.title != UNKNOWN else ""
    b = clean(row.brand, 30)  if row.brand != UNKNOWN else ""
    c = clean(row.L1, 30)     if row.L1   != UNKNOWN else ""
    return " ".join(p for p in [t, b, c] if p) or "urun"

print("Item metinleri hazırlanıyor...", flush=True)
item_texts = [item_text(r) for r in items.itertuples(index=False)]
train_q_texts = [clean(tid_to_q.get(t, ""), 80) or "sorgu" for t in train_tids]
test_q_texts  = [clean(tid_to_q.get(t, ""), 80) or "sorgu" for t in test_tids]

# ─── MODEL YÜKLEME STRATEJİSİ ────────────────────────────────────────
def try_load(trust_rc, device):
    """Modeli yükle, 10-item test encode yap. Başarılıysa (tok, mdl, device) döndür."""
    from transformers import AutoTokenizer, AutoModel
    label = f"trust_rc={trust_rc}, device={device}"
    try:
        print(f"  Deneniyor: {label} ...", flush=True)
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        mdl = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=trust_rc)
        mdl = mdl.to(device).eval()
        # Kısa test
        enc = tok(["test ürün laptop"], padding=True, truncation=True,
                  max_length=64, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = mdl(**enc)
        print(f"  ✓ BAŞARILI: {label}", flush=True)
        return tok, mdl, device
    except Exception as e:
        print(f"  ✗ BAŞARISIZ ({e.__class__.__name__}): {str(e)[:120]}", flush=True)
        try: del mdl
        except: pass
        if device == "cuda": torch.cuda.empty_cache()
        gc.collect()
        return None

def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return F.normalize((last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9), p=2, dim=1)

def encode_all(texts, tok, mdl, device, batch_size, max_len, label=""):
    n = len(texts); all_embs = []
    t0 = time.time()
    for i in range(0, n, batch_size):
        enc = tok(texts[i:i+batch_size], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl(**enc)
        all_embs.append(mean_pool(out.last_hidden_state, enc["attention_mask"]).cpu().float().numpy())
        done = min(i + batch_size, n)
        if (i // batch_size) % 50 == 0 or done == n:
            elapsed = time.time() - t0
            speed = done / elapsed if elapsed > 0 else 0
            eta = (n - done) / speed if speed > 0 else 0
            print(f"    {label}: {done:,}/{n:,} | {speed:.0f} item/s | ETA {eta/60:.1f} dk", flush=True)
    return np.concatenate(all_embs, axis=0).astype(np.float32)

# ─── STRATEJİ SIRASI ─────────────────────────────────────────────────
print("\n[1] MODEL YÜKLEME STRATEJİSİ...", flush=True)
result = None

if torch.cuda.is_available():
    # Strateji 1: GPU + custom kod YOK
    result = try_load(trust_rc=False, device="cuda")

if result is None and torch.cuda.is_available():
    # Strateji 2: GPU + custom kod VAR (Blackwell bug riski)
    result = try_load(trust_rc=True, device="cuda")

if result is None:
    # Strateji 3: CPU (yavaş ama kesin çalışır)
    print("  GPU stratejileri başarısız → CPU'ya geçiliyor...", flush=True)
    result = try_load(trust_rc=True, device="cpu")

if result is None:
    print("HATA: Hiçbir strateji çalışmadı!", flush=True)
    sys.exit(1)

tok, mdl, device = result
batch_items  = 512  if device == "cuda" else 256
batch_query  = 1024 if device == "cuda" else 512
print(f"\n  Kullanılan device: {device} | item batch: {batch_items} | query batch: {batch_query}", flush=True)

# ─── ITEM EMBEDDİNGLERİ ──────────────────────────────────────────────
if ITEM_EMB.exists() and ITEM_IDS.exists():
    cached = np.load(str(ITEM_EMB))
    if len(cached) == len(items):
        print(f"\n[2] Item emb cache'den: {len(cached):,} ✓", flush=True)
        item_embs = cached
    else:
        print(f"\n[2] Cache boyutu uyuşmuyor ({len(cached)} vs {len(items)}), yeniden hesaplanıyor...", flush=True)
        item_embs = None
else:
    item_embs = None

if item_embs is None:
    print(f"\n[2] ITEM EMBEDDİNGLERİ ({len(items):,} item)...", flush=True)
    t1 = time.time()
    item_embs = encode_all(item_texts, tok, mdl, device, batch_items, max_len=128, label="items")
    print(f"  → {len(item_embs):,} emb | {(time.time()-t1)/60:.1f} dk", flush=True)
    np.save(str(ITEM_EMB), item_embs)
    np.save(str(ITEM_IDS), items["item_id"].values)
    print(f"  Kaydedildi: {ITEM_EMB}", flush=True)

# ─── TRAIN QUERY EMBEDDİNGLERİ ───────────────────────────────────────
if TRAIN_Q_EMB.exists():
    cached_q = np.load(str(TRAIN_Q_EMB))
    if len(cached_q) == len(train_tids):
        print(f"\n[3] Train query emb cache'den: {len(cached_q):,} ✓", flush=True)
        train_q_embs = cached_q
    else:
        train_q_embs = None
else:
    train_q_embs = None

if train_q_embs is None:
    print(f"\n[3] TRAIN QUERY EMBEDDİNGLERİ ({len(train_tids):,})...", flush=True)
    t1 = time.time()
    train_q_embs = encode_all(train_q_texts, tok, mdl, device, batch_query, max_len=64, label="train_q")
    print(f"  → {len(train_q_embs):,} emb | {(time.time()-t1)/60:.1f} dk", flush=True)
    np.save(str(TRAIN_Q_EMB), train_q_embs)
    np.save(str(TRAIN_Q_IDS), np.array(train_tids))

# ─── TEST QUERY EMBEDDİNGLERİ ────────────────────────────────────────
if TEST_Q_EMB.exists():
    cached_tq = np.load(str(TEST_Q_EMB))
    if len(cached_tq) == len(test_tids):
        print(f"\n[4] Test query emb cache'den: {len(cached_tq):,} ✓", flush=True)
        test_q_embs = cached_tq
    else:
        test_q_embs = None
else:
    test_q_embs = None

if test_q_embs is None:
    print(f"\n[4] TEST QUERY EMBEDDİNGLERİ ({len(test_tids):,})...", flush=True)
    t1 = time.time()
    test_q_embs = encode_all(test_q_texts, tok, mdl, device, batch_query, max_len=64, label="test_q")
    print(f"  → {len(test_q_embs):,} emb | {(time.time()-t1)/60:.1f} dk", flush=True)
    np.save(str(TEST_Q_EMB), test_q_embs)
    np.save(str(TEST_Q_IDS), np.array(test_tids))

# ─── ÖZET ────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print(f"TY-ECOMM-EMBED CACHE HAZIR", flush=True)
print(f"  Device        : {device}", flush=True)
print(f"  Item embs     : {item_embs.shape}", flush=True)
print(f"  Train q embs  : {train_q_embs.shape}", flush=True)
print(f"  Test q embs   : {test_q_embs.shape}", flush=True)
print(f"  Cache dizin   : {EMB_CACHE}", flush=True)

# Hızlı cosine test
print(f"\nÖrnek cosine test (5 query, top-3 item):", flush=True)
item_ids_arr = np.load(str(ITEM_IDS))
iid_to_idx = {str(iid): i for i, iid in enumerate(item_ids_arr)}

q_sample = test_q_embs[:5]  # 5 test query
cos_sample = q_sample @ item_embs.T  # (5, 962873)
for qi in range(5):
    top3 = np.argsort(cos_sample[qi])[::-1][:3]
    q_text = test_q_texts[qi]
    print(f"  Q: '{q_text}'", flush=True)
    for rank, idx in enumerate(top3):
        iid = item_ids_arr[idx]
        row = items[items["item_id"] == iid].iloc[0] if (items["item_id"] == iid).any() else None
        title = row["title"][:50] if row is not None else "?"
        print(f"    #{rank+1} cos={cos_sample[qi][idx]:.3f} | {title}", flush=True)

print(f"\n✓ Tamamlandı. v26 pipeline bu cache'i kullanabilir.", flush=True)
