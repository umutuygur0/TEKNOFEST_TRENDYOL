"""
36_tyembed_cpuonly.py — TY-ecomm-embed CPU-Only Encoder
========================================================
DÜZELTMELER (35 vs 36):
  1. CUDA_VISIBLE_DEVICES="" → torch hiç CUDA başlatmaz → process corruption YOK
  2. CLS pooling (pooling_mode_cls_token=True) — mean pooling DEĞİL!
  3. sentence-transformers API (model'in native API'si, daha güvenilir)

Çıktı (claude only/emb_cache/):
  item_embs_tyembed.npy     : (962873, 768) float32, normalized
  item_ids_tyembed.npy      : (962873,) int
  train_q_embs_tyembed.npy  : (17968, 768) float32, normalized
  train_q_ids_tyembed.npy   : (17968,) str
  test_q_embs_tyembed.npy   : (32185, 768) float32, normalized
  test_q_ids_tyembed.npy    : (32185,) str

Süre tahmini (CPU): item ~90-150 dk, queries ~5-10 dk
"""

# ═══════════════════════════════════════════════════════════════════
# NOT: Eski (Asus) makinede CUDA ile bu modelde process-corruption crash'i
# vardı, bu yüzden CPU zorlanmıştı (~90-150 dk item embedding). Bu yeni
# makinede GPU (RTX 3060) smoke-test'te sorunsuz ve ~40x daha hızlı
# (1649 item/s) çalıştığı doğrulandı → GPU kullanılıyor.
# ═══════════════════════════════════════════════════════════════════
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # tokenizer warning'i kapat

import gc, sys, time, re
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Torch device: {DEVICE}")

BASE      = Path(__file__).resolve().parents[1]
DATA      = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
EMB_CACHE = BASE / "claude only" / "emb_cache"
EMB_CACHE.mkdir(exist_ok=True)

MODEL_NAME = "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0"
UNKNOWN    = "unknown"
LOWER      = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")
def trl(s): return str(s).translate(LOWER).lower().strip()

ITEM_EMB    = EMB_CACHE / "item_embs_tyembed.npy"
ITEM_IDS    = EMB_CACHE / "item_ids_tyembed.npy"
TRAIN_Q_EMB = EMB_CACHE / "train_q_embs_tyembed.npy"
TRAIN_Q_IDS = EMB_CACHE / "train_q_ids_tyembed.npy"
TEST_Q_EMB  = EMB_CACHE / "test_q_embs_tyembed.npy"
TEST_Q_IDS  = EMB_CACHE / "test_q_ids_tyembed.npy"

# ─── VERİ YÜKLEMESİ ──────────────────────────────────────────────
print("\n[A] Veri yükleniyor...", flush=True)
items  = pd.read_csv(DATA / "items.csv")
terms  = pd.read_csv(DATA / "terms.csv")
train  = pd.read_csv(DATA / "training_pairs.csv")
sub    = pd.read_csv(DATA / "submission_pairs.csv")

for col in ["title", "category", "brand"]:
    items[col] = items[col].fillna(UNKNOWN).apply(trl)
terms["query"] = terms["query"].fillna("").apply(trl)
items["L1"] = items["category"].str.split("/").str[0].fillna(UNKNOWN).apply(trl)
tid_to_q = dict(zip(terms["term_id"].astype(str), terms["query"]))

train_tids = sorted(train["term_id"].astype(str).unique())
test_tids  = sorted(sub["term_id"].astype(str).unique())
print(f"  {len(items):,} item | {len(train_tids):,} train query | {len(test_tids):,} test query", flush=True)

# ─── METİN HAZIRLIĞI ─────────────────────────────────────────────
_ctrl = re.compile(r'[\x00-\x1f\x7f-\x9f]')

def clean(s, n=200):
    s = _ctrl.sub(' ', str(s))
    return s[:n].strip()

def item_text(row):
    t = clean(row.title, 100) if row.title != UNKNOWN else ""
    b = clean(row.brand,  30) if row.brand != UNKNOWN else ""
    c = clean(row.L1,     30) if row.L1   != UNKNOWN else ""
    return " ".join(p for p in [t, b, c] if p) or "urun"

print("[B] Metinler hazırlanıyor...", flush=True)
item_texts      = [item_text(r) for r in items.itertuples(index=False)]
train_q_texts   = [clean(tid_to_q.get(t, ""), 80) or "sorgu" for t in train_tids]
test_q_texts    = [clean(tid_to_q.get(t, ""), 80) or "sorgu" for t in test_tids]

print(f"  item örnekleri: {item_texts[:3]}", flush=True)
print(f"  query örnekleri: {test_q_texts[:3]}", flush=True)

# ─── MODEL YÜKLEME ────────────────────────────────────────────────
print("\n[C] Model yükleniyor (sentence-transformers, CPU)...", flush=True)
t_load = time.time()

def fix_position_ids(hf_model):
    """
    BUG FIX: PyTorch 2.11.0 + bu model kombinasyonunda position_ids buffer'ının
    ilk elemanı çöp bellek adresi içeriyor (örn. 3140862869504 yerine 0 olmalı).
    Model yüklendikten sonra buffer'ı doğru değerlerle yeniden oluştur.
    """
    n = hf_model.config.max_position_embeddings  # 8192
    correct = torch.arange(n, dtype=torch.int64)
    hf_model.embeddings.position_ids = correct
    # register_buffer da güncelle ki .to(device) çalışsın
    hf_model.embeddings.register_buffer("position_ids", correct, persistent=False)
    print(f"  BUG FIX uygulandı: position_ids[0:5]={hf_model.embeddings.position_ids[:5].tolist()}", flush=True)

from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        MODEL_NAME,
        trust_remote_code=True,
        device=DEVICE,
    )
    # BUG FIX: position_ids buffer'ını düzelt
    for mod in model.modules():
        if hasattr(mod, "auto_model"):
            fix_position_ids(mod.auto_model)
            break
    USE_ST = True
    print(f"  ✓ sentence-transformers yüklendi ({time.time()-t_load:.1f}s)", flush=True)
    print(f"  Model device: {model.device}", flush=True)
    print(f"  Max seq length: {model.max_seq_length}", flush=True)

    # Kısa test
    print("  Test encode...", flush=True)
    t_test = time.time()
    test_embs = model.encode(
        ["test ürün laptop", "spor ayakkabı erkek"],
        batch_size=2,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"  ✓ Test encode OK: shape={test_embs.shape}, dtype={test_embs.dtype} ({time.time()-t_test:.2f}s)", flush=True)
    print(f"  Örnek norm: {np.linalg.norm(test_embs[0]):.4f} (1.0 olmalı)", flush=True)

except Exception as e:
    import traceback
    print(f"  ✗ sentence-transformers BAŞARISIZ ({e.__class__.__name__}): {e}", flush=True)
    traceback.print_exc()
    print("  AutoModel ile deneniyor...", flush=True)

    mdl = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mdl = mdl.to(DEVICE).eval()
    # BUG FIX
    fix_position_ids(mdl)
    USE_ST = False

    # CLS pooling — config'de pooling_mode_cls_token=True
    def cls_pool(last_hidden_state):
        return F.normalize(last_hidden_state[:, 0, :], p=2, dim=-1)

    # Test
    enc_test = tok(["test ürün laptop"], padding=True, truncation=True,
                   max_length=64, return_tensors="pt")
    enc_test = {k: v.to(DEVICE) for k, v in enc_test.items()}
    with torch.no_grad():
        out = mdl(**enc_test)
    test_out = cls_pool(out.last_hidden_state).cpu().numpy()
    print(f"  ✓ AutoModel test OK: shape={test_out.shape}", flush=True)

# ─── ENCODE FONKSİYONLARI ────────────────────────────────────────
def encode_st(texts, batch_size, max_len, label=""):
    """sentence-transformers ile encode (tercih edilen yol)."""
    old_max = model.max_seq_length
    model.max_seq_length = max_len
    t0 = time.time()
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    model.max_seq_length = old_max
    speed = len(texts) / (time.time() - t0)
    print(f"  {label}: {len(texts):,} → speed={speed:.0f} item/s", flush=True)
    return embs.astype(np.float32)

def encode_auto(texts, batch_size, max_len, label=""):
    """AutoModel fallback ile encode."""
    import torch.nn.functional as F
    n = len(texts)
    all_embs = []
    t0 = time.time()
    for i in range(0, n, batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = mdl(**enc)
        emb = F.normalize(out.last_hidden_state[:, 0, :], p=2, dim=-1).cpu().numpy()
        all_embs.append(emb)
        done = min(i + batch_size, n)
        if (i // batch_size) % 100 == 0 or done == n:
            elapsed = time.time() - t0
            speed = done / elapsed if elapsed > 0 else 0
            eta = (n - done) / speed / 60 if speed > 0 else 0
            print(f"    {label}: {done:,}/{n:,} | {speed:.0f}/s | ETA {eta:.1f} dk", flush=True)
    return np.concatenate(all_embs, axis=0).astype(np.float32)

encode_fn = encode_st if USE_ST else encode_auto

# GPU'da daha büyük batch daha verimli (smoke test: RTX 3060 6GB, batch=128 sorunsuz)
BATCH_ITEM  = 128 if DEVICE == "cuda" else 64
BATCH_QUERY = 256 if DEVICE == "cuda" else 128

# ─── ITEM EMBEDDİNGLERİ ──────────────────────────────────────────
print(f"\n[D] ITEM EMBEDDİNGLERİ ({len(items):,} item)...", flush=True)
if ITEM_EMB.exists() and ITEM_IDS.exists():
    cached = np.load(str(ITEM_EMB))
    if len(cached) == len(items):
        print(f"  Cache'den yüklendi: {cached.shape} ✓", flush=True)
        item_embs = cached
    else:
        print(f"  Cache boyutu uyuşmuyor ({len(cached)} vs {len(items)}), yeniden.", flush=True)
        item_embs = None
else:
    item_embs = None

if item_embs is None:
    t1 = time.time()
    item_embs = encode_fn(item_texts, BATCH_ITEM, max_len=128, label="items")
    elapsed = (time.time() - t1) / 60
    print(f"  → {item_embs.shape} | {elapsed:.1f} dk", flush=True)
    np.save(str(ITEM_EMB), item_embs)
    np.save(str(ITEM_IDS), items["item_id"].values)
    print(f"  Kaydedildi: {ITEM_EMB}", flush=True)

# ─── TRAIN QUERY EMBEDDİNGLERİ ───────────────────────────────────
print(f"\n[E] TRAIN QUERY EMBEDDİNGLERİ ({len(train_tids):,})...", flush=True)
if TRAIN_Q_EMB.exists():
    cached_q = np.load(str(TRAIN_Q_EMB))
    if len(cached_q) == len(train_tids):
        print(f"  Cache'den: {cached_q.shape} ✓", flush=True)
        train_q_embs = cached_q
    else:
        train_q_embs = None
else:
    train_q_embs = None

if train_q_embs is None:
    t1 = time.time()
    train_q_embs = encode_fn(train_q_texts, BATCH_QUERY, max_len=64, label="train_q")
    print(f"  → {train_q_embs.shape} | {(time.time()-t1)/60:.1f} dk", flush=True)
    np.save(str(TRAIN_Q_EMB), train_q_embs)
    np.save(str(TRAIN_Q_IDS), np.array(train_tids))
    print(f"  Kaydedildi: {TRAIN_Q_EMB}", flush=True)

# ─── TEST QUERY EMBEDDİNGLERİ ────────────────────────────────────
print(f"\n[F] TEST QUERY EMBEDDİNGLERİ ({len(test_tids):,})...", flush=True)
if TEST_Q_EMB.exists():
    cached_tq = np.load(str(TEST_Q_EMB))
    if len(cached_tq) == len(test_tids):
        print(f"  Cache'den: {cached_tq.shape} ✓", flush=True)
        test_q_embs = cached_tq
    else:
        test_q_embs = None
else:
    test_q_embs = None

if test_q_embs is None:
    t1 = time.time()
    test_q_embs = encode_fn(test_q_texts, BATCH_QUERY, max_len=64, label="test_q")
    print(f"  → {test_q_embs.shape} | {(time.time()-t1)/60:.1f} dk", flush=True)
    np.save(str(TEST_Q_EMB), test_q_embs)
    np.save(str(TEST_Q_IDS), np.array(test_tids))
    print(f"  Kaydedildi: {TEST_Q_EMB}", flush=True)

# ─── KALİTE KONTROLÜ ─────────────────────────────────────────────
print(f"\n[G] KALİTE KONTROLÜ...", flush=True)
item_ids_arr = np.load(str(ITEM_IDS), allow_pickle=True)

# 5 test query için top-3 item
q_sample   = test_q_embs[:5]
cos_sample = q_sample @ item_embs.T   # (5, 962873)
iid_to_row = {str(r.item_id): r for r in items.itertuples(index=False)}

print(f"\n  Cosine sim örnekleri (5 test query, top-3 item):", flush=True)
for qi in range(5):
    top3 = np.argsort(cos_sample[qi])[::-1][:3]
    print(f"  Q: '{test_q_texts[qi]}'", flush=True)
    for rank, idx in enumerate(top3):
        iid  = str(item_ids_arr[idx])
        row  = iid_to_row.get(iid)
        ttl  = row.title[:60] if row else "?"
        print(f"    #{rank+1} cos={cos_sample[qi][idx]:.3f} | {ttl}", flush=True)

# ─── ÖZET ────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print(f"TY-ECOMM-EMBED CACHE TAMAMLANDI", flush=True)
print(f"  Item embs     : {item_embs.shape}", flush=True)
print(f"  Train q embs  : {train_q_embs.shape}", flush=True)
print(f"  Test q embs   : {test_q_embs.shape}", flush=True)
print(f"  Pooling       : CLS token (normalize=True)", flush=True)
print(f"  Similarity    : cosine", flush=True)
print(f"  Cache dizin   : {EMB_CACHE}", flush=True)
print(f"\n✓ v26'da bu cache'i kullanmaya hazır.", flush=True)
