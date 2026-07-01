# Trendyol Kaggle 2026 — Master Plan: 0.90+ Macro F1

**Mevcut en iyi:** 0.49 (v9)  
**Hedef:** 0.90+  
**Yarışmada 0.90+ yapanlar var → erişilebilir**  
**Deadline:** 17 Temmuz 2026

---

## BÖLÜM 1 — GEÇMİŞTEN ÖĞRENDIKLERIMIZ (v1-v10 Autopsi)

### Tam Versiyon Tarihi

| Ver | Dosya | Yaklaşım | Skor | Kritik Hata |
|-----|-------|----------|------|------------|
| v1 | `01_first_submission.py` | BM25 Top-K=14 | **0.48** | Keyword matching 100 benzer aday içinde discriminative değil |
| v2b | `02b_lgbm_topk_calibrated.py` | LightGBM (BM25 feature) K=14 | **0.48** | Kötü feature üzerine ML → yine kötü sonuç |
| v3 | `03_semantic_embedding.py` | MiniLM zero-shot cosine | **0.45** | Küçük model (22M), Türkçe e-ticaret bilmez |
| v5 | `05_e5_finetuned_v5.py` | E5 fine-tuned **K=8** | **0.43** | (1) K=8 yanlış → F1 ≈ 0 için sınır | (2) in-batch neg → embedding collapse |
| v6 | `06_category_knn_v6.py` | Category KNN + token overlap | **0.47** | Heuristic tavan ~0.47 |
| v7 | `07_crossencoder_v7.py` | XLM-R + TF-IDF hard neg | DNF | TF-IDF neg = kontaminasyon (pozitif ürünler negatif sayıldı) |
| v8 | `08_brand_cat_bm25_v8.py` | Brand detect + Cat + BM25 | **0.47** | Kural-tabanlı yaklaşımlar 0.47'de tavan |
| v9 | `09_crossquery_neg_v9.py` | XLM-R + cross-query neg | **0.49** | Cross-query neg = TOO EASY (farklı kategori) → dağılım uyuşmazlığı |
| v10 | `10_ibqt_v10.py` | Query transfer TF-IDF | ~0.48* | Kolay CV 0.753 şişirilmiş; gerçek CV = 0.481 |

*v10 Kaggle'a yüklenmedi, gerçekçi CV = 0.481

### Kazanılan Bilgiler (Asla Unutma)

```
✓ K = 14 SABIT (percentage değil, mutlak sayı)
✓ Kolay CV ≠ gerçek CV (decoy random ise CV şişer)
✓ Cross-query negatives = yanlış dağılım = düşük skor
✓ TF-IDF negatives = kontaminasyon riski
✓ Heuristic (BM25/brand/category): tavan 0.47
✓ Küçük model (22M MiniLM) zero-shot: 0.45 < büyük model zero-shot
✗ Embedding collapse: in-batch neg + E5 → tüm skorlar yüksek
✗ K=8: macro F1 için felaket (recall çok düşük)
✗ Kolay negatifler = model zor vakaları görmüyor = test'te fail
```

### Ana Kök Neden

```
Test candidates (Trendyol production):
  "luck" → [avon luck 50ml, avon luck 30ml, avon love, avon surrender, ...]
  → Hepsi aynı BRAND, benzer KEYWORD → ayırt etmek çok ZOR

Bizim training negatives (v9 cross-query):
  "luck" + [granül kahve, mobilya, spor ayakkabı]
  → Farklı kategori → ayırt etmek çok KOLAY

Dağılım uyuşmazlığı → model training'de kolayı öğrendi → test'te zorla karşılaştı
```

**Çözüm:** Within-query hard negatives (aynı marka/kategori ama yanlış ürün)

---

## BÖLÜM 2 — NEDEN 0.90 MÜMKÜN?

### Yarışmada 0.90+ Yapanların Muhtemelen Kullandığı

**Senaryo A: Strong Zero-Shot**
- `BAAI/bge-reranker-v2-m3` (568M) veya `mDeBERTa-v3-large` cross-encoder
- HIÇBIR eğitim olmadan 3.36M çifti doğrudan skorla
- Model zaten: "luck + avon" bağlantısını biliyor (İngilizce/evrensel)
- Beklenti: **0.65-0.75 zero-shot**

**Senaryo B: Fine-tuned Cross-Encoder + Within-Query Neg**
- Aynı kategoriden negatifler: ("luck", avon_love) → 0
- Aynı markadan negatifler: ("pandora ring", pandora_necklace) → 0  
- Model öğreniyor: spesifik query intent'i
- Beklenti: **0.78-0.88 fine-tuned**

**Senaryo C: LLM (Qwen/Llama 7B+)**
- Turkish world knowledge built-in
- "luck" → "Avon" markası olduğunu zaten biliyor
- Beklenti: **0.80-0.90 fine-tuned**

**Senaryo D: Ensemble A+B+C**
- Beklenti: **0.88-0.92**

---

## BÖLÜM 3 — 10 SUBMISSION PLANI

### Öncelik Felsefesi

```
S1-S2: Zero-shot güçlü modeller → Tabanı sıfırla, gerçek potansiyeli gör
S3-S4: Dense hard neg mining → Kritik sıçrama (+0.10-0.15)
S5-S6: LLM fine-tune → Türkçe dünya bilgisi (+0.08-0.12)
S7-S8: Ensemble + self-training → Peak (+0.05-0.08)
S9-S10: Distillation + polish → Son optimizasyon
```

---

### S1: Zero-Shot BGE Reranker — Gerçek Taban

**Dosya:** `11_bge_s1.py`  
**Model:** `BAAI/bge-reranker-v2-m3` (568M param, cross-encoder)  
**Eğitim:** YOK — tamamen zero-shot  
**Beklenti:** 0.60-0.72  

**Neden v3 (0.45 MiniLM)'dan çok farklı:**
- v3: MiniLM (22M) cosine sim → bi-encoder, query-item ayrı encode
- S1: bge-reranker (568M) → cross-encoder, query+item BIRLIKTE encode → semantik etkileşim
- 25× büyük model + farklı mimari → çok daha güçlü

**Pipeline:**
```python
# Her test query için 100 aday var
# Direkt (query_text, item_text) → score
# Top-14 → positive

from sentence_transformers import CrossEncoder
model = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=256)

# Batch inference (3.36M pair, ~4-6 saat GPU'da)
scores = model.predict(pairs, batch_size=256, show_progress_bar=True)
```

**CV Stratejisi:**
```
1. 500 training sorgu holdout
2. Her holdout sorgu için BM25 top-100 aday getir (gerçek test yapısını taklit)
3. BGE ile score et
4. Macro F1 hesapla → gerçek test tahmini
5. Eğer CV > 0.58 → Kaggle'a yükle
```

---

### S2: Zero-Shot mMarco-MiniLMv2 — Hız Karşılaştırması

**Dosya:** `12_mmarco_s2.py`  
**Model:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (125M)  
**Beklenti:** 0.55-0.65  

**Amaç:** BGE (568M) vs MiniLMv2 (125M) hız/kalite tradeoff'unu ölç.  
MiniLMv2 ~4× hızlı. Eğer skor farkı < 0.05 → sonraki adımlar için MiniLMv2 kullan.

---

### S3: Dense Hard Negative Mining + BGE Fine-Tune (KRİTİK)

**Dosya:** `13_dense_neg_s3.py`  
**Beklenti:** 0.70-0.80  

**Neden Bu Adım En Kritik:**
```
v9 hatası: cross-query neg → kolay dağılım
S3 çözümü: dense retrieved neg → test dağılımına yakın

Test'teki "luck" negatives:
  - avon love (aynı marka, farklı ürün)
  - oriflame charm (farklı marka, benzer kategori)
  - hugo boss woman (farklı marka, benzer kategori)

S3'ün "luck" negatives (E5 dense retrieval ile):
  - E5("luck") → nearest neighbors → parfüm ürünleri
  - Bunlar gerçek test dağılımına benziyor!
```

**İki Aşamalı Pipeline:**

**Aşama 1 — Dense Index:**
```python
from sentence_transformers import SentenceTransformer
import faiss

# Tüm 962K item encode et
bi_encoder = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
item_embeddings = bi_encoder.encode(all_item_texts, batch_size=256, normalize=True)

# FAISS index
index = faiss.IndexFlatIP(1024)
index.add(item_embeddings)
```

**Aşama 2 — Hard Neg Seçimi:**
```python
for tid, pos_iids in train_pos.items():
    q_emb = bi_encoder.encode([query_text], normalize=True)
    D, I = index.search(q_emb, 200)  # top-200 dense neighbor
    
    candidates = [all_iids[i] for i in I[0]]
    hard_negs = [iid for iid in candidates if iid not in pos_iids][:15]
    
    # Training pair: (query, pos) → 1; (query, hard_neg) → 0
```

**Aşama 3 — BGE Fine-Tune:**
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=256)

# Training veri:
# 250K pozitif + 750K dense hard negative
# 1 epoch, lr=2e-5, batch=32
```

---

### S4: Iterative Hard Negative Mining (2. Tur)

**Dosya:** `14_iter_neg_s4.py`  
**Beklenti:** 0.73-0.82  

S3 modeliyle training queries üzerinde inference yap → modelin yanlış yüksek puan verdiği itemlar = "süper hard negative" → 2. tur eğitim.

```python
# S3 modeliyle training predictions al
s3_scores = s3_model.predict(training_pairs)

# "Zor" false positives = süper hard negative
super_hard_negs = [
    (query, item) 
    for (query, item), score in zip(training_pairs, s3_scores)
    if score > 0.6 and not is_positive(query, item)
]

# S4: S3 üzerine 2. tur, super_hard_negs ile
```

---

### S5: Qwen2.5-7B Zero-Shot Turkish Intelligence

**Dosya:** `15_qwen_zeroshot_s5.py`  
**Beklenti:** 0.72-0.82  

**Neden LLM:**
- "luck" → Qwen2.5-7B biliyor = Avon marka parfümü
- "pandora" → mücevher markası
- "fono yayınları" → eğitim yayınevi
- Bu bilgi AĞIRLIKLARDA zaten var → zero-shot bile güçlü

**Prompt:**
```python
prompt = """Bir Türk e-ticaret arama sisteminde, aşağıdaki ürünün verilen arama sorgusuna 
uygun olup olmadığını değerlendir. Sadece "evet" ya da "hayır" yaz.

Arama sorgusu: {query}
Ürün adı: {title}
Marka: {brand}
Kategori: {category}

Uygun mu?"""

# logit("evet") - logit("hayır") = relevance score
```

---

### S6: Qwen2.5-7B LoRA Fine-Tune

**Dosya:** `16_qwen_lora_s6.py`  
**Beklenti:** 0.78-0.86  

```python
# LoRA config
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    task_type="SEQ_CLS"
)

# Data:
# Pozitif: 250K training pairs → "evet"
# Negatif: S3 dense hard negatives → "hayır"
# Format: instruction tuning

# Training: 1 epoch, lr=2e-4, batch=8, gradient_acc=4
# VRAM: ~12GB (4bit quant + LoRA)
```

---

### S7: Self-Training (Test Pseudo-Labels)

**Dosya:** `17_selftrain_s7.py`  
**Beklenti:** 0.80-0.87  

```python
# S6 modeli ile tüm test pair'leri score et
test_scores = s6_model.predict(all_test_pairs)

# Yüksek güven pseudo-labels
pseudo_pos = [(q, i) for (q, i), s in zip(test_pairs, test_scores) if s > 0.90]
pseudo_neg = [(q, i) for (q, i), s in zip(test_pairs, test_scores) if s < 0.05]

# Training + pseudo-labels ile retrain
# Weight: original=1.0, pseudo_pos=0.7, pseudo_neg=0.5
```

---

### S8: Çok-Model Ensemble

**Dosya:** `18_ensemble_s8.py`  
**Beklenti:** 0.83-0.90  

```python
# Her model için normalize edilmiş scores
scores_bge   = normalize(bge_reranker_scores)     # S4 model
scores_e5    = normalize(e5_bi_encoder_scores)    # dense retrieval
scores_qwen  = normalize(qwen_lora_scores)        # S6 model
scores_brand = normalize(brand_heuristic_scores)  # v8 mantığı

# Ensemble
final_score = (
    0.35 * scores_bge +
    0.15 * scores_e5 +
    0.40 * scores_qwen +
    0.10 * scores_brand
)

# Meta-learner alternatifi:
# LightGBM ile ağırlıkları 1000 training sorgu üzerinde öğren
```

---

### S9: GPT-4 / Claude Distillation (Opsiyonel)

**Dosya:** `19_distill_s9.py`  
**Beklenti:** 0.85-0.92  

```python
# S8 modelinin düşük güvenli tahminleri (0.4-0.6 score)
uncertain_pairs = [(q, i) for (q, i), s in zip(test_pairs, s8_scores) 
                   if 0.35 < s < 0.65]

# Sadece ~5K çift → GPT-4/Claude API
# Maliyet: ~$5-15

for query, item in uncertain_pairs[:5000]:
    label = call_claude_api(query, item)  # "evet"/"hayır"
    # Sonuçları training setine ekle

# Retrain S6 modeli + distillation labels
```

---

### S10: Threshold Optimizasyonu + Final Polish

**Dosya:** `20_threshold_s10.py`  
**Beklenti:** 0.87-0.93  

```python
# K=14 sabit yerine per-query adaptive K
# Training queries üzerinde optimal K öğren

for tid in train_queries:
    scores = model.score(tid, train_candidates)
    best_k = argmax_f1_over_k(scores, true_positives[tid])
    
# Query feature'larından K tahmin et
k_features = {
    "max_score": max(scores),
    "score_gap_14_15": scores[13] - scores[14],  # eşik noktası
    "brand_detected": bool(brand),
    "query_length": len(query.split()),
    "category_confidence": cat_conf
}
k_model = LightGBM()
k_model.fit(train_features, optimal_ks)
```

---

## BÖLÜM 4 — TEKNİK STACK

### Modeller

| Model | Boyut | Görev | Versiyon |
|-------|-------|-------|---------|
| `BAAI/bge-reranker-v2-m3` | 568M | Cross-encoder (S1, S3, S4) | Birincil |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 125M | Hızlı cross-encoder (S2) | Alternatif |
| `intfloat/multilingual-e5-large-instruct` | 560M | Dense retrieval + hard neg mining (S3) | Zorunlu |
| `Qwen/Qwen2.5-7B-Instruct` | 7B (4bit) | LLM scoring/fine-tune (S5-S6) | Büyük sıçrama |
| `microsoft/mdeberta-v3-base` | 86M | Lightweight fallback | Opsiyonel |

### Altyapı

```bash
pip install sentence-transformers faiss-gpu transformers peft bitsandbytes
```

### FAISS Dense Index

```python
import faiss
dim = 1024  # E5-large embedding boyutu
index = faiss.IndexFlatIP(dim)  # Cosine sim (normalize edilmiş vektörler için)
index.add(item_embeddings)  # 962K item, ~4GB RAM
D, I = index.search(query_emb, 200)  # 1ms per query
```

---

## BÖLÜM 5 — GEÇMİŞ HATA KORUMA KURALLARI

| Kural | Kaynak Hata |
|-------|------------|
| Her versiyonda CV gerçek BM25 top-100 üzerinde yap | v10 kolay CV (0.753 sahte) |
| K = 14 asla değiştirme | v5 K=8 felaketi |
| In-batch neg tek başına kullanma | v5 embedding collapse |
| TF-IDF neg kullanma | v7 kontaminasyon |
| Cross-query neg kullanma (yanlış dağılım) | v9 0.49 tavanı |
| CV makro F1 < 0.55 ise submit etme | slot israfı |

---

## BÖLÜM 6 — ZAMAN PLANI (17 Temmuz'a kadar)

```
Hafta 1 (30 Haziran - 6 Temmuz):
  ✓ S1: BGE zero-shot → submit → baseline belirle
  ✓ S2: MiniLMv2 → hız karşılaştırması
  → S3: Dense neg mining pipeline (en uzun adım)

Hafta 2 (7-13 Temmuz):
  → S3 submit (kritik büyük sıçrama)
  → S4: Iterative hard neg → submit
  → S5: Qwen zero-shot → submit

Hafta 3 (14-17 Temmuz):
  → S6: Qwen LoRA → submit
  → S7+S8: Self-train + Ensemble → final submit
```

---

## BÖLÜM 7 — BAŞARI METRİĞİ

| Aşama | Hedef F1 | Durum |
|-------|---------|-------|
| Şu an | 0.49 | Mevcut en iyi |
| S1 sonrası | 0.60-0.72 | BGE zero-shot |
| S3 sonrası | 0.73-0.82 | Dense hard neg |
| S6 sonrası | 0.80-0.88 | LLM fine-tune |
| S8 sonrası | 0.85-0.92 | Ensemble |
| **Hedef** | **0.90+** | **Yarışmada olan skor** |

---

## SONRAKI ADIM: S1 BAŞLIYOR

```bash
# Terminal'de çalıştır:
python "claude only/11_bge_s1.py"

# 1. Adım: BGE model indir (~2GB)
# 2. Adım: CV hesapla (500 sorgu, ~5 dakika)
# 3. Adım: Skor > 0.58 ise test inference başlat (~4-6 saat)
# 4. Adım: submission_v11_bge_zeroshot.csv oluştur
```
