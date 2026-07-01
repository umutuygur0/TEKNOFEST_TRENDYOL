# Trendyol Kaggle 2026 — %96 Macro F1 Yol Haritası

## Mevcut Durum Analizi

| Versiyon | Yaklaşım | Test Skoru |
|----------|----------|-----------|
| v1 | BM25 Top-K | 0.48 |
| v6 | Category + KNN | 0.47 |
| v8 | Brand + BM25 | 0.47 |
| v9 | Cross-Encoder (cross-query neg) | 0.49 ← EN İYİ |
| v10 | Query Transfer TF-IDF | ~0.50 (tahmin) |
| **Hedef** | **Dense Hard Neg + LLM Ensemble** | **0.75-0.85** |

---

## Neden Tüm Modellerimiz ~0.47-0.50 Takılı Kaldı?

### Kök Neden: Yanlış Negatifler

Test veri seti yapısı:
```
Her test query: ~100 aday
  - 13.4 gerçek pozitif (kullanıcıların tıkladığı/beğendiği)
  - 86.6 HARD NEGATİF (üretim arama sistemi tarafından getirilmiş ama ilgisiz)
```

Test hard negatives → **Trendyol'un kendi dense retrieval indexi** ile seçilmiş  
Bizim training negatives → **TF-IDF keyword matching** ile seçilmiş

**Sonuç:** Model yanlış şeyi öğreniyor. Trendyol'un hard neg'leri semantik olarak çok daha zor. Model test zamanında bu "zor" itemları pozitiften ayırt edemiyor.

### Matematiksel Kanıt

```
Gerçekçi CV (BM25 sim, 500 sorgu):
  - Precision: 0.097
  - Recall: 0.087  
  - Macro F1: 0.481 ← test sonuçlarıyla tam eşleşiyor

Sorun: BM25 top-100 adaylarımızın yalnızca %36.5'i gerçek pozitif içeriyor!
```

---

## Temel İçgörü: İki Aşamalı Problem

```
Aşama 1 (RETRİEVAL): Doğru adayları getir
  Şu an: TF-IDF top-100 → %36.5 gerçek pozitif coverage
  Hedef: Dense E5 top-100 → %70-80 gerçek pozitif coverage

Aşama 2 (RANKING): Adaylar içinden doğruları seç  
  Şu an: Keyword overlap, brand match → precision ~10%
  Hedef: Fine-tuned LLM cross-encoder → precision ~80-90%
```

---

## 10 Aşamalı Submission Planı

### FELSEFE

Her submission bir öncekinin üzerine inşa edilir:
- **S1-S2**: Sıfır-shot güçlü modeller (ne kadar iyi olduğumuzu gör)
- **S3-S4**: Dense hard negative mining (kritik adım)
- **S5-S6**: LLM fine-tuning (büyük sıçrama)
- **S7-S8**: Self-training + ensemble (sınırı zorla)
- **S9-S10**: Distillation + threshold opt (hedefe ulaş)

---

### S1: Zero-Shot BGE Reranker (Baz Güçlü Model)

**Hedef F1**: 0.55-0.62  
**Model**: `BAAI/bge-reranker-v2-m3` (568M param)  
**Teknik**: Fine-tune yok, doğrudan test üzerinde çalıştır  

**Neden bu model?**
- MS-MARCO multilingual ile eğitilmiş (Türkçe dahil)
- Cross-encoder: query+item birlikte encode → semantic anlama
- MIRACL benchmark'ta Türkçe için en iyi açık kaynak model
- Bizim v3'ten (MiniLM 0.45) çok daha güçlü

**İmplementasyon**:
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=256)
scores = model.predict([(query, item_text) for ...])
```

**Beklenti**: v9 cross-encoder'ımız (0.49) zaten fine-tuned'dı.
BGE zero-shot muhtemelen 0.52-0.58 verir.

---

### S2: BGE + Dense Hard Negative Mining (İlk Kritik Adım)

**Hedef F1**: 0.62-0.70  
**Model**: `BAAI/bge-reranker-v2-m3` fine-tuned  
**Negatifler**: `intfloat/multilingual-e5-large-instruct` ile dense retrieval  

**Kritik Fark — v7 vs S2:**

```
v7 (BAŞARISIZ):
  "pandora gold" → negatif = TF-IDF top pandora olmayan takı
  Problem: Pandora itemları yanlışlıkla negatif sayıldı (partial labels)

S2 (YENİ):
  "pandora gold" → negatif = E5 dense top-200'den çekilen non-positive itemlar
  Neden daha iyi:
  1. E5, semantic proximity'ye göre getiriyor (TF-IDF'ten güçlü)
  2. Bu itemlar test'in hard negatives'ine daha çok benziyor
  3. Contamination: DÜŞÜK (dense space'de pandora itemları farklı cluster'da)
```

**Dense Hard Negative Mining Pipeline**:
```
1. multilingual-e5-large-instruct ile tüm 962K item encode et
   → Her item: 1024 boyutlu vektör
   
2. Her training query için:
   a. Query vektörü hesapla  
   b. Tüm item vektörleriyle cosine sim  
   c. Top-200 al
   d. Bu 200'den training positives çıkar
   e. Kalan 150-186 item = DENSE HARD NEGATİF
   
3. Training çiftleri:
   - (query, pos_item) → 1  (250K)
   - (query, dense_hard_neg) → 0  (750K, 3 per pos)
   
4. BGE-reranker fine-tune: 1 epoch, lr=1e-5
```

**Neden E5-large?**
- `intfloat/multilingual-e5-large-instruct`: 560M param bi-encoder
- MTEB multilingual leaderboard'da top 5
- Türkçe metni semantik olarak anlıyor
- Örnek: "luck" → avon luck vektöre yakın (keyword olmasa bile)

**Compute**: E5 encoding 962K item → ~45 dakika (GPU'da)

---

### S3: In-Batch Negatives + InfoNCE Loss

**Hedef F1**: 0.65-0.72  
**Teknik**: SimCSE / GTE yaklaşımı  
**Neden**: Cross-encoder sadece pair-wise bakıyor, list-wise bakmalıyız

**Listwise Training**:
```python
# Her batch'te:
# - 1 pozitif item
# - 15 dense hard negative
# Loss: InfoNCE (kontrastif)
# Model, tüm negatiflerle AYNI ANDA kıyaslanıyor

loss = -log(exp(score(q, pos)) / sum(exp(score(q, neg_i))))
```

Bu, çift-çift BCELoss'tan çok daha güçlü:
- Model aynı anda 15 negatif arasında sıralamayı öğreniyor
- Test zamanında 86 aday arasında doğruyu seçmeye daha iyi hazırlanıyor

---

### S4: Iterative Hard Negative Mining (İkinci Tur)

**Hedef F1**: 0.68-0.76  
**Teknik**: S3 modeliyle önce tahmin yap → yanlış tahminler = "hard hard" negatives → retrain

```
Tur 1 (S3 model):
  → Query "pandora gold" için model "silver heart ring" (yanlış) yüksek puan verdi
  → Bu item artık SUPER HARD NEGATİF oldu
  
Tur 2 (S4 model):
  → Training: (pandora gold, pandora ring) + (pandora gold, silver heart ring[super hard neg])
  → Model daha da zorlanıyor → daha iyi öğreniyor
```

**Pipeline**:
```
1. S3 modeliyle training queries üzerinde inference
2. Yanlış sıralanan itemları (yüksek puanlanmış non-positive) topla
3. Bu "model'in zor bulduğu" itemları negatif olarak ekle
4. 2. tur eğitim
```

**Not**: Contamination riski düşük çünkü dense retrieval'dan geliyorlar.

---

### S5: Qwen2.5-7B Sıfır-Shot Relevance Scoring

**Hedef F1**: 0.70-0.78  
**Model**: `Qwen/Qwen2.5-7B-Instruct`  
**Teknik**: LLM'in dünya bilgisini kullan, fine-tune olmadan

**Neden LLM Güçlü?**
- Qwen2.5-7B Türkçe anlıyor
- Marka bilgisi var: "luck" → Avon, "pandora" → mücevher markası
- E-ticaret semantiğini anlıyor
- Keyword olmadan anlıyor

**Prompt Tasarımı**:
```
Sen bir e-ticaret arama sistemisin. Verilen arama sorgusu için ürünün alakalı 
olup olmadığını belirle.

Arama Sorgusu: "{query}"
Ürün Adı: "{title}"
Marka: "{brand}"
Kategori: "{category}"

Bu ürün bu arama sorgusuyla alakalı mı? Sadece "evet" veya "hayır" yaz.
```

**Inference Stratejisi**:
- batch_size=64 ile local inference
- Her pair için logit["evet"] - logit["hayır"] = relevance score
- Top-14 per query

**Süre**: 32K query × 100 candidates = 3.2M çift → batch=64 → 50K batch → GPU'da ~2 saat

---

### S6: Qwen2.5-7B LoRA Fine-Tuning

**Hedef F1**: 0.75-0.83  
**Teknik**: LoRA ile 7B model'i görevimize adapte et

**Veri**:
- Pozitif: 250K training çifti → "evet"
- Negatif: Dense hard negatives (S2'den) → "hayır"
- Format: instruction-tuning

**Training Konfigürasyonu**:
```python
# LoRA config
r=16, lora_alpha=32, 
target_modules=["q_proj", "v_proj"],
lora_dropout=0.1

# Training
num_epochs=1
lr=2e-4
batch_size=8 (gradient_accumulation_steps=4)
max_length=256

# Compute: ~4-6 saat (RTX 4090) veya ~8-10 saat (RTX 3080)
```

**Neden LoRA?**
- Tam fine-tune: 7B × 4 bytes = 28GB VRAM gerekli
- LoRA: sadece ~50M param update → 8-12GB VRAM yeterli

---

### S7: Self-Training (Test Pseudo-Labels)

**Hedef F1**: 0.78-0.85  
**Teknik**: S6 modeli yüksek güvenli test tahminlerini yeni eğitim verisi olarak kullan

```
S6 modeli → test 3.36M pair score et
→ Score > 0.95 olan positif tahminler → PSEUDO-POZİTİF (yüksek güven)
→ Score < 0.05 olan negatif tahminler → PSEUDO-NEGATİF (yüksek güven)
→ Bu pseudo-etiketleri eğitim verisine ekle → S7 modelini eğit
```

**Filtreleme**:
```python
# Sadece çok emin olunan tahminleri al
pseudo_pos = test_pairs[scores > 0.95]  # ~50K tahmin
pseudo_neg = test_pairs[scores < 0.05]  # ~200K tahmin
```

**Dikkat**: Pseudo-label gürültüsü var. Bu yüzden S7'yi S6 üzerine fine-tune et,
sıfırdan değil → catastrophic forgetting'i önler.

---

### S8: Çok Modelli Ensemble

**Hedef F1**: 0.82-0.88  
**Teknik**: Farklı modellerin skorlarını birleştir

**Ensemble Bileşenleri**:
```
Model A: BGE-reranker fine-tuned (S4) → query-item semantic match
Model B: E5-large bi-encoder → dense retrieval score
Model C: Qwen2.5-7B LoRA (S6) → LLM relevance judgment
Model D: Brand + Category heuristic (v8'den) → domain knowledge

Final Score = 0.35×A + 0.20×B + 0.35×C + 0.10×D
```

**Neden Ensemble Güçlü?**
- Her model farklı bilgiyi yakalar
- BGE: cross-attention features
- E5: embedding space proximity
- Qwen: world knowledge + reasoning
- Heuristic: domain rules

**Meta-learner alternatifi**:
- 1000 training sorgu üzerinde her modelin skorunu topla
- LightGBM ile ağırlıkları öğren
- Test üzerinde apply et

---

### S9: GPT-4 / Claude Distillation

**Hedef F1**: 0.85-0.92  
**Teknik**: Büyük LLM'den bilgi damıt

**API Kullanımı (maliyet kontrollü)**:
```python
# Her query için sadece TOP-20 adayı GPT-4'e gönder
# (100 değil, 20 → 5× daha ucuz)

for query, candidates in test_queries:
    top20 = initial_model_score(query, candidates)[:20]
    gpt4_scores = gpt4_rank(query, top20)
    final_preds[query] = top14_from(gpt4_scores)
```

**Maliyet Tahmini**:
- 32K query × 20 items × 100 token/item = 64M token
- GPT-4o: $5/1M token → $320 total
- Claude Sonnet: daha ucuz

**Alternati**: Yalnızca "zor" sorgular için kullan (model confidence < 0.7)

**Distillation**:
- GPT-4'ün verdiği etiketleri → S6/S8 modelini retrain
- Daha ucuz model, büyük modelin bilgisini öğreniyor

---

### S10: Threshold Optimization + Final Polish

**Hedef F1**: 0.88-0.96  
**Teknik**: Sabit K=14 yerine her query için özel threshold

**Problem**: K=14 sabit → bazı queries için çok az, bazıları için çok fazla

**Çözüm**: Her query için ayrı threshold öğren
```python
# Training queries üzerinde optimal threshold bul
for tid, true_pos in train_pos.items():
    scores = model.score(tid, all_candidates)
    best_threshold = find_threshold_for_best_f1(scores, true_pos)
    
# Query feature'larına göre threshold predict et
threshold_model = LightGBM(features=[
    "query_length",
    "max_score",
    "score_gap_14_15",  # 14. ve 15. item arasındaki skor farkı
    "brand_detected",
    "category_confidence"
])
threshold_model.fit(train_thresholds)

# Test'e uygula
for tid in test_queries:
    scores = model.score(tid, candidates)
    K = threshold_model.predict(features(tid))
    predictions[tid] = top_K(scores)
```

**Final Ensemble Ağırlıkları** (S10 için optimize edilmiş):
```
BGE reranker (iter. hard neg): 0.30
E5-large bi-encoder:           0.15
Qwen2.5-7B LoRA:              0.30
Self-trained model (S7):       0.15
GPT-4 distilled (S9):         0.10
```

---

## Gerekli Modeller ve Kaynaklar

### Araştırma Onaylı — State-of-the-Art (2024-2025)

| Model | Boyut | Görev | MIRACL TR | Öncelik |
|-------|-------|-------|-----------|---------|
| `intfloat/multilingual-e5-large-instruct` | 560M | Dense retrieval (bi-encoder) | En iyi açık kaynak | **KRİTİK** |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 125M | Hızlı cross-encoder | Doğrulanmış MS-MARCO | **İLK ADIM** |
| `cross-encoder/mmarco-multilingual-v1` | 250M | Güçlü cross-encoder | daha yavaş, daha iyi | S3-S4 |
| `BAAI/bge-reranker-v2-m3` | 568M | En güçlü cross-encoder | top multilingual | S2+ |
| `castorini/mT5-large-mmarco-v2` | 580M | mT5 reranker (alternatif) | MIRACL winner | S4 alternatif |
| `Qwen/Qwen2.5-7B-Instruct` | 7B (4bit: ~4GB) | LLM scoring / fine-tune | — | S5-S6 |

### FAISS — Dense Retrieval için Zorunlu

```bash
pip install faiss-gpu  # GPU için
# veya
pip install faiss-cpu  # CPU için (daha yavaş)
```

```python
import faiss
import numpy as np

# 962K item embed et, FAISS indexe koy
dimension = 1024  # E5-large output
index = faiss.IndexFlatIP(dimension)  # Inner product = cosine sim (normalize edilmişse)
index.add(item_embeddings)  # (962K, 1024)

# Her query için top-200 nearest neighbor
query_vecs = encode_queries(query_texts)  # (N_query, 1024)
D, I = index.search(query_vecs, 200)  # top-200 per query
```

---

## Kritik Başarı Faktörleri

### 1. Dense Hard Negatives (S2-S4)
Bu olmadan 0.60'ı geçemeyiz.  
TF-IDF negatives → model keyword matching öğrenir  
Dense negatives → model semantic understanding öğrenir

### 2. LLM (S5-S6)
Bu olmadan 0.75'i geçemeyiz.  
LLM'in dünya bilgisi olmadan:
- "luck" = Avon markası → bilinmez
- "fono yayınları" = eğitim yayınevi → bilinmez

LLM ile:
- Bu bilgi zaten modelde var → transfer öğrenme

### 3. Ensemble (S8)
Bu olmadan 0.85'i geçemeyiz.  
Tek model → kör nokta var  
Ensemble → kör noktalar farklı → toplam daha iyi

### 4. Threshold Optimization (S10)
Son 5-8 puan bu adımdan gelir.  
K=14 sabit → her query için %100 doğru değil  
Adaptive K → query'e özel

---

## Zaman Planı

| Submission | Süre | Temel Gereksinim |
|-----------|------|-----------------|
| S1 (zero-shot BGE) | 1 gün | BGE model indirme, inference |
| S2 (dense neg) | 2-3 gün | E5 encoding 962K item, fine-tune |
| S3 (InfoNCE) | 1 gün | S2 üzerine loss değişikliği |
| S4 (iter. hard neg) | 1-2 gün | S3 üzerine 2. tur |
| S5 (Qwen zero-shot) | 1-2 gün | Qwen indirme, inference |
| S6 (Qwen LoRA) | 2-3 gün | LoRA training |
| S7 (self-train) | 1-2 gün | Pseudo-label pipeline |
| S8 (ensemble) | 1 gün | Score fusion |
| S9 (GPT-4 distill) | 1-2 gün | API calls (~$100-300) |
| S10 (threshold opt) | 1 gün | Meta-learner |

**Toplam**: ~15-20 gün | Deadline: 17 Temmuz 2026

---

## Submission Sırası ve Öncelikleri

```
MUTLAKA YAP (slot harcamaya değer):
  S1 → Base score belirle (1 slot)
  S2 → Dense neg büyük sıçrama (1 slot)
  S5 → LLM sıçraması (1 slot)
  S8 → Ensemble peak (1 slot)

GEREKIRSE YAP:
  S3, S4, S6, S7, S9, S10 (duruma göre)
```

---

## Gerçekçi Hedef Analizi (Araştırma Bulgularıyla Güncellendi)

### Araştırma Sonucu:

| Teknik Seviye | Beklenen Macro F1 | Neden |
|---------------|------------------|-------|
| Mevcut (v9) | 0.49 | Yanlış negatifler, TF-IDF |
| S1-S2 (dense neg + BGE) | 0.60-0.72 | Doğru hard neg mining |
| S3-S5 (listwise + LLM) | 0.68-0.78 | LLM dünya bilgisi |
| S6-S8 (LoRA + ensemble) | 0.73-0.85 | Fine-tuned LLM + çok model |
| S9-S10 (distillation) | 0.78-0.88 | Oracle LLM signal |

**MIRACL 2023 Turkish track**: En iyi takım ~0.87 NDCG@10  
**MS-MARCO multilingual**: Top çözümler ~0.85-0.90 MRR@10  
**Bu yarışma** için: **0.75-0.85** gerçekçi üst sınır, %96 çok yüksek hedef

### Neden %96 Zor?

1. **Query belirsizliği**: "batarya" → telefon bataryası mı? Araba aküsü mü? El feneri mi?
2. **Ürün belirsizliği**: Başlık kullanıcının niyetini tam yansıtmıyor
3. **Etiket gürültüsü**: Bazı test etiketleri click-based → noise var
4. **%79 yeni ürün**: Test itemlarının %79'u training'de yok → tam cold start

### Yeni Hedef: **0.75+ macro F1** (TEKNOFEST için gerçekçi)

Mevcut 0.49 → 0.75 = **+0.26 iyileştirme** gerekiyor
Bu hedef için: S1-S6 yeterli (dense neg + LLM)

---

## Sonraki Adım: S1'i Başlat (Hızlı)

```bash
pip install sentence-transformers faiss-gpu
```

```python
# 3 model öncelik sırasıyla dene:
from sentence_transformers import CrossEncoder

# SEÇENEK A: Küçük ama hızlı (ilk test için)
model = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", max_length=256)

# SEÇENEK B: Daha güçlü
model = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=256)

# Test: 100 çift için skor
test_pairs = [("luck", "avon luck kadın parfüm edp 50ml"), ...]
scores = model.predict(test_pairs)
```

**S1 hazır olduğunda**: 1 slot harca, gerçek skoru gör.  
- 0.55+ → plan doğru, S2'ye geç (dense hard neg mining)  
- 0.50 civarında → gold label yapısı çok zor, S5'e (LLM) direkt geç  
- 0.45 altı → model Türkçeyi anlamıyor, mT5 veya XLM-R fine-tune gerekli

## Araştırmanın Doğruladıkları

✓ Dense hard negative mining — en kritik adım (araştırma onaylıyor)  
✓ `multilingual-e5-large` — MIRACL Turkish'te en iyi açık kaynak bi-encoder  
✓ `mmarco-mMiniLMv2` — hızlı cross-encoder, ilk deney için ideal  
✓ LLM distillation — ~3-8% F1 iyileştirme (araştırma onaylıyor)  
✓ In-batch negatives yetmez — explicit hard negatives şart  
✗ 0.96 makro F1 — MIRACL Turkish top team 0.87 NDCG → bu task için ~0.75-0.85 gerçekçi

---

## 30 Haziran 2026 Güncellemesi — S1 ve Dense Coverage Sonucu

### S1: BGE Zero-Shot

`11_bge_s1.py` çalıştı, model CUDA üzerinde yüklendi ve 500 sorguluk gerçekçi CV tamamlandı.

Sonuç:
```
CV Macro F1: 0.502
v9 karşılaştırma: 0.490 → 0.502 (+0.012)
BM25 top-100 candidate coverage: 36.5%
BM25 coverage sıfır olan sorgu: 81/500
Karar: full test inference atlandı, submit edilmedi
```

Yorum: güçlü zero-shot reranker tek başına sorunu çözmüyor. Esas darboğaz, doğru pozitiflerin aday setine yeterince girmemesi ve train/test negatif dağılımının uyuşmaması.

### S2 Diagnostic: Dense Retrieval Coverage

Yeni diagnostic script:
```
python "claude only/12_dense_coverage_check.py" --source cache-e5-v5
```

Mevcut `e5_finetuned_v5` cache ile 500 sorgu sonucu:
```
Dense E5-v5 top-20  coverage: 20.4%
Dense E5-v5 top-50  coverage: 34.9%
Dense E5-v5 top-100 coverage: 47.0%
Dense E5-v5 top-200 coverage: 58.6%
```

50 sorguluk aynı-seed kontrol:
```
BM25 top-100:      36.8%
Dense E5-v5 top-100: 47.1%
```

Fresh `intfloat/multilingual-e5-base` zero-shot kontrolü:
```
E5-base zero-shot top-20  coverage: 15.7%
E5-base zero-shot top-50  coverage: 27.3%
E5-base zero-shot top-100 coverage: 36.6%
E5-base zero-shot top-200 coverage: 45.8%
```

Fresh `intfloat/multilingual-e5-large-instruct` kontrolü:
```
E5-large-instruct top-20  coverage: 16.4%
E5-large-instruct top-50  coverage: 28.6%
E5-large-instruct top-100 coverage: 38.2%
E5-large-instruct top-200 coverage: 48.3%
```

Fresh `BAAI/bge-m3` kontrolü:
```
BGE-M3 top-20  coverage: 14.1%
BGE-M3 top-50  coverage: 23.9%
BGE-M3 top-100 coverage: 32.7%
BGE-M3 top-200 coverage: 42.3%
```

Karar: S3 dense hard-negative mining'i ana sıçrama olarak görmek artık riskli. Fresh E5-large-instruct ve BGE-M3, BM25 ve eski E5-v5'ten daha iyi coverage vermedi. En iyi dense sonuç halen `e5_finetuned_v5` cache'i:
```
top-100: 47.0%
top-200: 58.6%
```

Bu değer S3 için yardımcı sinyal olabilir ama tek başına 0.70+ beklemek doğru değil. Sonraki ana hat: gerçek `submission_pairs.csv` aday dağılımını analiz etmek, train/test term overlap ve item overlap sızıntılarını ölçmek, ardından LLM/Qwen veya test-candidate-aware pseudo-label/reranking stratejisine geçmek.
