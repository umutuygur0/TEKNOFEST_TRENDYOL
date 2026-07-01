# TEKNOFEST TRENDYOL 2026 — Araştırma Raporu & v5 Plan

## 1. KÖK NEDEN ANALİZİ (Neden 0.45-0.48?)

### Matematiksel Gerçek
- All-zeros tahmini → Macro F1 ≈ 0.464 (pozitif oran ~%7-13 ile)
- v1 BM25 top-14 → 0.48 ≈ rastgele
- v2b LightGBM top-14 → 0.48 ≈ rastgele
- v3 MiniLM semantic → 0.45 < rastgele (NEGATİF KORELASYON!)

**Sonuç:** Bizim "en alakalı" dediğimiz itemlar test'te YANLIŞ.

### Problem Anatomisi
Test setindeki 100 aday Trendyol'un BM25/retrieval sistemiyle SEÇİLMİŞ.
- Yüksek BM25 benzerlik = Trendyol'un göstermeyi düşündüğü ama kullanıcının TIKLAMADİĞİ itemlar
- Gerçek pozitifler: kullanıcının tıkladığı ürünler — bunlar her zaman BM25 top değil
- Sonuç: BM25 yüksek skoru = NEGATIF label (gösterildi ama tıklanmadı)

### Neden MiniLM 0.45 Aldı (BM25'ten Bile Kötü)?
1. **Yanlış model tipi**: `paraphrase-multilingual-MiniLM-L12-v2` → cümle BENZERLİĞİ için.
   - Bu model: "ayakkabı koşu" ≈ "koşu ayakkabısı" (paraphrase, SİMETRİK)
   - Arama: "kadın spor ayakkabı" → "Nike Air Max 270 Kadın Koşu Ayakkabısı" (ASİMETRİK)
2. **Domain yok**: Türkçe e-ticaret verisi üzerinde eğitilmemiş
3. **Yanlış görev formülasyonu**: Paraphrase ≠ Retrieval/Ranking

---

## 2. TÜRKÇE NLP — TÜRKİYE E-TİCARET ÖZGÜ SORUNLAR

### 2.1 Türkçe Morfoloji (Eklemeli Dil)
```
ayakkabı    → ayakkabısı, ayakkabıların, ayakkabıya, ayakkabıcı
spor         → sporcu, sporcular, sporca
kadın        → kadının, kadınlar, kadınca
```
- BM25 "ayakkabı" aramasında "ayakkabısı"yı BULAMAZ (farklı token)
- BERT tabanlı modeller subword tokenization ile bu sorunu çözer
- ZemberekNLP (Java) ile tam morfolojik analiz mümkün ama ağır

### 2.2 Karakter Normalizasyonu
```
Doğru: İ→i, I→ı, Ş→ş, Ğ→ğ, Ü→ü, Ö→ö, Ç→ç
Hatalı Python lowercase: İ→İ (lowercase yapmaz!), I→i (ı yerine i)
```
- Python str.lower() Türkçe'de yanlış: "İSTANBUL".lower() = "i̇stanbul" değil "istanbul"
- `str.maketrans` ile manuel düzeltme şart

### 2.3 E-ticaret Özgü Terimler
- Marka adları genelde İngilizce (Nike, Adidas, Samsung) — query'de Türkçe
- Boyut kodları (36-40, XS-XXL), renk isimleri
- "2li 3lü paket", "1+1", "kombine set" gibi ticari ifadeler

---

## 3. SOTA ARAŞTIRMA (2024-2025)

### 3.1 En İyi Modeller (Product Search için)

| Model | Tip | Dil | Retrieval Kalitesi |
|-------|-----|-----|---------------------|
| intfloat/multilingual-e5-large | Bi-encoder | 100+ | ★★★★★ (MTEB #1) |
| intfloat/multilingual-e5-base | Bi-encoder | 100+ | ★★★★☆ (hızlı, iyi) |
| BAAI/bge-m3 | Dense+Sparse+ColBERT | 100+ | ★★★★★ (SOTA) |
| sentence-transformers/paraphrase-multilingual | Paraphrase | 50+ | ★★☆☆☆ (yanlış görev) |
| dbmdz/bert-base-turkish-cased | MLM | TR | ★★★☆☆ (fine-tune gerekli) |

**Seçim: `intfloat/multilingual-e5-base`**
- NEDEN: Özellikle asimetrik sorgu-belge eşleştirme için eğitilmiş
- Talimat prefiksleri: "query: " + sorgu, "passage: " + ürün metni
- 278M parametre, RTX 5080'de çok hızlı
- MS-MARCO, BEIR, NQ ve çok dilli veri üzerinde eğitilmiş

### 3.2 Fine-Tuning Teknikleri

**MultipleNegativesRankingLoss (In-Batch Negatives)**
```python
# Batch içinde her pozitif çifti için diğer itemlar negatif olarak kullanılır
# Batch_size=128 → her sorgu için 127 negatif
# Efficient, no explicit negatives needed
```

**BM25 Hard Negative Mining (İleride)**
```python
# Eğitim verimini artırmak için:
# Her pozitif çift için BM25 ile top-50 non-positive item → hard negative
# Triplet loss: (query, positive, hard_negative)
```

**In-Batch + Hard Negatives (En İyi)**
- Batch'e hem random hem hard negatives ekle
- Hem kolay hem zor ayrımı öğrenir

### 3.3 KNN Collaborative Filtering
- Eğitim labellarını doğrudan kullanan yaklaşım
- Test sorgusu → benzer eğitim sorguları → onların pozitif itemları → tahmin
- V4'te uyguladık: 92K/3.36M paire skor verdi

### 3.4 Hibrit Yaklaşım (Dense + Sparse)
- SPLADE: TF-IDF benzeri ama neural expansion ile (ayakkabı → spor ayakkabı, sneaker)
- BGE-M3: aynı modelde dense + sparse + colbert
- Bizim durumda: sparse (BM25) TERS KORELASYONlu → sadece dense kullan

---

## 4. ÖNERILEN MİMARİ: v5 SUBMISSION

### 4.1 Bileşenler
```
[Test Sorgu] 
    ↓ "query: " + turkish_normalize(sorgu)
    ↓ E5-base Fine-tuned encoder
    ↓ 768-dim embedding (L2 normalized)
    
[Test Aday Item]
    ↓ "passage: " + title + brand + kategori_L1
    ↓ E5-base Fine-tuned encoder  
    ↓ 768-dim embedding (L2 normalized)
    
[Cosine Similarity] = dot_product(query_emb, item_emb)

+

[KNN Collaborative Score]
    ↓ Test sorgusu ↔ Eğitim sorguları benzerliği
    ↓ En yakın 50 eğitim sorgusu → onların pozitif itemları
    ↓ Item kaç kez yakın eğitim sorgularında pozitif? → KNN_score

[Combined Score] = α * KNN_norm + (1-α) * E5_sim
    α = 0.7 eğer KNN sinyali varsa, 0.1 yoksa
    
[Per-Query Top-K] → K=8 pozitif tahmin
```

### 4.2 Türkçe Preprocessing
```python
LOWER_MAP = str.maketrans("İIŞĞÜÖÇ", "iışğüöç")

def tr_normalize(text):
    return text.translate(LOWER_MAP).lower().strip()

# E5 için format:
# Sorgu: "query: kadın spor ayakkabı"
# Item:  "passage: nike air max 270 | nike | ayakkabı"
```

### 4.3 Fine-Tuning Detayları
- Model: `intfloat/multilingual-e5-base`
- Data: 250K pozitif çift (term_text, item_title+brand+cat)
- Loss: MultipleNegativesRankingLoss (in-batch negatives)
- Epochs: 1 (overfitting olmasın, domain adaptasyonu yeterli)
- Batch size: 128 (RTX 5080 için uygun, 127 in-batch negative)
- Learning rate: 2e-5 (standart fine-tuning için)
- Mixed precision: AMP (FP16, 2x hız)
- Warmup: 200 steps

### 4.4 Zaman Bütçesi (RTX 5080)
```
Veri yükleme:            ~30s
Turkish preprocessing:   ~30s (vectorized)
E5 model download:       ~2-5dk (1.1GB, sadece ilk seferinde)
Fine-tuning (250K, 1ep): ~2-4dk (AMP ile)
Item encoding (962K):    ~1-2dk (batch_size=512)
Query encoding:          ~5s
KNN similarity (GPU):    ~1s
Scoring loop:            ~30s
Kayıt:                   ~30s
TOPLAM:                  ~10-20 dakika
```

---

## 5. K DEĞERİ KALİBRASYONU

### 5.1 Veri
- Train: 250K pozitif / 17968 sorgu = 13.9 pozitif/sorgu
- Test 50/50 split → ~250K pozitif / 32185 sorgu = 7.8 pozitif/sorgu
- Test adayları: 100 item/sorgu → pozitif oran: 7.8/100 = %7.8

### 5.2 Optimal K Seçimi
K=8 → %8 pozitif (50/50 matematiğinden)
K=14 → %14 pozitif (eğitim ortalamasından)

**Karar: K=8 (daha güvenli)**
- 50/50 split matematiği daha güvenilir
- Over-prediction daha az → yüksek false positive = düşük precision = düşük F1

---

## 6. RİSK DEĞERLENDİRME

| Risk | Olasılık | Etki | Azaltma |
|------|----------|------|---------|
| E5 fine-tuned de anti-korelasyon | Düşük | Yüksek | KNN ile hybrid |
| K değeri yanlış | Orta | Orta | K=8 ile 10 arası dene |
| Fine-tuning overfitting | Düşük | Orta | Sadece 1 epoch |
| Model download başarısız | Düşük | Yüksek | Offline HuggingFace cache |

---

## 7. SUBMISSION SONRASI (Gelecek Adımlar)

1. **BM25 Hard Negative Mining**: Eğitim verisi için BM25 top-50 → triplet loss
2. **BGE-M3**: Dense+Sparse+ColBERT kombinasyonu (daha güçlü ama yavaş)
3. **Cross-encoder reranking**: Top-50 adayı yeniden sırala (çok yavaş ama çok iyi)
4. **Synthetic Dataset**: GPT-4 ile Türkçe ürün-sorgu çifti üretimi
5. **ZemberekNLP**: Gerçek Türkçe morfolojik analiz
6. **ColBERT**: Her token için embedding (daha yüksek kalite)

---

## 8. ÖZET KARAR

**v5 Submission = multilingual-e5-base fine-tuned + KNN collaborative**

Önceki hatalar:
- ❌ BM25/overlap özellikleri (ters korelasyon)
- ❌ Paraphrase model (yanlış görev)
- ❌ Zero-shot (domain yok)

v5 doğru yaklaşımlar:
- ✅ Retrieval modeli (E5, asimetrik query-document)
- ✅ Fine-tuning (Türkçe e-ticaret domain adaptasyonu)
- ✅ Training labels (KNN collaborative)
- ✅ Türkçe karakter normalizasyonu
- ✅ Kalibre K=8

Beklenen public score: 0.60-0.80 (önceki max 0.48)
