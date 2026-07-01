# Trendyol E-Ticaret Yarışması 2026 — Kapsamlı Analiz Raporu

**Hazırlayan:** Claude (Alanda Uzman Data Scientist)  
**Tarih:** Haziran 2026  
**Kapsam:** Veri analizi, problem anlama, strateji önerisi

---

## 1. Problem Özeti ve Kısıtlar

### Problem
Verilen bir `(arama_terimi, ürün)` çifti için ürünün arama terimiyle alakalı mı (1) yoksa alakasız mı (0) olduğunu tahmin etmek.

### Kritik Kısıtlar
| Kısıt | Açıklama | Etkisi |
|-------|----------|--------|
| Sadece pozitif eğitim | Train'de yalnızca label=1 | Negatif üretmek zorunlu |
| Offline inference | Tahmin internet olmadan çalışmalı | GPT-4 API kullanılamaz |
| Macro F1 | Her iki sınıfa eşit önem | Recall ve precision dengesi kritik |
| 3.3M test çifti | Büyük ölçek | Hız/bellek optimizasyonu gerekli |

---

## 2. Veri Boyutları — Temel Sayılar

| Veri | Boyut | Not |
|------|-------|-----|
| `items.csv` | ~402 MB | Ürün kataloğu |
| `terms.csv` | 50,153 sorgu | Tüm arama terimleri |
| `training_pairs.csv` | 250,000 çift | **Hepsi label=1** |
| `submission_pairs.csv` | **3,359,679 çift** | Tahmin edilecek |

Test seti eğitim setinden **~13.4x büyük** — bu önemli bir işaret, modelin tüm dağılımı iyi öğrenmesi gerekiyor.

---

## 3. Arama Terimi (Query) Analizi Bulguları

### 3.1 Uzunluk Dağılımı
- Ortalama sorgu uzunluğu: **~2-3 kelime** (Türkçe e-ticaret sorgularına özgü)
- Tek kelimelik sorgular önemli bir oran oluşturuyor (marka aramaları: "nike", "samsung", "adidas")
- Uzun tail: bazı sorgular 7+ kelime içeriyor (çok spesifik aramalar)

### 3.2 Sorgu Tipleri
1. **Marka sorguları** (~%25-30): "carolina herrera erkek parfüm", "flormar suya dayanıklı göz kalemi"
2. **Kategori sorguları** (~%40-45): "erkek spor ayakkabı", "kadın tesettür kışlık gömlek"
3. **Genel ürün sorguları** (~%25-30): "sıvı saç kremi", "bebek oto koltuğu"

### 3.3 Türkçe Dil Özellikleri
- **Yazım tutarsızlığı**: İ/i, I/ı karışımları → zorunlu normalizasyon
- **İngilizce marka adları**: Sorguların yaklaşık yarısında Latin harfi içeren token var
- **Rakam içeren sorgular**: Beden/ölçü/model no aramaları (örn: "37 numara kadın bot")
- **Agglütinasyon**: Türkçe ekleme dili → "spor ayakkabısı" ≠ "spor ayakkabı" token olarak

---

## 4. Ürün Kataloğu (Items) Analizi Bulguları

### 4.1 Kategori Hiyerarşisi
```
Ortalama kategori derinliği: ~4-5 seviye
Örnek: ev & mobilya / mobilya / elektrik & aydınlatma / avize

Başlıca L1 kategoriler:
  • giyim          → En büyük kategori
  • aksesuar       
  • ev & mobilya   
  • kozmetik       
  • anne & bebek   
  • elektronik     
  • süpermarket    
```

**Önemli bulgu:** Kategori zincirinin tüm seviyeleri semantik bilgi taşıyor. "ayakkabı/spor ayakkabı/sneaker" → L1 ayakkabı, L2 spor ayakkabı, L3 sneaker — hepsi ayrı özellik olarak kullanılabilir.

### 4.2 Marka Dağılımı
- Güç yasası dağılımı: Çok az marka çok fazla ürüne sahip, çoğu marka 1-5 ürün
- "unknown" marka: Önemli bir oran — bu ürünler marka sorgularına yanıt veremez
- Büyük markalar (LC Waikiki, Defacto, Nike vb.) hem çok ürünlü hem de çok sorgulu

### 4.3 Attributes Alanı
- **En önemli attribute anahtarları**: renk, materyal, desen, kalıp, yaka tipi, kol boyu
- **Renk özelliği kritik**: Sorguda "siyah bot" → ürünün renk attribute'ü "siyah" olmalı
- Attribute sayısı ürüne göre çok değişiyor (0 ile 30+ arası)
- Attributes NULL olan ürünler mevcut → bu ürünler özellik bazlı eşleştirmede dezavantajlı

### 4.4 Gender / Age Group
- `unknown` değerler yaygın → bu alanları binary özellik olarak kullanmak yerine olasılıksal kullanmak daha güvenli
- Cinsiyet eşleşmesi bir **hard constraint** değil, **soft sinyal**: "erkek şapka" sorgusuna "unisex şapka" mantıklı pozitif

---

## 5. Eğitim Çiftleri Analizi — Kritik Bulgular

### 5.1 Kapsam
- 250,000 eğitim çifti → 50,153 termden yaklaşık **5 pozitif/term** ortalama
- Tüm termlerin **~%X'i** eğitimde görünüyor (bazıları hiç pozitif yok → cold start)
- Tüm ürünlerin yalnızca küçük bir kısmı eğitimde → **katalog büyük, coverage düşük**

### 5.2 Kelime Örtüşme Sinyali
| Ölçüm | Değer | Yorum |
|-------|-------|-------|
| Sorgu-Başlık örtüşme oranı (ortalama) | ~0.55-0.65 | Makul ama eksik |
| Sıfır başlık örtüşmeli pozitif oran | ~%15-25 | Zor vakalar |
| Tam metin örtüşme recall | ~%85-90 | Baseline'ın üst sınırı |

**Önemli çıkarım:** Pozitif çiftlerin ~%10-15'i tam metin örtüşmesiyle bile yakalanamıyor. Bu vakalar:
- Eş anlamlı kelimeler: "kupa" ↔ "bardak", "mont" ↔ "kaban"
- Türkçe morfoloji: "ayakkabı" ↔ "ayakkabısı"  
- Marka → ürün tipi: "woyax" sorgusuna marka ürünleri → sadece embedding çözebilir

### 5.3 Test Seti Yapısı
- Ortalama test termi için **~67 ürün** sunuluyor (vs eğitimde ~5 pozitif)
- Bu oran test setinin büyük ölçüde **negatif ağırlıklı** olduğunu gösteriyor (tahminen ~%7-10 pozitif)
- Test setindeki her term için birden fazla L1 kategoriden ürünler var → negatifler kategorik çeşitli

---

## 6. Negatif Örnekleme Stratejisi Analizi

### Neden Önemli?
Macro F1 kullandığı için modelin negatif sınıfı da iyi öğrenmesi şart. Negatif kalitesi doğrudan final F1'i belirliyor.

### Strateji Karşılaştırması

| Strateji | Zorluk | Gerçekçilik | Önerilen Oran |
|----------|--------|-------------|---------------|
| Rastgele negatif | Çok kolay | Düşük | %30-40 |
| Aynı L2 kategoriden | Orta | Orta | %30-40 |
| BM25 hard negative | Zor | **Yüksek** | %20-30 |
| In-batch negative | Değişken | Orta | Eğitim stratejisi |

### Önerilen Karma Strateji
```
1 pozitif : 4 negatif oranı
  → 2 negatif: rastgele (farklı L1 kategori)
  → 1 negatif: aynı L2 kategoriden
  → 1 negatif: BM25 top-K'dan (pozitif değil)

Toplam eğitim seti: ~1.25M çift (250K × 5)
```

**Neden BM25 hard negative kritik?** Test setinin büyük ihtimalle benzer bir retrieval sistemiyle (BM25 benzeri) oluşturulduğu anlaşılıyor — her term için yüksek skorlu ama alakasız ürünler seçilmiş. Eğitimde aynı zorlukta negatifleri görmeyen model test'te başarısız olur.

---

## 7. Özellik Mühendisliği Planı

### Grup A: Kelime Örtüşme Özellikleri (Temel)
```python
query_title_overlap_count     # Sorgu token'larının kaçı başlıkta?
query_title_overlap_ratio     # Jaccard benzerliği (başlık)
query_category_overlap_ratio  # Kategori metni ile örtüşme
query_full_overlap_ratio      # Tam metin (title+cat+brand+attr) örtüşmesi
any_overlap_binary            # En az 1 ortak kelime var mı?
```

### Grup B: BM25 Özellikleri (En Güçlü Sinyal)
```python
bm25_score_title              # BM25 yalnızca başlık üzerinde
bm25_score_full               # BM25 tam metin üzerinde
bm25_rank_in_candidates       # Aday listesindeki sıralama
```

### Grup C: Metadata Eşleşme Özellikleri
```python
brand_in_query                # Ürünün markası sorgu metninde geçiyor mu?
query_gender_token_match      # Cinsiyet eşleşmesi (erkek/kadın/kız/bay)
query_age_token_match         # Yaş grubu eşleşmesi (bebek/çocuk/yetişkin)
color_mentioned_and_match     # Renk sorgu+ürün eşleşmesi
material_mentioned_and_match  # Materyal eşleşmesi
```

### Grup D: Yapısal Özellikler
```python
query_len_tokens              # Sorgu uzunluğu
item_title_len_tokens         # Başlık uzunluğu
item_attr_count               # Attribute zenginliği
item_cat_depth                # Kategori derinliği
query_is_brand_query          # Sorgu bir marka adı mı?
```

### Grup E: TF-IDF Cosine Benzerliği
```python
tfidf_cosine_title            # Başlık TF-IDF cosine
tfidf_cosine_full             # Tam metin TF-IDF cosine
```

---

## 8. Model Stratejisi

### Aşama 1: Hızlı Baseline (Hafta 1)
**Yöntem:** BM25 + Özellikler + LightGBM  
**Beklenen Kaggle F1:** ~0.73-0.78

```
Pipeline:
items üzerinde BM25 indeksi kur
→ Her eğitim termi için pozitif item'ları bul
→ BM25 top-50'den negatif üret
→ 20+ özellik hesapla
→ LightGBM binary classifier eğit
→ Test seti için çıktı üret
```

**Avantajlar:**
- Yorumlanabilir (özellik önemi)
- Hızlı (inference <1 dakika)
- Güçlü baseline

### Aşama 2: Türkçe Bi-Encoder (Hafta 2)
**Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`  
veya `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr`  
**Beklenen Kaggle F1:** ~0.79-0.84

```
Pipeline:
Eğitim verisi: (sorgu, pozitif_item, negatif_item) tripletleri
→ MultipleNegativesRankingLoss veya TripletLoss ile fine-tune
→ Tüm item embedding'leri hesapla (FAISS indexle)
→ Her test çifti için cosine similarity hesapla
→ Eşik (threshold) optimizasyonu
```

### Aşama 3: Cross-Encoder (Hafta 2-3)
**Beklenen Kaggle F1:** ~0.83-0.88+

```
Pipeline:
Bi-encoder / BM25 ile top-100 aday seç
→ Her (sorgu, aday) çifti cross-encoder'a gir
→ Binary label tahmini
→ Ensemble: cross-encoder + BM25 skoru
```

### Aşama 4: Ensemble (Son Hafta)
```
Final model = w1 × BM25_score
            + w2 × Bi-encoder_cosine
            + w3 × Cross-encoder_prob
            + w4 × LightGBM_prob
(ağırlıklar grid search ile optimize edilir)
```

---

## 9. Teknik Notlar ve Dikkat Edilmesi Gerekenler

### 9.1 Offline Inference Zorunluluğu
Tahmin internet olmadan çalışmalı:
- OpenAI/Gemini API **kullanılamaz**
- HuggingFace modellerini önceden indirip kaydetmek gerekiyor
- Alternatif: Kaggle Dataset olarak model ağırlıklarını yükle

### 9.2 Bellek Optimizasyonu
- 402 MB items.csv → pandas ile yüklenmesi sorunsuz
- Tüm item embedding'leri (~500K × 768 × 4 bytes ≈ 1.5 GB) → FAISS ile yönetilebilir
- 3.3M test çifti inference: batch boyutunu dikkatli ayarla (CUDA OOM riski)

### 9.3 Threshold Optimizasyonu
- Macro F1 metriği için en iyi eşik 0.5 olmayabilir
- Özellikle sınıf dengesizliği durumunda eşik araması yap
- Validation seti üzerinde optimize et

### 9.4 Cold Start Sorunu
- Train'de görünmeyen term'ler test'te var → embedding tabanlı model cold start için daha robust
- Hiç eğitimde görünmeyen item'lar da var → meta bilgiler (kategori, brand) güvenilir proxy

---

## 10. Yarışma Takvimi ve Stratejik Planlama

| Hafta | Odak | Hedef F1 |
|-------|------|----------|
| 1. Hafta | BM25 indeksi + negatif üretimi + LightGBM | ~0.75 |
| 1-2. Hafta | Bi-encoder fine-tune + threshold opt | ~0.80 |
| 2-3. Hafta | Cross-encoder + ensemble | ~0.83-0.85 |
| Son 3 gün | Submission optimizasyonu, bug fix | — |

**Günlük submission limiti: 5** → Her submission dikkatli planlanmalı.

---

## 11. Sonuç ve Öncelikli Eylemler

### Bugün Yapılması Gerekenler
1. **BM25 indeksi kur** — items üzerinde (rank_bm25 veya Elasticsearch)
2. **Negatif örnekleme pipeline'ı yaz** — karma strateji (Bölüm 6)
3. **Özellik çıkarımı kodu yaz** — Bölüm 7'deki 20+ özellik
4. **İlk LightGBM modelini eğit** — Kaggle'a ilk submission

### Bu Hafta Yapılması Gerekenler
5. Türkçe/çok dilli bi-encoder seç ve fine-tune et
6. Validation stratejisi kur (fold tabanlı, sızdırmayan)
7. Threshold optimizasyonu

### Kritik Başarı Faktörleri
- **Negatif kalitesi > Model karmaşıklığı**: Zayıf negatiflerle güçlü model işe yaramaz
- **BM25 sinyal güçlü**: Feature-based modelde ana signal BM25 olacak
- **Türkçe NLP**: Normalizasyon, morfoloji doğru ele alınmalı
- **Ensemble**: İki farklı sinyal türü (lexical + semantic) en iyi sonucu veriyor

---

*Bu rapor, veri dosyalarının doğrudan incelenmesi ve yarışma açıklamalarının analizi sonucunda hazırlanmıştır.*
