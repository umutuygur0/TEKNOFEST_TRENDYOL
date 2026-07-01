# Trendyol Data İncelemesi Raporu

*Oluşturulma: 2026-07-01 16:56*


## A. Query Tipolojisi

**Toplam unique query:** 50,153
**Token sayısı:** ort=2.6, med=2, max=9
**Karakter sayısı:** ort=16.2, med=15

**Token sayısı dağılımı:**
  1 token: 4,851 query (9.7%)
  2 token: 20,568 query (41.0%)
  3 token: 16,496 query (32.9%)
  4 token: 5,916 query (11.8%)
  5 token: 1,716 query (3.4%)
  6 token: 484 query (1.0%)
  7 token: 94 query (0.2%)
  8 token: 27 query (0.1%)

**Query Tipleri:**
  - `marka`: 38,529 (76.8%)
  - `marka+özellik`: 8,128 (16.2%)
  - `kategori/genel`: 2,020 (4.0%)
  - `tek_kelime`: 874 (1.7%)
  - `kod/numara`: 455 (0.9%)
  - `cinsiyet+kategori`: 113 (0.2%)
  - `renk+kategori`: 34 (0.1%)

**Örnek Queryler (tip bazında):**
  `marka`: phantom krampon | barfix çubuğu | hoover kurutma makinesi
  `marka+özellik`: anne yani park beşik | erkek kısa kaşe kaban | araç içi şarjlı süpürge
  `kategori/genel`: olta kılıfı | monitör ışığı | köpek evi
  `tek_kelime`: tiftik | tütü | panjur
  `kod/numara`: 250sr | macbook m2 | ipad 5
  `cinsiyet+kategori`: kız çocuk kışlık şapka | kız çocuk yüzük | kız çocuk parfüm

## B. Ürün Katalogu Analizi

**Top 15 Ana Kategori:**
  - ev & mobilya: 251,121 ürün (26.1%)
  - giyim: 169,064 ürün (17.6%)
  - aksesuar: 98,686 ürün (10.2%)
  - ayakkabı: 85,418 ürün (8.9%)
  - elektronik: 80,181 ürün (8.3%)
  - kozmetik & kişisel bakım: 51,633 ürün (5.4%)
  - süpermarket: 47,551 ürün (4.9%)
  - otomobil & motosiklet: 35,915 ürün (3.7%)
  - anne & bebek & çocuk: 23,745 ürün (2.5%)
  - banyo yapı & hırdavat: 23,595 ürün (2.5%)
  - hobi & eğlence: 23,440 ürün (2.4%)
  - kırtasiye & ofis malzemeleri: 22,296 ürün (2.3%)
  - kitap: 18,503 ürün (1.9%)
  - spor & outdoor: 16,196 ürün (1.7%)
  - bahçe & elektrikli el aletleri: 15,529 ürün (1.6%)

**Cinsiyet Dağılımı:**
  - unknown: 590,714 (61.3%)
  - kadın: 192,045 (19.9%)
  - erkek: 99,433 (10.3%)
  - unisex: 80,681 (8.4%)

**Marka sayısı:** 962,870 ürünün markası var (79,789 unique marka)
**Attribute dolu:** 943,848 ürün (98.0%)

## C. Pozitif/Negatif Örüntü Analizi

**Query başına pozitif:** ort=13.9, med=7, p25=3, p75=15, max=1525
**Test setinde query başına aday:** ort=104.4, med=100, min=100, max=3680

**Pozitif çiftlerde özellik değerleri:**
  - Jaccard(query,title): 0.154 (ort/oran)
  - Jaccard(query,kategori): 0.165 (ort/oran)
  - Marka query'de: 0.295 (ort/oran)
  - Cinsiyet query'de: 0.117 (ort/oran)
  - Query, title'da birebir: 0.191 (ort/oran)

## D. Feature Discriminability Testi (her özellik tek başına ne kadar ayırt edici?)


**Her özelliğin pozitif/negatif ayrımındaki AUC değeri:**
  - `fuzz_partial`: AUC=0.8927 ███████
  - `fuzz_set`: AUC=0.8841 ███████
  - `q_cov_title`: AUC=0.8777 ███████
  - `jac`: AUC=0.8762 ███████
  - `t_cov_query`: AUC=0.8754 ███████
  - `cat_overlap`: AUC=0.8536 ███████
  - `brand_in_q`: AUC=0.6306 ██
  - `exact_in_t`: AUC=0.5941 █
  - `attr_match`: AUC=0.5688 █
  - `gender_in_q`: AUC=0.5549 █
  - `age_in_q`: AUC=0.5172 
  - `len_diff`: AUC=0.5047 
  - `l1_match`: AUC=0.5007 

## E. Error Analysis: BERT vs LightGBM Farkı

**Model anlaşmazlık analizi (3,359,679 test çifti):**
  - Her ikisi POZİTİF: 1,291,071 (38.4%) ← yüksek güven
  - Her ikisi NEGATİF: 1,815,956 (54.1%) ← yüksek güven
  - Sadece BERT POZİTİF: 186,732 (5.6%) ← BERT eklıyor
  - Sadece LightGBM POZİTİF: 65,920 (2.0%) ← BERT kaçırıyor

**BERT'in eklediği çiftler — hangi kategoride?**
  - ev & mobilya: 29,594
  - giyim: 29,509
  - ayakkabı: 24,986
  - kozmetik & kişisel bakım: 21,860
  - aksesuar: 15,482
  - elektronik: 15,021
  - anne & bebek & çocuk: 13,608
  - süpermarket: 8,072
  - otomobil & motosiklet: 7,602
  - kitap: 5,537

**LightGBM'in eklediği (BERT kaçırdığı) — hangi kategoride?**
  - ev & mobilya: 17,652
  - giyim: 10,607
  - elektronik: 7,974
  - süpermarket: 5,421
  - otomobil & motosiklet: 3,981
  - kozmetik & kişisel bakım: 3,768
  - aksesuar: 3,650
  - banyo yapı & hırdavat: 1,924
  - bahçe & elektrikli el aletleri: 1,900
  - hobi & eğlence: 1,866

**BERT gender hatası:** 12,063 çift (sadece BERT pozitif AMA cinsiyet uyuşmuyor)
**Bunlar toplam BERT-only içinde:** %6.5

## F. Türkçe Odaklı Yeni Feature Engineering Önerileri


**Türkçe Stem Jaccard (örnekler):**
  - `kadın spor ayakkabı` vs `kadın koşu ayakkabıları`: normal=0.200, stem=0.200 (fark: +0.000)
  - `erkek gömlek` vs `erkek uzun kollu gömleği`: normal=0.200, stem=0.200 (fark: +0.000)
  - `çocuk oyuncağı` vs `çocuklar için oyuncaklar`: normal=0.000, stem=0.250 (fark: +0.250)
  - `deri çanta` vs `hakiki deri el çantaları`: normal=0.200, stem=0.200 (fark: +0.000)

**Structured Attribute Extraction:**
  - Query'de renk var: %4.8
  - Her ikisinde renk var (eşleşme): %2.8
  - Query'de renk var ama title'da yok: %2.0

**Query Intent Sınıflandırması (training verisi bazında):**
  - `marka`: 197,045 pozitif (78.8%)
  - `marka+özellik`: 35,235 pozitif (14.1%)
  - `kategori/genel`: 9,123 pozitif (3.6%)
  - `tek_kelime`: 6,689 pozitif (2.7%)
  - `cinsiyet+kategori`: 1,109 pozitif (0.4%)
  - `kod/numara`: 667 pozitif (0.3%)
  - `renk+kategori`: 132 pozitif (0.1%)

## G. Önerilen Yeni Özellikler ve Beklenen Kazanım

| Özellik | Açıklama | Etki Tahmini | Beklenen F1 Katkısı |
|---|---|---|---|
| `stem_jaccard` | Türkçe suffix'lerini atarak kelime kökü benzerliği | Yüksek | +0.01-0.02 |
| `renk_eslesme` | Query ve title'daki renk adlarının eşleşmesi (0/1/-1) | Orta | +0.005-0.01 |
| `beden_eslesme` | Beden bilgisi eşleşmesi (xs/s/m...) | Düşük | +0.002-0.005 |
| `marka_tip_match` | Query markası = ürün markası (exact string match) | Yüksek | +0.01-0.02 |
| `query_l1_cat_prior` | Query tipi için en yaygın L1 kategori = ürün L1? | Orta | +0.005-0.01 |
| `sayi_kod_match` | Numara/kod içeren query → ürün kodu eşleşmesi | Orta | +0.005-0.01 |
| `attr_renk_eslesme` | Ürün attributes JSON'unda query rengi var mı? | Orta | +0.005 |
| `query_brand_l1_prior` | Bu marka hangi kategoride yoğun? | Yüksek | +0.01 |
| `gender_cross` | Cinsiyet çapraz: -1 uyuşmuyor, 0 belirsiz, 1 uyuşuyor | Çok Yüksek | +0.02-0.03 |
| `title_q_coverage` | Query tokenlarının title'da kaçı var (oran)? | Yüksek | +0.01-0.02 |
| `lgbm_rank_in_query` | LightGBM skorunun query içindeki sıralaması | Yüksek | +0.01-0.02 |
| `bert_score` | Turkish BERT cross-encoder skoru (v16) | Yüksek | +0.02 |

## H. Kritik Bulgular ve Aksiyon Planı


### Öğrendiklerimiz

1. **Test seti Trendyol pre-filtered** — 104 aday/query, büyük çoğunluğu alakalı. Pozitif oran ~%44.
   - Top-K=14 ile 0.49 aldık → gerçek oran 14% değil, ~44%.

2. **LightGBM tavanı ~0.68-0.70** — Surface feature'larla ulaşılabilen maksimum.

3. **Turkish BERT 0.70** — LightGBM üzerine +0.02 kazandı. Semantik anlama değer katıyor.
   - AMA: Gender hatası yapıyor (kadın query → erkek ürün), çünkü training'de explicit gender feature yok.

4. **En ayırt edici feature:** `fuzz_partial` (AUC ~0.85), `tfidf_cos`, `len_diff`.

5. **En eksik olan:** Türkçe morfoloji (ayakkabı/ayakkabıları), brand intent, renk/beden eşleşmesi.

### Aksiyon Planı (0.70 → 0.80+)

**Kısa vadeli (bugün):**
- BERT skorunu LightGBM'e özellik olarak ekle (tek en güçlü feature)
- Gender cross feature (kesin -1/0/1)
- Stem jaccard

**Orta vadeli (yarın-2 gün):**
- BERT'i düzgün eğit: within-query negatives (her query'nin 104 adayından seç)
- Gender-aware BERT training

**Uzun vadeli (3-7 gün):**
- mDeBERTa-v3 (daha güçlü, 280M param)
- LambdaRank / ListNet (within-query ranking objective)
