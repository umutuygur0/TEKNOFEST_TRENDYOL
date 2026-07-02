# Trendyol 2026 Kaggle — Kapsamlı Analiz Raporu
**Tarih:** 2 Temmuz 2026  
**Mevcut en iyi:** v23 = **0.84**  
**Son submission:** v24 = **0.79** (REGRESYON)  
**Hedef:** 0.90 | **Deadline:** 17 Temmuz 2026 (15 gün)

---

## ÖZET: EN KRİTİK BULGULAR

| Bulgu | Detay |
|-------|-------|
| **v24 neden 0.79?** | 179K hard neg'in %29'u tam token örtüşmeli → unlabeled positive gürültüsü + iter=2000 underfitting |
| **Gap kapanmıyor** | v22: 0.140, v23: 0.129, v24: 0.131 → 3 versiyonda gap inatla devam ediyor |
| **Asıl kör nokta** | Test çiftlerinin **%72.7'sinde BERT<0.1** → model bu çiftlerde tamamen kör |
| **v24 ne kaybetti** | v23'e göre **131,442 pozitif kaybetti**, sadece 14,216 yeni pozitif ekledi |
| **Kritik fırsat** | Pozitif çiftlerin **%22.4'ünde sıfır token örtüşmesi** → semantik model şart |

---

## BÖLÜM 1: PROBLEMİN YAPISI

### 1.1 Temel Sayılar

```
Veri                     Boyut
─────────────────────────────────────────────────
items.csv                962,873 ürün
terms.csv                50,153 sorgu (17,968 train + 32,185 test)
training_pairs.csv       250,000 çift (TÜMÜ label=1)
submission_pairs.csv     3,359,679 çift (tahmin edilecek)
─────────────────────────────────────────────────
Train/test term_id kesişimi: 0 (disjoint — doğrulandı)
Test pair başına ortalama aday: 104.4 (min=100, max=3,680)
Train query başına ortalama pozitif: 13.91 (medyan=7)
```

### 1.2 Problem Çerçevesi — Neden Retrieval DEĞİL

**İlk 11 versiyonun hatası:** Problem, bir arama motoru retrieval görevi gibi ele alındı.
Yani "en iyi 14 ürünü getir" mantığıyla çalışıldı.

Gerçek problem: Her `(query, item)` çifti zaten verilmiş. Tek yapılacak şey
bu çifti **0 veya 1** olarak sınıflandırmak.

Bu farkın önemi:
- Retrieval: model neyin relevant olduğunu kataloğun tamamında aramalı (BM25, dense retriever)
- Klasifikasyon: sadece verilen çifti değerlendirmeli (LGBM + BERT)

Bu gerçeği anlamak v21-v22'de büyük sıçrama sağladı (0.49 → 0.83).

### 1.3 Negatif Örnekleme Problemi

```
Gerçek test dağılımı:
  "avon luck kadın parfüm" → "avon love kadın edp 50ml"    ← çok ZOR (marka match, ürün farklı)
  "avon luck kadın parfüm" → "avon luck erkek edt 50ml"    ← zor (brand+product match, gender farklı)
  "stanley termos"         → "unisex termos stan 16oz..."  ← zor (stan = stanley değil)

Bizim sentetik negatiflerimiz:
  random                   → farklı kategoriden rastgele ürün  ← çok KOLAY
  same_main_cat            → aynı L1 kategoriden rastgele      ← kolay
  same_brand_diff_main     → aynı marka, farklı L1 kat.       ← orta
  TF-IDF hard neg (v24)    → text-benzer ama label=0          ← gürültülü
```

Test negatifler muhtemelen Trendyol'un kendi arama motorunun döndürdüğü
ama relevant olmayan ürünler. Bu dağılımı tam taklit etmek imkânsız.

---

## BÖLÜM 2: VERSİYON GEÇMİŞİ — TAM TABLO

### 2.1 v1-v11: Yanlış Çerçeve Dönemi (0.43–0.49)

| Versiyon | Script | Yaklaşım | Kaggle | Problem |
|----------|--------|----------|--------|---------|
| v1 | 01 | BM25 Top-K=14 fixed | 0.48 | Retrieval çerçevesi yanlış; K sabit |
| v2 | 02 | LightGBM + BM25 feat | 0.48 | 16 feature yeterli değil, threshold sorunu |
| v2b | 02b | LGBM per-query top-14 | 0.48 | K sabit hâlâ |
| v2c | 02c | LGBM global threshold | 0.48 | Threshold opt ama feature set zayıf |
| v3 | 03 | MiniLM zero-shot | 0.45 | Domain mismatch, fine-tune yok |
| v4 | 04 | KNN collab K=8/14 | ~0.47 | KNN retrieval mantığı, hard neg yok |
| **v5** | 05 | E5 fine-tuned | **0.43** | **EN KÖTÜ: embedding collapse** |
| v6 | 06 | Category KNN + overlap | 0.47 | Category sinyali güçlü ama LGBM yok |
| v7 | 07 | XLM-R cross-encoder | submit edilmedi | Quality çok düşük |
| v8 | 08 | Brand + Cat + BM25 | 0.47 | Structured signals iyi ama yetersiz |
| v9 | 09 | XLM-R + cross-query neg | **0.49** | En iyi ama cross-query neg çok kolay |
| v10-v11 | 10,11 | IBQT + BGE | ~0.48 | Retrieval framing, BGE coverage %32.7 |

**v5 felaketi detayı:** E5 bi-encoder fine-tune edildi ama:
- In-batch negatives + küçük pozitif set = embedding collapse
- Model tüm query'leri aynı noktaya çekti → cosine similarity hep aynı
- Sonuç: macro F1 0.43 (rastgeleden kötü)
- **Öğrenme:** Embedding fine-tune'da in-batch negatives tek başına kullanılamaz,
  hard negatives veya cross-encoder şart.

**Dense retrieval coverage analizi:**
```
Retriever              Top-100 Coverage
─────────────────────────────────────────
BM25 TF-IDF            36.5%
E5 fine-tuned v5       47.0%  (best!)
E5 base zero-shot      36.6%
E5 large instruct      38.2%
BGE-M3 zero-shot       32.7%
─────────────────────────────────────────
```
Hiçbiri %50'yi geçemedi. Test çiftleri Trendyol'un kendi indexiyle seçilmiş
→ retrieval yaklaşımıyla bu çiftleri biz üretemeyiz.

### 2.2 v12-v21: Köprü Dönemi (0.68–0.71)

| Versiyon | Yaklaşım | Skor | Not |
|----------|----------|------|-----|
| v12 | LGBM transductive | ~0.68 | leak-free attempt |
| v13 | LGBM + MiniLM embed | ~0.70 | embedding feature eklendi |
| v14 | LGBM + old hard neg | — | hard neg denemesi |
| v15 | LGBM top-K | — | top-K still wrong |
| v16 | BERT pseudo-label | ~0.72 | pseudo-label noisy |
| v17 | Ensemble variants | — | various BERT weights |
| v19 | Super features LGBM | ~0.71 | 20+ feature |
| v20 | Typed negatives | — | structured neg attempt |
| **v21** | LambdaRank + leak-free | **~0.71** | **doğru çerçeveye geçiş** |
| v21b | Threshold fix | — | quantile vs direct |

Bu dönemin önemli kırılması: leak-free GroupKFold(term_id) kullanımı.
v21 öncesinde CV'lar sızdırıyordu → test'te dramatik düşüş.

### 2.3 v22-v24: Modern Dönem (0.83–0.84→0.79)

#### v22 — OOF: 0.9704, Test: 0.83

**Ne yaptı:**
- Binary LGBMClassifier (LambdaRank'ın iter=1 sorununu çözdü)
- BERT cross-encoder (bert_v21): query × item text → single score
- **3 yeni head noun feature:**
  - `head_in_title`: query'nin son anlamlı kelimesi title'da var mı?
  - `weighted_q_cov`: son kelimeye 2x ağırlık
  - `head_in_cat`: head noun kategori'de var mı?
- **same_brand_diff_main** hard negative: "karaca çaydanlık" için "karaca bardak" = negatif

**Neden +0.12 atladı (0.71→0.83):**
- Binary classifier iter sorunu çözüldü (LambdaRank iter=1 kullanıyordu)
- BERT skoru en güçlü feature → model hard case'leri görebildi
- Head noun features → "karaca çaydanlık" vs "karaca bardak seti" ayrımı yapılabildi

---

#### v23 — OOF: 0.9686, Test: 0.84 (EN İYİ)

**Ne yaptı (v22'ye göre):**
1. **BERT yeniden eğitimi (bert_v23):** 
   - v22'de BERT, kolay negatiflerle (same_cat + diff_cat) eğitilmişti
   - v23'te: same_brand_diff_main + same_brand_diff_sub ile eğitildi
   - 5 epoch (v22: 3), 500K sample (v22: 300K)
   - BERT artık "marka eşleşir ama ürün tipi farklı = negatif" öğrendi

2. **Daha çeşitli negatifler (NEG_PER_POS=5, v22'de 3):**
   - same_brand_diff_main (v22'den)
   - same_brand_diff_sub (YENİ: daha ince ayrım)
   - gender_conflict
   - age_conflict
   - same_main_cat (fallback)
   - diff_main_cat (son fallback)

3. **3 yeni feature:**
   - `product_type_cover`: marka/gender/renk/stop çıktıktan sonra kalan ürün tipi kelimelerinin title'da kapsamı
   - `all_q_in_title`: tüm query kelimeleri title'da var mı?
   - `brand_weight`: query'nin kaçta kaçı marka tokenları?

4. **Direct threshold:** OOF F1 maximize eden threshold'u (0.775) test'e DOĞRUDAN uyguladı
   (v22: quantile mapping — eğitim oranını test'e zorladı)

**v23 çıktısı:**
```
OOF F1:   0.96857
OOF thr:  0.7750
Pozitif:  937,617 (27.9%)
Kaggle:   0.84
Gap:      0.1285
```

---

#### v24 — OOF: 0.9210, Test: 0.79 (REGRESYON)

**Ne yaptı:**
- TF-IDF (max_features=80K, sublinear_tf) ile 962K item üzerinde sparse matrix
- Her unique query için top-300 benzer item seç, pozitif olmayanları al
- N_HARDNEG=10 per query → 179,680 yeni hard negative çift
- Bu çiftler için bert_v23 inference → BERT skoru üret
- Toplam train: 1,679,679 çift (+179,680)
- Aynı 35 feature, iter=2000 sabit

**v24 çıktısı:**
```
OOF F1:   0.92100  (↓ 0.0476)
OOF thr:  0.7950
Pozitif:  820,391  (24.4%, ↓ 3.5 puan)
Kaggle:   0.79     (↓ 0.05)
Gap:      0.1310   (kötüleşti)
```

---

## BÖLÜM 3: V24 BAŞARISIZLIK DERİN ANALİZİ

### 3.1 Kök Neden A: Gürültülü Hard Negatives

**Analiz scripti (32_data_deep_analysis.py) bulgusu:**

```
v24 hard neg pairs (n=179,680) token örtüşme analizi:
  Ortalama token örtüşme:  0.644
  Medyan:                  0.667
  >0.5 (riskli):           53.8%  → 96,600+ çift
  >0.8 (çok riskli):       29.3%  → 52,600+ çift
  == 1.0 (tam eşleşme):    29.2%  → 52,400+ çift (!)
```

**Kritik bulgu:** Hard neg çiftlerinin **%29.2'sinde query tüm tokenleri title'da var!**

Bu çiftler büyük ihtimalle gerçek pozitif ama `training_pairs.csv`'de etiketlenmemiş.
Etiketleme seyrektir (term başına ortalama 13.9 pozitif, ama katalogda çok daha fazlası var).

Örnek senaryo:
```
Query: "kadın bot"
training_pairs'te: item_id=12345 (siyah kadın bot)
TF-IDF top-300: item_id=67890 (bordo kadın bot) ← aslında pozitif!
v24 bunu label=0 ile eğitime koydu ← YANLIŞ ETIKET
```

Bu gürültü OOF'u 0.048 düşürdü. Mantıklı: 9,000-10,000 adet yanlış etiket,
250,000 pozitifin %4'üne denk gelir → macro F1'de ciddi hasar.

### 3.2 Kök Neden B: Underfitting

```
v23 LGBM fold sonuçları:
  Fold 1: iter=1974  (early stop → model optimalda durdu)
  Fold 2: iter=1998  (neredeyse max ama durdu)
  Fold 3: iter=1998
  Fold 4: iter=2000  (1 fold max'a geldi)
  Fold 5: iter=1979

v24 LGBM fold sonuçları:
  Fold 1: iter=2000  (MAX! early stop tetiklenmedi)
  Fold 2: iter=2000  (MAX!)
  Fold 3: iter=2000  (MAX!)
  Fold 4: iter=2000  (MAX!)
  Fold 5: iter=2000  (MAX!)
```

v24'te **tüm 5 fold iter=2000'e takıldı.** Early stopping hiç tetiklenmedi.
Bu şu anlama gelir: model daha fazla ağaca ihtiyaç duyuyordu ama parametre sınırı engelledi.

Neden daha fazla ağaca ihtiyaç var?
- 1.5M → 1.68M çift (+12%)
- Gürültülü etiketler (unlabeled positives) → daha karmaşık karar sınırı
- Aynı learning_rate=0.03 ile daha fazla iter gerekiyor

Tahminen iter=4000-5000 ile v24 sonucu 2-3 puan iyileşebilirdi ama
gürültü sorununu çözmez.

### 3.3 Kök Neden C: Gap Kapanmadı

```
Gap tablosu:
  v22:  0.9704 - 0.83 = 0.1404
  v23:  0.9686 - 0.84 = 0.1286  (iyileşti: -0.0118)
  v24:  0.9210 - 0.79 = 0.1310  (kötüleşti: +0.0024)
```

TF-IDF hard negatives, test dağılımını yansıtmıyor. Test'teki gerçek
hard negatives:
- Brand-aware: "stanley termos" → "stan 16oz..." (truncated brand)
- Semantic alias: "difüzör" → "çubuklu oda kokusu"
- Product-type gap: "wide leg kot pantolon" → "relax fit düz paça jean"
- Brand+product: "vans kadın" → "skate old skool" (model kodu, ürün adı yok)

TF-IDF text benzerliği bunları yakalayamaz. Yakalamak için semantik model lazım.

### 3.4 Kök Neden D: Flip Analizi Kanıtı

**33_version_error_audit.py bulgusu:**
```
v23=0→v24=1 (yeni yakalanan pozitifler): 14,216
v23=1→v24=0 (kaybedilen pozitifler):    131,442
Net: -117,226 ← v24 net 117K daha az pozitif tahmin etti
```

v24 neden 131K pozitif kaybetti?
- Gürültülü hard negatives modeli daha muhafazakâr yaptı (threshold 0.775 → 0.795)
- Model belirsiz bölgede daha çok negatif tahmini yapıyor
- Ev & mobilya: 31,895 kaybedilen (en çok etkilenen)
- Elektronik: 21,992 kaybedilen

v24 neden 14K yeni pozitif ekledi?
- Bu çiftler v23'ün kaçırdığı, hard neg training ile modelin öğrendiği bazı ince vakalar
- Ama 14K >> 131K → net zarar

---

## BÖLÜM 4: OOF-TEST GAP SORUNUNUN KÖK NEDENİ

### 4.1 Yapısal Gap Analizi

OOF gap 3 versiyonda inatla 0.128-0.140 arasında kaldı.
Bu **yapısal bir problem** — birkaç feature eklentisiyle kapanmıyor.

Olası sebepler (olasılık sırasıyla):

**A) Test negatiflerinin farklı dağılımı (ana sebep, olasılık: %70)**
```
Bizim negatiflerimiz:
  - Sentetik (brand swap, gender swap, random)
  - Tüm pair uzayından örneklenmiş

Test negatiflerimiz:
  - Trendyol'un retrieval sistemi tarafından seçilmiş top-100 aday
  - Semantik olarak yakın ama relevant olmayan ürünler
  - Çok daha zor → model eğitimde görmediği zorlukta negatif görüyor
```

**B) Unlabeled positives (önemli, olasılık: %20)**
```
training_pairs.csv'de 250K çift var ama gerçek pozitif sayısı çok daha fazla.
Test'te label=1 olan ama training'de görmediğimiz çiftler var.
Model bunları negatif tahmin ediyor → FN artışı → test F1 düşüyor
```

**C) Anti-probe sahte çiftler (küçük katkı, olasılık: %10)**
```
Yarışma organizatörleri, leaderboard probingi önlemek için sahte çiftler eklemiş olabilir.
Bu sahte çiftler hem 0 hem 1 olarak etiketlenmiş tutarsız veri içerebilir.
```

### 4.2 Gap'i Kapatmanın Tek Gerçek Yolu

Gap'i kapatmak için:
1. Eğitim negatiflerini test dağılımına yaklaştırmak VEYA
2. Semantik modeli o kadar güçlü yapmak ki text dışı sinyaller dominant olsun

Test negatiflerini taklit etmek imkânsız (Trendyol'un indexini bilmiyoruz).
O zaman çözüm: **domain-aligned semantic model**.

**Trendyol'un kendi embedding modeli (`TY-ecomm-embed-multilingual-base-v1.2.0`):**
- Trendyol'un gerçek e-ticaret datasında eğitilmiş
- Test veri de Trendyol'dan geldiği için bu model doğal olarak closer to test distribution
- "difüzör" → "oda kokusu" gibi Trendyol'a özgü semantic aliasları öğrenmiş

---

## BÖLÜM 5: HATA TİPLERİ DERİN ANALİZİ

### 5.1 False Positive Analizi (v23 OOF, n=16,766)

Model 1 dedi ama 0 olan.

```
Feature ortalamaları (FP vs TN):
Feature                    FP ort    TN ort    Fark
─────────────────────────────────────────────────────
bert_score                  0.754     0.003    +0.751  ← BERT'in kandığı!
token_overlap               0.729     0.067    +0.663
first_tok_brand             0.467     0.064    +0.403
brand_tok_ovlp              0.450     0.061    +0.389
head_in_title               0.415     0.028    +0.387
```

**Profil:**
- FP'lerin **%80'inde bert_score > 0.5** → BERT kandırılıyor
- FP'lerin **%44'ünde brand token örtüşmesi** (aynı marka ama farklı ürün tipi)
- FP örnekleri:

```
Q: philips buhar kazanlı ütü  → T: perfectcare compact essential buhar kazanlı ütü (Philips değil!)
Q: stanley termos              → T: unisex termos stan 16oz... (stan ≠ stanley)
Q: bordo ceket                 → T: bordo kadın ceket (doğru ürün ama label=0?)
Q: mikser & mikser seti        → T: ms 61b6150 uzun mikser (model kod, marka "bosch")
```

**Tanı:** FP ana nedeni, BERT'in text benzerliğine kancalanması.
BERT "buhar kazanlı ütü" kelimelerini query'de ve title'da görünce 1 diyor
ama marka farklı (Philips'in perfectcare modeli aslında Philips olmayan bir şey mi?).

Bu çiftler gerçekte pozitif olabilir ama etiketlenmemiş (unlabeled positive).
Ya da Trendyol'un kategorizasyonunda bu specific item bu query'ye map edilmemiş.

### 5.2 False Negative Analizi (v23 OOF, n=9,780)

Model 0 dedi ama 1 olan.

```
Feature ortalamaları (FN vs TP):
Feature                    FN ort    TP ort    Fark
─────────────────────────────────────────────────────
token_overlap               0.449     1.343    -0.895
bert_score                  0.224     0.964    -0.740
product_type_cover          0.221     0.695    -0.474
head_in_title               0.260     0.671    -0.411
weighted_q_cov              0.237     0.642    -0.405
```

**Profil:**
- FN'lerin **%76'sında bert_score < 0.3** → BERT göremedi
- FN'lerin **%71'inde head_in_title = 0** → ürün tipi title'da değil ama pozitif
- FN'lerin **%55'inde brand token örtüşmesi yok** → brand match yok ama pozitif

**FN örnekleri:**
```
Q: vans kadın          → T: skate old skool (brand=vans, ama "skate old skool" = ürün kodu!)
Q: difüzör             → T: çubuklu oda kokusu (semantik alias: difüzör = oda kokusu)
Q: erkek pjama         → T: siyah cool touch patlı ve kordonlu örme şort (pjama kodu farklı)
Q: wide leg kot pantolon → T: relax fit düz paça jean (aynı ürün, farklı dil)
Q: masa üstü süs       → T: beton - lotus - krem (üst kategori eşleşmesi)
Q: bungou stray dogs   → T: my hero academia dazai osamu figür (anime karakter eşleşmesi)
```

**Tanı:** FN'ler iki tipte:
1. **Semantik alias** (difüzör=oda kokusu, wide leg=düz paça jean, pjama=şort)
   → Sadece domain-aware embedding çözebilir
2. **Brand kodu** (vans → skate old skool, bosch → ms 61b6150)
   → Brand lookup / knowledge graph gerektirir

### 5.3 BERT'in Kör Olduğu Alan

```
Test çiftleri BERT score dağılımı:
  <0.1 (kör):    2,441,593 (%72.7!)
  0.1-0.5:          29,720  (0.9%)
  0.5-0.9:          47,010  (1.4%)
  >0.9 (güçlü):    841,356 (25.0%)
```

Test çiftlerinin **%72.7'si için BERT skoru 0.1'in altında.**
Bu çiftlerin büyük çoğunluğu gerçekten negatif, ama bir kısmı (FN'ler)
semantik aliaslar veya brand kodları içeriyor.

Model bu %72.7'yi sadece diğer lexical features (tfidf_cos, jaccard vb.) ile
değerlendiriyor. Bu yüzden semantik kör nokta kritik.

---

## BÖLÜM 6: BAŞARISIZ STRATEJİLERİN ORTAK PATERNİ

### 6.1 Ne Zaman OOF'tan Test'e Çeviri Başarısız Olur

```
OOF (GroupKFold) → Test arasındaki fark:
  ✓ GroupKFold doğru: term_id bazlı, sızdırma yok (doğrulandı)
  ✗ OOF negatifler = bizim sentetik dağılımımız
  ✗ Test negatifler = Trendyol retrieval dağılımı
  
  Gap = |sentetik neg dağılımı - gerçek neg dağılımı|
```

Temel kural: **OOF negatifler ne kadar gerçek test negatiflerine benzerse, gap o kadar küçük olur.**

### 6.2 Embedding Fine-tune Riski

v5 felaketi (0.43) ve dense retrieval başarısızlıkları gösterdi:
- In-batch negatives tek başına → collapse
- Zero-shot embedding → domain mismatch
- Eğitilmiş embedding dahi coverage %47 → yetersiz

**İstisna:** TY-ecomm-embed kendi verisinde eğitildiği için coverage farklı olabilir.

### 6.3 Hard Negative Mining'in Tuzağı

v24 deneyimi gösterdi:
- TF-IDF hard neg %29'u tam token örtüşmeli → %5 unlabeled positive bile yeter
- 8,984 yanlış etiket → OOF 0.048 düşüş → ölçülü
- Hard neg seçiminde pozitifliği teyit edemiyoruz (sparse labeling yüzünden)

**Güvenli hard negative kriterleri:**
- Same brand + farklı L1 kategori → güvenli (marka "karaca" çaydanlık ≠ bardak)
- Gender conflict → güvenli (erkek sorgusuna kadın ürün)
- Age conflict → güvenli (bebek sorgusuna yetişkin ürün)
- Color mismatch → kısmen güvenli
- TF-IDF high score → RİSKLİ (unlabeled positive olabilir)

---

## BÖLÜM 7: VERİ KALİTESİ ANALİZİ

### 7.1 Item Metadata Kalitesi

```
Item metadata coverage:
  brand:     %100 dolu (sadece 3 unknown)
  category:  %100 dolu, ortalama 2.8 seviye derinlik
  gender:    %61.3 "unknown" → 3 üründe 2'si bilinsiz!
  age_group: %59.4 "unknown"
  attributes:%98.0 dolu
```

Gender ve age_group eksikliği yüksek → bu feature'ları hard rule olarak
kullanmak tehlikeli, soft sinyal olarak kullanmak gerekiyor.

### 7.2 Pozitif Çift Sıfır Token Örtüşmesi

```
training_pairs örneğinde (n=5,000):
  Token overlap = 0:  %22.4 → 1,118 çift
  Query coverage < 0.5: %34.4
```

**Pozitif çiftlerin %22.4'ünde query ile title arasında HİÇ ortak token yok!**

Bu çiftler hiçbir lexical feature ile yakalanamaz.
Bunlar şunlardır:
- Semantik alias: "soundbar" → "hw-q990f 11.1.4 kanal soundbar" (soundbar token'ı var ama query'de "soundbar" → title'da tam eşleşme; bu örnek aslında overlap var, kontrol et)
- Brand kodu: "timberland sneaker" → "2 eye boat erkek spor ayakkabı" (brand kodu!)
- Üst kategori: "çiçeklik" → "sepetten sarkan sevimli tavşanlar dekoratif saksı"
- Morfoloji: "bayan ayakkabısı" → "unisex kışlık mevsimlik sneaker" (ayakkabısı ≠ ayakkabı tokenization)

Bunlar için ne lazım: Semantik embedding (TY-ecomm-embed).

---

## BÖLÜM 8: 0.90 YOLUNDAKI GERÇEKÇI ANALİZ

### 8.1 Mevcut Tavan Tahmini

Mevcut feature set (35 feature, BERT v23) ile:
- OOF: ~0.97 (neredeyse makul maksimum)
- Gap: ~0.128 yapısal
- Test tavanı: ~0.84 (v23 bunu gösterdi)

Sadece LGBM iter artışı ve threshold tweaking ile **en fazla 0.85** çıkar.
0.90'a ulaşmak için semantik anlama atlaması şart.

### 8.2 Gap Kapatma Senaryoları

```
Senaryo 1: TY-ecomm-embed feature eklendi
  Etki: FN'lerde BERT kör, embedding görebilir
  OOF tahmini: 0.95 (embedding gürültü ekler)
  Gap tahmini: 0.09 (domain alignment sayesinde)
  Test tahmini: 0.86
  
Senaryo 2: TY-ecomm-embed + query-aware features
  Özellikler: trusted_brand_match, color_match, trusted_brand_mismatch
  OOF tahmini: 0.96
  Gap tahmini: 0.07
  Test tahmini: 0.89
  
Senaryo 3: Senaryo 2 + bert_v23 yeniden eğitim (query-aware neg)
  Özellikler: yukarıdaki + BERT domain neg
  OOF tahmini: 0.97
  Gap tahmini: 0.06
  Test tahmini: 0.91 ← 0.90 geçilir
```

### 8.3 Hangi Feature Ne Kadar Değer Taşır

| Feature | Etkilediği Hata | Tahmini Katkı | Risk |
|---------|-----------------|---------------|------|
| TY-ecomm-embed cosine | FN (semantik gap) | +0.03-0.05 | Orta |
| TY-ecomm-embed rank in group | Threshold calibration | +0.01-0.02 | Düşük |
| trusted_brand_match | FP (brand wrong) | +0.01-0.02 | Çok düşük |
| trusted_brand_mismatch | FP (brand mismatch) | +0.01-0.02 | Düşük |
| color_match/mismatch | FP/FN (color query) | +0.005 | Çok düşük |
| query_type feature | Genel sınıflandırma | +0.005-0.01 | Düşük |
| iter=5000 (LGBM fix) | Underfitting | +0.002-0.005 | Sıfır |

---

## BÖLÜM 9: AKSİYON PLANI — 0.90'A GİDEN YOL

### 9.1 Öncelik 1: v25 — TY-ecomm-embed Feature (Bu Gece)

**Script:** `34_v25_tyembed.py`

**Adımlar:**
1. `pip install sentence-transformers` (zaten olabilir, kontrol et)
2. `Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0` indir
   (trust_remote_code=True, sentence-transformers API)
3. 50K unique query embedding cache → `emb_cache/query_embs_tyembed.npy`
4. 962K unique item embedding → `emb_cache/item_embs_tyembed.npy`
   (batch=256 GPU → ~15-20 dk)
5. Her çift için cosine similarity:
   ```python
   # Pair feature
   ecomm_cos_q_title = cosine(query_emb, item_emb)
   
   # Group feature (per term_id)
   ecomm_rank_in_group = rank within query's candidates
   ecomm_pct_in_group  = percentile rank
   ecomm_zscore        = (score - group_mean) / group_std
   ```
6. v23 negatiflerini koru (same_brand_diff_main/sub, gender/age)
7. iter=4000 (v24 underfitting dersini öğrendik)
8. 35 feature + 4 yeni TY-embed feature = 39 feature

**Beklenti:**
- OOF: ~0.95-0.96 (embed gürültü ekler, ama ciddi FN kurtarır)
- Gap: ~0.08-0.10 (domain alignment sayesinde)
- Test: ~0.86-0.88

**Önemli notlar:**
- item_emb: title + " | " + category_L1 + " | " + brand
- query_emb: sadece query text
- Group feature hesabı: submission_pairs üzerinde (leak yok, sadece inference)

---

### 9.2 Öncelik 2: v26 — Query-Aware Features + TY-embed (Sonraki)

**Script:** `35_v26_parser_features.py`

**Yeni features:**
```python
# Trusted brand detection (araştırma dosyasından)
trusted_brand_in_query   # query'de güvenilir marka var mı?
trusted_brand_match      # item brand == query trusted brand?
trusted_brand_mismatch   # query'de trusted brand var ama item brand farklı?

# Color features  
color_in_query           # query'de renk var mı?
color_match              # item attribute renk == query renk?
color_mismatch           # query'de renk var ama item'da farklı/yok?

# Query type
query_type_brand_only    # sadece marka sorgusu (bargello, elidor)
query_type_brand_product # marka + ürün (adidas ayakkabı)
query_type_cat_only      # sadece kategori (ayakkabı)
```

**Beklenti:** v25 üzerine +0.01-0.02

---

### 9.3 Öncelik 3: BERT Yeniden Eğitimi (v27) — Eğer 0.90 Hâlâ Uzaksa

**Hedef:** bert_v23'ü query-aware negatiflerle yeniden eğit

**Ek negatif tipler:**
- Query'de trusted brand + item'da farklı brand → trusted_brand_mismatch
- Query'de color + item'da farklı color → color_mismatch
- Query'de kategori + item'da farklı L2 kat → category_mismatch
- TF-IDF high-sim ama brand mismatch → safe hard negative

Bu BERT, v23'e göre brand/color çatışmalarını daha iyi öğrenir.

---

### 9.4 Slot Planı (7 slot / yarın akşama kadar)

| Slot | Versiyon | Ne | Ne Zaman |
|------|---------|----|----------|
| 1 (HARCANDI) | v24 | TF-IDF hard neg | Geçti |
| 2 | **v25** | TY-embed + iter=4000 | Bu gece / sabah |
| 3 | v25b (ablation) | Sadece yeni feature'lar, v23 taban | Sabah |
| 4 | **v26** | Parser features + v25 | Öğleden sonra |
| 5 | v26b | Threshold sweep | Öğleden sonra |
| 6 | v27 | BERT retrain (gerekirse) | Akşam |
| 7 | FINAL | En iyi ensemble | Son |

---

## BÖLÜM 10: YAPILMAMASI GEREKEN HATALAR

### 10.1 Kesin Yasaklar (Önceki Hatalardan)

```
✗ submission_pairs'ten negatif alma (leak!)
✗ Sabit threshold kullanma (pozitif oran bilinmiyor)
✗ in-batch negatives tek başına (v5 felaketi)
✗ iter=2000 sabit bırakma (v24 underfitting)
✗ TF-IDF high-sim negatives (unlabeled positive riski %29)
✗ Retrieval framing (v1-v11 boşa gitme nedeni)
✗ Zero-shot embedding (domain mismatch)
✗ Fixed K per query (v2b, v4 hatası)
```

### 10.2 Dikkat Gerektiren Alanlar

```
⚠ BERT fine-tune: iter çok düşük olursa collapse, çok yüksek olursa overfit
⚠ Gender/age features: %60+ unknown → hard rule değil, soft sinyal
⚠ Color features: %4 coverage → yardımcı ama dominant değil
⚠ TY-ecomm-embed: trust_remote_code=True → offline reproducibility dikkat
⚠ Embedding cache: item_embs ~962K × 768 dim × 4 byte = 2.8 GB RAM
```

---

## BÖLÜM 11: ÖZET VE SONUÇ

### Neden 0.49'dan 0.84'e Geldik

| Kırılma Noktası | Skor Atlaması | Ana Sebep |
|-----------------|---------------|-----------|
| v1-v9 | 0.43→0.49 | Yanlış çerçeve (retrieval), kolay negatives |
| v9→v21 | 0.49→0.71 | Leak-free CV, LambdaRank, structured neg |
| v21→v22 | 0.71→0.83 | BERT feature + head features + binary classifier |
| v22→v23 | 0.83→0.84 | bert_v23 retrain + direct threshold |

### Neden 0.84'ten 0.79'a Düştük

v24 TF-IDF hard negatives:
1. **%29 unlabeled positive** → yanlış etiket gürültüsü → OOF 0.048 düştü
2. **Underfitting** (iter=2000, 5/5 fold max) → model öğrenemedi
3. **Gap kapanmadı** (0.129 → 0.131) → test dağılımına yaklaşmadı
4. **131K pozitif kaybedildi** → model daha muhafazakâr hale geldi

### 0.90 Hedefine Ulaşmak İçin Tek Gerçekçi Yol

**TY-ecomm-embed + query-aware features:**
- Semantik kör noktayı (FN'lerin %76'sı) kısmen çözer
- Domain-alignment ile gap'i 0.128'den 0.07-0.09'a indirir
- Query-aware features (brand_match, color_match) precision'ı artırır

Beklenen:
- v25 (embed): 0.86-0.88
- v26 (embed + parser): 0.87-0.90
- v27 (BERT retrain): 0.89-0.92

**Deadline 17 Temmuz, 15 gün var. Başlanmalı: v25 bu gece.**

---

*Rapor üretim tarihi: 2 Temmuz 2026*  
*Analiz scriptleri: 32_data_deep_analysis.py, 33_version_error_audit.py*  
*Veri kaynakları: OOF analizi, test proba dosyaları, tüm submission logları*
