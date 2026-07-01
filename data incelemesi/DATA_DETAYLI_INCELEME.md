# Data Detaylı İncelemesi — Trendyol 2026 Kaggle
*Son güncelleme: 2026-07-01 | Mevcut en iyi: 0.71 (leak-free) | Hedef: 0.90*

---

## 0. Submission Geçmişi ve Öğrenilenler

| Versiyon | Yöntem | Kaggle F1 | Durum | Not |
|---|---|---|---|---|
| v2 (0.68) | LightGBM + TF-IDF | **0.68** | LEAK VAR | submission_pairs'tan negatif |
| v15 | LightGBM Top-K=14 | **0.49** | HATALI HİPOTEZ | %14 değil %44 pozitif |
| 0.71 leak-free | LightGBM (item katalog neg) | **0.71** | TEMİZ | Sadece train terms + item catalog |
| v16b | Turkish BERT threshold=0.89 | **0.70** | LEAK VAR | submission_pairs neg + pseudo-label |
| v20 | LightGBM + BERT + typed feat | OOF=0.969 | HENÜZ SUBMIT YOK | Leak düzeltilmedi |
| **v21** | Leak-free + BERT-attr + LambdaRank | TBD | HEDEF | Bu versiyonda tümü birleşiyor |

---

## 1. KRİTİK: Data Leak Tespiti ve Düzeltme

### Leak Neydi?

0.68 submission ve BERT v16 dahil tüm eski versiyonlar negatif örnekleri
`submission_pairs.csv`'dan çekiyordu:

```python
# YANLIŞ (leaked):
pool = sub_pairs.merge(terms, on="term_id").merge(items, on="item_id")
neg_pool = pool[mask_g | pool["mask_z"]][["term_id","item_id"]]
```

**Neden bu leak?**
- `submission_pairs.csv` Trendyol'un test seti → modelin hiç görmemesi gerekiyor
- Bu adaylar Trendyol tarafından pre-filter edilmiş → HEPSI potansiyel pozitif
- Bunları negatif olarak kullanmak modele "alakalı-görünen item → negatif" öğretiyor
- Bu hem leak hem de yanlış sinyal

### Leak Fix (0.71 kodu)

```
leak_fix: "Negatives are generated only from training term_ids plus the item catalog;
           submission_pairs.csv is read only during inference."
```

**Negatif kaynak mix (500K negatif):**
```
same_main_category   : 244,806  (%49)
different_main_category: 200,920  (%40)
gender_conflict      :  40,096  (%8)
age_conflict         :  14,178  (%3)
```

**Sonuç:** 0.68 → **0.71** (+%4.4 iyileşme) sadece leak düzeltilerek.

---

## 2. Problem ve Genel Tablo

| Bilgi | Değer |
|---|---|
| Problem | Binary relevance (0/1), macro F1 |
| Test seti | 3,359,679 çift |
| Unique test query | ~32,185 |
| Query başına aday | ~104 (Trendyol pre-filtered) |
| Gerçek pozitif oranı | **~%44** |
| 0.71 submission pozitif oranı | %38.1 (biraz düşük tahmin) |
| Hedef | 0.90 |

**Kritik öğrenme**: Test seti pre-filtered → her 100 adayın ~44'ü pozitif.
v15'te top-K=14 yaptık (0.49 aldık) — yanlış hipotez.

---

## 3. Query Anlama — 5 Katmanlı Analiz

### L1: Marka Tespiti

**50,153 unique query üzerinde:**

| Yöntem | Query Sayısı | % |
|---|---|---|
| Exact match | 31,805 | %63.4 |
| Token match | 6,320 | %12.6 |
| Tespit edilemeyen | 12,028 | %24.0 |

**Marka pozisyonu:** %84'ü queryin başında (0. pozisyon)
**Marka uzunluğu:** %87.6 tek kelime, %10.4 iki kelime

#### Kritik Hatalar (Parser)

| Hatalı Tespit | Sorun |
|---|---|
| `kadın topuklu ayakkabı` → marka=[kadın] | "kadın" brand alanında görünüyor |
| `bebek battaniyesi` → marka=[bebek] | "bebek" brand alanında var |
| `iphone 15 pro` → marka=[pro] | Apple brand=Apple ama "iphone" title'da |
| `mavi kot pantolon` → marka=[mavi] | Hem renk hem gerçek marka |
| `us polo tişört` → marka=[polo] | "u.s. polo assn." normalize edilmiyor |

**Çözüm (v21):** `brand_tok_ovlp` feature → noktalama kaldır, token overlap oranı

---

### L2: Demografik Katman

| Kategori | % |
|---|---|
| Cinsiyet belirsiz | **%90.5** |
| Kadın | %4.1 |
| Erkek | %3.8 |
| Kız/Erkek çocuk | %1.7 |

BERT v16'nın gender hatası: **%6.5** (sadece BERT pozitif çiftlerin %6.5'i cinsiyet uyuşmazlığı)
→ BERT cinsiyet sinyalini öğrenemedi (çünkü gender label training'de eksikti)

---

### L3: Ürün Tipi

| Tip | % |
|---|---|
| Kategori | %32.5 |
| Soyut | %29.8 |
| Spesifik | %25.1 |
| Belirsiz | %12.6 |

**Query Tip Matrisi (top 4):**
```
MARKA + ÜRÜN              : 25.3%  ← ana segment
marka-yok genel           : 23.4%
MARKA + ÜRÜN (spesifik)  : 17.8%
MARKA only                : 16.5%
```

---

### L4: Nitelik Katmanı

| Nitelik | % | Önemi |
|---|---|---|
| Renk | %5.2 | Yüksek — renk varsa çok belirleyici |
| Stil | %2.1 | Orta |
| Malzeme | %1.6 | Orta |
| Beden | %0.2 | Düşük |

---

### L5: Türkçe Morfoloji

| Özellik | Değer |
|---|---|
| Suffix içeren token oranı | **%43** |
| Bileşik kelime içeren query | %3.9 |

**Stem Jaccard farkı:**
```
"çocuk oyuncağı" vs "çocuklar için oyuncaklar"
  Normal:  0.000 → Stem: 0.250  (+0.250!)
```

---

## 4. Ürün (items.csv) Analizi

### KRİTİK: Attribute Yapısı JSON DEĞİL

```
"renk: siyah, color detail: antrasit, menşei: tr, materyal: pamuklu"
```

**Doğru parser (v20'de düzeltildi):**
```python
def parse_attrs(s):
    if not s or s in ("unknown", ""): return {}
    d = {}
    for part in s.split(","):
        if ":" in part:
            k, _, v = part.partition(":")
            d[k.strip()] = v.strip()
    return d
```

**v19/0.68 hatası:** `json.loads()` deneniyor → exception → ham string jaccard'a düşüyor
→ `attr_match` AUC = 0.5688 (neredeyse random)

### Top Attribute Key'ler

| Key | Kapsam % | En Sık Değerler |
|---|---|---|
| renk | **%77.4** | siyah, beyaz, çok renkli |
| color detail | **%76.1** | siyah, beyaz, gri |
| menşei | %72.4 | tr, cn, vn |
| materyal | %58.2 | pamuklu, polyester |
| materyal bileşeni | %22.1 | 100% pamuk, 100% polyester |
| kol boyu | %12.7 | uzun, kısa, kolsuz |

---

## 5. Renk Analizi

### İki Katmanlı Renk Sistemi

| Alan | Unique | Açıklama |
|---|---|---|
| `renk` | **164** | Ana renk (normalize edilebilir) |
| `color detail` | **12,611** | Spesifik ton (çok dağınık) |

### Renk Korelasyon Sonuçları (9,249 pozitif çiftte renk var)

- **%72.5** eşleşiyor → güçlü sinyal
- **%27.5** eşleşmiyor → bunlar neden pozitif?

**Eşleşmeyen örnekler:**
```
gold çay kaşığı    → item renk: altın   (İngilizce=Türkçe sorunu)
mavi jeans erkek   → item renk: siyah   (kullanıcı rengi değil ürünü arıyor)
krem fon perde     → item renk: boş     (item renk alanı doldurulmamış)
```

### Renk Aileleri

```
Gri ailesi:  gri → antrasit → metalik gri → koyu gri → platin → gri melanj
Mavi ailesi: mavi → lacivert → indigo → saks mavi → petrol → bebe mavisi
```

### Renk Synonym Sorunları

| Query | Item Attribute | Sorun |
|---|---|---|
| gold | altın | İngilizce → Türkçe |
| silver | gümüş | İngilizce → Türkçe |
| krem | bej | Eşanlamlı |
| mavi | lacivert | Aynı aile, farklı ton |

---

## 6. Malzeme Analizi

**Standardizasyon sorunu:**
```
100% pamuk   : 2,766
%100 pamuk   :   594  ← aynı anlam
pamuk        :   375  ←
```

**Query → attribute key eşleşmesi (v21 için):**
```
pamuklu, pamuk → materyal bileşeni içinde "pamuk"
deri, hakiki   → materyal bileşeni içinde "deri"
polyester      → "polyester"
```

---

## 7. Feature AUC Analizi

| Feature | AUC | Durum |
|---|---|---|
| `fuzz_partial` | **0.8927** | En iyi |
| `fuzz_set` | 0.8841 | İyi |
| `q_cov_title` | 0.8777 | İyi |
| `jaccard` | 0.8762 | İyi |
| `cat_overlap` | 0.8536 | İyi |
| `brand_in_q` | 0.6306 | **İyileştirme gerekiyor** |
| `attr_match` | 0.5688 | **BUGLU** (JSON parse → fix ile artacak) |
| `len_diff` | 0.5047 | **KALDIRILD** (gürültü) |

---

## 8. BERT vs LightGBM Karşılaştırması

| Durum | Çift Sayısı | % |
|---|---|---|
| Her ikisi POZİTİF | 1,291,071 | **38.4%** — güven bölgesi |
| Her ikisi NEGATİF | 1,815,956 | **54.1%** — güven bölgesi |
| Sadece BERT POZİTİF | 186,732 | 5.6% — semantik değer |
| Sadece LGBM POZİTİF | 65,920 | 2.0% — surface değer |

**BERT gender hatası:** %6.5 (attribute-aware eğitimle düzeltilebilir)

---

## 9. LightGBM vs Alternatifler — Doğru Araç Seçimi

### LightGBM bu problemde YANLIŞ MI?

Hayır, **LightGBM doğru araç** — ama kullanım şekli önemli:

```
Binary Classification (mevcut) → sadece "bu çift alakalı mı?"
LambdaRank (v21)              → "bu 104 aday içinde hangisi daha yukarda?"
```

**Neden LambdaRank daha iyi?**

- Test seti: her query için 104 aday, biz sıralamalarını değiştiriyoruz
- Binary: her çifti bağımsız değerlendiriyor
- LambdaRank: aynı query'nin adaylarını birlikte optimize ediyor
- Doğrudan ranking metriki optimize eder (NDCG) → F1'e daha yakın

### Alternatifler ve Değerlendirme

| Yaklaşım | Güç | Zayıflık | v21'de? |
|---|---|---|---|
| Binary LightGBM | Hızlı, stabil | Query-level context yok | HAYIR (LambdaRank'e geçildi) |
| **LambdaRank LightGBM** | Within-query ranking | Grup yapısı gerektirir | **EVET** |
| Turkish BERT (cross-encoder) | Semantik anlama | Yavaş, eğitim maliyetli | EVET (yeniden eğitildi) |
| Bi-encoder | Hızlı inference | Cross-encoder'dan zayıf | Sonraki versiyon |
| mDeBERTa-v3 | BERT'ten güçlü | 280M param, yavaş | Sonraki versiyon |
| ColBERT | Token-level interaction | No Turkish support | Uzak vade |

**Pratik sonuç:** Model değiştirmek yerine BERT eğitimini düzeltmek çok daha fazla kazandırır.

---

## 10. Tespit Edilen Kritik Hatalar

| # | Hata | Etki | v21'de Çözüm |
|---|---|---|---|
| H1 | attr_jac JSON parse | AUC=0.57 (random) | comma-sep parser |
| H2 | submission_pairs'tan negatif | DATA LEAK | item katalog negatifleri |
| H3 | brand detection false positive (kadın, bebek) | Yanlış brand signal | brand_tok_ovlp |
| H4 | brand normalizasyon eksik (us polo≠u.s. polo) | brand_in_q AUC düşük | token overlap ratio |
| H5 | renk synonym eksik (gold≠altın) | renk feature gürültülü | color_typed feature |
| H6 | BERT training easy negatives | Zor vakaları öğrenemiyor | same-category hard neg |
| H7 | BERT input'ta attribute yok | Renk/materyal bilgisi görmüyor | enhanced input v21 |
| H8 | Binary objective (ranking değil) | Within-query context yok | LambdaRank |

---

## 11. V21 Tasarım Planı

### Beklenti: 0.71 → 0.77-0.81

### Pipeline

```
[A] Leak-free negatives (item katalog: same_cat + gender_conflict + age_conflict + diff_cat)
[B] BERT fine-tune
    - Input: "query | title | brand | cat | renk:X | mat:Y"  
    - Negatif: same-category hard negatives (NOT submission_pairs)
    - Epoch: 3, batch: 32, lr: 2e-5
[C] Feature engineering (v20 typed features + BERT v21 scores)
[D] LambdaRank LightGBM 5-fold GroupKFold(term_id)
    - objective="lambdarank"
    - eval_at=[5, 10]
    - group = query group sizes
[E] Threshold optimize + submission
```

### Yeni Feature'lar (v21 = v20 + fix)

| Feature | v19/0.68 | v20 | v21 |
|---|---|---|---|
| attr_jac | JSON (BUGLU) | comma-sep | comma-sep |
| negatif kaynak | submission_pairs (LEAK) | submission_pairs (LEAK) | item katalog |
| color matching | title text | attr typed (-1/0/1/2) | attr typed (-1/0/1/2) |
| brand match | substring | token overlap | token overlap |
| material | - | attr parse | attr parse |
| BERT input | title+brand+cat | title+brand+cat | title+brand+cat+**renk+mat** |
| BERT negatives | zero-overlap/sub_pairs | same (sub_pairs) | **same-category hard neg** |
| objective | binary | binary | **lambdarank** |

---

## 12. Türkçe NLP Zorlukları

1. **Eklemeli dil** — stem_jaccard bu sorunu kısmen çözüyor
2. **Belirsiz tokenlar** — "kadın", "mavi" hem isim/renk hem marka
3. **Yazım tutarsızlığı** — "%100 pamuk" vs "100% pamuk"
4. **İngilizce renk adları** — gold, silver, rose (synonym gerekli)
5. **Türkçe lowercase** — `İ→i, I→ı` (doğru implementasyon önemli)
6. **Kısaltmalar** — "us polo" ≡ "u.s. polo assn." ≡ "us polo assn"

---

*Tüm analizler: `data incelemesi/` klasöründe*
*01_kapsamli_analiz.py → RAPOR.md*
*02_query_parser_analiz.py → parser_output.txt*
*03_attribute_analiz.py → (terminal çıktısı)*
*0.68submission+0.71leakfree/ → leak fix kodu ve metrikleri*
