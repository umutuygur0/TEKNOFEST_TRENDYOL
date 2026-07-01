# Query Intent Analysis

Bu not, leak-free `0.71` civari skor alan pipeline uzerine yapilan veri profilidir. Ana fikir: problemi sadece `query-item binary classification` olarak degil, query icindeki zorunlu/opsiyonel niyet katmanlarini cozup adaylari buna gore siralayan bir arama alaka sistemi gibi ele almak.

## Veri Ozeti

- `terms.csv`: 50,153 unique query.
- Train term sayisi: 17,968.
- Submission term sayisi: 32,185.
- Train ve submission `term_id` kesişimi: 0. Query parser genellemek zorunda.
- `items.csv`: 962,873 item.
- Marka sayisi: 79,789.
- Kategori sayisi: 2,932.
- Query uzunlugu kisa: train ortalama 2.60 token, submission ortalama 2.63 token. Query'lerin cogu 2-3 token.

## Item Metadata Coverage

- Gender:
  - `unknown`: 590,714
  - `kadın`: 192,045
  - `erkek`: 99,433
  - `unisex`: 80,681
- Age group:
  - `unknown`: 572,028
  - `yetişkin`: 280,876
  - `çocuk`: 52,876
  - `genç`: 31,246
  - `bebek`: 18,426
  - `bebek & çocuk`: 7,421

Metadata eksikligi yuksek oldugu icin age/gender sinyallerini hard rule olarak degil, soft feature ve sadece bariz conflict durumlarinda negatif sinyal olarak kullanmak daha guvenli.

## Marka Tespiti

Saf marka sozlugu cok gurultulu. Katalogda `bebek`, `kadın`, `çanta`, `siyah`, `kahve`, `mutfak`, `mont`, `mavi`, `polo` gibi urun/renk/kategori kelimeleri de marka olarak geciyor.

Iki marka kanali daha saglikli:

- `loose_brand`: katalogda marka olarak gecen her ifade.
- `trusted_brand`: jenerik tek kelimeleri eleyen, train pozitiflerinde item brand ile tutarliligi yuksek olan veya guvenilir cok kelimeli markalari tutan sozluk.

Sonuclar:

- Loose brand query-level detection:
  - Train: %63.15
  - Submission: %62.99
  - Train pozitif row brand match: %47.01
- Trusted brand query-level detection:
  - Train: %28.87
  - Submission: %26.92
  - Train pozitif row brand match: %92.93

Trusted brand pozisyonu:

- Submission trusted brand'lerin buyuk kismi prefix: 7,788 / 8,663 term.
- Ama suffix ve middle durumlari da var: `205 55 16 lassa`, `... samsung`, vb. Bu yuzden sadece ilk token kontrolu yetersiz.

Submission adaylari:

- Trusted brand iceren submission row: 883,891.
- Bu row'larda aday item brand match orani: %22.59.
- Trusted brand iceren term'lerin %92.96'sinda en az bir brand-match aday var.

Yorum: Query'de trusted brand yakalandiysa item brand mismatch cok guclu negatif sinyal. Ancak `uyumlu`, `yedek parça`, `aksesuar`, `kılıf`, `şarj aleti` gibi durumlarda item brand farkli olabilir; bu yuzden model feature'i olarak kullanmak hard filter'dan daha iyi.

## Kategori / Urun Kimligi

Kategori phrase tespiti:

- Train term rate: %54.34
- Submission term rate: %53.55
- Train pozitiflerde query category phrase'in item category icinde gecme orani: %86.46
- Submission adaylarinda category phrase'in candidate category icinde gecme orani: %44.37

Yorum: Kategori/urun kimligi ana sinyal. `ayakkabı`, `spor ayakkabı`, `ceket`, `pantolon`, `parfüm`, `süpürge`, `çanta`, `halı`, `gömlek` gibi phrase'ler query core'u cikarmada kullanilmali.

## Renk

Curated renk sozlugu ile:

- Train term color rate: %4.03
- Submission term color rate: %4.07
- Train pozitiflerde query renginin item title/category/attributes icinde gecme orani: %90.99
- Submission adaylarinda candidate text renginin tutma orani: %37.92

Yorum: Renk seyrek ama yakalandiginda cok temiz. `siyah`, `beyaz`, `mavi`, `krem`, `gold`, `gri`, `kahverengi`, `pembe`, `bordo`, `lacivert` iyi feature olur. Renk mismatch ozellikle ayni urun/kategori icinde hard negative uretmek icin cok degerli.

## Gender / Age

Gender query'de seyrek:

- Submission explicit gender term sayisi: yaklasik 1,542 / 32,185.
- Train explicit gender row sayisi: 22,418.
- Male querylerde item gender:
  - `erkek`: 17,049
  - `unisex`: 3,771
  - `kadın`: 45
  - `unknown`: 84
- Female querylerde item gender:
  - `kadın`: 1,234
  - `unisex`: 199
  - `unknown`: 26

Yorum: `erkek` query + `kadın` item bariz negatif; ama `unisex` item'lar relevant olabilir. Bu nedenle conflict feature guclu, match feature soft olmali.

Age:

- Submission age explicit neredeyse tamamen `bebek`: 634 term.
- Train baby querylerde pozitif item age dagilimi:
  - `bebek`: 2,396
  - `bebek & çocuk`: 517
  - `çocuk`: 269
  - `unknown`: 1,200

Yorum: `bebek` / `çocuk` sinyali onemli ama metadata eksikligi yuksek. Hard reject yerine soft match/conflict daha iyi.

## Onerilen Query Parser Katmanlari

1. Turkish normalization:
   - `İ/I` ayrimi dogru cozulmeli.
   - Noktalama, slash, tire, model numaralari korunmali.
   - Brand/model/renk span'leri bulunmadan agresif stemming yapilmamali.

2. Protected span extraction:
   - trusted brand
   - loose/ambiguous brand
   - renk
   - gender
   - age
   - sayi/model/beden: `iphone 13`, `205 55 r16`, `xl`, `6 li`, `50 ml`

3. Category/product phrase extraction:
   - Kategori path component'lerinden longest-match.
   - `spor ayakkabı`, `şarj aleti`, `ekran kartı`, `balık yağı`, `topuklu ayakkabı` gibi multiword phrase'ler tek unit gibi ele alinmali.

4. Residual core query:
   - Protected span'ler cikarildiktan sonra kalan tokenlar item title/category/attributes ile TF-IDF, char-ngram, BM25 veya embedding similarity icin kullanilmali.

5. Query type:
   - brand-only: `bargello`, `elidor`
   - brand + product: `adidas ayakkabı`
   - category-only: `ayakkabı`
   - product-specific: `topuklu ayakkabı`
   - attribute-heavy: `siyah deri ceket`
   - model/compatibility: `iphone 13 kılıf`, `bosch süpürge başlığı`

## Modelleme Sirasi

1. Mevcut LGBM'ye parser feature'lari ekle.
2. Ayni feature set ile CatBoostClassifier ve XGBoost/CatBoost ranker dene.
3. Query group bazli ranker skorunu LGBM skoru ile ensemble et.
4. Transformer cross-encoder'i once sadece reranker/stacking feature olarak dene.
5. En son per-query calibration ve threshold dene.

En dusuk riskli sonraki adim: `trusted_brand`, `brand_match`, `brand_mismatch`, `color_match`, `color_mismatch`, `category_phrase_match`, `gender_conflict`, `age_conflict`, `query_type` feature'larini mevcut leak-free pipeline'a eklemek.
