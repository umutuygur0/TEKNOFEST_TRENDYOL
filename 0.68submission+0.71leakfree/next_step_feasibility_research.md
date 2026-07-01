# Next Step Feasibility Research

Bu notun amaci onceki fikirlerin hangisinin gercekten skor getirme potansiyeli oldugunu ayirmak. Yorumlar hem lokal veri audit'ine hem de resmi dokuman/model kaynaklarina dayaniyor.

## Yarismaya Gore Problem Yapisi

Yarisma binary relevance olarak degerlendiriliyor ve metrik macro F1. Train verisi yalnizca pozitiflerden olusuyor; negatifleri biz uretiyoruz. Submission ise her `(term_id, item_id)` ciftine `0/1` bekliyor.

Onemli pratik sonuc:

- Ranker kullanmak mantikli olabilir, cunku submission'da her query icin cok sayida aday var.
- Ama nihai metrik ranking degil macro F1 oldugu icin ranker skoru tek basina yetmez; per-row binary threshold veya ranker + classifier ensemble gerekir.
- Train'de real negative olmadigi icin offline validation asiri iyimser olabilir. Public LB ile kontrollu ablation sart.

## Lokal Kanit: Query Facet Feature'lari

Mevcut `submission_leak_free_lgbm.csv` uzerinde audit:

- Toplam submission row: 3,359,679
- Predicted positive: 1,279,637
- Pozitif oran: 38.09%

### Trusted Brand

Train pozitiflerde trusted brand mismatch riski:

- Applicable positive row: 61,137
- Brand mismatch flagged positive row: 2,916
- Tum pozitiflerde false reject riski: 1.17%
- Applicable row icinde risk: 4.77%

Mevcut submission'da:

- Predicted positive olup trusted brand mismatch olan row: 142,451
- Predicted positive icindeki oran: 11.13%
- `brand_match` row'larinda modelin pozitif basma orani: 98.9%
- `brand_mismatch` row'larinda modelin pozitif basma orani: 23.01%

Karar: **Kesinlikle denenmeli.** Hard postprocess bile denenebilir ama en saglisi model feature + query-aware hard negative.

### Color

Train pozitiflerde color missing riski:

- Applicable positive row: 10,102
- Color missing positive row: 257
- Tum pozitiflerde false reject riski: 0.10%
- Applicable row icinde risk: 2.54%

Mevcut submission'da:

- Predicted positive olup color missing olan row: 15,642
- `color_match` row'larinda modelin pozitif basma orani: 66.18%
- `color_missing` row'larinda modelin pozitif basma orani: 17.79%

Karar: **Kesinlikle denenmeli.** Renk az kapsar ama yakalandiginda temiz sinyal.

### Gender / Age

Train pozitiflerde gender conflict ve age conflict false reject audit'inde 0 cikti. Submission'da gender conflict row sayisi az:

- Gender conflict row: 1,654
- Bu row'larda modelin pozitif basma orani: 1.51%
- Predicted positive gender conflict: 25

Karar: **Feature olarak kalsin, hard rule etkisi kucuk.** Skor getirisi sinirli ama precision temizligi saglar.

### Category Phrase

Train pozitiflerde category phrase missing:

- Applicable positive row: 167,189
- Missing positive row: 69,796
- Tum pozitiflerde false reject riski: 27.92%
- Applicable row icinde risk: 41.75%

Mevcut submission'da:

- Predicted positive olup category missing olan row: 400,508
- `cat_match` row'larinda modelin pozitif basma orani: 74.07%
- `cat_missing` row'larinda modelin pozitif basma orani: 30.28%

Karar: **Hard rule olarak kullanma.** Feature olarak cok iyi, hard filter olarak tehlikeli.

## Model Secenekleri

### 1. Parser Feature + LGBM/CatBoost Ensemble

Olma ihtimali: **Yuksek**

Neden:

- Facet sinyalleri lokal olarak guclu ve mevcut modelin acik hatalarini yakaliyor.
- Risk dusuk; mevcut pipeline uzerine feature eklenebilir.
- CatBoost resmi olarak numerical, categorical, text ve embedding feature destekliyor. Text feature'lari tokenizers/dictionaries/feature_calcers ile numeric feature'a ceviriyor.

Risk:

- CatBoost text pipeline'i Turkce morfoloji bilmez; raw text yerine bizim normalize/parser feature'larini vermek daha guvenli.

### 2. Query-Aware Negative Mining

Olma ihtimali: **Cok yuksek**

En mantikli negatifler:

- Query trusted brand + same/near category + wrong brand
- Query color + same product/category + wrong color
- Query gender + opposite gender
- Query baby/child + adult-only item
- Query category phrase + sibling category distractor

Neden:

- Mevcut modelde 142k brand-mismatch positive var; bu tip negatifleri daha cok gormesi lazim.
- Renk false reject riski cok dusuk.

Risk:

- `uyumlu`, `kılıf`, `aksesuar`, `yedek parça`, `başlık`, `şarj aleti` query'lerinde marka mismatch her zaman negatif degil.

### 3. XGBRanker / Ranking Objective

Olma ihtimali: **Orta-yuksek**

Neden:

- Submission yapisi query basina aday listesi gibi. XGBoost resmi dokumani `qid` ile query group verilip `rank:ndcg`, `rank:map`, `rank:pairwise` kullanilabildigini belirtiyor.
- Binary relevance icin `rank:map` dogal aday; `rank:ndcg` daha genel ve default.

Risk:

- Final metrik macro F1, ranking metriği degil.
- Train'de real negative candidate list yok; sentetik negatif listeleri submission dagilimini iyi taklit etmezse ranker offline iyi gorunup LB'de dusuk kalabilir.

Kullanim sekli:

- Tek model degil, `rank_score` feature'i veya ensemble komponenti olarak kullan.
- Query basina score distribution feature'lari ekle: percentile, rank, top-k gap, zscore.

### 4. CatBoostRanker

Olma ihtimali: **Orta**

Neden:

- CatBoostRanker default `YetiRank` ile geliyor ve categorical/text feature destegi var.
- Marka/category gibi kategorik feature'lari iyi kullanabilir.

Risk:

- XGBRanker'a gore daha yavas olabilir.
- Yine sentetik negatif problemi var.

Kullanim sekli:

- Once CatBoostClassifier ile ayni feature setini dene.
- Ranker'i ikinci adimda dene.

### 5. Transformer CrossEncoder / BERTurk

Olma ihtimali: **Orta, ama potansiyel tavan yuksek**

Neden:

- CrossEncoder query ve item text'i beraber isleyip pair skoru uretebilir; reranking problemi icin dogru mimari.
- BERTurk Turkce icin egitilmis cased BERT modeli; Transformers ile kullanilabiliyor.

Risk:

- 3.36M pair inference pahali.
- Train negatifleri sentetik oldugu icin fine-tune overfit edebilir.
- Internet kapali ortamda reproduce icin model agirliklari lokal paketlenmeli.

Kullanim sekli:

- Tum submission'a direkt calistirma.
- Once mevcut modelin belirsiz bandina uygula: ornegin probability 0.35-0.65 veya query basina top adaylar.
- CrossEncoder skorunu LGBM/CatBoost stack feature'i yap.

### 6. Zemberek / Turkish Morphology

Olma ihtimali: **Dusuk-orta**

Neden:

- Zemberek Turkce morphology, tokenization ve normalization sunuyor.
- Turkce ekler icin teorik olarak iyi.

Risk:

- Java tabanli ve repo slow maintenance modunda.
- E-ticaret query'leri marka/model/renk agirlikli. Agresif kok bulma marka/model bilgisini bozabilir.
- Query'ler cok kisa; char ngram + protected span extraction daha pratik.

Kullanim sekli:

- Tum pipeline'i morphology'ye baglama.
- Sadece noisy normalization veya ekli kategori kelimelerini yakalama icin opsiyonel deney.

## Net Oncelik Sirasi

1. Parser feature'larini mevcut LGBM pipeline'a ekle.
2. Query-aware negative mining ekle.
3. Bu feature setiyle CatBoostClassifier dene.
4. XGBRanker'i qid=term_id ile egit; score/rank feature'larini ensemble'a kat.
5. Public LB ablation: base, +parser, +parser+hard negatives, +CatBoost ensemble, +ranker ensemble.
6. Transformer'i sadece belirsiz/top aday reranker olarak dene.

## Kaynaklar

- XGBoost Learning to Rank docs: `rank:ndcg`, `rank:map`, `qid`, LambdaMART.
- CatBoost text features docs: numerical/categorical/text/embedding feature destegi.
- CatBoostRanker docs: ranking estimator, default `YetiRank`.
- SentenceTransformers CrossEncoder docs: iki metni beraber isleyip score/class probability uretir.
- Hugging Face BERTurk model card: Turkish cased BERT, Transformers ile kullanilabilir.
- Zemberek GitHub: Turkish morphology, tokenization, normalization modules.

## Ek Arastirma: BERT-Enhanced + LambdaRank + Trendyol HF Modelleri

Kullanici fikri: BERT/embedding ile query-item anlamsal skoru uretmek, bunu LambdaRank ile query icinde aday siralamaya cevirmek ve mevcut leak-free LGBM ile birlestirmek.

### Trendyol Hugging Face Modelleri

Trendyol'un Hugging Face organizasyonu verified gorunuyor ve 20 model listeliyor. Bu yarismaya en alakali model LLM degil, e-commerce embedding modeli:

#### `Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0`

Model kartina gore:

- Task: Sentence Similarity.
- Model tipi: Sentence Transformer.
- E-commerce semantic search, classification ve retrieval icin fine-tune edilmis.
- Turkish ve multilingual query understanding vurgulanmis.
- Query rephrasing, product tagging, attribute extraction, clustering ve product categorization icin uygun oldugu belirtilmis.
- 384 token input support.
- 768-dimensional dense vector output.
- Cosine similarity ile inference.
- Alibaba-NLP/gte-multilingual-base'in distilled versiyonu uzerine fine-tune edilmis.
- License: Apache-2.0.
- `sentence-transformers` kullanimi oneriliyor ve model `trust_remote_code=True` istiyor.

Bu model bu problem icin cok alakali. Cunku bizim veri tamamen query/product semantic similarity, category/product matching ve attribute matching uzerine kurulu.

Yerel ortam notu:

- `sentence_transformers` kurulu degil.
- `transformers` ve `torch` kurulu.
- Model indirimi ve `trust_remote_code` finalde offline reproduce icin dikkat ister. Kullanacaksak model snapshot'i lokal paketlenmeli ve versiyon/hash sabitlenmeli.

Kullanim fikri:

- Query text embedding.
- Item text embedding: `title + category + brand + gender + age_group + attributes`.
- Feature'lar:
  - `ecomm_embed_cos_query_title`
  - `ecomm_embed_cos_query_item_text`
  - `ecomm_embed_cos_query_category`
  - Query group icinde embedding rank / percentile / zscore
  - Elementwise L1/L2 distance opsiyonel ama 768 dim direkt GBDT'ye vermek agir olabilir.

En pratik baslangic:

- Tum pair icin CrossEncoder yerine embedding cosine hesapla.
- Cosine score'u LGBM/CatBoost feature'i yap.
- Query basina cosine rank'i LambdaRank veya post-calibration icin kullan.

#### `Trendyol/Trendyol-LLM-8B-T1`

Model kartina gore:

- 8B chat model.
- Qwen3-8B uzerine kurulmus.
- Trendyol tarafindan curate edilen large-scale Turkish e-commerce datasets ile egitilmis.
- Turkish reasoning ve English reasoning destekli.
- Instruction following, summarisation/paraphrasing, coding, text/review classification gibi multitask kullanimlar hedeflenmis.
- License: Apache-2.0.

Bu model yarismada dogrudan inference classifier olarak uygun degil:

- 8B oldugu icin 3.36M pair inference pratik degil.
- Generative model oldugu icin deterministic binary scoring zordur.
- Offline reproduce icin agir.

Ama iki yerde kullanilabilir:

- Query parser icin az sayida offline etiketleme/prototip: query type, brand ambiguity, compatibility intent.
- Synthetic instruction/data generation veya rule discovery. Tahmin adiminda kullanmak yerine feature engineering gelistirme araci olarak kullanmak daha mantikli.

#### `Trendyol/Trendyol-LLM-Asure-12B`

Model kartina gore:

- Image-text-to-text / multimodal.
- Gemma3 tabanli, Turkish/English, e-commerce, vision/conversational etiketli.
- 12B oldugu icin daha agir.

Bu yarismada mevcut veri image icermedigi icin dogrudan uygun degil. Sadece gelecekte image dataseti verilirse anlamli.

#### Trendyol image encoders

`Trendyol/e-commerce-product-image-encoder`:

- ConvNeXt-based image embedding model.
- Product unification, visual search, duplicate detection, product similarity ranking icin.
- 512-dim embedding.
- ArcFace loss.

`Trendyol/trendyol-dino-v2-ecommerce-256d`:

- DinoV2 ViT-B/14 + ArcFace.
- 256-dim image embedding.
- E-commerce image similarity/retrieval icin.

Mevcut yarismada image kolonu yok. Bu modeller bu veriyle dogrudan kullanilamaz.

### BERT-Enhanced + LambdaRank + Leak-Free LGBM Gercekten Olur mu?

Karar: **Olur, ama en dogru formu CrossEncoder degil, once bi-encoder embedding + LambdaRank + leak-free LGBM/CatBoost ensemble.**

Neden olur:

- Trendyol embedding modeli domain-aligned: e-commerce semantic search ve Turkish query understanding icin fine-tune edilmis.
- Query'ler kisa, item text'leri zengin. Dense semantic cosine, TF-IDF/char ngram'in kacirdigi synonym/paraphrase durumlarini yakalayabilir.
- LambdaRank query icindeki adaylari siralamayi ogrenir. Submission dogal olarak query-grouped candidate list yapisinda.

Neden tek basina yetmez:

- Kaggle metriği macro F1, ranking metriği degil.
- Ranker score'u query icinde iyi siralama verebilir ama global `0/1` threshold ayri kalibre edilmeli.
- Train'de gercek negatif yok; LambdaRank sentetik negatif dagilimina fazla uyum saglayabilir.

Uygulanabilir tasarim:

1. Leak-free train set:
   - Pozitifler: `training_pairs.csv`.
   - Negatifler: query-aware katalogdan uretilir.
   - Submission pair'leri training label'i olarak kullanilmaz.

2. Embedding feature:
   - `TY-ecomm-embed` ile unique query ve unique item text embedding cache'lenir.
   - Pair feature: cosine similarity.
   - Query group feature: candidate cosine rank, percentile, zscore.

3. LambdaRank:
   - `qid=term_id`.
   - XGBoost icin `rank:ndcg` default iyi baslangic; binary relevance icin `rank:map` da denenir.
   - LightGBM icin `lambdarank`; dokumanda `rank_xendcg` daha hizli ve benzer performansli olarak belirtiliyor.
   - Cikti: `rank_score`, `rank_percentile`, `rank_top_gap`.

4. Final classifier/ensemble:
   - LGBM/CatBoost binary classifier mevcut parser + TF-IDF + embedding + rank features ile egitilir.
   - Ranker tek basina submission uretmez.
   - Final threshold macro F1 icin optimize edilir.

### Leak-Free Birlestirme Meselesi

Burada asil hedef test/submission ciftlerini label'li train verisi yapmadan, leak-free sekilde birden fazla sinyali birlestirmek.

#### Eski leaky negatifleri tekrar kullanmak degil

Karar: **Onermiyorum.**

Eski yontemde `submission_pairs.csv` icinden hard negative uretilip train'e `label=0` olarak ekleniyordu. Bu public LB'de skor getirebilir ama:

- Test pair'lerini label'li train ornegi gibi kullanmis olursun.
- Private/generalization ve cozum tekrar uretilebilirligi riskli olur.
- Final asamada cozum incelenirse aciklanmasi zor.

#### Submission verisini labelsiz inference/ranking icin kullanmak

Karar: **Olur.**

`submission_pairs.csv`:

- Tahmin edilecek aday listesidir.
- Query group rank/percentile feature'lari inference'da bu adaylar uzerinden hesaplanabilir.
- Ancak bu satirlara egitimde label atanmamali.

Guvenli form:

- Train: sadece train pozitifleri + katalogdan uretilen sentetik negatifler.
- Inference: submission candidate list uzerinde embedding/rank score hesaplanir.
- Postprocess: brand/color/gender gibi feature'lara dayali threshold ayari yapilabilir, ama test row'lari training label'i olmaz.

Leak-free final kombinasyon:

- Leak-free LGBM classifier score.
- Parser/facet features.
- `TY-ecomm-embed` cosine features.
- Query group icinde cosine/ranker rank features.
- LambdaRank score feature.
- Final binary threshold veya meta-classifier.

### Beklenen Skor Etkisi

En olasi katkilar:

1. `TY-ecomm-embed` cosine + parser feature:
   - Muhtemel skor artisi: orta.
   - Risk: dusuk-orta.

2. Embedding rank features + LambdaRank:
   - Muhtemel skor artisi: orta.
   - Risk: orta; macro F1 threshold kalibrasyonu kritik.

3. CrossEncoder/BERT classifier:
   - Muhtemel skor tavani: yuksek.
   - Maliyet/risk: yuksek. 3.36M pair inference pahali.

4. Trendyol LLM ile direct judging:
   - Muhtemel skor: belirsiz.
   - Maliyet/risk: cok yuksek. Tavsiye edilmez.

### Guncellenmis Oncelik

1. Parser feature + query-aware hard negatives.
2. `TY-ecomm-embed` bi-encoder cosine feature.
3. Cosine rank/percentile/zscore feature.
4. LGBM/CatBoost ensemble.
5. XGBRanker veya LightGBM LambdaRank score feature.
6. Sadece belirsiz band/top-k icin CrossEncoder.
7. Trendyol LLM sadece offline analiz/query parser gelistirme yardimcisi.

### Ek Kaynaklar

- Trendyol HF org: https://huggingface.co/Trendyol
- Trendyol e-commerce embedding model: https://huggingface.co/Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0
- Trendyol LLM-8B-T1: https://huggingface.co/Trendyol/Trendyol-LLM-8B-T1
- Trendyol LLM Asure 12B: https://huggingface.co/Trendyol/Trendyol-LLM-Asure-12B
- Trendyol product image encoder: https://huggingface.co/Trendyol/e-commerce-product-image-encoder
- Trendyol DinoV2 image similarity model: https://huggingface.co/Trendyol/trendyol-dino-v2-ecommerce-256d
- LightGBM parameters, ranking objectives: https://lightgbm.readthedocs.io/en/latest/Parameters.html
- XGBoost learning to rank: https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html
