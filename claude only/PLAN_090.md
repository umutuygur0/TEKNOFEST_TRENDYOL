# Plan: 0.68 → 0.90 (5 Phase)

**Tarih:** 2026-07-01  
**Deadline:** 17 Temmuz 2026 (16 gün)  
**Mevcut:** 0.68 | **Hedef:** 0.90

---

## Geçmişten Alınan Dersler

| Hata | Öğrenme |
|------|---------|
| v1-v11: BM25 retrieval + reranker | Problem retrieval DEĞİL — 3.36M çift zaten hazır, sınıflandır |
| BGE fine-tune easy neg (same_cat) | Kolay negatifler → CV 0.95 ama test 0.17 → model hard case öğrenemez |
| Random/cross-query negatifler | Gerçek test dağılımını yansıtmıyor → CV-test gap büyür |
| OOF CV 0.956 vs test 0.68 | Eğitim negativleri çok kolay → threshold altında kalan hard case'ler |

---

## Phase 1 — LightGBM Baseline Güçlendir (BUGÜN)
**Script:** `16_lgbm_transductive_v12.py` (çalışıyor)  
**Değişiklikler:** 7 özellik → 13 özellik  
**Yeni özellikler:** tfidf_cos_sim, fuzz_partial, fuzz_token_sort, fuzz_basic, attr_match, len_diff  
**Beklenen:** 0.68 → 0.72  
**Submission:** `submission_v12_lgbm_transductive.csv`

**Neden işe yarayacak:** TF-IDF cosine similarity en güçlü yeni sinyal (raporda belirtilen ama kodda yoktu).

---

## Phase 2 — Embedding Features (1-2 gün)
**Script:** `17_lgbm_embedding_v13.py`

### Yaklaşım
LightGBM feature olarak multilingual embedding cosine similarity ekle.

### Adımlar
1. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (278MB, hızlı)
2. Tüm unique query'leri encode et (~50K sorgu)
3. Tüm unique title'ları encode et (~900K title → önceden hesapla, cache'le)
4. Her çift için cosine similarity → yeni feature `embed_cos_sim`
5. Aynı LightGBM'e ekle, retrain

### Neden Bu Model
- Multilingual (Türkçeyi tanır)
- Küçük (fast encode)
- sentence-transformers → encode() tek satır

### Beklenen: 0.72 → 0.77

---

## Phase 3 — Hard Negative Mining (2-3 gün)
**Script:** `18_lgbm_hardneg_v14.py`

### Sorun
Şu anki negativler çok kolay:
- Gender mismatch → model bu kadar basit bir kuralı öğrenir
- Sıfır kelime overlap → model sadece kelime paylaşımına bakar

Gerçek test datası ÇOK DAHA ZOR negatifler içeriyor.

### Yaklaşım: BM25 Hard Negatives (Same-Query)
Her pozitif çift (term_id, item_id) için:
1. O term_id için training_pairs'te olmayan items'ı bul
2. TF-IDF cosine ile bu items'ı query'ye göre sırala
3. Top-3 most similar ama NOT positive → hard negative

Bu "hard negatives" model için gerçek zorluktur: query'ye benzeyen ama alakasız ürünler.

### Neden Daha İyi
- Mevcut: "kadın kol saati" sorgusu için "erkek kol saati" → negatif (çok kolay)
- Hard neg: "kadın kol saati" için "kadın bileklik" → negatif (gerçekten zor)
- Model artık ince farkları öğrenir

### Beklenen: 0.77 → 0.82

---

## Phase 4 — Neural Cross-Encoder (3-4 gün)
**Script:** `19_mdeberta_crossencoder_v15.py`

### Model
`microsoft/mdeberta-v3-base` (280MB, multilingual, güçlü)  
Alternatif: `dbmdz/bert-base-turkish-cased` (Türkçe-özel)

### Yaklaşım
1. Phase 3'ün hard negativleriyle mDeBERTa fine-tune
2. 500K pozitif + 500K hard negative → balanced
3. Doğrudan 3.36M test çifti üzerinde score
4. mDeBERTa skoru → yeni feature olarak Phase 1 LightGBM'e ekle

### Farkı
BGE (568M, yavaş) yerine mDeBERTa (280M, daha hızlı).  
İlk hatayla farkı: DOĞRU hard negativlerle eğitim.

### Beklenen: 0.82 → 0.87

---

## Phase 5 — Ensemble + Pseudo-Labeling (2-3 gün)
**Script:** `20_ensemble_pseudolabel_v16.py`

### Adımlar

**5a. Stacking Ensemble**
- Phase 1 (LightGBM v12): proba_1
- Phase 2 (LightGBM + embed): proba_2  
- Phase 4 (mDeBERTa): proba_4
- Ağırlıklı ortalama: 0.3 * proba_1 + 0.3 * proba_2 + 0.4 * proba_4

**5b. Pseudo-Labeling**
- Ensemble skoru > 0.95 → pseudo-pozitif (çok güvenli)
- Ensemble skoru < 0.05 → pseudo-negatif (çok güvenli)
- Bu pseudo-label'ları training'e ekle → Phase 4'ü retrain
- 1-2 iterasyon

**5c. Threshold Fine-Tuning**
- Per-category threshold (gender-related queryler için farklı threshold)
- GroupKFold içinde: ayrı threshold her fold için

### Beklenen: 0.87 → 0.90+

---

## Zaman Çizelgesi

| Tarih | Phase | Aksiyon |
|-------|-------|---------|
| 1 Temmuz (bugün) | Phase 1 | v12 çalışıyor → submit → 0.72 beklenti |
| 2-3 Temmuz | Phase 2 | Embedding features → submit → 0.77 beklenti |
| 4-6 Temmuz | Phase 3 | Hard negative mining → retrain → 0.82 beklenti |
| 7-11 Temmuz | Phase 4 | mDeBERTa fine-tune → submit → 0.87 beklenti |
| 12-16 Temmuz | Phase 5 | Ensemble + pseudo-label → 0.90 hedef |
| 17 Temmuz | DEADLINE | Final submit |

---

## Risk Analizi

| Risk | Olasılık | Azaltma |
|------|----------|---------|
| Phase 4 mDeBERTa yavaş (9 saat) | Yüksek | fp16 + MAX_LEN=128, batch=64 |
| Pseudo-labeling noise | Orta | Sadece >0.95 ve <0.05 confidence |
| CV-test gap devam eder | Orta | Hard neg mining ile azaltılır |
| 0.90 ulaşılamaz | Orta | Realistic target: 0.85 |

---

## Gerçekçi Beklenti

| Phase | Hedef | Gerçekçi |
|-------|-------|---------|
| 1 | 0.72 | 0.70-0.72 |
| 2 | 0.77 | 0.73-0.77 |
| 3 | 0.82 | 0.77-0.81 |
| 4 | 0.87 | 0.81-0.85 |
| 5 | 0.90 | 0.84-0.88 |

**Not:** 0.90 çok yüksek bir hedef ama yönlendirme olarak tutuyoruz. 0.85 kesinlikle ulaşılabilir.
