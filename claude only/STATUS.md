# Trendyol Kaggle 2026 — Proje Durum Dosyası

**Metrik:** Macro-averaged F1  
**Deadline:** 17 Temmuz 2026  
**Hedef:** 0.90+

---

## 1. KAGGLE GERÇEK SKORLAR

| Versiyon | Script | Yaklaşım | Kaggle Skoru |
|----------|--------|----------|-------------|
| v1 | 01 | BM25 Top-K=14 | **0.48** |
| v2 | 02 | LightGBM (BM25 feature) | **0.48** |
| v2b | 02b | LightGBM Top-K=14 | **0.48** |
| v3 | 03 | MiniLM zero-shot cosine | **0.45** |
| v4_k8 | 04 | KNN collab K=8 | submit edildi |
| v4_k14 | 04 | KNN collab K=14 | submit edildi |
| v5 | 05 | E5-base fine-tuned K=8 | **0.43** ← EN KÖTÜ |
| v6 | 06 | Category KNN + overlap | **0.47** |
| v7 | 07 | XLM-R + TF-IDF neg | kalite yok, submit edilmedi |
| v8 | 08 | Brand + Cat + BM25 | **0.47** |
| v9 | 09 | XLM-R + cross-query neg | **0.49** ← EN İYİ |
| v10 | 10 | Query transfer (IBQT) | ~0.48 (submit edilmedi) |

---

## 2. TANISAL DENEYLER (Kaggle'a yüklenmedi)

### S1 — BGE Reranker Zero-Shot (11_bge_s1.py)
- Model: BAAI/bge-reranker-v2-m3 (568M)
- CV Macro F1: **0.502** (500 sorgu, BM25 top-100 aday)
- BM25 coverage: 36.5%
- Karar: submit edilmedi (eşik 0.56 altında)

### Dense Coverage Ölçümleri (12_dense_coverage_check.py)
| Retriever | Top-100 Coverage |
|-----------|-----------------|
| BM25 TF-IDF | 36.5% |
| E5-v5 (fine-tuned, collapsed) | 47.0% |
| E5-base zero-shot | 36.6% |
| E5-large-instruct zero-shot | 38.2% |
| BGE-M3 zero-shot | 32.7% |

**Sonuç:** Hiçbir dense retriever BM25'i belirgin şekilde geçemedi.
Test adayları Trendyol'un kendi indexiyle seçilmiş → biz taklit edemiyoruz.

---

## 3. NEDEN 0.47-0.50'DE TAKILDIK — KÖK NEDEN

### Yanlış Negatif Dağılımı

```
Bizim training negatiflerimiz (v9):
  "avon luck kadın parfüm"  →  "granül kahve 500gr"   (çok KOLAY)
  "avon luck kadın parfüm"  →  "mobilya takımı"       (çok KOLAY)

Test'teki gerçek negatifler:
  "avon luck kadın parfüm"  →  "avon love kadın edp 50ml"   (çok ZOR)
  "avon luck kadın parfüm"  →  "oriflame charm edp 50ml"    (çok ZOR)
  "avon luck kadın parfüm"  →  "avon luck erkek edt 50ml"   (gender yanlış)
```

Model training'de kolayı öğrendi → test'te zoru göremedi → 0.49 tavanı.

### Query Anlama Eksikliği

Türkçe queryler karmaşık yapıda:
```
"avon luck kadın edp 50ml"
  brand=avon | product=luck | gender=kadın | type=edp | size=50ml

"bebek bezi 3 numara 4-9 kg"
  ürün=bezi | beden=3 | yaş=4-9kg

"pandora gold ring kadın gümüş"
  brand=pandora | color=gold | ürün=ring | gender=kadın | material=gümüş
```

Şu ana kadar bu özellikleri hiç çıkarmadık. Her şeyi karakter n-gram olarak işledik.

---

## 4. YENİ YAKLAŞIM — Yapısal Query Anlama + Yapısal Negatifler

### Fikir

1. **Query Parser** — Her query'den şu özellikleri çıkar:
   - brand (marka)
   - product_type (ürün türü: parfüm, ayakkabı, telefon...)
   - gender (kadın/erkek/unisex/bebek/çocuk)
   - color (renk)
   - size/capacity (50ml, 128GB, 3 numara...)
   - age_range (bebek/çocuk/yetişkin)
   - material (gümüş/gold/deri...)

2. **Yapısal Hard Negative Üretimi** — Belirli kurallara göre:
   - BRAND_SWAP: aynı ürün, farklı marka → kesinlikle negatif
   - GENDER_SWAP: bebek ürünü ↔ yetişkin ürünü → kesinlikle negatif
   - PRODUCT_SWAP: aynı marka, farklı ürün kategorisi → kesinlikle negatif
   - COLOR_SWAP: renk belirtilmişse farklı renk → negatif
   - Bu negativler kontaminasyon riski SIFIR çünkü kurallar belirgin

3. **Fine-Tune** — Yapısal negatiflerle BGE veya Qwen'i eğit

### Neden Bu Yaklaşım Doğru

| Eski Negatif | Yeni Negatif |
|-------------|-------------|
| Farklı kategoriden rastgele | Aynı kategoride, bir özelliği farklı |
| "avon" + "mobilya" | "avon luck" + "avon love" |
| Model kolay şey öğrendi | Model gerçek ayırım öğreniyor |
| Test dağılımından uzak | Test dağılımına yakın |

---

## 5. SUBMISSION PLANI (v11 sonrası)

| Versiyon | Script | Yaklaşım | CV Hedefi | Submit? |
|----------|--------|----------|-----------|---------|
| — | 13_query_parser.py | Query özellik çıkarma (araç) | — | Hayır |
| — | 14_structured_neg_gen.py | Yapısal negatif üretici (araç) | — | Hayır |
| **v11** | 15_bge_structured_v11.py | BGE + yapısal neg fine-tune | **0.62+** | CV>0.58 ise |
| **v12** | 16_qwen_zeroshot_v12.py | Qwen2.5-7B zero-shot + attr | **0.68+** | CV>0.62 ise |
| **v13** | 17_qwen_lora_v13.py | Qwen LoRA + yapısal neg | **0.75+** | CV>0.70 ise |
| **v14** | 18_ensemble_v14.py | BGE + Qwen ensemble | **0.82+** | CV>0.78 ise |

### Submission Kuralı
- Her submission dosyası: `submissions/submission_vXX_isim.csv`
- Submit kararı: **CV Macro F1 > eşik** ise Kaggle'a yükle
- Her adımda CV önce çalışır, sonra karar verilir

---

## 6. DOSYA DÜZENİ

```
claude only/
├── STATUS.md                 ← BU DOSYA (tek kaynak)
├── requirements.txt
│
├── scripts/   (araçlar, submission değil)
│   ├── 13_query_parser.py
│   └── 14_structured_neg_gen.py
│
├── submissions/             (tüm v1..vN çıktıları)
│   ├── submissions_log.csv  ← her versiyon için kayıt
│   ├── submission_v1_*.csv
│   ...
│   └── submission_vN_*.csv
│
├── coverage_reports/        (tanısal raporlar)
├── models/                  (kaydedilmiş modeller)
├── emb_cache/               (embedding cache)
├── figures/                 (EDA grafikleri)
│
└── archive/                 (eski plan dosyaları)
    ├── PLAN_TO_096.md
    ├── MASTER_PLAN.md
    └── ...
```

---

## 7. HATA ÖNLEME KURALLARI (Geçmişten Öğrenildi)

| Kural | Kaynak Hata |
|-------|------------|
| K = **14 sabit** (değiştirme) | v5 K=8 felaketi (0.43) |
| CV = BM25 top-100 üzerinde (kolay decoy değil) | v10 sahte CV 0.75 |
| TF-IDF negatif kullanma | v7 kontaminasyon |
| Cross-query negatif yetmez (test dağılımına uzak) | v9 0.49 tavanı |
| CV < 0.58 → Kaggle slot harcama | slot israfını önle |
| In-batch negatives tek başına kullanma | v5 embedding collapse |

---

## 8. AKTİF GÖREV

**Şu an yapılacak:**
1. `13_query_parser.py` — Query özellikleri çıkar, validate et
2. `14_structured_neg_gen.py` — Yapısal negatif üret
3. `15_bge_structured_v11.py` — İlk yeni submission adayı

**Sonra:**
- v12 (Qwen zero-shot), v13 (Qwen LoRA), v14 (ensemble)
