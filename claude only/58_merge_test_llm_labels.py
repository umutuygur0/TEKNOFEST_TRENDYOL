"""
58_merge_test_llm_labels.py — v36 Aşama a: test-dağılımı LLM doğrulama sonuçlarını birleştir
================================================================================================
51_merge_llm_labels.py'nin test-dağılımı versiyonu. Tüm batch_*_result.csv dosyalarını
(Claude'un doğruladığı pair_id -> verdict) orijinal candidates.csv (term_id, item_id, query,
title, ...) ile birleştirir.

Çıktı: claude only/57_llm_labels_test/merged_test_llm_labels.csv
  Kolonlar: term_id, item_id, label, src
    - verdict=1 → label=1, src="llm_recovered_positive_test"
    - verdict=0 → label=0, src="llm_verified_negative_test"

src isimleri v35'in train-kaynaklı etiketlerinden ("llm_recovered_positive" /
"llm_verified_negative") kasıtlı olarak farklı — kaynak izlenebilirliği için.

Bu dosya v36a script'inde train_df'e eklenecek yeni satırları oluşturur.
"""

import sys, glob
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parents[1]
LABELS_DIR = BASE / "claude only" / "57_llm_labels_test"
RESULTS_DIR = LABELS_DIR / "results"

candidates = pd.read_csv(LABELS_DIR / "candidates.csv")
print(f"Toplam aday: {len(candidates):,}")

result_files = sorted(glob.glob(str(RESULTS_DIR / "batch_*_result.csv")))
print(f"Bulunan sonuç dosyası: {len(result_files)}")

all_results = []
bad_files = []
for f in result_files:
    try:
        try:
            df = pd.read_csv(f)
        except Exception:
            df = pd.read_csv(f, engine="python", on_bad_lines="skip")
        if not {"pair_id", "verdict"}.issubset(df.columns):
            bad_files.append((f, "eksik kolon"))
            continue
        df = df[pd.to_numeric(df["pair_id"], errors="coerce").notna()]
        df = df[pd.to_numeric(df["verdict"], errors="coerce").notna()]
        all_results.append(df[["pair_id", "verdict"]])
    except Exception as e:
        bad_files.append((f, str(e)))

if bad_files:
    print(f"\n⚠ Sorunlu dosyalar ({len(bad_files)}):")
    for f, err in bad_files:
        print(f"  {f}: {err}")

results = pd.concat(all_results, ignore_index=True)
results = results.drop_duplicates(subset="pair_id", keep="first")
print(f"\nToplam doğrulanmış çift: {len(results):,} / {len(candidates):,} "
      f"({100*len(results)/len(candidates):.1f}%)")

merged = candidates.merge(results, on="pair_id", how="inner")
print(f"Birleştirilen: {len(merged):,}")

merged["verdict"] = merged["verdict"].astype(int)
n_pos = (merged["verdict"] == 1).sum()
n_neg = (merged["verdict"] == 0).sum()
print(f"\n  llm_recovered_positive_test (verdict=1): {n_pos:,} ({100*n_pos/len(merged):.1f}%)")
print(f"  llm_verified_negative_test  (verdict=0): {n_neg:,} ({100*n_neg/len(merged):.1f}%)")

out = pd.DataFrame({
    "term_id": merged["term_id"],
    "item_id": merged["item_id"],
    "label": merged["verdict"],
    "src": merged["verdict"].map({1: "llm_recovered_positive_test", 0: "llm_verified_negative_test"}),
})

out_path = LABELS_DIR / "merged_test_llm_labels.csv"
out.to_csv(str(out_path), index=False)
print(f"\nDosya: {out_path}")
print(f"Benzersiz sorgu sayısı: {out['term_id'].nunique():,}")
