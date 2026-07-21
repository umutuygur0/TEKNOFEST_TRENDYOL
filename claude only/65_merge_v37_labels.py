"""
65_merge_v37_labels.py — v37: hedefli LLM doğrulama sonuçlarını birleştir (train-side + test-side)
=======================================================================================================
51_merge_llm_labels.py / 58_merge_test_llm_labels.py'nin v37 (hedefli/stratified) versiyonu.
Hem train-side (63_llm_labels_v37_train) hem test-side (64_llm_labels_v37_test) sonuçlarını
ayrı ayrı birleştirir, iki ayrı CSV üretir (src isimleri v37 ile ayırt edilsin diye farklı).

Çıktı:
  claude only/63_llm_labels_v37_train/merged_v37_train_labels.csv
  claude only/64_llm_labels_v37_test/merged_v37_test_labels.csv
"""

import sys, glob
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parents[1]


def merge_one(labels_dir: Path, src_pos: str, src_neg: str, out_name: str):
    results_dir = labels_dir / "results"
    candidates = pd.read_csv(labels_dir / "candidates.csv")
    print(f"\n[{labels_dir.name}] Toplam aday: {len(candidates):,}")

    result_files = sorted(glob.glob(str(results_dir / "batch_*_result.csv")))
    print(f"  Bulunan sonuç dosyası: {len(result_files)}")

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
        print(f"  ⚠ Sorunlu dosyalar ({len(bad_files)}):")
        for f, err in bad_files:
            print(f"    {f}: {err}")

    results = pd.concat(all_results, ignore_index=True)
    results = results.drop_duplicates(subset="pair_id", keep="first")
    print(f"  Toplam doğrulanmış çift: {len(results):,} / {len(candidates):,} "
          f"({100*len(results)/len(candidates):.1f}%)")

    merged = candidates.merge(results, on="pair_id", how="inner")
    merged["verdict"] = merged["verdict"].astype(int)
    n_pos = (merged["verdict"] == 1).sum()
    n_neg = (merged["verdict"] == 0).sum()
    print(f"  {src_pos} (verdict=1): {n_pos:,} ({100*n_pos/len(merged):.1f}%)")
    print(f"  {src_neg} (verdict=0): {n_neg:,} ({100*n_neg/len(merged):.1f}%)")

    out = pd.DataFrame({
        "term_id": merged["term_id"],
        "item_id": merged["item_id"],
        "label": merged["verdict"],
        "src": merged["verdict"].map({1: src_pos, 0: src_neg}),
    })

    out_path = labels_dir / out_name
    out.to_csv(str(out_path), index=False)
    print(f"  Dosya: {out_path}")
    print(f"  Benzersiz sorgu sayısı: {out['term_id'].nunique():,}")
    return out


train_out = merge_one(
    BASE / "claude only" / "63_llm_labels_v37_train",
    "llm_recovered_positive_v37_train", "llm_verified_negative_v37_train",
    "merged_v37_train_labels.csv",
)

test_out = merge_one(
    BASE / "claude only" / "64_llm_labels_v37_test",
    "llm_recovered_positive_v37_test", "llm_verified_negative_v37_test",
    "merged_v37_test_labels.csv",
)

print(f"\n{'='*65}")
print(f"TOPLAM v37 LLM etiketli çift: {len(train_out) + len(test_out):,}")
print(f"{'='*65}")
