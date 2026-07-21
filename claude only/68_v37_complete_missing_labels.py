"""
68_v37_complete_missing_labels.py

Fill only the missing v37 batch result CSVs with a conservative
LLM-distilled labeler.

Why this exists:
  The original batch result files were written by manual/LLM semantic
  judging. In this Codex continuation we first learn that judgement pattern
  from all completed LLM batches, including the existing v37 subset, then
  write the remaining v37 result files only if grouped CV quality is high.

Inputs:
  claude only/51_llm_labels
  claude only/57_llm_labels_test
  claude only/63_llm_labels_v37_train
  claude only/64_llm_labels_v37_test

Outputs, with --write:
  Missing files under:
    claude only/63_llm_labels_v37_train/results
    claude only/64_llm_labels_v37_test/results
"""

from __future__ import annotations

import argparse
import gc
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from rapidfuzz import fuzz as rfuzz
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer


sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "claude only"
MODELS_DIR = ROOT / "models"
BERT_V36 = MODELS_DIR / "bert_v36"
CACHE_DIR = ROOT / "llm_distill_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DIRS = [
    ROOT / "51_llm_labels",
    ROOT / "57_llm_labels_test",
    ROOT / "63_llm_labels_v37_train",
    ROOT / "64_llm_labels_v37_test",
]
TARGET_DIRS = [
    ROOT / "63_llm_labels_v37_train",
    ROOT / "64_llm_labels_v37_test",
]

TR_MAP = str.maketrans({
    "\u0130": "i", "I": "i", "\u0131": "i",
    "\u015e": "s", "\u015f": "s",
    "\u011e": "g", "\u011f": "g",
    "\u00dc": "u", "\u00fc": "u",
    "\u00d6": "o", "\u00f6": "o",
    "\u00c7": "c", "\u00e7": "c",
})

UNKNOWN = "unknown"
STOP = {
    "ve", "ile", "icin", "bu", "bir", "de", "da", "den", "dan", "en",
    "cok", "az", "gibi", "olan", "set", "adet", "li", "lu", "model",
}
GENDER_WORDS = {"kadin", "bayan", "erkek", "kiz", "unisex"}
AGE_WORDS = {"bebek", "cocuk", "yetiskin", "genc"}
COLORS = {
    "siyah", "beyaz", "mavi", "kirmizi", "yesil", "sari", "pembe", "mor",
    "gri", "turuncu", "lacivert", "bej", "kahverengi", "gold", "altin",
    "gumus", "silver", "rose", "ekru", "krem", "bordo", "haki", "fume",
    "antrasit", "indigo", "petrol",
}
MATERIALS = {
    "pamuk", "pamuklu", "deri", "polyester", "yun", "keten", "naylon",
    "celik", "plastik", "ahsap", "akrilik", "viskon", "modal", "bambu",
}


def norm(s) -> str:
    if pd.isna(s):
        return ""
    s = str(s).translate(TR_MAP).casefold()
    return re.sub(r"\s+", " ", s).strip()


def toks(s) -> list[str]:
    return re.findall(r"[a-z0-9]+", norm(s))


def tokset(s) -> set[str]:
    return set(toks(s))


def jac(a: set[str], b: set[str]) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def cov(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a) if a else 0.0


def query_head(q_tokens: list[str]) -> str:
    words = [w for w in q_tokens if w not in STOP and len(w) > 1]
    return words[-1] if words else (q_tokens[-1] if q_tokens else "")


def code_tokens(tokens: list[str]) -> set[str]:
    return {t for t in tokens if any(c.isdigit() for c in t)}


def spec_code_match(q_tokens: list[str], target_tokens: list[str]) -> float:
    q_codes = code_tokens(q_tokens)
    if not q_codes:
        return 0.0
    t_codes = code_tokens(target_tokens)
    if not t_codes:
        return -1.0
    hits = len(q_codes & t_codes)
    if hits == len(q_codes):
        return 1.0
    if hits == 0:
        return -1.0
    return hits / len(q_codes)


def gender_signal(q_tokens: set[str], gender: str, title_tokens: set[str]) -> float:
    q_k = bool(q_tokens & {"kadin", "bayan", "kiz"})
    q_e = "erkek" in q_tokens
    g = norm(gender)
    target_k = g in {"kadin", "bayan", "kiz"} or bool(title_tokens & {"kadin", "bayan", "kiz"})
    target_e = g == "erkek" or "erkek" in title_tokens
    if q_k and target_e:
        return -1.0
    if q_e and target_k:
        return -1.0
    if q_k and target_k:
        return 1.0
    if q_e and target_e:
        return 1.0
    return 0.0


def age_signal(q_tokens: set[str], age: str, title_tokens: set[str]) -> float:
    q_baby = "bebek" in q_tokens
    q_child = "cocuk" in q_tokens
    q_adult = "yetiskin" in q_tokens
    a = norm(age)
    target_baby = a == "bebek" or "bebek" in title_tokens
    target_child = a in {"cocuk", "bebek & cocuk", "genc"} or "cocuk" in title_tokens
    target_adult = a == "yetiskin" or "yetiskin" in title_tokens
    if q_baby and target_baby:
        return 1.0
    if q_child and target_child:
        return 1.0
    if q_adult and target_adult:
        return 1.0
    if q_baby and target_adult:
        return -1.0
    if q_child and target_adult:
        return -1.0
    if q_adult and (target_baby or target_child):
        return -1.0
    return 0.0


def color_signal(q_tokens: set[str], target_tokens: set[str]) -> float:
    q_colors = q_tokens & COLORS
    if not q_colors:
        return 0.0
    t_colors = target_tokens & COLORS
    if not t_colors:
        return 0.0
    return 1.0 if q_colors & t_colors else -1.0


def product_cover(q_tokens: list[str], title_tokens: set[str], brand_tokens: set[str]) -> float:
    keep = [
        w for w in q_tokens
        if w not in STOP
        and w not in GENDER_WORDS
        and w not in AGE_WORDS
        and w not in COLORS
        and w not in MATERIALS
        and w not in brand_tokens
        and len(w) > 1
    ]
    if not keep:
        return cov(set(q_tokens), title_tokens)
    return sum(1 for w in keep if w in title_tokens) / len(keep)


def read_result_file(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception as exc:
            print(f"  ! cannot read {path}: {exc}")
            return None
    if not {"pair_id", "verdict"}.issubset(df.columns):
        print(f"  ! skipping {path}: missing pair_id/verdict")
        return None
    df = df[["pair_id", "verdict"]].copy()
    df["pair_id"] = pd.to_numeric(df["pair_id"], errors="coerce")
    df["verdict"] = pd.to_numeric(df["verdict"], errors="coerce")
    df = df.dropna(subset=["pair_id", "verdict"])
    df["pair_id"] = df["pair_id"].astype(int)
    df["verdict"] = df["verdict"].astype(int)
    df = df[df["verdict"].isin([0, 1])]
    return df


def batch_num(path: Path) -> str:
    m = re.search(r"batch_(\d+)", path.name)
    return m.group(1) if m else "unknown"


def read_known(labels_dir: Path) -> pd.DataFrame:
    cand = pd.read_csv(labels_dir / "candidates.csv")
    parts = []
    for result_path in sorted((labels_dir / "results").glob("batch_*_result.csv")):
        res = read_result_file(result_path)
        if res is None or res.empty:
            continue
        res["batch"] = batch_num(result_path)
        parts.append(res)
    if not parts:
        return pd.DataFrame()
    res_all = pd.concat(parts, ignore_index=True).drop_duplicates("pair_id", keep="first")
    out = cand.merge(res_all, on="pair_id", how="inner")
    out["label"] = out["verdict"].astype(int)
    out["source_dir"] = labels_dir.name
    out["group"] = labels_dir.name + "_batch_" + out["batch"].astype(str)
    return out


def missing_batches(labels_dir: Path) -> list[Path]:
    batches = sorted((labels_dir / "batches").glob("batch_*.csv"))
    done = {
        batch_num(p)
        for p in (labels_dir / "results").glob("batch_*_result.csv")
    }
    return [p for p in batches if batch_num(p) not in done]


def read_missing(labels_dir: Path) -> pd.DataFrame:
    parts = []
    for batch_path in missing_batches(labels_dir):
        df = pd.read_csv(batch_path)
        df["source_dir"] = labels_dir.name
        df["batch"] = batch_num(batch_path)
        df["group"] = labels_dir.name + "_batch_" + df["batch"].astype(str)
        df["batch_path"] = str(batch_path)
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def item_text_for_bert(df: pd.DataFrame) -> list[str]:
    out = []
    for row in df.itertuples(index=False):
        cat = norm(getattr(row, "category", ""))
        main_cat = cat.split("/")[0] if cat else ""
        chunks = [
            norm(getattr(row, "title", "")),
            norm(getattr(row, "brand", "")),
            main_cat,
            norm(getattr(row, "attributes_short", "")),
        ]
        out.append(" ".join(c for c in chunks if c and c != UNKNOWN))
    return out


def bert_scores_for(df: pd.DataFrame, cache_key: str, batch_size: int = 256) -> np.ndarray:
    cache_path = CACHE_DIR / f"bert_v36_{cache_key}.npy"
    if cache_path.exists():
        arr = np.load(str(cache_path))
        if len(arr) == len(df):
            print(f"  BERT cache: {cache_path.name} ({len(arr):,})")
            return arr.astype(np.float32)

    print(f"  BERT inference: {cache_key} ({len(df):,})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(BERT_V36))
    model = AutoModelForSequenceClassification.from_pretrained(str(BERT_V36)).to(device)
    model.eval()

    queries = [norm(x) for x in df["query"].fillna("").tolist()]
    items = item_text_for_bert(df)
    scores = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            enc = tokenizer(
                queries[start:end],
                items[start:end],
                max_length=128,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.reshape(-1)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            scores.append(prob)
            if end == len(df) or (end // batch_size) % 20 == 0:
                print(f"    {end:,}/{len(df):,}", flush=True)

    arr = np.concatenate(scores).astype(np.float32)
    np.save(str(cache_path), arr)
    print(f"  -> {cache_path.name} | {(time.time() - t0) / 60:.1f} dk")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return arr


def build_features(df: pd.DataFrame, bert_scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        q = norm(getattr(row, "query", ""))
        t = norm(getattr(row, "title", ""))
        b = norm(getattr(row, "brand", ""))
        c = norm(getattr(row, "category", ""))
        g = norm(getattr(row, "gender", ""))
        a = norm(getattr(row, "age_group", ""))
        attrs = norm(getattr(row, "attributes_short", ""))

        qt = toks(q)
        tt = toks(t)
        bt = toks(b)
        ct = toks(c.replace("/", " "))
        at = toks(attrs)
        qs = set(qt)
        ts = set(tt)
        bs = set(bt)
        cs = set(ct)
        ats = set(at)
        target = ts | bs | cs | ats
        head = query_head(qt)

        rows.append({
            "cosine": float(getattr(row, "cosine", 0.0) or 0.0),
            "bert_score": float(bert_scores[i]),
            "q_len": len(qt),
            "t_len": len(tt),
            "brand_len": len(bt),
            "cat_len": len(ct),
            "token_overlap": len(qs & ts),
            "target_overlap": len(qs & target),
            "jaccard_title": jac(qs, ts),
            "jaccard_target": jac(qs, target),
            "q_cov_title": cov(qs, ts),
            "q_cov_target": cov(qs, target),
            "t_cov_query": cov(ts, qs),
            "cat_cov": cov(qs, cs),
            "attr_cov": cov(qs, ats),
            "brand_cov_query": cov(bs, qs),
            "brand_cov_title": cov(bs, ts),
            "query_brand_cov": cov(qs, bs),
            "exact_in_title": 1.0 if q and q in t else 0.0,
            "all_q_in_title": 1.0 if qs and qs.issubset(ts) else 0.0,
            "head_in_title": 1.0 if head and head in ts else 0.0,
            "head_in_cat": 1.0 if head and head in cs else 0.0,
            "first_tok_title": 1.0 if qt and qt[0] in ts else 0.0,
            "first_tok_brand": 1.0 if qt and qt[0] in bs else 0.0,
            "fuzz_ratio": rfuzz.ratio(q, t) / 100.0,
            "fuzz_partial": rfuzz.partial_ratio(q, t) / 100.0,
            "fuzz_sort": rfuzz.token_sort_ratio(q, t) / 100.0,
            "fuzz_set": rfuzz.token_set_ratio(q, t) / 100.0,
            "gender_signal": gender_signal(qs, g, ts),
            "age_signal": age_signal(qs, a, ts),
            "color_signal": color_signal(qs, target),
            "spec_code_match": spec_code_match(qt, tt + at),
            "product_cover": product_cover(qt, ts, bs),
            "has_code": 1.0 if code_tokens(qt) else 0.0,
            "has_color": 1.0 if qs & COLORS else 0.0,
            "has_gender": 1.0 if qs & GENDER_WORDS else 0.0,
            "has_age": 1.0 if qs & AGE_WORDS else 0.0,
        })
    return pd.DataFrame(rows).fillna(0.0)


def best_threshold(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.951, 0.005):
        f1 = f1_score(y_true, (score >= thr).astype(int), average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, best_f1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write missing result CSVs")
    ap.add_argument("--min-f1", type=float, default=0.88, help="minimum grouped OOF macro F1 needed to write")
    args = ap.parse_args()

    print("=" * 70)
    print("v37 missing batch completion via LLM-distilled labeler")
    print("=" * 70)

    known_parts = []
    for d in TRAIN_DIRS:
        k = read_known(d)
        print(f"  known {d.name}: {len(k):,}")
        if not k.empty:
            known_parts.append(k)
    known = pd.concat(known_parts, ignore_index=True)
    known = known.drop_duplicates(["source_dir", "pair_id"], keep="first")

    missing_parts = []
    for d in TARGET_DIRS:
        m = read_missing(d)
        print(f"  missing {d.name}: {len(m):,} rows in {m['batch'].nunique() if not m.empty else 0} batches")
        if not m.empty:
            missing_parts.append(m)
    if not missing_parts:
        print("\nNo missing v37 batches. Nothing to do.")
        return 0
    missing = pd.concat(missing_parts, ignore_index=True)

    print(f"\nKnown labels: {len(known):,} | pos={int(known['label'].sum()):,} "
          f"neg={int((known['label'] == 0).sum()):,}")
    print(f"Missing rows: {len(missing):,}")

    # Keep cache keys stable by source.
    all_for_bert = []
    for name, part in list(known.groupby("source_dir")) + list(missing.groupby("source_dir")):
        key = f"{name}_{'known' if 'label' in part.columns and part['label'].notna().any() else 'missing'}"
        part = part.copy()
        part["_bert_cache_key"] = key
        all_for_bert.append(part)

    known_scores = np.zeros(len(known), dtype=np.float32)
    missing_scores = np.zeros(len(missing), dtype=np.float32)
    for name, part in known.groupby("source_dir", sort=False):
        idx = part.index.to_numpy()
        arr = bert_scores_for(part.reset_index(drop=True), f"{name}_known")
        known_scores[idx] = arr
    for name, part in missing.groupby("source_dir", sort=False):
        idx = part.index.to_numpy()
        arr = bert_scores_for(part.reset_index(drop=True), f"{name}_missing")
        missing_scores[idx] = arr

    print("\nFeature engineering...")
    X = build_features(known.reset_index(drop=True), known_scores)
    y = known["label"].astype(int).to_numpy()
    groups = known["group"].astype(str).to_numpy()
    X_missing = build_features(missing.reset_index(drop=True), missing_scores)

    print(f"  X known: {X.shape} | X missing: {X_missing.shape}")
    print("  grouped CV by source/batch...")

    oof = np.zeros(len(known), dtype=np.float32)
    gkf = GroupKFold(n_splits=5)
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups), start=1):
        model = LGBMClassifier(
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=35,
            subsample=0.90,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            class_weight="balanced",
            random_state=4200 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1].astype(np.float32)
        thr_f, f1_f = best_threshold(y[va], oof[va])
        print(f"    fold {fold}: val macroF1={f1_f:.4f} thr={thr_f:.3f} rows={len(va):,}")

    thr, macro_f1 = best_threshold(y, oof)
    pred = (oof >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    acc = float((pred == y).mean())
    print("\nOOF summary")
    print(f"  macro F1 : {macro_f1:.5f}")
    print(f"  threshold: {thr:.3f}")
    print(f"  accuracy : {acc:.5f}")
    print(f"  neg P/R/F: {p[0]:.4f}/{r[0]:.4f}/{f[0]:.4f}")
    print(f"  pos P/R/F: {p[1]:.4f}/{r[1]:.4f}/{f[1]:.4f}")

    if macro_f1 < args.min_f1:
        print(f"\nSTOP: OOF macro F1 {macro_f1:.5f} < --min-f1 {args.min_f1:.5f}.")
        print("No files were written.")
        return 2

    final_model = LGBMClassifier(
        n_estimators=900,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=35,
        subsample=0.90,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        class_weight="balanced",
        random_state=777,
        n_jobs=-1,
        verbose=-1,
    )
    final_model.fit(X, y)
    miss_proba = final_model.predict_proba(X_missing)[:, 1].astype(np.float32)
    miss_pred = (miss_proba >= thr).astype(int)

    missing = missing.reset_index(drop=True).copy()
    missing["distill_proba"] = miss_proba
    missing["verdict"] = miss_pred
    missing["bert_score"] = missing_scores

    print("\nMissing prediction summary")
    for source, part in missing.groupby("source_dir"):
        pos = int(part["verdict"].sum())
        print(f"  {source}: rows={len(part):,} pos={pos:,} ({100 * pos / len(part):.1f}%)")
    print("  by batch:")
    for (source, batch), part in missing.groupby(["source_dir", "batch"]):
        pos = int(part["verdict"].sum())
        print(f"    {source}/batch_{batch}: {len(part):4d} rows | pos={pos:4d} ({100 * pos / len(part):5.1f}%)")

    if not args.write:
        print("\nDry run only. Re-run with --write to create missing result CSVs.")
        return 0

    print("\nWriting missing result CSVs...")
    for (source, batch), part in missing.groupby(["source_dir", "batch"], sort=True):
        labels_dir = ROOT / source
        out_path = labels_dir / "results" / f"batch_{batch}_result.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            print(f"  skip existing {out_path}")
            continue
        out = pd.DataFrame({
            "pair_id": part["pair_id"].astype(int),
            "verdict": part["verdict"].astype(int),
            "reason": [
                f"llm_distill p={p:.3f} bert={b:.3f}"
                for p, b in zip(part["distill_proba"], part["bert_score"])
            ],
        })
        out.to_csv(out_path, index=False)
        print(f"  wrote {out_path} ({len(out):,})")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
