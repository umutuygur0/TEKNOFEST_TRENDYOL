# -*- coding: utf-8 -*-
"""
S2 diagnostic: dense retrieval coverage check
=============================================

Purpose
-------
S1 showed that a strong zero-shot reranker is not enough when the candidate
generator misses many true positives. This script checks whether dense retrieval
improves candidate coverage before spending time on dense hard-negative mining.

Fast cached run:
  python "claude only/12_dense_coverage_check.py" --source cache-e5-v5

Fresh model run, slower:
  python "claude only/12_dense_coverage_check.py" ^
    --source model ^
    --model-name intfloat/multilingual-e5-large-instruct ^
    --cache-prefix e5_large_instruct_s2

What to look for
----------------
If dense top-100 coverage is clearly above BM25 top-100, S3 dense hard-negative
mining is worth doing. If dense coverage is still around BM25, jump to LLM/Qwen
or rethink candidate construction.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "trendyol-e-ticaret-yarismasi-2026-kaggle"
WORK = BASE / "claude only"
CACHE = WORK / "emb_cache"
REPORTS = WORK / "coverage_reports"
REPORTS.mkdir(parents=True, exist_ok=True)

SEED = 42
K_PREDICT = 14

LOWER_MAP = str.maketrans(
    {
        "\u0130": "i",
        "I": "\u0131",
        "\u015e": "\u015f",
        "\u011e": "\u011f",
        "\u00dc": "\u00fc",
        "\u00d6": "\u00f6",
        "\u00c7": "\u00e7",
    }
)


def tr_lower(text: object) -> str:
    return str(text).translate(LOWER_MAP).lower().strip()


def e5_query_text(query: object, model_name: str) -> str:
    q = tr_lower(query)
    if "instruct" in model_name.lower():
        return (
            "Instruct: Given a Turkish e-commerce search query, "
            "retrieve relevant product listings.\nQuery: "
            + q
        )
    return "query: " + q


def e5_item_text(row: pd.Series, model_name: str) -> str:
    title = tr_lower(row.get("title", ""))
    brand = tr_lower(row.get("brand", ""))
    category = tr_lower(str(row.get("category", "")).split("/")[0])
    gender = tr_lower(row.get("gender", ""))
    age_group = tr_lower(row.get("age_group", ""))

    parts = [p for p in (title, brand, category, gender, age_group) if p and p != "nan"]
    text = " | ".join(parts)
    if "e5" in model_name.lower() and "instruct" not in model_name.lower():
        return "passage: " + text
    return text


def bm25_item_text(row: pd.Series) -> str:
    return tr_lower(str(row.get("title", "")) + " " + str(row.get("brand", "")))


def parse_topks(value: str) -> list[int]:
    topks = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not topks:
        raise ValueError("At least one top-k value is required")
    return topks


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, set[str]]]:
    print("[1] Loading data...")
    t0 = time.time()
    items = pd.read_csv(DATA / "items.csv")
    terms = pd.read_csv(DATA / "terms.csv")
    train = pd.read_csv(DATA / "training_pairs.csv")

    tid_to_query = dict(zip(terms["term_id"], terms["query"]))
    train_pos: dict[str, set[str]] = defaultdict(set)
    for tid, iid in zip(train["term_id"].values, train["item_id"].values):
        train_pos[tid].add(iid)

    print(
        f"  {time.time() - t0:.1f}s | items={len(items):,} "
        f"| train_pairs={len(train):,} | train_queries={len(train_pos):,}"
    )
    return items, terms, train, tid_to_query, train_pos


def choose_holdout(train_pos: dict[str, set[str]], samples: int) -> list[str]:
    all_tids = list(train_pos.keys())
    random.seed(SEED)
    random.shuffle(all_tids)
    return all_tids[: min(samples, len(all_tids))]


def evaluate_top_indices(
    name: str,
    top_indices: np.ndarray,
    item_ids: np.ndarray,
    holdout_tids: list[str],
    train_pos: dict[str, set[str]],
    topks: list[int],
) -> list[dict[str, object]]:
    rows = []
    n_queries = len(holdout_tids)
    total_true = sum(len(train_pos[tid]) for tid in holdout_tids)

    for k in topks:
        hits_total = 0
        selected_tp_total = 0
        any_hit = 0
        zero_hit = 0
        full_hit = 0
        hit_counts = []

        for row_i, tid in enumerate(holdout_tids):
            true_pos = train_pos[tid]
            candidates = set(item_ids[top_indices[row_i, :k]].tolist())
            hit_count = len(true_pos & candidates)
            hit_counts.append(hit_count)
            hits_total += hit_count
            selected_tp_total += min(hit_count, K_PREDICT)
            any_hit += int(hit_count > 0)
            zero_hit += int(hit_count == 0)
            full_hit += int(hit_count == len(true_pos))

        pair_cov = hits_total / total_true if total_true else 0.0
        query_any = any_hit / n_queries if n_queries else 0.0
        query_full = full_hit / n_queries if n_queries else 0.0
        precision_ub = selected_tp_total / (n_queries * K_PREDICT) if n_queries else 0.0
        recall_ub = selected_tp_total / total_true if total_true else 0.0
        f1_ub = (
            2 * precision_ub * recall_ub / (precision_ub + recall_ub)
            if precision_ub + recall_ub > 0
            else 0.0
        )

        rows.append(
            {
                "method": name,
                "top_k": k,
                "n_queries": n_queries,
                "total_true": total_true,
                "hits": hits_total,
                "pair_coverage": pair_cov,
                "query_any_hit": query_any,
                "query_full_hit": query_full,
                "zero_hit_queries": zero_hit,
                "avg_hits_per_query": float(np.mean(hit_counts)) if hit_counts else 0.0,
                "median_hits_per_query": float(np.median(hit_counts)) if hit_counts else 0.0,
                "oracle_pos_f1_fixed14": f1_ub,
            }
        )

    return rows


def bm25_top_indices(
    items: pd.DataFrame,
    tid_to_query: dict[str, str],
    holdout_tids: list[str],
    max_topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    print(f"\n[2] BM25-like char TF-IDF top-{max_topk} baseline...")
    t0 = time.time()
    item_ids = items["item_id"].values
    titles = items["title"].fillna("").astype(str)
    brands = items["brand"].fillna("").astype(str)
    texts = (titles + " " + brands).map(tr_lower).tolist()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=3,
        max_features=200_000,
        sublinear_tf=True,
    )
    item_mat = vectorizer.fit_transform(texts)
    print(f"  item_matrix={item_mat.shape} | fit={time.time() - t0:.1f}s")

    top_indices = np.empty((len(holdout_tids), max_topk), dtype=np.int64)
    for qi, tid in enumerate(holdout_tids):
        q_text = tr_lower(tid_to_query.get(tid, ""))
        q_vec = vectorizer.transform([q_text])
        sims = (q_vec * item_mat.T).toarray()[0]
        idx = np.argpartition(sims, -max_topk)[-max_topk:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        top_indices[qi] = idx
        if qi and qi % 100 == 0:
            print(f"  BM25 {qi}/{len(holdout_tids)}")

    print(f"  BM25 done: {time.time() - t0:.1f}s")
    return top_indices, item_ids


def load_cached_e5_v5() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = {
        "item_embs": CACHE / "item_embs_e5_finetuned_v5.npy",
        "item_ids": CACHE / "item_ids_e5_finetuned_v5.npy",
        "query_embs": CACHE / "train_q_embs_e5_v5.npy",
        "query_ids": CACHE / "train_q_ids_e5_v5.npy",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing cache files:\n" + "\n".join(missing))

    print("\n[3] Loading cached E5 v5 embeddings...")
    item_embs = np.load(paths["item_embs"], mmap_mode="r")
    item_ids = np.load(paths["item_ids"], allow_pickle=True)
    query_embs = np.load(paths["query_embs"])
    query_ids = np.load(paths["query_ids"], allow_pickle=True)
    print(f"  item_embs={item_embs.shape} | query_embs={query_embs.shape}")
    return item_embs, item_ids, query_embs, query_ids


def model_cache_paths(prefix: str) -> dict[str, Path]:
    return {
        "item_embs": CACHE / f"item_embs_{prefix}.npy",
        "item_ids": CACHE / f"item_ids_{prefix}.npy",
        "query_embs": CACHE / f"train_q_embs_{prefix}.npy",
        "query_ids": CACHE / f"train_q_ids_{prefix}.npy",
    }


def encode_or_load_model_embeddings(
    items: pd.DataFrame,
    train: pd.DataFrame,
    tid_to_query: dict[str, str],
    model_name: str,
    cache_prefix: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = model_cache_paths(cache_prefix)
    if all(path.exists() for path in paths.values()):
        print(f"\n[3] Loading cached model embeddings: {cache_prefix}")
        return (
            np.load(paths["item_embs"], mmap_mode="r"),
            np.load(paths["item_ids"], allow_pickle=True),
            np.load(paths["query_embs"]),
            np.load(paths["query_ids"], allow_pickle=True),
        )

    print(f"\n[3] Encoding embeddings with {model_name}...")
    print("  This may download the model on first run and can take a while.")
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device={device}")
    model = SentenceTransformer(model_name, device=device)

    item_texts = [e5_item_text(row, model_name) for _, row in items.iterrows()]
    item_ids = items["item_id"].values
    t0 = time.time()
    item_embs = model.encode(
        item_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"  item encoding done: {time.time() - t0:.1f}s | {item_embs.shape}")
    np.save(paths["item_embs"], item_embs)
    np.save(paths["item_ids"], item_ids)

    train_tids = train["term_id"].unique()
    query_texts = [e5_query_text(tid_to_query.get(tid, ""), model_name) for tid in train_tids]
    t0 = time.time()
    query_embs = model.encode(
        query_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"  query encoding done: {time.time() - t0:.1f}s | {query_embs.shape}")
    np.save(paths["query_embs"], query_embs)
    np.save(paths["query_ids"], train_tids)
    return item_embs, item_ids, query_embs, train_tids


def dense_top_indices_torch(
    item_embs: np.ndarray,
    query_embs: np.ndarray,
    max_topk: int,
    query_batch_size: int,
    device_name: str,
) -> np.ndarray:
    import torch

    device = torch.device(device_name)
    print(f"\n[4] Dense top-{max_topk} search with torch on {device}...")
    t0 = time.time()
    with torch.no_grad():
        item_t = torch.as_tensor(item_embs, dtype=torch.float32, device=device)
        query_t = torch.as_tensor(query_embs, dtype=torch.float32, device=device)
        chunks = []
        for start in range(0, query_t.shape[0], query_batch_size):
            q_batch = query_t[start : start + query_batch_size]
            scores = q_batch @ item_t.T
            top_idx = torch.topk(scores, k=max_topk, dim=1).indices
            chunks.append(top_idx.cpu().numpy())
            print(f"  dense {min(start + query_batch_size, query_t.shape[0])}/{query_t.shape[0]}")

        del item_t, query_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = np.vstack(chunks)
    print(f"  dense search done: {time.time() - t0:.1f}s")
    return result


def dense_top_indices_numpy(
    item_embs: np.ndarray,
    query_embs: np.ndarray,
    max_topk: int,
    query_batch_size: int,
    item_batch_size: int,
) -> np.ndarray:
    print(f"\n[4] Dense top-{max_topk} search with numpy CPU blocks...")
    t0 = time.time()
    all_top = []
    n_items = item_embs.shape[0]

    for q_start in range(0, query_embs.shape[0], query_batch_size):
        q_batch = np.asarray(
            query_embs[q_start : q_start + query_batch_size],
            dtype=np.float32,
        )
        best_scores = np.full((q_batch.shape[0], max_topk), -np.inf, dtype=np.float32)
        best_indices = np.full((q_batch.shape[0], max_topk), -1, dtype=np.int64)

        for item_start in range(0, n_items, item_batch_size):
            item_end = min(item_start + item_batch_size, n_items)
            block = np.asarray(item_embs[item_start:item_end], dtype=np.float32)
            scores = q_batch @ block.T
            local_idx = np.argpartition(scores, -max_topk, axis=1)[:, -max_topk:]
            local_scores = np.take_along_axis(scores, local_idx, axis=1)
            local_indices = local_idx + item_start

            merged_scores = np.concatenate([best_scores, local_scores], axis=1)
            merged_indices = np.concatenate([best_indices, local_indices], axis=1)
            keep = np.argpartition(merged_scores, -max_topk, axis=1)[:, -max_topk:]
            best_scores = np.take_along_axis(merged_scores, keep, axis=1)
            best_indices = np.take_along_axis(merged_indices, keep, axis=1)

        order = np.argsort(best_scores, axis=1)[:, ::-1]
        best_indices = np.take_along_axis(best_indices, order, axis=1)
        all_top.append(best_indices)
        print(f"  dense {min(q_start + query_batch_size, query_embs.shape[0])}/{query_embs.shape[0]}")

    result = np.vstack(all_top)
    print(f"  dense search done: {time.time() - t0:.1f}s")
    return result


def get_holdout_query_embeddings(
    holdout_tids: list[str],
    query_embs: np.ndarray,
    query_ids: np.ndarray,
) -> np.ndarray:
    query_id_to_idx = {str(tid): i for i, tid in enumerate(query_ids)}
    missing = [tid for tid in holdout_tids if tid not in query_id_to_idx]
    if missing:
        raise KeyError(f"{len(missing)} holdout query ids are missing from query embeddings")
    indices = [query_id_to_idx[tid] for tid in holdout_tids]
    return np.asarray(query_embs[indices], dtype=np.float32)


def print_report(rows: list[dict[str, object]]) -> None:
    print("\n" + "=" * 88)
    print("Coverage report")
    print("=" * 88)
    print(
        f"{'method':<18} {'topk':>5} {'pair_cov':>9} {'any_hit':>9} "
        f"{'zero_q':>7} {'avg_hit':>8} {'oracle_f1':>10}"
    )
    for row in rows:
        print(
            f"{str(row['method']):<18} {int(row['top_k']):>5} "
            f"{float(row['pair_coverage']):>9.1%} "
            f"{float(row['query_any_hit']):>9.1%} "
            f"{int(row['zero_hit_queries']):>7} "
            f"{float(row['avg_hits_per_query']):>8.2f} "
            f"{float(row['oracle_pos_f1_fixed14']):>10.3f}"
        )


def save_report(rows: list[dict[str, object]], source_label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS / f"dense_coverage_{source_label}_{stamp}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def print_decision(rows: list[dict[str, object]], dense_name: str) -> None:
    bm25_100 = next((r for r in rows if r["method"] == "bm25_char" and int(r["top_k"]) == 100), None)
    dense_100 = next(
        (r for r in rows if r["method"] == dense_name and int(r["top_k"]) == 100),
        None,
    )
    if not dense_100:
        return

    dense_cov = float(dense_100["pair_coverage"])

    print("\nDecision hint")
    print("-" * 88)
    print(f"Dense top-100 coverage: {dense_cov:.1%}")

    if not bm25_100:
        print("BM25 top-100 coverage : skipped in this run")
        if dense_cov >= 0.60:
            print("Recommendation: Dense coverage is strong enough to start S3 mining.")
        elif dense_cov >= 0.50:
            print("Recommendation: Borderline; try a stronger fresh retriever before S3.")
        else:
            print("Recommendation: This retriever is not enough; try fresh E5-large/BGE-M3 or LLM.")
        return

    bm25_cov = float(bm25_100["pair_coverage"])
    lift = dense_cov - bm25_cov
    print(f"BM25 top-100 coverage : {bm25_cov:.1%}")
    print(f"Lift                  : {lift:+.1%}")

    if dense_cov >= 0.60 and lift >= 0.10:
        print("Recommendation: GO to S3 dense hard-negative mining with this retriever.")
    elif dense_cov >= 0.50 and lift >= 0.05:
        print("Recommendation: Promising, but check top-200 and a stronger fresh retriever.")
    else:
        print("Recommendation: This retriever is not enough; try fresh E5-large/BGE-M3 or LLM.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["cache-e5-v5", "model"], default="cache-e5-v5")
    parser.add_argument("--model-name", default="intfloat/multilingual-e5-large-instruct")
    parser.add_argument("--cache-prefix", default="e5_large_instruct_s2")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--topks", default="20,50,100,200")
    parser.add_argument("--skip-bm25", action="store_true")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--item-batch-size", type=int, default=100_000)
    parser.add_argument("--encode-batch-size", type=int, default=512)
    args = parser.parse_args()

    topks = parse_topks(args.topks)
    max_topk = max(topks)

    print("=" * 88)
    print("S2 dense retrieval coverage diagnostic")
    print("=" * 88)
    print(f"source={args.source} | samples={args.samples} | topks={topks}")

    items, _terms, train, tid_to_query, train_pos = load_data()
    holdout_tids = choose_holdout(train_pos, args.samples)
    print(f"Holdout queries: {len(holdout_tids):,} | seed={SEED}")

    rows: list[dict[str, object]] = []
    if not args.skip_bm25:
        bm25_idx, bm25_item_ids = bm25_top_indices(items, tid_to_query, holdout_tids, max_topk)
        rows.extend(
            evaluate_top_indices(
                "bm25_char",
                bm25_idx,
                bm25_item_ids,
                holdout_tids,
                train_pos,
                topks,
            )
        )

    if args.skip_dense:
        print_report(rows)
        out_path = save_report(rows, "bm25_only")
        print(f"\nSaved report: {out_path}")
        return

    if args.source == "cache-e5-v5":
        dense_name = "dense_e5_v5"
        item_embs, item_ids, query_embs, query_ids = load_cached_e5_v5()
    else:
        dense_name = args.cache_prefix
        item_embs, item_ids, query_embs, query_ids = encode_or_load_model_embeddings(
            items,
            train,
            tid_to_query,
            args.model_name,
            args.cache_prefix,
            args.encode_batch_size,
        )

    holdout_q_embs = get_holdout_query_embeddings(holdout_tids, query_embs, query_ids)

    if args.device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = args.device

    if device == "cuda":
        dense_idx = dense_top_indices_torch(
            item_embs,
            holdout_q_embs,
            max_topk,
            args.query_batch_size,
            "cuda",
        )
    else:
        dense_idx = dense_top_indices_numpy(
            item_embs,
            holdout_q_embs,
            max_topk,
            args.query_batch_size,
            args.item_batch_size,
        )

    rows.extend(
        evaluate_top_indices(
            dense_name,
            dense_idx,
            item_ids,
            holdout_tids,
            train_pos,
            topks,
        )
    )

    print_report(rows)
    out_path = save_report(rows, dense_name)
    print(f"\nSaved report: {out_path}")
    print_decision(rows, dense_name)


if __name__ == "__main__":
    main()
