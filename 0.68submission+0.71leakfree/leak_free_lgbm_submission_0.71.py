from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


SEED = 42
UNKNOWN = "unknown"

TOKEN_PATTERN = r"(?u)\b[0-9a-z\u00e7\u011f\u0131\u00f6\u015f\u00fc]{2,}\b"
TOKEN_SPLIT_RE = re.compile(r"[^0-9a-z\u00e7\u011f\u0131\u00f6\u015f\u00fc]+")

KADIN = "kad\u0131n"
KIZ = "k\u0131z"
ERKEK = "erkek"
BAYAN = "bayan"
BEBEK = "bebek"
COCUK = "\u00e7ocuk"
GENC = "gen\u00e7"
YETISKIN = "yeti\u015fkin"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def normalize_value(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = value.replace("\u0130", "i").replace("I", "\u0131")
    value = value.lower().replace("i\u0307", "i")
    return value


def normalize_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace("\u0130", "i", regex=False)
        .str.replace("I", "\u0131", regex=False)
        .str.lower()
        .str.replace("i\u0307", "i", regex=False)
    )


def tokenize(value: object) -> set[str]:
    text = normalize_value(value)
    return {token for token in TOKEN_SPLIT_RE.split(text) if len(token) >= 2}


def read_core_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log("Loading train, terms and item catalog")
    train = pd.read_csv(data_dir / "training_pairs.csv", dtype=str)
    train["label"] = 1

    terms = pd.read_csv(data_dir / "terms.csv", dtype=str)
    terms["query"] = normalize_series(terms["query"])

    item_cols = ["item_id", "title", "category", "brand", "gender", "age_group"]
    items = pd.read_csv(data_dir / "items.csv", usecols=item_cols, dtype=str)
    for col in ["title", "category", "brand", "gender", "age_group"]:
        items[col] = normalize_series(items[col])
        items.loc[items[col].eq(""), col] = UNKNOWN

    items["main_category"] = items["category"].str.split("/", n=1).str[0].fillna(UNKNOWN)
    items["category_text"] = items["category"].str.replace("/", " ", regex=False)
    items["item_text"] = (
        items["title"]
        + " "
        + items["category_text"]
        + " "
        + items["brand"]
        + " "
        + items["gender"]
        + " "
        + items["age_group"]
    )
    return train, terms, items


def build_group_index(items: pd.DataFrame, cols: list[str]) -> dict[object, np.ndarray]:
    if len(cols) == 1:
        return {key: values for key, values in items.groupby(cols[0], sort=False).indices.items()}
    return {key: values for key, values in items.groupby(cols, sort=False).indices.items()}


def query_gender(query: str) -> str:
    has_female = (KADIN in query) or (KIZ in query) or (BAYAN in query)
    has_male = ERKEK in query
    if has_female and not has_male:
        return KADIN
    if has_male and not has_female:
        return ERKEK
    return ""


def query_age(query: str) -> str:
    if BEBEK in query:
        return BEBEK
    if COCUK in query:
        return COCUK
    if GENC in query:
        return GENC
    if YETISKIN in query:
        return YETISKIN
    return ""


def sample_index(
    pool: np.ndarray | None,
    term_id: str,
    pos_item_id: str,
    item_ids: np.ndarray,
    positive_keys: set[str],
    used_keys: set[str],
    rng: np.random.Generator,
    max_tries: int = 40,
) -> int | None:
    if pool is None or len(pool) == 0:
        return None
    for _ in range(max_tries):
        idx = int(pool[rng.integers(0, len(pool))])
        item_id = item_ids[idx]
        key = term_id + "\t" + item_id
        if item_id != pos_item_id and key not in positive_keys and key not in used_keys:
            return idx
    return None


def sample_global_different_main(
    current_main: str,
    term_id: str,
    pos_item_id: str,
    item_ids: np.ndarray,
    item_mains: np.ndarray,
    positive_keys: set[str],
    used_keys: set[str],
    rng: np.random.Generator,
    max_tries: int = 80,
) -> int | None:
    n_items = len(item_ids)
    for _ in range(max_tries):
        idx = int(rng.integers(0, n_items))
        item_id = item_ids[idx]
        key = term_id + "\t" + item_id
        if (
            item_mains[idx] != current_main
            and item_id != pos_item_id
            and key not in positive_keys
            and key not in used_keys
        ):
            return idx
    return None


def build_leak_free_negatives(
    train: pd.DataFrame,
    terms: pd.DataFrame,
    items: pd.DataFrame,
    negatives_per_positive: int,
    seed: int,
) -> pd.DataFrame:
    log("Building leak-free negatives from training terms and item catalog")
    pos = train[["id", "term_id", "item_id", "label"]].merge(terms, on="term_id", how="left")
    pos = pos.merge(
        items[["item_id", "main_category", "gender", "age_group"]],
        on="item_id",
        how="left",
    )

    item_ids = items["item_id"].to_numpy()
    item_mains = items["main_category"].to_numpy()
    by_main = build_group_index(items, ["main_category"])
    by_gender = build_group_index(items, ["gender"])
    by_main_gender = build_group_index(items, ["main_category", "gender"])
    by_age = build_group_index(items, ["age_group"])
    by_main_age = build_group_index(items, ["main_category", "age_group"])

    positive_keys = set(train["term_id"].astype(str) + "\t" + train["item_id"].astype(str))
    used_keys: set[str] = set()
    rng = np.random.default_rng(seed)

    neg_term_ids: list[str] = []
    neg_item_ids: list[str] = []
    neg_sources: list[str] = []

    for row in pos.itertuples(index=False):
        term_id = row.term_id
        pos_item_id = row.item_id
        main = row.main_category if isinstance(row.main_category, str) else UNKNOWN
        query = row.query if isinstance(row.query, str) else ""

        selected: list[tuple[int, str]] = []

        gender = query_gender(query)
        if gender == ERKEK:
            pool = by_main_gender.get((main, KADIN))
            idx = sample_index(pool, term_id, pos_item_id, item_ids, positive_keys, used_keys, rng)
            if idx is None:
                idx = sample_index(by_gender.get(KADIN), term_id, pos_item_id, item_ids, positive_keys, used_keys, rng)
            if idx is not None:
                selected.append((idx, "gender_conflict"))
        elif gender == KADIN:
            pool = by_main_gender.get((main, ERKEK))
            idx = sample_index(pool, term_id, pos_item_id, item_ids, positive_keys, used_keys, rng)
            if idx is None:
                idx = sample_index(by_gender.get(ERKEK), term_id, pos_item_id, item_ids, positive_keys, used_keys, rng)
            if idx is not None:
                selected.append((idx, "gender_conflict"))

        age = query_age(query)
        if len(selected) < negatives_per_positive and age in {BEBEK, COCUK}:
            age_pool = by_main_age.get((main, YETISKIN))
            idx = sample_index(age_pool, term_id, pos_item_id, item_ids, positive_keys, used_keys, rng)
            if idx is None:
                idx = sample_index(by_age.get(YETISKIN), term_id, pos_item_id, item_ids, positive_keys, used_keys, rng)
            if idx is not None:
                selected.append((idx, "age_conflict"))
        elif len(selected) < negatives_per_positive and age in {GENC, YETISKIN}:
            for conflict_age in (BEBEK, COCUK):
                age_pool = by_main_age.get((main, conflict_age))
                idx = sample_index(age_pool, term_id, pos_item_id, item_ids, positive_keys, used_keys, rng)
                if idx is None:
                    idx = sample_index(by_age.get(conflict_age), term_id, pos_item_id, item_ids, positive_keys, used_keys, rng)
                if idx is not None:
                    selected.append((idx, "age_conflict"))
                    break

        if len(selected) < negatives_per_positive:
            idx = sample_index(by_main.get(main), term_id, pos_item_id, item_ids, positive_keys, used_keys, rng)
            if idx is not None:
                selected.append((idx, "same_main_category"))

        while len(selected) < negatives_per_positive:
            idx = sample_global_different_main(
                main,
                term_id,
                pos_item_id,
                item_ids,
                item_mains,
                positive_keys,
                used_keys,
                rng,
            )
            if idx is None:
                break
            selected.append((idx, "different_main_category"))

        for idx, source in selected[:negatives_per_positive]:
            item_id = item_ids[idx]
            key = term_id + "\t" + item_id
            if key in used_keys or key in positive_keys:
                continue
            used_keys.add(key)
            neg_term_ids.append(term_id)
            neg_item_ids.append(item_id)
            neg_sources.append(source)

    negatives = pd.DataFrame(
        {
            "id": [f"NEG_{i:08d}" for i in range(len(neg_term_ids))],
            "term_id": neg_term_ids,
            "item_id": neg_item_ids,
            "label": 0,
            "negative_source": neg_sources,
        }
    )
    log(f"Generated {len(negatives):,} negatives; source mix: {negatives['negative_source'].value_counts().to_dict()}")
    return negatives.drop(columns=["negative_source"])


def assemble_training_frame(
    train: pd.DataFrame,
    negatives: pd.DataFrame,
    terms: pd.DataFrame,
    items: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    train_ready = pd.concat(
        [train[["id", "term_id", "item_id", "label"]], negatives[["id", "term_id", "item_id", "label"]]],
        ignore_index=True,
    )
    train_ready = train_ready.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    assert not train_ready["id"].astype(str).str.startswith("TST_").any(), "submission ids leaked into train"

    df = train_ready.merge(terms, on="term_id", how="left")
    df = df.merge(items, on="item_id", how="left")
    df["query"] = df["query"].fillna("")
    df["title"] = df["title"].fillna(UNKNOWN)
    df["category"] = df["category"].fillna(UNKNOWN)
    df["brand"] = df["brand"].fillna(UNKNOWN)
    df["gender"] = df["gender"].fillna(UNKNOWN)
    df["age_group"] = df["age_group"].fillna(UNKNOWN)
    df["main_category"] = df["main_category"].fillna(UNKNOWN)
    df["category_text"] = df["category_text"].fillna(UNKNOWN)
    df["item_text"] = df["item_text"].fillna(UNKNOWN)
    return df


def fit_text_vectorizers(df: pd.DataFrame) -> dict[str, object]:
    log("Fitting text vectorizers on leak-free training rows")
    train_corpus = pd.concat(
        [df["query"], df["title"], df["category_text"], df["item_text"]],
        ignore_index=True,
    ).drop_duplicates()

    word_vec = TfidfVectorizer(
        lowercase=False,
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 2),
        min_df=2,
        max_features=180_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    word_vec.fit(train_corpus)

    char_corpus = pd.concat([df["query"], df["title"]], ignore_index=True).drop_duplicates()
    char_vec = TfidfVectorizer(
        lowercase=False,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=70_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char_vec.fit(char_corpus)

    count_vec = CountVectorizer(
        lowercase=False,
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 2),
        vocabulary=word_vec.vocabulary_,
        binary=True,
        dtype=np.float32,
    )
    log(f"Word vocab: {len(word_vec.vocabulary_):,}; char vocab: {len(char_vec.vocabulary_):,}")
    return {"word": word_vec, "char": char_vec, "count": count_vec}


def fit_category_encoders(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    encoders: dict[str, dict[str, int]] = {}
    for col in ["main_category", "gender", "age_group"]:
        values = sorted(v for v in df[col].fillna(UNKNOWN).unique() if isinstance(v, str))
        encoders[col] = {value: idx for idx, value in enumerate(values)}
    return encoders


def pairwise_dot(left: sparse.spmatrix, right: sparse.spmatrix) -> np.ndarray:
    return np.asarray(left.multiply(right).sum(axis=1)).ravel().astype(np.float32)


def row_contains(values: np.ndarray, needles: np.ndarray, min_len: int = 2) -> np.ndarray:
    return np.fromiter(
        (
            1.0 if isinstance(n, str) and len(n) >= min_len and n != UNKNOWN and n in v else 0.0
            for v, n in zip(values, needles)
        ),
        dtype=np.float32,
        count=len(values),
    )


def build_features(
    df: pd.DataFrame,
    vectorizers: dict[str, object],
    encoders: dict[str, dict[str, int]],
) -> pd.DataFrame:
    n_rows = len(df)
    query = df["query"].fillna("").astype(str)
    title = df["title"].fillna(UNKNOWN).astype(str)
    category_text = df["category_text"].fillna(UNKNOWN).astype(str)
    item_text = df["item_text"].fillna(UNKNOWN).astype(str)

    word_vec: TfidfVectorizer = vectorizers["word"]  # type: ignore[assignment]
    char_vec: TfidfVectorizer = vectorizers["char"]  # type: ignore[assignment]
    count_vec: CountVectorizer = vectorizers["count"]  # type: ignore[assignment]

    q_word = word_vec.transform(query)
    title_word = word_vec.transform(title)
    cat_word = word_vec.transform(category_text)
    item_word = word_vec.transform(item_text)

    q_count = count_vec.transform(query)
    item_count = count_vec.transform(item_text)
    overlap = pairwise_dot(q_count, item_count)
    q_terms = q_count.getnnz(axis=1).astype(np.float32)
    item_terms = item_count.getnnz(axis=1).astype(np.float32)

    q_char = char_vec.transform(query)
    title_char = char_vec.transform(title)

    query_values = query.to_numpy()
    title_values = title.to_numpy()
    brand_values = df["brand"].fillna(UNKNOWN).astype(str).to_numpy()
    gender_values = df["gender"].fillna(UNKNOWN).astype(str).to_numpy()
    age_values = df["age_group"].fillna(UNKNOWN).astype(str).to_numpy()

    q_len = query.str.len().to_numpy(dtype=np.float32)
    title_len = title.str.len().to_numpy(dtype=np.float32)
    min_len = np.minimum(q_len, title_len)
    max_len = np.maximum(q_len, title_len)

    q_has_male = query.str.contains(ERKEK, regex=False).to_numpy()
    q_has_female = (
        query.str.contains(KADIN, regex=False)
        | query.str.contains(KIZ, regex=False)
        | query.str.contains(BAYAN, regex=False)
    ).to_numpy()
    q_has_baby = query.str.contains(BEBEK, regex=False).to_numpy()
    q_has_child = query.str.contains(COCUK, regex=False).to_numpy()
    q_has_young = query.str.contains(GENC, regex=False).to_numpy()
    q_has_adult = query.str.contains(YETISKIN, regex=False).to_numpy()

    gender_match = ((q_has_male & (gender_values == ERKEK)) | (q_has_female & (gender_values == KADIN))).astype(np.float32)
    gender_conflict = ((q_has_male & (gender_values == KADIN)) | (q_has_female & (gender_values == ERKEK))).astype(np.float32)
    age_strings = age_values.astype(str)
    age_has_baby = np.char.find(age_strings, BEBEK) >= 0
    age_has_child = np.char.find(age_strings, COCUK) >= 0
    age_has_young = np.char.find(age_strings, GENC) >= 0
    age_has_adult = np.char.find(age_strings, YETISKIN) >= 0
    age_match = (
        (q_has_baby & age_has_baby)
        | (q_has_child & age_has_child)
        | (q_has_young & age_has_young)
        | (q_has_adult & age_has_adult)
    ).astype(np.float32)
    age_conflict = (
        ((q_has_baby | q_has_child) & ((age_values == YETISKIN) | (age_values == GENC)))
        | ((q_has_adult | q_has_young) & age_has_baby)
    ).astype(np.float32)

    features = pd.DataFrame(index=np.arange(n_rows))
    features["word_cos_title"] = pairwise_dot(q_word, title_word)
    features["word_cos_category"] = pairwise_dot(q_word, cat_word)
    features["word_cos_item"] = pairwise_dot(q_word, item_word)
    features["char_cos_title"] = pairwise_dot(q_char, title_char)
    features["overlap_count"] = overlap
    features["overlap_query_ratio"] = np.divide(overlap, q_terms + 1e-6).astype(np.float32)
    features["overlap_jaccard"] = np.divide(overlap, q_terms + item_terms - overlap + 1e-6).astype(np.float32)
    features["query_len"] = q_len
    features["title_len"] = title_len
    features["len_abs_diff"] = np.abs(q_len - title_len).astype(np.float32)
    features["len_ratio"] = np.divide(min_len, max_len + 1e-6).astype(np.float32)
    features["brand_in_query"] = row_contains(query_values, brand_values, min_len=2)
    features["query_in_title"] = row_contains(title_values, query_values, min_len=4)
    features["gender_match"] = gender_match
    features["gender_conflict"] = gender_conflict
    features["age_match"] = age_match
    features["age_conflict"] = age_conflict

    for col in ["main_category", "gender", "age_group"]:
        mapping = encoders[col]
        features[f"{col}_code"] = df[col].fillna(UNKNOWN).map(mapping).fillna(-1).astype(np.int32)

    return features


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    categorical_features: list[str],
    seed: int,
) -> tuple[list[lgb.LGBMClassifier], np.ndarray, float, float]:
    log("Training LightGBM with GroupKFold")
    models: list[lgb.LGBMClassifier] = []
    oof = np.zeros(len(X), dtype=np.float32)
    gkf = GroupKFold(n_splits=5)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=1800,
            learning_rate=0.035,
            num_leaves=96,
            max_depth=-1,
            min_child_samples=45,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=seed + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            eval_metric="binary_logloss",
            categorical_feature=categorical_features,
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(period=0)],
        )
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        models.append(model)
        fold_pred = (oof[val_idx] > 0.5).astype(np.int8)
        fold_f1 = f1_score(y.iloc[val_idx], fold_pred, average="macro")
        log(f"Fold {fold}: best_iter={model.best_iteration_}, macro_f1@0.50={fold_f1:.5f}")

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.15, 0.85, 141):
        score = f1_score(y, (oof > threshold).astype(np.int8), average="macro")
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    log(f"Best OOF threshold={best_threshold:.3f}; macro_f1={best_f1:.5f}")
    return models, oof, best_threshold, best_f1


def prepare_pair_chunk(chunk: pd.DataFrame, terms: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    df = chunk.merge(terms, on="term_id", how="left")
    df = df.merge(items, on="item_id", how="left")
    for col in ["query", "title", "category", "brand", "gender", "age_group", "main_category", "category_text", "item_text"]:
        df[col] = df[col].fillna(UNKNOWN if col != "query" else "")
    return df


def predict_submission(
    data_dir: Path,
    output_path: Path,
    terms: pd.DataFrame,
    items: pd.DataFrame,
    vectorizers: dict[str, object],
    encoders: dict[str, dict[str, int]],
    models: list[lgb.LGBMClassifier],
    threshold: float,
    chunk_size: int,
) -> dict[str, int | float]:
    log("Writing submission in chunks")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp.csv")
    if temp_path.exists():
        temp_path.unlink()

    total_rows = 0
    positives = 0
    first = True
    for chunk_no, chunk in enumerate(
        pd.read_csv(data_dir / "submission_pairs.csv", dtype=str, chunksize=chunk_size),
        start=1,
    ):
        ids = chunk["id"].copy()
        df = prepare_pair_chunk(chunk, terms, items)
        X_chunk = build_features(df, vectorizers, encoders)

        proba = np.zeros(len(X_chunk), dtype=np.float32)
        for model in models:
            proba += model.predict_proba(X_chunk)[:, 1].astype(np.float32) / len(models)
        pred = (proba > threshold).astype(np.int8)

        out = pd.DataFrame({"id": ids.to_numpy(), "prediction": pred})
        out.to_csv(temp_path, index=False, mode="w" if first else "a", header=first)
        first = False

        total_rows += len(out)
        positives += int(pred.sum())
        log(f"Chunk {chunk_no}: rows={total_rows:,}, positives={positives:,} ({positives / total_rows:.4f})")

        del chunk, df, X_chunk, proba, pred, out
        gc.collect()

    if output_path.exists():
        output_path.unlink()
    temp_path.rename(output_path)
    return {"rows": total_rows, "positives": positives, "positive_rate": positives / max(total_rows, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Leak-free Trendyol submission pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("trendyol-e-ticaret-yarismasi-2026-kaggle"))
    parser.add_argument("--output", type=Path, default=Path("0.68submission") / "submission_leak_free_lgbm.csv")
    parser.add_argument("--metrics-output", type=Path, default=Path("0.68submission") / "leak_free_lgbm_metrics.json")
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    np.random.seed(args.seed)
    start = time.time()

    train, terms, items = read_core_tables(args.data_dir)
    negatives = build_leak_free_negatives(train, terms, items, args.negatives_per_positive, args.seed)
    train_df = assemble_training_frame(train, negatives, terms, items, args.seed)
    log(f"Training rows: {len(train_df):,}; label mix: {train_df['label'].value_counts().to_dict()}")

    vectorizers = fit_text_vectorizers(train_df)
    encoders = fit_category_encoders(train_df)
    X = build_features(train_df, vectorizers, encoders)
    y = train_df["label"].astype(np.int8)
    groups = train_df["term_id"]
    categorical_features = ["main_category_code", "gender_code", "age_group_code"]

    models, oof, best_threshold, best_f1 = train_models(X, y, groups, categorical_features, args.seed)
    prediction_stats = predict_submission(
        args.data_dir,
        args.output,
        terms,
        items,
        vectorizers,
        encoders,
        models,
        best_threshold,
        args.chunk_size,
    )

    metrics = {
        "seed": args.seed,
        "negatives_per_positive": args.negatives_per_positive,
        "training_rows": int(len(train_df)),
        "label_counts": {str(k): int(v) for k, v in train_df["label"].value_counts().to_dict().items()},
        "best_oof_threshold": best_threshold,
        "best_oof_macro_f1": best_f1,
        "oof_positive_rate": float((oof > best_threshold).mean()),
        "submission": prediction_stats,
        "elapsed_seconds": round(time.time() - start, 2),
        "leak_fix": "Negatives are generated only from training term_ids plus the item catalog; submission_pairs.csv is read only during inference.",
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"Done. Submission: {args.output}; metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()
