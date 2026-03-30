"""
Realistic benchmark for PlaceholderScan: text feature engineering pipeline.

Pipeline: keyword contains -> recency bucketing -> group_by.agg -> prefix

Compares two approaches:
  1. Direct: build full pipeline from scratch each iteration
  2. PlaceholderScan: build template once, bind different data each iteration
"""
from __future__ import annotations

import random
import time
from collections import Counter

import pytest

import polars as pl
from polars.testing import assert_frame_equal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_SCHEMA = {
    "user_id": pl.Int64,
    "recency": pl.Int64,
    "text": pl.String,
}

RECENCY_BUCKETS = ["recent", "mid", "all"]

# ~1000 word vocabulary pool for synthetic text generation
VOCAB_POOL = (
    "account active address admin advance alert amount annual api app apply approve "
    "archive audit balance bank batch billing block bonus budget bulk cache cancel card "
    "cash catalog change charge check claim clear click close code collect comment commit "
    "complete config confirm connect contact convert copy count coupon create credit "
    "custom daily data date debit decline default delay delete deliver demo deploy "
    "deposit detail device direct disable discount display done download draft due "
    "duration edit email enable encrypt end engine entry error estimate event execute "
    "expire export extend fail feature fee fetch field file filter final flag flow "
    "folder follow force format forward free fund generate global grant group guide "
    "handle hash header health help hide history hold host icon image import inactive "
    "index info initial input insert install instant integrate interest internal invalid "
    "invite invoice issue item job journal keep key label language last launch layer "
    "lead leave legacy level library limit link list load local lock log login lookup "
    "maintain manage manual map mark match media merge message method metric migrate "
    "minimum mobile mode model modify module money monitor month move multi name network "
    "new next node note notice notify number object offer offset online open operate "
    "option order origin output override owner package page paid panel parameter parse "
    "partner password path pause pay penalty pending percent period permit person phone "
    "plan platform point policy pool port position post power prefix premium prepare "
    "preview price primary print priority private process product profile program project "
    "promote proof protect provide publish purchase push quality query queue quick quota "
    "random range rate reach read ready reason receive recent record recover redirect "
    "reduce refer refresh refund register reject release remain remind remove renew rent "
    "repair repeat replace report request require reserve reset resolve resource respond "
    "restore result retail retain retrieve return review revise reward role rollback root "
    "route rule safe salary sample save scale scan schedule score screen script search "
    "secure select send sequence server service session setup share shift show sign simple "
    "site size skip smart sort source space split stable stage standard start state static "
    "status step stock stop store stream string strong submit success summary support "
    "suspend switch sync system table tag target task team template tenant term test text "
    "theme ticket time title token tool total touch track trade traffic train transfer "
    "transform trend trial trigger trust type unique unit update upgrade upload usage "
    "user valid value vendor verify version view virtual visit volume wait wallet warning "
    "watch webhook weight window withdraw work write yearly yield zero zone"
).split()


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------


def generate_data(
    n_rows: int, n_users: int, seed: int = 42
) -> pl.DataFrame:
    """Generate synthetic text data with user_id, recency, and text columns."""
    rng = random.Random(seed)

    user_ids = [rng.randint(0, n_users - 1) for _ in range(n_rows)]
    recencies = [rng.randint(0, 365) for _ in range(n_rows)]
    texts = [
        " ".join(rng.choices(VOCAB_POOL, k=rng.randint(5, 30)))
        for _ in range(n_rows)
    ]

    return pl.DataFrame({
        "user_id": user_ids,
        "recency": recencies,
        "text": texts,
    }).cast({"user_id": pl.Int64, "recency": pl.Int64})


def extract_top_keywords(texts: list[str], n: int = 500) -> list[str]:
    """Extract top-N keywords by frequency from text corpus."""
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(text.split())
    return [word for word, _ in counter.most_common(n)]


# ---------------------------------------------------------------------------
# Pipeline Helpers
# ---------------------------------------------------------------------------


def get_recency_bucket_columns() -> list[pl.Expr]:
    """Create boolean columns for recency buckets."""
    r = pl.col("recency")
    return [
        (r <= 30).alias("recent"),   # last 30 days
        (r <= 90).alias("mid"),      # last 90 days
        pl.lit(True).alias("all"),   # everything
    ]


def get_agg_list(
    bucket_name: str, keyword: str | None = None
) -> list[pl.Expr]:
    """Build aggregation expressions for a given bucket and optional keyword."""
    text = pl.col("text")
    cond = pl.col(bucket_name)
    prefix = bucket_name

    if keyword is not None:
        cond = cond & pl.col(f"kw_{keyword}")
        prefix += f"_{keyword}"

    text_filtered = pl.when(cond).then(text)
    word_count = text_filtered.str.split(" ").list.len()

    return [
        text_filtered.count().alias(f"{prefix}_cnt"),
        word_count.sum().alias(f"{prefix}_word_sum"),
    ]


# ---------------------------------------------------------------------------
# Pipeline Builders
# ---------------------------------------------------------------------------


def build_text_pipeline(
    lf: pl.LazyFrame, keywords: list[str]
) -> pl.LazyFrame:
    """
    Full text feature pipeline. Works with both concrete LazyFrame and placeholder.

    Steps:
    1. Keyword contains (boolean columns)
    2. Recency bucketing (boolean columns)
    3. Group by user_id with wide aggregation
    4. Rename with prefix
    """
    # 1. Keyword contains
    lf = lf.with_columns(
        *[
            pl.col("text").str.contains(kw).alias(f"kw_{kw}")
            for kw in keywords
        ]
    )

    # 2. Recency buckets
    lf = lf.with_columns(*get_recency_bucket_columns())

    # 3. Group by + aggregate
    agg_list: list[pl.Expr] = [
        pl.len().alias("row_cnt"),
        pl.col("text").str.split(" ").list.len().mean().alias("avg_word_len"),
    ]
    for bucket in RECENCY_BUCKETS:
        agg_list.extend(get_agg_list(bucket))
        for kw in keywords:
            agg_list.extend(get_agg_list(bucket, keyword=kw))

    features = lf.group_by("user_id").agg(*agg_list)

    # 4. Rename with prefix
    rename_map = {
        c: f"feat_{c}"
        for c in features.collect_schema().names()
        if c != "user_id"
    }
    return features.rename(rename_map)


def build_text_template(keywords: list[str]) -> pl.LazyFrame:
    """Build the pipeline template once using PlaceholderScan."""
    lf = pl.LazyFrame.placeholder("input", INPUT_SCHEMA)
    return build_text_pipeline(lf, keywords)


# ---------------------------------------------------------------------------
# Correctness Tests
# ---------------------------------------------------------------------------


def test_pipeline_correctness() -> None:
    """Direct and PlaceholderScan approaches produce identical results."""
    df = generate_data(n_rows=1000, n_users=50)
    keywords = extract_top_keywords(df["text"].to_list(), n=20)

    direct_result = (
        build_text_pipeline(df.lazy(), keywords).sort("user_id").collect()
    )

    template = build_text_template(keywords)
    placeholder_result = (
        template.bind({"input": df.lazy()}).sort("user_id").collect()
    )

    assert_frame_equal(direct_result, placeholder_result)


def test_template_reuse_correctness() -> None:
    """Same template, different data produces valid but different results."""
    df_ref = generate_data(n_rows=2000, n_users=100, seed=0)
    keywords = extract_top_keywords(df_ref["text"].to_list(), n=30)
    template = build_text_template(keywords)

    df1 = generate_data(n_rows=500, n_users=20, seed=1)
    df2 = generate_data(n_rows=800, n_users=30, seed=2)

    r1 = template.bind({"input": df1.lazy()}).sort("user_id").collect()
    r2 = template.bind({"input": df2.lazy()}).sort("user_id").collect()

    assert r1.columns == r2.columns
    assert r1.height != r2.height


def test_output_schema_width() -> None:
    """Verify output column count matches expected formula."""
    n_keywords = 50
    df = generate_data(n_rows=200, n_users=20)
    keywords = extract_top_keywords(df["text"].to_list(), n=n_keywords)

    template = build_text_template(keywords)
    result = template.bind({"input": df.lazy()}).collect()

    # user_id(1) + base_aggs(2) + buckets * (1 + n_keywords) * aggs_per_combo(2)
    n_buckets = len(RECENCY_BUCKETS)
    expected_cols = 1 + 2 + n_buckets * (1 + n_keywords) * 2
    assert result.width == expected_cols, (
        f"Expected {expected_cols} columns, got {result.width}"
    )


# ---------------------------------------------------------------------------
# Benchmark Tests
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.parametrize(
    ("n_rows", "n_keywords", "n_iterations"),
    [
        (1_000, 50, 5),
        (10_000, 100, 5),
        (10_000, 500, 3),
    ],
    ids=["1K-50kw", "10K-100kw", "10K-500kw"],
)
def test_benchmark_direct_vs_placeholder(
    n_rows: int, n_keywords: int, n_iterations: int
) -> None:
    """
    Compare direct pipeline construction vs PlaceholderScan template reuse.

    Measures execution time only (data generation and vocab extraction excluded).
    """
    # --- Setup (not measured) ---
    ref_data = generate_data(n_rows=n_rows * 2, n_users=max(50, n_rows // 20))
    keywords = extract_top_keywords(ref_data["text"].to_list(), n=n_keywords)
    datasets = [
        generate_data(
            n_rows=n_rows,
            n_users=max(50, n_rows // 20),
            seed=i + 100,
        )
        for i in range(n_iterations)
    ]

    # --- Direct approach: rebuild pipeline each time ---
    direct_times = []
    for ds in datasets:
        t0 = time.perf_counter()
        build_text_pipeline(ds.lazy(), keywords).collect()
        direct_times.append(time.perf_counter() - t0)

    # --- PlaceholderScan approach ---
    t0 = time.perf_counter()
    template = build_text_template(keywords)
    template_build_time = time.perf_counter() - t0

    placeholder_times = []
    for ds in datasets:
        t0 = time.perf_counter()
        template.bind({"input": ds.lazy()}).collect()
        placeholder_times.append(time.perf_counter() - t0)

    # --- Report ---
    direct_total = sum(direct_times)
    ph_bind_total = sum(placeholder_times)
    ph_total = template_build_time + ph_bind_total
    direct_avg = direct_total / n_iterations
    ph_avg = ph_bind_total / n_iterations

    print(f"\n{'=' * 70}")
    print(
        f"Benchmark: n_rows={n_rows}, n_keywords={n_keywords}, "
        f"n_iter={n_iterations}"
    )
    print(f"{'=' * 70}")
    print(f"Direct approach:")
    print(f"  Total:   {direct_total:.4f}s")
    print(f"  Average: {direct_avg:.4f}s per iteration")
    print(f"PlaceholderScan approach:")
    print(f"  Template build:  {template_build_time:.4f}s (one-time)")
    print(
        f"  Bind+collect:    {ph_bind_total:.4f}s "
        f"({n_iterations} iterations)"
    )
    print(f"  Average:         {ph_avg:.4f}s per bind+collect")
    print(f"  Total:           {ph_total:.4f}s (including template build)")
    if ph_avg > 0:
        print(f"Speedup (bind+collect vs direct): {direct_avg / ph_avg:.2f}x")
    if ph_total > 0:
        print(f"Speedup (total):                  {direct_total / ph_total:.2f}x")
    print(f"{'=' * 70}")

    # Correctness check on last dataset
    direct_result = (
        build_text_pipeline(datasets[-1].lazy(), keywords)
        .sort("user_id")
        .collect()
    )
    placeholder_result = (
        template.bind({"input": datasets[-1].lazy()}).sort("user_id").collect()
    )
    assert_frame_equal(direct_result, placeholder_result)


# ---------------------------------------------------------------------------
# Stress Test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_large_scale() -> None:
    """Stress test with 100K rows and 500 keywords."""
    df = generate_data(n_rows=100_000, n_users=5000, seed=99)
    keywords = extract_top_keywords(df["text"].to_list(), n=500)

    template = build_text_template(keywords)
    result = template.bind({"input": df.lazy()}).collect()

    assert result.height <= 5000
    assert result.height > 0
    # Verify schema width
    n_kw = len(keywords)
    n_buckets = len(RECENCY_BUCKETS)
    expected_cols = 1 + 2 + n_buckets * (1 + n_kw) * 2
    assert result.width == expected_cols, (
        f"Expected {expected_cols} columns (n_kw={n_kw}), got {result.width}"
    )
