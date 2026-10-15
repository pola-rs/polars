from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal

ENGINES = ["in-memory", "streaming"]


def order_sensitive_fold() -> pl.Expr:
    # A fold whose result depends on the order of every earlier row.
    return pl.foldv(
        lambda state: (
            pl.when(state >= 1000)
            .then(state - 1000 + pl.col("l"))
            .otherwise(state * 2 + pl.col("l"))
        ),
        initial_state=0,
    )


def order_sensitive_reference(values: list[int]) -> list[int]:
    # The same fold as a Python loop.
    state = 0
    result = []
    for value in values:
        state = state - 1000 + value if state >= 1000 else state * 2 + value
        result.append(state)
    return result


@pytest.fixture
def grouped() -> pl.DataFrame:
    return pl.DataFrame({"g": ["a", "b", "a", "b", "a"], "l": [1, 2, 3, 4, 5]})


def test_foldv_is_named_foldv_in_query_plans(grouped: pl.DataFrame) -> None:
    # Both forms appear by name in the plan instead of as an anonymous function.
    plan = (
        grouped.lazy()
        .select(
            pl.foldv(lambda state: state + pl.col("l"), initial_state=0).alias("f"),
            pl.col("l").foldv(lambda state, value: state + value, initial_state=0),
        )
        .explain()
    )

    assert plan.count(".foldv(") == 2


@pytest.mark.parametrize("engine", ENGINES)
def test_foldv_folds_each_group_separately_in_group_by(
    grouped: pl.DataFrame, engine: pl.EngineType
) -> None:
    # Every group starts again from the initial state and keeps its own row order.
    result = (
        grouped.lazy()
        .group_by("g", maintain_order=True)
        .agg(
            pl.foldv(lambda state: state + pl.col("l"), initial_state=0).alias("f"),
            pl.col("l").foldv(lambda state, value: state + value, initial_state=0),
        )
        .collect(engine=engine)
    )

    expected = pl.DataFrame(
        {"g": ["a", "b"], "f": [[1, 4, 9], [2, 6]], "l": [[1, 4, 9], [2, 6]]}
    )
    assert_frame_equal(result, expected)


def test_foldv_takes_each_groups_own_initial_state(grouped: pl.DataFrame) -> None:
    # An initial state computed per group starts that group's fold.
    result = grouped.group_by("g", maintain_order=True).agg(
        pl.foldv(
            lambda state: state + pl.col("l"),
            initial_state=pl.col("l").first() * 100,
        ).alias("f")
    )

    assert result["f"].to_list() == [[101, 104, 109], [202, 206]]


def test_foldv_folds_string_states_per_group(grouped: pl.DataFrame) -> None:
    # Owned string states are kept apart between groups.
    result = grouped.group_by("g", maintain_order=True).agg(
        pl.foldv(lambda state: state + pl.col("g"), initial_state="").alias("f")
    )

    assert result["f"].to_list() == [["a", "aa", "aaa"], ["b", "bb"]]


@pytest.mark.parametrize("engine", ENGINES)
def test_foldv_folds_each_window_separately_with_over(
    grouped: pl.DataFrame, engine: pl.EngineType
) -> None:
    # Each row receives the state of its own group's fold, in the frame's row order.
    result = (
        grouped.lazy()
        .with_columns(
            pl.foldv(lambda state: state + pl.col("l"), initial_state=0)
            .over("g")
            .alias("f"),
            pl.col("l")
            .foldv(lambda state, value: state + value, initial_state=0)
            .over("g")
            .alias("m"),
        )
        .collect(engine=engine)
    )

    assert result["f"].to_list() == [1, 2, 4, 6, 9]
    assert result["m"].to_list() == [1, 2, 4, 6, 9]


def test_foldv_adds_a_column_with_with_columns(grouped: pl.DataFrame) -> None:
    # The fold has one value per row, so it can sit beside the frame's other columns.
    result = grouped.with_columns(
        pl.foldv(lambda state: state + pl.col("l"), initial_state=0).alias("r")
    )

    assert result.columns == ["g", "l", "r"]
    assert result["r"].to_list() == [1, 3, 6, 10, 15]


@pytest.mark.parametrize("engine", ENGINES)
def test_foldv_folds_only_the_rows_remaining_after_a_filter(
    grouped: pl.DataFrame, engine: pl.EngineType
) -> None:
    # A filter before the fold removes rows from the fold, not just from its output.
    result = (
        grouped.lazy()
        .filter(pl.col("g") == "a")
        .select(pl.foldv(lambda state: state + pl.col("l"), initial_state=0))
        .collect(engine=engine)
    )

    assert result.to_series().to_list() == [1, 4, 9]


@pytest.mark.parametrize("engine", ENGINES)
def test_foldv_carries_the_state_across_chunks_and_streaming_morsels(
    engine: pl.EngineType,
) -> None:
    # Large chunked input is split into several morsels by the streaming engine.
    # The fold must see every row once, in order, with one carried state.
    values = [i % 7 for i in range(200_000)]
    parts = [
        pl.DataFrame({"l": values[i : i + 50_000]}) for i in range(0, 200_000, 50_000)
    ]
    df = pl.concat(parts, rechunk=False)
    assert df.n_chunks() > 1

    result = df.lazy().select(order_sensitive_fold()).collect(engine=engine)

    assert result.to_series().to_list() == order_sensitive_reference(values)


@pytest.mark.parametrize("engine", ENGINES)
def test_foldv_folds_the_rows_of_a_join_in_order(engine: pl.EngineType) -> None:
    # A join before the fold supplies its rows in the join's output order.
    left = pl.LazyFrame({"k": [3, 1, 2, 1], "a": [1, 2, 3, 4]})
    right = pl.LazyFrame({"k": [1, 2, 3], "b": [10, 20, 30]})

    result = (
        left.join(right, on="k", how="left", maintain_order="left")
        .select(
            pl.foldv(lambda state: state + pl.col("a") * pl.col("b"), initial_state=0)
        )
        .collect(engine=engine)
    )

    assert result.to_series().to_list() == [30, 50, 110, 150]


def test_foldv_plans_cannot_be_serialized() -> None:
    # A fold is an opaque function, which Polars cannot serialize.
    lf = pl.LazyFrame({"l": [1]}).select(
        pl.foldv(lambda state: state + pl.col("l"), initial_state=0)
    )

    with pytest.raises(pl.exceptions.ComputeError, match="serialization not supported"):
        lf.serialize()


@pytest.mark.parametrize("engine", ENGINES)
def test_foldv_filter_after_fold_preserves_prior_rows(engine: pl.EngineType) -> None:
    result = (
        pl.LazyFrame({"l": [1, 2, 3, 4]})
        .with_columns(
            pl.foldv(lambda state: state * 10 + pl.col("l"), initial_state=0).alias("f")
        )
        .filter(pl.col("l") > 2)
        .collect(engine=engine)
    )

    assert result["f"].to_list() == [123, 1234]


@pytest.mark.parametrize("engine", ENGINES)
def test_foldv_slice_after_fold_preserves_prior_rows(engine: pl.EngineType) -> None:
    result = (
        pl.LazyFrame({"l": [1, 2, 3, 4]})
        .select(
            pl.col("l").foldv(lambda state, value: state * 10 + value, initial_state=0)
        )
        .slice(2, 1)
        .collect(engine=engine)
    )

    assert result.to_series().to_list() == [123]


def test_foldv_requires_jit_support(monkeypatch: pytest.MonkeyPatch) -> None:
    import polars._plr as plr

    monkeypatch.delattr(plr, "foldv")
    monkeypatch.delattr(plr, "foldv_over")
    with pytest.raises(RuntimeError, match="requires a Polars build with JIT support"):
        pl.foldv(lambda state: state + pl.col("l"), initial_state=0)
    with pytest.raises(RuntimeError, match="requires a Polars build with JIT support"):
        pl.col("l").foldv(lambda state, value: state + value, initial_state=0)
