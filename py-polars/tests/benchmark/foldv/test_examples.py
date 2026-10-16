from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal
from tests.benchmark.foldv import battery, inventory, trading

if TYPE_CHECKING:
    from types import ModuleType

EXAMPLES = [trading, inventory, battery]


@pytest.mark.parametrize(
    ("example", "expected"),
    [
        (trading, [1123.75, 1262.47, 1199.16, 1199.16, 2000.0]),
        (inventory, [3, 3, 2, 0, 0]),
        (battery, [7.0, 10.0, 40.0 / 9.0, 0.0, 1.8]),
    ],
)
def test_hand_checked_examples(example: ModuleType, expected: list[float]) -> None:
    for expression in (example.foldv(), example.best_effort()):
        actual = example.sample().select(expression).to_series()
        assert_series_equal(
            actual,
            pl.Series(actual.name, expected, dtype=actual.dtype),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


@pytest.mark.parametrize("example", EXAMPLES)
@pytest.mark.parametrize("rows", [0, 1, 31, 1000])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_examples_match_baseline(
    example: ModuleType,
    rows: int,
    seed: int,
    engine: pl.EngineType,
) -> None:
    df = example.generate(rows, seed)
    expected = df.select(example.best_effort())
    assert_frame_equal(
        df.select(example.foldv()), expected, rel_tol=1e-12, abs_tol=1e-12
    )
    assert_frame_equal(
        df.lazy().select(example.foldv()).collect(engine=engine),
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize("example", EXAMPLES)
def test_sliced_and_chunked_examples(example: ModuleType) -> None:
    df = example.generate(105, seed=7).slice(3, 100)
    # Different boundaries for each input, retaining nonzero slice offsets.
    df = pl.DataFrame(
        [
            pl.concat([series[: i + 1], series[i + 1 :]], rechunk=False)
            for i, series in enumerate(df)
        ]
    )
    assert_frame_equal(
        df.select(example.foldv()),
        df.select(example.best_effort()),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize("example", EXAMPLES)
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_independent_entities(example: ModuleType, engine: pl.EngineType) -> None:
    df = example.generate(101).with_columns(entity=pl.int_range(pl.len()) % 3)
    # Verify each group's initial state and event order independently of UDF grouping.
    expected = (
        pl.concat(
            [
                group.with_columns(example.best_effort())
                for group in df.with_row_index().partition_by(
                    "entity", maintain_order=True
                )
            ]
        )
        .sort("index")
        .drop("index")
    )
    actual = (
        df.lazy().with_columns(example.foldv().over("entity")).collect(engine=engine)
    )
    assert_frame_equal(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    actual_groups = (
        df.lazy().group_by("entity").agg(example.foldv()).collect(engine=engine)
    )
    expected_groups = expected.group_by("entity").agg(pl.col(expected.columns[-1]))
    assert_frame_equal(actual_groups.sort("entity"), expected_groups.sort("entity"))


def test_trading_resizes_and_stops_when_capital_is_insufficient() -> None:
    df = pl.DataFrame({"price": [100.0] * 5, "return": [-1.0] * 5})
    # One $200 trade, then $100 trades as the bankroll shrinks.
    expected = pl.Series("cash", [798.8, 697.7, 596.6, 495.5, 394.4])
    # Continue until the 25% budget cannot buy a $100 share.
    df = pl.concat([df, df])
    for expression in (trading.foldv(), trading.best_effort()):
        actual = df.select(expression).to_series()
        assert_series_equal(actual.head(5), expected, rel_tol=1e-12, abs_tol=1e-12)
        assert actual.tail(5).to_list() == [actual[4]] * 5


def test_trading_sizes_each_trade_from_the_current_bankroll() -> None:
    # Sizing every trade from the initial bankroll, as a cumulative sum would have to,
    # gives a different state sequence on the benchmark data.
    df = trading.generate(10_000)
    notional = (trading.INITIAL_STATE * 0.25 / pl.col("price")).floor() * pl.col(
        "price"
    )
    profit = notional * pl.col("return") - 1.0 - notional * 0.001
    fixed_size = df.select(trading.INITIAL_STATE + profit.cum_sum()).to_series()
    actual = df.select(trading.foldv()).to_series()
    assert (fixed_size - actual).abs().max() > 1000.0
    assert actual.max() == trading.WITHDRAWAL_THRESHOLD


def test_inventory_rejection_preserves_stock_for_a_later_order() -> None:
    df = pl.DataFrame({"delivery": [0, 0], "demand": [11, 9]})
    assert df.select(inventory.foldv()).to_series().to_list() == [10, 1]


def test_battery_does_not_recover_spilled_energy() -> None:
    df = pl.DataFrame(
        {"solar_kw": [20.0, 0.0], "load_kw": [0.0, 9.0], "hours": [10.0, 1.0]}
    )
    actual = df.select(battery.foldv()).to_series()
    assert actual.to_list() == pytest.approx([10.0, 40.0 / 9.0])
    naive = df.select(
        (battery.INITIAL_STATE + battery.energy_change().cum_sum()).clip(0.0, 10.0)
    )
    assert naive.to_series()[1] == 10.0


@pytest.mark.parametrize("rows", [0, 1, 31, 1000])
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_battery_capacity_comparison(rows: int, engine: pl.EngineType) -> None:
    df = battery.generate(rows)
    expected = df.select(battery.best_effort_sizing())
    assert_frame_equal(
        df.select(battery.foldv_sizing()), expected, rel_tol=1e-12, abs_tol=1e-12
    )
    assert_frame_equal(
        df.lazy().select(battery.foldv_sizing()).collect(engine=engine),
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_battery_candidates_keep_independent_state() -> None:
    # Fill every candidate, then discharge once: each capacity must carry its own state.
    df = pl.DataFrame(
        {"solar_kw": [20.0, 0.0], "load_kw": [0.0, 5.0], "hours": [10.0, 1.0]}
    )
    expected = pl.DataFrame(
        {
            name: [capacity, max(capacity - 5.0 / 0.9, 0.0)]
            for name, capacity in zip(battery.NAMES, battery.CAPACITIES, strict=True)
        }
    )
    for expressions in (battery.foldv_sizing(), battery.best_effort_sizing()):
        assert_frame_equal(
            df.select(expressions), expected, rel_tol=1e-12, abs_tol=1e-12
        )
