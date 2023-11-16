from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pytest

import polars as pl
from polars.testing import assert_frame_equal

pytestmark = pytest.mark.xdist_group("streaming")


def test_streaming_joins() -> None:
    n = 100
    dfa = pd.DataFrame(
        {
            "a": np.random.randint(0, 40, n),
            "b": np.arange(0, n),
        }
    )

    n = 100
    dfb = pd.DataFrame(
        {
            "a": np.random.randint(0, 40, n),
            "b": np.arange(0, n),
        }
    )
    dfa_pl = pl.from_pandas(dfa).sort("a")
    dfb_pl = pl.from_pandas(dfb)

    join_strategies: list[Literal["inner", "left"]] = ["inner", "left"]
    for how in join_strategies:
        pd_result = dfa.merge(dfb, on="a", how=how)
        pd_result.columns = pd.Index(["a", "b", "b_right"])

        pl_result = (
            dfa_pl.lazy()
            .join(dfb_pl.lazy(), on="a", how=how)
            .sort(["a", "b"])
            .collect(streaming=True)
        )

        a = pl.from_pandas(pd_result).with_columns(pl.all().cast(int)).sort(["a", "b"])
        assert_frame_equal(a, pl_result, check_dtype=False)

        pd_result = dfa.merge(dfb, on=["a", "b"], how=how)

        pl_result = (
            dfa_pl.lazy()
            .join(dfb_pl.lazy(), on=["a", "b"], how=how)
            .sort(["a", "b"])
            .collect(streaming=True)
        )

        # we cast to integer because pandas joins creates floats
        a = pl.from_pandas(pd_result).with_columns(pl.all().cast(int)).sort(["a", "b"])
        assert_frame_equal(a, pl_result, check_dtype=False)


def test_sorted_flag_after_streaming_join() -> None:
    # streaming left join
    df1 = pl.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 6]}).set_sorted("x")
    df2 = pl.DataFrame({"x": [4, 2, 3, 1], "z": [1, 4, 9, 1]})
    assert (
        df1.lazy()
        .join(df2.lazy(), on="x", how="left")
        .collect(streaming=True)["x"]
        .flags["SORTED_ASC"]
    )


def test_streaming_cross_join_empty() -> None:
    df1 = pl.LazyFrame(
        data={
            "col1": ["a"],
        }
    )

    df2 = pl.LazyFrame(
        data={
            "col1": [],
        },
        schema={
            "col1": str,
        },
    )

    out = df1.join(
        df2,
        how="cross",
        on="col1",
    ).collect(streaming=True)
    assert out.shape == (0, 2)
    assert out.columns == ["col1", "col1_right"]


def test_streaming_join_rechunk_12498() -> None:
    rows = pl.int_range(0, 2)

    a = pl.select(A=rows).lazy()
    b = pl.select(B=rows).lazy()

    q = a.join(b, how="cross")
    assert q.collect(streaming=True).to_dict(as_series=False) == {
        "A": [0, 1, 0, 1],
        "B": [0, 0, 1, 1],
    }
