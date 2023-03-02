from __future__ import annotations

import itertools
from dataclasses import dataclass
from decimal import Decimal as D
from typing import NamedTuple

import polars as pl


def permutations_int_dec_none() -> list[tuple[D | int | None, ...]]:
    return list(
        itertools.permutations(
            [
                D("-0.01"),
                D("1.2345678"),
                D("500"),
                -1,
                None,
            ]
        )
    )


def test_series_from_pydecimal_and_ints() -> None:
    # TODO: check what happens if there are strings, floats arrow scalars in the list
    for data in permutations_int_dec_none():
        s = pl.Series("name", data)
        assert s.dtype == pl.Decimal(None, 7)  # inferred scale = 7, precision = None
        assert s.name == "name"
        assert s.null_count() == 1
        for i, d in enumerate(data):
            assert s[i] == d


def test_frame_from_pydecimal_and_ints() -> None:
    class X(NamedTuple):
        a: int | D | None

    @dataclass
    class Y:
        a: int | D | None

    for data in permutations_int_dec_none():
        row_data = [(d,) for d in data]
        for cls in (X, Y):
            for ctor in (pl.DataFrame, pl.from_records):
                df = ctor(data=list(map(cls, data)))  # type: ignore[operator]
                assert df.schema == {
                    "a": pl.Decimal(None, 7),
                }
                assert df.rows() == row_data


def test_to_from_pydecimal_and_format() -> None:
    dec_strs = [
        "0",
        "-1",
        "0.01",
        "-1.123801239123981293891283123",
        "12345678901.234567890123458390192857685",
        "-99999999999.999999999999999999999999999",
    ]
    formatted = (
        str(pl.Series(list(map(D, dec_strs))))
        .split("[", 1)[1]
        .split("\n", 1)[1]
        .strip()[1:-1]
        .split()
    )
    assert formatted == dec_strs
