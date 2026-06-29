from __future__ import annotations

import operator
from typing import Any

import pytest

import polars as pl
from polars.exceptions import SchemaError


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([], False),
        ([None], False),
        ([False], False),
        ([False, None], False),
        ([True], True),
        ([True, None], True),
        ([None] * 199 + [True], True),
        ([None] * 200, False),
        ([None] * 200 + [False] * 200, False),
    ],
)
def test_any(data: list[bool | None], expected: bool) -> None:
    assert pl.Series(data, dtype=pl.Boolean).any() is expected


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([], False),
        ([None], None),
        ([False], False),
        ([False, None], None),
        ([True], True),
        ([True, None], True),
    ],
)
def test_any_kleene(data: list[bool | None], expected: bool | None) -> None:
    assert pl.Series(data, dtype=pl.Boolean).any(ignore_nulls=False) is expected


def test_any_wrong_dtype() -> None:
    with pytest.raises(SchemaError, match="expected `Boolean`"):
        pl.Series([0, 1, 0]).any()


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([], True),
        ([None], True),
        ([False], False),
        ([False, None], False),
        ([True], True),
        ([True, None], True),
        ([None] * 200 + [True] * 199 + [False], False),
        ([True] * 200 + [None] * 200 + [True] * 200, True),
    ],
)
def test_all(data: list[bool | None], expected: bool) -> None:
    assert pl.Series(data, dtype=pl.Boolean).all() is expected


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([], True),
        ([None], None),
        ([False], False),
        ([False, None], False),
        ([True], True),
        ([True, None], None),
    ],
)
def test_all_kleene(data: list[bool | None], expected: bool | None) -> None:
    assert pl.Series(data, dtype=pl.Boolean).all(ignore_nulls=False) is expected


def test_all_wrong_dtype() -> None:
    with pytest.raises(SchemaError, match="expected `Boolean`"):
        pl.Series([0, 1, 0]).all()


F = False
U = None
T = True


@pytest.mark.parametrize(
    ("op_impl", "truth_table"),
    [
        # https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics
        (
            operator.and_,
            {  #   [F, U, T]
                F: [F, F, F],
                U: [F, U, U],
                T: [F, U, T],
            },
        ),
        (
            operator.or_,
            {  #   [F, U, T]
                F: [F, U, T],
                U: [U, U, T],
                T: [T, T, T],
            },
        ),
        (
            operator.xor,
            {  #   [F, U, T]
                F: [F, U, T],
                U: [U, U, U],
                T: [T, U, F],
            },
        ),
    ],
)
@pytest.mark.parametrize("swap_args", [True, False])
def test_binary_bitwise_kleene_24809(
    op_impl: Any,
    swap_args: bool,
    truth_table: dict[bool, list[bool]],
) -> None:
    def op(lhs: Any, rhs: Any) -> Any:
        return op_impl(rhs, lhs) if swap_args else op_impl(lhs, rhs)

    rhs = pl.Series([F, U, T], dtype=pl.Boolean)

    class _:
        def f(scalar: bool | None) -> Any:  # type: ignore[misc]
            lhs = pl.lit(scalar, dtype=pl.Boolean)
            return pl.select(op(lhs, rhs)).to_series().to_list()

        assert {
            F: f(F),
            U: f(U),
            T: f(T),
        } == truth_table

    class _:  # type: ignore[no-redef]
        def f(scalar: bool | None) -> Any:  # type: ignore[misc]
            lhs = pl.Series([scalar], dtype=pl.Boolean)
            return pl.select(op(lhs, rhs)).to_series().to_list()

        assert {
            F: f(F),
            U: f(U),
            T: f(T),
        } == truth_table

    class _:  # type: ignore[no-redef]
        def f(scalar: bool | None) -> Any:  # type: ignore[misc]
            lhs = pl.Series([scalar, scalar, scalar], dtype=pl.Boolean)
            return op(lhs, rhs).to_list()

        assert {
            F: f(F),
            U: f(U),
            T: f(T),
        } == truth_table

    class _:  # type: ignore[no-redef]
        def f(scalar: bool | None) -> Any:  # type: ignore[misc]
            lhs = pl.lit(pl.Series([scalar]))
            return pl.select(op(lhs, rhs)).to_series().to_list()

        assert {
            F: f(F),
            U: f(U),
            T: f(T),
        } == truth_table

    class _:  # type: ignore[no-redef]
        def f(scalar: bool | None) -> Any:  # type: ignore[misc]
            lhs = pl.lit(pl.Series([scalar, scalar, scalar]))
            return pl.select(op(lhs, rhs)).to_series().to_list()

        assert {
            F: f(F),
            U: f(U),
            T: f(T),
        } == truth_table
