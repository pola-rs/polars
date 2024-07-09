from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
import polars.selectors as cs
from polars.dependencies import _lazy_import

# don't import jax until an actual test is triggered (the decorator already
# ensures the tests aren't run locally; this avoids premature local import)
jx, _ = _lazy_import("jax")
jxn, _ = _lazy_import("jax.numpy")

pytestmark = pytest.mark.ci_only

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


@pytest.fixture()
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [1, 2, 2, 3],
            "y": [1, 0, 1, 0],
            "z": [1.5, -0.5, 0.0, -2.0],
        },
        schema_overrides={"x": pl.Int8, "z": pl.Float32},
    )


def assert_array_equal(actual: Any, expected: Any, nans_equal: bool = True) -> None:
    assert isinstance(actual, jx.Array)
    jxn.array_equal(actual, expected, equal_nan=nans_equal)


@pytest.mark.parametrize(
    ("dtype", "expected_jax_dtype"),
    [
        (pl.Int8, "int8"),
        (pl.Int16, "int16"),
        (pl.Int32, "int32"),
        (pl.Int64, "int32"),
        (pl.UInt8, "uint8"),
        (pl.UInt16, "uint16"),
        (pl.UInt32, "uint32"),
        (pl.UInt64, "uint32"),
    ],
)
def test_to_jax_from_series(
    dtype: PolarsDataType,
    expected_jax_dtype: str,
) -> None:
    s = pl.Series("x", [1, 2, 3, 4], dtype=dtype)
    for dvc in (None, "cpu", jx.devices("cpu")[0]):
        assert_array_equal(
            s.to_jax(device=dvc),
            jxn.array([1, 2, 3, 4], dtype=getattr(jxn, expected_jax_dtype)),
        )


def test_to_jax_array(df: pl.DataFrame) -> None:
    a1 = df.to_jax()
    a2 = df.to_jax("array")
    a3 = df.to_jax("array", device="cpu")
    a4 = df.to_jax("array", device=jx.devices("cpu")[0])

    expected = jxn.array(
        [
            [1.0, 1.0, 1.5],
            [2.0, 0.0, -0.5],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, -2.0],
        ],
        dtype=jxn.float32,
    )
    for a in (a1, a2, a3, a4):
        assert_array_equal(a, expected)


def test_to_jax_dict(df: pl.DataFrame) -> None:
    arr_dict = df.to_jax("dict")
    assert list(arr_dict.keys()) == ["x", "y", "z"]

    assert_array_equal(arr_dict["x"], jxn.array([1, 2, 2, 3], dtype=jxn.int8))
    assert_array_equal(arr_dict["y"], jxn.array([1, 0, 1, 0], dtype=jxn.int32))
    assert_array_equal(
        arr_dict["z"],
        jxn.array([1.5, -0.5, 0.0, -2.0], dtype=jxn.float32),
    )

    arr_dict = df.to_jax("dict", dtype=pl.Float32)
    for a, expected_data in zip(
        arr_dict.values(),
        ([1.0, 2.0, 2.0, 3.0], [1.0, 0.0, 1.0, 0.0], [1.5, -0.5, 0.0, -2.0]),
    ):
        assert_array_equal(a, jxn.array(expected_data, dtype=jxn.float32))


@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="jax.numpy.bool requires Python >= 3.9",
)
def test_to_jax_feature_label_dict(df: pl.DataFrame) -> None:
    df = pl.DataFrame(
        {
            "age": [25, 32, 45, 22, 34],
            "income": [50000, 75000, 60000, 58000, 120000],
            "education": ["bachelor", "master", "phd", "bachelor", "phd"],
            "purchased": [False, True, True, False, True],
        }
    ).to_dummies("education", separator=":")

    lbl_feat_dict = df.to_jax(return_type="dict", label="purchased")
    assert list(lbl_feat_dict.keys()) == ["label", "features"]

    assert_array_equal(
        lbl_feat_dict["label"],
        jxn.array([[False], [True], [True], [False], [True]], dtype=jxn.bool),
    )
    assert_array_equal(
        lbl_feat_dict["features"],
        jxn.array(
            [
                [25, 50000, 1, 0, 0],
                [32, 75000, 0, 1, 0],
                [45, 60000, 0, 0, 1],
                [22, 58000, 1, 0, 0],
                [34, 120000, 0, 0, 1],
            ],
            dtype=jxn.int32,
        ),
    )


def test_misc_errors(df: pl.DataFrame) -> None:
    with pytest.raises(
        ValueError,
        match="invalid `return_type`: 'stroopwafel'",
    ):
        _res0 = df.to_jax("stroopwafel")  # type: ignore[call-overload]

    with pytest.raises(
        ValueError,
        match="`label` is required if setting `features` when `return_type='dict'",
    ):
        _res2 = df.to_jax("dict", features=cs.float())

    with pytest.raises(
        ValueError,
        match="`label` and `features` only apply when `return_type` is 'dict'",
    ):
        _res3 = df.to_jax(label="stroopwafel")
