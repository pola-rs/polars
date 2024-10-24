import polars as pl
from polars.testing import assert_series_equal


def test_array() -> None:
    s0 = pl.Series("a", [1.0, 2.0], dtype=pl.Float64)
    s1 = pl.Series("b", [3.0, 4.0], dtype=pl.Float64)
    expected_f64 = pl.Series(
        "z", [[1.0, 3.0], [2.0, 4.0]], dtype=pl.Array(pl.Float64, 2)
    )
    expected_i32 = pl.Series(
        "z", [[1.0, 3.0], [2.0, 4.0]], dtype=pl.Array(pl.Int32, 2), strict=False
    )
    df = pl.DataFrame([s0, s1])
    print("\n")

    result = df.select(pl.array(["a", "b"]).alias("z"))["z"]
    print("No Cast")
    print(result)
    print()
    assert_series_equal(result, expected_f64)

    result = df.select(
        pl.array(["a", "b"], dtype='{"DtypeColumn":["Float64"]}').alias("z")
    )["z"]
    print("Cast to Float64")
    print(result)
    print()
    assert_series_equal(result, expected_f64)

    result = df.select(
        pl.array(["a", "b"], dtype='{"DtypeColumn":["Int32"]}').alias("z")
    )["z"]
    print("Cast to Int32")
    print(result)
    print()
    assert_series_equal(result, expected_i32)


def test_array_empty() -> None:
    s0 = pl.Series("a", [1.0, 2.0], dtype=pl.Float64)
    s1 = pl.Series("b", [3.0, 4.0], dtype=pl.Float64)
    expected_f64 = pl.Series(
        "z", [[1.0, 3.0], [2.0, 4.0]], dtype=pl.Array(pl.Float64, 2)
    )
    df = pl.DataFrame([s0, s1])
    print("\n")

    result = df.select(pl.array([], dtype='{"DtypeColumn":["Float64"]}').alias("z"))["z"]
    print("Empty")
    print(result)
    print()
    # assert_series_equal(result, expected_f64)



def test_array_nulls() -> None:
    s0 = pl.Series("a", [1.0, None], dtype=pl.Float64)
    s1 = pl.Series("b", [None, 4.0], dtype=pl.Float64)
    expected_f64 = pl.Series(
        "z", [[1.0, None], [None, 4.0]], dtype=pl.Array(pl.Float64, 2)
    )
    df = pl.DataFrame([s0, s1])
    print("\n")

    result = df.select(pl.array(["a", "b"]).alias("z"))["z"]
    print("No Cast")
    print(result)
    print()
    assert_series_equal(result, expected_f64)


def test_array_string() -> None:
    s0 = pl.Series("a", ["1", "2"])
    s1 = pl.Series("b", ["3", "4"])
    expected_f64 = pl.Series(
        "z", [[1.0, 3.0], [2.0, 4.0]], dtype=pl.Array(pl.Float64, 2)
    )
    df = pl.DataFrame([s0, s1])
    print("\n")

    result = df.select(pl.array(["a", "b"]).alias("z"))["z"]
    print("No Cast")
    print(result)
    print()
    assert_series_equal(result, expected_f64)

    result = df.select(
        pl.array(["a", "b"], dtype='{"DtypeColumn":["Float64"]}').alias("z")
    )["z"]
    print("Cast to Float64")
    print(result)
    print()
    assert_series_equal(result, expected_f64)
