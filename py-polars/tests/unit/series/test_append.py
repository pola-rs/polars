import io

import pytest

import polars as pl
from polars.exceptions import SchemaError
from polars.testing import assert_series_equal


def test_append() -> None:
    a = pl.Series("a", [1, 2])
    b = pl.Series("b", [8, 9, None])

    result = a.append(b)

    expected = pl.Series("a", [1, 2, 8, 9, None])
    assert_series_equal(a, expected)
    assert_series_equal(result, expected)
    assert a.n_chunks() == 2


def test_append_self_3915() -> None:
    a = pl.Series("a", [1, 2])

    a.append(a)

    expected = pl.Series("a", [1, 2, 1, 2])
    assert_series_equal(a, expected)
    assert a.n_chunks() == 2


def test_append_bad_input() -> None:
    a = pl.Series("a", [1, 2])
    b = a.to_frame()

    with pytest.raises(
        TypeError,
        match=r"expected `other` .*to be a 'Series'.* not 'DataFrame'",
    ):
        a.append(b)  # type: ignore[arg-type]

    with pytest.raises(
        TypeError,
        match=r"expected `other` .*to be a 'Series'.* not 'LazyFrame'",
    ):
        a.append(b.lazy())  # type: ignore[arg-type]


def test_struct_schema_on_append_extend_3452() -> None:
    housing1_data = [
        {
            "city": "Chicago",
            "address": "100 Main St",
            "price": 250000,
            "nbr_bedrooms": 3,
        },
        {
            "city": "New York",
            "address": "100 First Ave",
            "price": 450000,
            "nbr_bedrooms": 2,
        },
    ]

    housing2_data = [
        {
            "address": "303 Mockingbird Lane",
            "city": "Los Angeles",
            "nbr_bedrooms": 2,
            "price": 450000,
        },
        {
            "address": "404 Moldave Dr",
            "city": "Miami Beach",
            "nbr_bedrooms": 1,
            "price": 250000,
        },
    ]
    housing1, housing2 = pl.Series(housing1_data), pl.Series(housing2_data)
    with pytest.raises(
        SchemaError,
    ):
        housing1.append(housing2)

    with pytest.raises(
        SchemaError,
    ):
        housing1.extend(housing2)


def test_append_mismatching_struct_with_null_22639() -> None:
    a = pl.Series([{"x": "foo", "y": "bar"}])
    b = pl.Series([{"z": "baz"}])
    c = pl.Series([{"z": None}])
    with pytest.raises(
        SchemaError,
    ):
        a.append(b)
    with pytest.raises(
        SchemaError,
    ):
        a.append(c)

    assert_series_equal(b.append(c), pl.Series([{"z": "baz"}, {"z": None}]))


def test_append_null_series() -> None:
    a = pl.Series("a", [1, 2], pl.Int64)
    b = pl.Series("b", [None, None], pl.Null)

    result = a.append(b)

    expected = pl.Series("a", [1, 2, None, None], pl.Int64)
    assert_series_equal(a, expected)
    assert_series_equal(result, expected)
    assert a.n_chunks() == 2


def test_append_enum_22764() -> None:
    f = io.BytesIO()
    g = io.BytesIO()
    pl.DataFrame({"someletter": ["A", "B"]}).write_csv(f)

    f.seek(0)

    schema = pl.Schema(
        {
            "someletter": pl.Enum(["A", "B"]),
        }
    )
    pl.scan_csv(f, schema=schema).sink_parquet(g)


def test_append_sliced_list_to_trimmed_list_30881() -> None:
    s = pl.Series("p", [["b"]])
    # Checks that `s` needs no trimming and flags it.
    s.list.eval(pl.element() == "a")
    s.append(pl.Series("p", [["a", ""], ["c"]]).head(1))

    out = s.list.eval(pl.element().filter(pl.element() != ""))
    assert out.to_list() == [["b"], ["a"]]


# In the tests below, `when/then` keeps the values under the null rows it creates, and
# a cast flags the nulls of the series it is called on as propagated.


def test_append_masked_list_to_propagated_30881() -> None:
    s = pl.Series("p", [["1"]])
    s.cast(pl.List(pl.Int64))
    masked = pl.DataFrame({"p": [["x"], ["2"]], "m": [False, True]})
    s.append(masked.select(pl.when("m").then("p"))["p"])
    assert s.cast(pl.List(pl.Int64)).to_list() == [[1], None, [2]]


def test_append_masked_array_to_propagated_30881() -> None:
    dtype = pl.Array(pl.Int64, 2)
    s = pl.Series("a", [[1, 2]], dtype=dtype)
    s.cast(pl.Array(pl.Float64, 2))
    masked = pl.DataFrame(
        {"a": pl.Series([[7, 8], [5, 6]], dtype=dtype), "m": [True, False]}
    )
    s.append(masked.select(pl.when("m").then("a"))["a"])
    assert s.reshape((-1,)).to_list() == [1, 2, 7, 8, None, None]


def test_append_masked_struct_to_propagated_30881() -> None:
    dtype = pl.Array(pl.Int64, 2)
    s = pl.DataFrame({"a": pl.Series([[1, 2]], dtype=dtype)}).select(
        pl.struct("a").alias("s")
    )["s"]
    s.cast(s.dtype)
    masked = pl.DataFrame(
        {
            "a": pl.Series([[7, 8], [5, 6], [3, 4]], dtype=dtype),
            "m": [True, False, False],
        }
    )
    s.append(masked.select(pl.struct(pl.when("m").then("a")).alias("s"))["s"])
    # The `a` field is null in both masked-out rows.
    assert s.n_unique() == 3


def test_append_to_empty_takes_chunks_of_other_30881() -> None:
    pa = pytest.importorskip("pyarrow")

    class EmptyChunksWithValues:
        # Two empty list chunks whose child arrays still hold a value.
        def __arrow_c_stream__(self, requested_schema: object = None) -> object:
            arr = pa.LargeListArray.from_arrays([0], pa.array(["x"]))
            return pa.chunked_array([arr, arr]).__arrow_c_stream__(requested_schema)

    other = pl.Series("p", EmptyChunksWithValues())
    assert other.n_chunks() == 2

    s = pl.Series("p", [], pl.List(pl.String))
    s.cast(pl.List(pl.Int64))
    # `s` is empty, so it takes the chunks of `other`.
    s.append(other)
    t = pl.Series("p", [["1"], ["2"]])
    t.cast(pl.List(pl.Int64))
    s.append(t)

    out = s.list.eval(pl.element().filter(pl.element() != ""))
    assert out.to_list() == [["1"], ["2"]]
