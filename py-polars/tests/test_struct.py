import polars as pl


def test_struct_various() -> None:
    df = pl.DataFrame(
        {"int": [1, 2], "str": ["a", "b"], "bool": [True, None], "list": [[1, 2], [3]]}
    )
    s = df.to_struct("my_struct")

    assert s.struct.fields == ["int", "str", "bool", "list"]
    assert s[0] == (1, "a", True, pl.Series([1, 2]))
    assert s[1] == (2, "b", None, pl.Series([3]))
    assert s.struct.field("list").to_list() == [[1, 2], [3]]
    assert s.struct.field("int").to_list() == [1, 2]

    assert df.to_struct("my_struct").struct.to_frame().frame_equal(df)


def test_struct_to_list() -> None:
    assert pl.DataFrame(
        {"int": [1, 2], "str": ["a", "b"], "bool": [True, None], "list": [[1, 2], [3]]}
    ).select([pl.struct(pl.all()).alias("my_struct")]).to_series().to_list() == [
        (1, "a", True, pl.Series([1, 2])),
        (2, "b", None, pl.Series([3])),
    ]


def test_apply_to_struct() -> None:
    df = (
        pl.Series([None, 2, 3, 4])
        .apply(lambda x: (x, x * 2, True, [1, 2], "foo"))
        .struct.to_frame()
    )

    expected = pl.DataFrame(
        {
            "field_0": [None, 2, 3, 4],
            "field_1": [None, 4, 6, 8],
            "field_2": [None, True, True, True],
            "field_3": [None, [1, 2], [1, 2], [1, 2]],
            "field_4": [None, "foo", "foo", "foo"],
        }
    )

    assert df.frame_equal(expected)


def test_rename_fields() -> None:
    df = pl.DataFrame({"int": [1, 2], "str": ["a", "b"], "bool": [True, None]})
    assert df.to_struct("my_struct").struct.rename_fields(["a", "b"]).struct.fields == [
        "a",
        "b",
    ]


def struct_unnesting() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    out = df.select(
        [
            pl.all().alias("a_original"),
            pl.col("a")
            .apply(lambda x: (x, x * 2, x % 2 == 0))
            .struct.rename_fields(["a", "a_squared", "mod2eq0"])
            .alias("foo"),
        ]
    ).unnest("foo")

    expected = pl.DataFrame(
        {
            "a_original": [1, 2],
            "a": [1, 2],
            "a_squared": [2, 4],
            "mod2eq0": [False, True],
        }
    )

    assert out.frame_equal(expected)

    out = (
        df.lazy()
        .select(
            [
                pl.all().alias("a_original"),
                pl.col("a")
                .apply(lambda x: (x, x * 2, x % 2 == 0))
                .struct.rename_fields(["a", "a_squared", "mod2eq0"])
                .alias("foo"),
            ]
        )
        .unnest("foo")
        .collect()
    )
    out.frame_equal(expected)


def test_struct_function_expansion() -> None:
    df = pl.DataFrame(
        {"a": [1, 2, 3, 4], "b": ["one", "two", "three", "four"], "c": [9, 8, 7, 6]}
    )
    assert df.with_column(pl.struct(pl.col(["a", "b"])))["a"].struct.fields == [
        "a",
        "b",
    ]


def test_value_counts_expr() -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "b", "b", "c", "c", "c"],
        }
    )

    out = (
        df.select(
            [
                pl.col("id").value_counts(),
            ]
        )
        .to_series()
        .to_list()
    )

    out = sorted(out)  # type: ignore
    assert out == [("a", 1), ("b", 2), ("c", 3)]


def test_nested_struct() -> None:
    df = pl.DataFrame({"d": [1, 2, 3], "e": ["foo", "bar", "biz"]})
    # Nest the datafame
    nest_l1 = df.to_struct("c").to_frame()
    # Add another column on the same level
    nest_l1 = nest_l1.with_column(pl.col("c").is_nan().alias("b"))
    # Nest the dataframe again
    nest_l2 = nest_l1.to_struct("a").to_frame()

    assert isinstance(nest_l2.dtypes[0], pl.datatypes.Struct)
    assert nest_l2.dtypes[0].inner_types == nest_l1.dtypes
    assert isinstance(nest_l1.dtypes[0], pl.datatypes.Struct)
