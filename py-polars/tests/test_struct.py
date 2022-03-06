import polars as pl


def test_struct_various() -> None:
    df = pl.DataFrame(
        {"int": [1, 2], "str": ["a", "b"], "bool": [True, None], "list": [[1, 2], [3]]}
    )
    s = df.to_struct("my_struct")

    assert s.struct.fields() == ["int", "str", "bool", "list"]
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
