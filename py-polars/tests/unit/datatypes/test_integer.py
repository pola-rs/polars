import polars as pl


def test_integer_float_functions() -> None:
    assert pl.DataFrame({"a": [1, 2]}).select(
        finite=pl.all().is_finite(),
        infinite=pl.all().is_infinite(),
        nan=pl.all().is_nan(),
        not_na=pl.all().is_not_nan(),
    ).to_dict(as_series=False) == {
        "finite": [True, True],
        "infinite": [False, False],
        "nan": [False, False],
        "not_na": [True, True],
    }


def test_int_negate_operation() -> None:
    assert pl.Series([1, 2, 3, 4, 50912341409]).not_().to_list() == [
        -2,
        -3,
        -4,
        -5,
        -50912341410,
    ]
