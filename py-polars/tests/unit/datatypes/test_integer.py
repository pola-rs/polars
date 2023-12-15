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
