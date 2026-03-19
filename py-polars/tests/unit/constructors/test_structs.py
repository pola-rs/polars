import polars as pl


def test_constructor_non_strict_schema_17956() -> None:
    schema = {
        "logged_event": pl.Struct(
            [
                pl.Field(
                    "completetask",
                    pl.Struct(
                        [
                            pl.Field(
                                "parameters",
                                pl.List(
                                    pl.Struct(
                                        [
                                            pl.Field(
                                                "numericarray",
                                                pl.Struct(
                                                    [
                                                        pl.Field(
                                                            "value", pl.List(pl.Float64)
                                                        ),
                                                    ]
                                                ),
                                            ),
                                        ]
                                    )
                                ),
                            ),
                        ]
                    ),
                ),
            ]
        ),
    }

    data = {
        "logged_event": {
            "completetask": {
                "parameters": [
                    {
                        "numericarray": {
                            "value": [
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                431,
                                430.5,
                                431,
                                431,
                                431,
                            ]
                        }
                    }
                ]
            }
        }
    }

    lazyframe = pl.LazyFrame(
        [data],
        schema=schema,
        strict=False,
    )
    assert lazyframe.collect().to_dict(as_series=False) == {
        "logged_event": [
            {
                "completetask": {
                    "parameters": [
                        {
                            "numericarray": {
                                "value": [
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    431.0,
                                    430.5,
                                    431.0,
                                    431.0,
                                    431.0,
                                ]
                            }
                        }
                    ]
                }
            }
        ]
    }
