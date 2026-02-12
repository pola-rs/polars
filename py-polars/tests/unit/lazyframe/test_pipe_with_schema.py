import polars as pl


def test_pipe_with_schema() -> None:
    def cast_to_float_if_necessary(lf: pl.LazyFrame, schema: pl.Schema) -> pl.LazyFrame:
        required_casts = [
            pl.col(name).cast(pl.Float64)
            for name, dtype in schema.items()
            if not dtype.is_float()
        ]
        return lf.with_columns(required_casts)

    lf = pl.LazyFrame(
        {"a": [1.0, 2.0], "b": ["1.0", "2.5"], "c": [2.0, 3.0]},
        schema={"a": pl.Float64, "b": pl.String, "c": pl.Float32},
    )
    result = lf.pipe_with_schema(cast_to_float_if_necessary).collect()

    assert result.schema == {"a": pl.Float64, "b": pl.Float64, "c": pl.Float32}


def test_pipe_with_schema_rewrite() -> None:
    def rewrite(lf: pl.LazyFrame, schema: pl.Schema) -> pl.LazyFrame:
        return pl.LazyFrame({"x": [1, 2, 3]})

    lf = pl.LazyFrame({"a": [1, 2]})
    result = lf.pipe_with_schema(rewrite).collect()

    assert result.schema == {"x": pl.Int64}
    assert result.height == 3
