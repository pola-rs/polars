import io

import pyarrow.parquet as pq
import pytest

import polars as pl
from polars.io.parquet import ParquetFieldOverwrites


def test_required_flat() -> None:
    f = io.BytesIO()
    pl.Series("a", [1, 2, 3]).to_frame().lazy().sink_parquet(
        f,
        field_overwrites=pl.io.parquet.ParquetFieldOverwrites(name="a", required=False),
    )

    f.seek(0)
    assert pq.read_schema(f).field(0).nullable

    f.seek(0)
    pl.Series("a", [1, 2, 3]).to_frame().lazy().sink_parquet(
        f,
        field_overwrites=pl.io.parquet.ParquetFieldOverwrites(name="a", required=True),
    )

    f.truncate()
    f.seek(0)
    assert not pq.read_schema(f).field(0).nullable

    f = io.BytesIO()
    with pytest.raises(pl.exceptions.InvalidOperationError, match="missing value"):
        pl.Series("a", [1, 2, 3, None]).to_frame().lazy().sink_parquet(
            f,
            field_overwrites=pl.io.parquet.ParquetFieldOverwrites(
                name="a", required=True
            ),
        )


@pytest.mark.parametrize("dtype", [pl.List(pl.Int64()), pl.Array(pl.Int64(), 1)])
def test_required_list(dtype: pl.DataType) -> None:
    f = io.BytesIO()
    pl.Series("a", [[1], [2], [3], [None]], dtype).to_frame().lazy().sink_parquet(
        f,
        field_overwrites=pl.io.parquet.ParquetFieldOverwrites(name="a", required=True),
    )
    f.seek(0)
    schema = pq.read_schema(f)
    assert not schema.field(0).nullable
    assert schema.field(0).type.value_field.nullable

    with pytest.raises(pl.exceptions.InvalidOperationError, match="missing value"):
        pl.Series("a", [[1], [2], [3], None], dtype).to_frame().lazy().sink_parquet(
            io.BytesIO(),
            field_overwrites=pl.io.parquet.ParquetFieldOverwrites(
                name="a", required=True
            ),
        )

    with pytest.raises(pl.exceptions.InvalidOperationError, match="missing value"):
        pl.Series("a", [[1], [2], [3], [None]], dtype).to_frame().lazy().sink_parquet(
            io.BytesIO(),
            field_overwrites=pl.io.parquet.ParquetFieldOverwrites(
                name="a",
                required=True,
                children=pl.io.parquet.ParquetFieldOverwrites(required=True),
            ),
        )

    f = io.BytesIO()
    pl.Series("a", [[1], [2], [3], [4]], dtype).to_frame().lazy().sink_parquet(
        f,
        field_overwrites=pl.io.parquet.ParquetFieldOverwrites(
            name="a",
            required=True,
            children=pl.io.parquet.ParquetFieldOverwrites(required=True),
        ),
    )
    f.seek(0)
    schema = pq.read_schema(f)
    assert not schema.field(0).nullable
    assert not schema.field(0).type.value_field.nullable


def test_required_struct() -> None:
    f = io.BytesIO()
    pl.Series(
        "a", [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}]
    ).to_frame().lazy().sink_parquet(
        f,
        field_overwrites=pl.io.parquet.ParquetFieldOverwrites(
            name="a",
            required=True,
        ),
    )
    f.seek(0)
    schema = pq.read_schema(f)
    assert not schema.field(0).nullable
    assert schema.field(0).type.fields[0].nullable

    f = io.BytesIO()
    pl.Series(
        "a", [{"x": 1}, {"x": None}, {"x": 2}, {"x": 3}]
    ).to_frame().lazy().sink_parquet(
        f,
        field_overwrites=pl.io.parquet.ParquetFieldOverwrites(
            name="a",
            required=True,
        ),
    )

    f.seek(0)
    schema = pq.read_schema(f)
    assert not schema.field(0).nullable
    assert schema.field(0).type.fields[0].nullable

    with pytest.raises(pl.exceptions.InvalidOperationError, match="missing value"):
        pl.Series(
            "a", [{"x": 1}, {"x": None}, {"x": 2}, {"x": 3}]
        ).to_frame().lazy().sink_parquet(
            io.BytesIO(),
            field_overwrites=ParquetFieldOverwrites(
                name="a",
                required=True,
                children={"x": ParquetFieldOverwrites(required=True)},
            ),
        )

    f = io.BytesIO()
    pl.Series(
        "a", [{"x": 1}, {"x": 2}, {"x": 2}, {"x": 3}]
    ).to_frame().lazy().sink_parquet(
        f,
        field_overwrites=ParquetFieldOverwrites(
            name="a",
            required=True,
            children={"x": ParquetFieldOverwrites(required=True)},
        ),
    )
    f.seek(0)
    schema = pq.read_schema(f)
    assert not schema.field(0).nullable
    assert not schema.field(0).type.fields[0].nullable
