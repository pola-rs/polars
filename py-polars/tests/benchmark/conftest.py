from pathlib import Path

import pytest

import polars as pl


@pytest.fixture(scope="module")
def data_path() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def h2aoi_groupby_data_path(data_path: Path) -> Path:
    return data_path / "G1_1e7_1e2_5_0.csv"


@pytest.fixture(scope="module")
def h2oai_groupby_data(h2aoi_groupby_data_path: Path) -> pl.DataFrame:
    if not h2aoi_groupby_data_path.is_file():
        pytest.skip("Dataset must be generated before running this test.")

    df = pl.read_csv(
        h2aoi_groupby_data_path,
        dtypes={
            "id4": pl.Int32,
            "id5": pl.Int32,
            "id6": pl.Int32,
            "v1": pl.Int32,
            "v2": pl.Int32,
            "v3": pl.Float64,
        },
    )
    return df
