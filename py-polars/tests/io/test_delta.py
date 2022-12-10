from pathlib import Path

from polars.testing import assert_frame_equal

import polars as pl


def test_read_delta() -> None:
    table_path = str(Path(__file__).parent.parent / "files" / "delta-table")
    df = pl.read_delta(table_path, version=0)

    expected = pl.DataFrame({"name": ["Joey", "Ivan"], "age": [14, 32]})
    assert_frame_equal(expected, df, check_dtype=False)
