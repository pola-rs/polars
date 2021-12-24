import os
from typing import Generator

import pytest

import polars as pl


@pytest.fixture()
def environ() -> Generator:
    """Fixture to restore the environment variables after the test"""
    old_environ = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(old_environ)


def test_tables(environ: None) -> None:
    os.environ.pop("POLARS_FMT_NO_UTF8", None)

    assert "POLARS_FMT_NO_UTF8" not in os.environ
    pl.Config.set_ascii_tables()
    assert os.environ["POLARS_FMT_NO_UTF8"] == "1"

    pl.Config.set_utf8_tables()
    assert "POLARS_FMT_NO_UTF8" not in os.environ


def test_tbl_width_chars(environ: None) -> None:
    pl.Config.set_tbl_width_chars(100)
    assert os.environ["POLARS_TABLE_WIDTH"] == "100"
    pl.Config.set_tbl_width_chars(200)
    assert os.environ["POLARS_TABLE_WIDTH"] == "200"


def test_tbl_cols_rows(environ: None) -> None:
    pl.Config.set_tbl_cols(50)
    assert os.environ["POLARS_FMT_MAX_COLS"] == "50"
    pl.Config.set_tbl_cols(60)
    assert os.environ["POLARS_FMT_MAX_COLS"] == "60"

    pl.Config.set_tbl_rows(50)
    assert os.environ["POLARS_FMT_MAX_ROWS"] == "50"
    pl.Config.set_tbl_rows(60)
    assert os.environ["POLARS_FMT_MAX_ROWS"] == "60"


@pytest.mark.skip("how to test this")
def test_string_cache(environ: None) -> None:
    pl.Config.set_global_string_cache()
