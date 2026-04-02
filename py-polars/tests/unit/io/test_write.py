from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing.asserts.frame import assert_frame_equal
from tests.unit.io.conftest import format_file_uri

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.conftest import PlMonkeyPatch

READ_WRITE_FUNC_PARAM = [
    (pl.read_parquet, pl.DataFrame.write_parquet),
    (lambda *a: pl.scan_csv(*a).collect(), pl.DataFrame.write_csv),
    (lambda *a: pl.scan_ipc(*a).collect(), pl.DataFrame.write_ipc),
    # Sink
    (pl.read_parquet, lambda df, path: pl.DataFrame.lazy(df).sink_parquet(path)),
    (
        lambda *a: pl.scan_csv(*a).collect(),
        lambda df, path: pl.DataFrame.lazy(df).sink_csv(path),
    ),
    (
        lambda *a: pl.scan_ipc(*a).collect(),
        lambda df, path: pl.DataFrame.lazy(df).sink_ipc(path),
    ),
    (
        lambda *a: pl.scan_ndjson(*a).collect(),
        lambda df, path: pl.DataFrame.lazy(df).sink_ndjson(path),
    ),
]


@pytest.mark.parametrize(
    ("read_func", "write_func"),
    READ_WRITE_FUNC_PARAM,
)
@pytest.mark.write_disk
def test_write_async(
    read_func: Callable[[Path], pl.DataFrame],
    write_func: Callable[[pl.DataFrame, Path], None],
    tmp_path: Path,
) -> None:
    tmp_path.mkdir(exist_ok=True)
    path = (tmp_path / "1").absolute()
    path = format_file_uri(path)  # type: ignore[assignment]

    df = pl.DataFrame({"x": 1})

    write_func(df, path)

    assert_frame_equal(read_func(path), df)


@pytest.mark.parametrize(
    ("read_func", "write_func"),
    READ_WRITE_FUNC_PARAM,
)
@pytest.mark.parametrize("opt_absolute_fn", [Path, Path.absolute])
@pytest.mark.write_disk
def test_write_async_force_async(
    read_func: Callable[[Path], pl.DataFrame],
    write_func: Callable[[pl.DataFrame, Path], None],
    opt_absolute_fn: Callable[[Path], Path],
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    plmonkeypatch.setenv("POLARS_FORCE_ASYNC", "1")
    tmp_path.mkdir(exist_ok=True)
    path = opt_absolute_fn(tmp_path / "1")

    df = pl.DataFrame({"x": 1})

    write_func(df, path)
    assert_frame_equal(read_func(path), df)


@pytest.mark.write_disk
def test_write_with_storage_options_22873(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / "data"

    df = pl.DataFrame({"x": 1})

    for func in [
        pl.DataFrame.write_parquet,
        pl.DataFrame.write_ipc,
        pl.DataFrame.write_csv,
        pl.DataFrame.write_ndjson,
    ]:
        # TODO: write_ndjson cloud support
        if func is pl.DataFrame.write_ndjson:
            with pytest.raises(
                TypeError, match="unexpected keyword argument 'storage_options'"
            ):
                func(df, path, storage_options={"test": "1"})  # type: ignore[operator]

            continue

        func(df, path, storage_options={"test": "1"})  # type: ignore[operator]

    lf = df.lazy()

    for func in [
        pl.LazyFrame.sink_parquet,
        pl.LazyFrame.sink_ipc,
        pl.LazyFrame.sink_csv,
        pl.LazyFrame.sink_ndjson,
    ]:
        func(lf, path, storage_options={"test": "1"})  # type: ignore[operator]
