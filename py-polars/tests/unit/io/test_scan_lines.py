from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing.asserts.frame import assert_frame_equal

if TYPE_CHECKING:
    from tests.conftest import PlMonkeyPatch


def lazified_read_lines(*a: Any, **kw: Any) -> pl.LazyFrame:
    return pl.read_lines(*a, **kw).lazy()


@pytest.mark.parametrize("patch_scan_lines", [True, False])
@pytest.mark.parametrize("force_unit_chunk_size", [True, False])
@pytest.mark.parametrize("carriage_return", [True, False])
def test_scan_lines(
    patch_scan_lines: bool,
    force_unit_chunk_size: bool,
    carriage_return: bool,
    capfd: pytest.CaptureFixture[str],
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    if patch_scan_lines:
        plmonkeypatch.setattr(pl, "scan_lines", lazified_read_lines)
        assert pl.scan_lines is lazified_read_lines

    if carriage_return:
        inner = pl.scan_lines
        last_bytes = b""

        def wrapped(data: Any, *a: Any, **kw: Any) -> Any:
            nonlocal last_bytes
            last_bytes = bytes.replace(data, b"\n", b"\r\n")
            return inner(last_bytes, *a, **kw)

        plmonkeypatch.setattr(pl, "scan_lines", wrapped)

        pl.scan_lines(b"\n\n")
        assert last_bytes == b"\r\n\r\n"

    if force_unit_chunk_size:
        plmonkeypatch.setenv("POLARS_FORCE_NDJSON_READ_SIZE", "1")

        with plmonkeypatch.context() as cx:
            capfd.readouterr()
            cx.setenv("POLARS_VERBOSE", "1")
            pl.scan_lines(b"").collect()
            capture = capfd.readouterr().err
            assert "fixed_read_size: Some(1)" in capture

    assert_frame_equal(
        pl.scan_lines(b"").collect(),
        pl.DataFrame(schema={"lines": pl.String}),
    )

    assert_frame_equal(
        pl.scan_lines(b"", name="A").collect(),
        pl.DataFrame(schema={"A": pl.String}),
    )

    assert_frame_equal(
        pl.scan_lines(b"").collect(),
        pl.DataFrame(schema={"lines": pl.String}),
    )

    lf = pl.scan_lines(b"""\
AAA
BBB
CCC
DDD
EEE
""")

    assert_frame_equal(
        lf.slice(2, 1).collect(),
        pl.DataFrame({"lines": ["CCC"]}),
    )

    assert_frame_equal(
        lf.with_row_index().slice(2, 1).collect(),
        pl.DataFrame(
            {"index": [2], "lines": ["CCC"]},
            schema_overrides={"index": pl.get_index_type()},
        ),
    )

    assert_frame_equal(
        lf.slice(-2, 1).collect(),
        pl.DataFrame({"lines": ["DDD"]}),
    )

    assert_frame_equal(
        lf.with_row_index().slice(-2, 1).collect(),
        pl.DataFrame(
            {"index": [3], "lines": ["DDD"]},
            schema_overrides={"index": pl.get_index_type()},
        ),
    )

    def f(n_spaces: int, use_file_eol: bool) -> None:
        v = n_spaces * " "
        file_eol = "\n" if use_file_eol else ""

        lf = pl.scan_lines(f"{v}\n{v}\n{v}\n{v}\n{v}{file_eol}".encode())

        q = lf

        assert_frame_equal(
            q.collect(),
            pl.DataFrame({"lines": 5 * [v]}),
        )

        assert q.select(pl.len()).collect().item() == 5

        q = lf.slice(4)

        assert_frame_equal(
            q.collect(),
            pl.DataFrame({"lines": [v]}),
        )

        assert q.select(pl.len()).collect().item() == 1

        q = lf.with_row_index().slice(4)

        assert_frame_equal(
            q.collect(),
            pl.DataFrame(
                {"index": [4], "lines": [v]},
                schema_overrides={"index": pl.get_index_type()},
            ),
        )

        assert q.select(pl.len()).collect().item() == 1

        q = lf.slice(5)

        assert_frame_equal(
            q.collect(),
            pl.DataFrame(schema={"lines": pl.String}),
        )

        assert q.select(pl.len()).collect().item() == 0

        q = lf.slice(-1)

        assert_frame_equal(
            q.collect(),
            pl.DataFrame({"lines": [v]}),
        )

        assert q.select(pl.len()).collect().item() == 1

        q = lf.with_row_index().slice(-1)

        assert_frame_equal(
            q.collect(),
            pl.DataFrame(
                {"index": [4], "lines": [v]},
                schema_overrides={"index": pl.get_index_type()},
            ),
        )

        assert q.select(pl.len()).collect().item() == 1

        q = lf.slice(-4)

        assert_frame_equal(
            q.collect(),
            pl.DataFrame({"lines": 4 * [v]}),
        )

        assert q.select(pl.len()).collect().item() == 4

        q = lf.slice(-99)

        assert_frame_equal(
            q.collect(),
            pl.DataFrame({"lines": 5 * [v]}),
        )

        assert q.select(pl.len()).collect().item() == 5

    f(n_spaces=0, use_file_eol=True)

    for n_spaces in [1, 100]:
        for use_file_eol in [True, False]:
            f(n_spaces, use_file_eol)


def test_scan_lines_negative_slice_reversed_read(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    plmonkeypatch.setenv("POLARS_FORCE_NDJSON_READ_SIZE", "1")
    q = pl.scan_lines(b"\xff" + 5000 * b"abc\n")

    with pytest.raises(ComputeError, match="invalid utf8"):
        q.collect()

    assert q.tail(1).collect().item() == "abc"
    assert q.tail(1).select(pl.len()).collect().item() == 1

    # This succeeds because the line counter simply counts '\n' bytes without
    # parsing to string.
    assert q.select(pl.len()).collect().item() == 5000
