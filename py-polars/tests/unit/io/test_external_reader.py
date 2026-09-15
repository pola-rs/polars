import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

import polars as pl
from polars._typing import SchemaDict
from polars.io.external_reader import (
    FileReader,
    FileReaderBuilder,
    FilterExpr,
    ReaderCapabilities,
)
from polars.lazyframe.resolver._resolver import (
    LazyFrameResolver,
    ResolvedLazyFrameProps,
)
from polars.testing.asserts.frame import assert_frame_equal


def test_external_reader() -> None:
    class ReaderBuilder(FileReaderBuilder):
        def __init__(
            self,
            *,
            calls: list[Any],
            collect_batches_calls: list[Any],
            reader_capabilities: ReaderCapabilities,
            collect_batches_impl: Any,
        ) -> None:
            self.calls = calls
            self.collect_batches_calls = collect_batches_calls
            self.reader_capabilities_ = reader_capabilities
            self.collect_batches_impl = collect_batches_impl

        def reader_capabilities(self) -> ReaderCapabilities:
            self.calls.append(("reader_capabilities",))
            return self.reader_capabilities_

        def build_file_reader(self, source: str | bytes) -> FileReader:
            self.calls.append(("build_file_reader", source))
            return Reader(
                source,
                calls=self.calls,
                collect_batches_calls=self.collect_batches_calls,
                collect_batches_impl=self.collect_batches_impl,
            )

    class Reader(FileReader):
        def __init__(
            self,
            source: str | bytes,
            *,
            calls: list[Any],
            collect_batches_calls: list[Any],
            collect_batches_impl: Any,
        ) -> None:
            assert isinstance(source, str)
            self.df = data[source]
            self.calls = calls
            self.collect_batches_calls = collect_batches_calls
            self.collect_batches_impl = collect_batches_impl

        def n_rows_in_file(self, limit: int | None = None) -> int:
            self.calls.append(("n_rows_in_file", limit))
            return self.df.height

        def schema(self) -> SchemaDict:
            self.calls.append(("schema",))
            return self.df.schema

        def fetch_metadata(self) -> None:
            self.calls.append(("fetch_metadata",))

        def collect_batches(
            self,
            *,
            columns: Sequence[str] = (),
            row_index: tuple[str, int] | None = None,
            pre_slice: tuple[int, int] | None = None,
            filters: Sequence[FilterExpr] = (),
        ) -> Iterator[pl.DataFrame]:
            self.collect_batches_calls.append(
                {
                    "columns": columns,
                    "row_index": row_index,
                    "pre_slice": pre_slice,
                    "filters": filters,
                }
            )

            return self.collect_batches_impl(  # type: ignore[no-any-return]
                slf=self,
                columns=columns,
                row_index=row_index,
                pre_slice=pre_slice,
                filters=filters,
            )

    data = {
        "0": pl.DataFrame({"a": [0, 1, 2], "b": [0, 1, 2]}),
        "x": pl.DataFrame({"a": [-1], "b": [-1]}),
    }

    class _:
        calls: Any = []  # noqa: RUF012
        collect_batches_calls: Any = []  # noqa: RUF012
        reader_capabilities = ReaderCapabilities()

        def collect_batches_impl(  # type: ignore[invalid-annotation, misc]
            slf: Reader,
            *,
            columns: Sequence[str],
            row_index: tuple[str, int] | None,
            pre_slice: tuple[int, int] | None,
            filters: Sequence[FilterExpr],
        ) -> Iterator[pl.DataFrame]:
            return (slf.df.slice(x, 1) for x in range(slf.df.height))

        q = pl.scan_external_reader(
            ReaderBuilder(
                calls=calls,
                collect_batches_calls=collect_batches_calls,
                reader_capabilities=reader_capabilities,
                collect_batches_impl=collect_batches_impl,
            ),
            sources=[*data],
            schema=next(iter(data.values())).schema,
        )

        assert_frame_equal(
            q.collect(),
            pl.DataFrame({"a": [0, 1, 2, -1], "b": [0, 1, 2, -1]}),
        )

        assert sum(1 for x in calls if x[0] == "fetch_metadata") == 2

    class _:  # type: ignore[no-redef]
        calls: Any = []  # noqa: RUF012
        collect_batches_calls: Any = []  # noqa: RUF012
        reader_capabilities = ReaderCapabilities()

        def collect_batches_impl(  # type: ignore[invalid-annotation, misc]
            slf: Reader,
            *,
            columns: Sequence[str],
            row_index: tuple[str, int] | None,
            pre_slice: tuple[int, int] | None,
            filters: Sequence[FilterExpr],
        ) -> Iterator[pl.DataFrame]:
            return (slf.df.slice(x, 1) for x in range(slf.df.height))

        q = pl.scan_external_reader(
            ReaderBuilder(
                calls=calls,
                collect_batches_calls=collect_batches_calls,
                reader_capabilities=reader_capabilities,
                collect_batches_impl=collect_batches_impl,
            ),
            sources=[*data],
            schema=next(iter(data.values())).schema,
        ).filter(pl.col("a") != -1)

        assert_frame_equal(
            q.collect(),
            pl.DataFrame({"a": [0, 1, 2], "b": [0, 1, 2]}),
        )

        assert (
            sum(not x["filters"] for x in collect_batches_calls)
            == 2
            == len(collect_batches_calls)
        )

    class _:  # type: ignore[no-redef]
        calls: Any = []  # noqa: RUF012
        collect_batches_calls: Any = []  # noqa: RUF012
        reader_capabilities = ReaderCapabilities(supported_filter="partial")

        def collect_batches_impl(  # type: ignore[invalid-annotation, misc]
            slf: Reader,
            *,
            columns: Sequence[str],
            row_index: tuple[str, int] | None,
            pre_slice: tuple[int, int] | None,
            filters: Sequence[FilterExpr],
        ) -> Iterator[pl.DataFrame]:
            return (slf.df.slice(x, 1) for x in range(slf.df.height))

        q = pl.scan_external_reader(
            ReaderBuilder(
                calls=calls,
                collect_batches_calls=collect_batches_calls,
                reader_capabilities=reader_capabilities,
                collect_batches_impl=collect_batches_impl,
            ),
            sources=[*data],
            schema=next(iter(data.values())).schema,
        ).filter(pl.col("a") != -1)

        assert_frame_equal(
            q.collect(),
            pl.DataFrame({"a": [0, 1, 2], "b": [0, 1, 2]}),
        )

        assert len(collect_batches_calls) == 2

        for call in collect_batches_calls:
            assert len(call["filters"]) == 1

            assert re.search(
                r"pyarrow\.compute\.Expression.*\(a != -1\)",
                repr(call["filters"][0].pyarrow_expr),
            )

            assert re.search(
                r"pa\.compute\.field.*\('a'\) != -1\)",
                call["filters"][0].pyarrow_str,
            )

    class _:  # type: ignore[no-redef]
        calls: Any = []  # noqa: RUF012
        collect_batches_calls: Any = []  # noqa: RUF012
        reader_capabilities = ReaderCapabilities(supported_filter="full")

        def collect_batches_impl(  # type: ignore[invalid-annotation, misc]
            slf: Reader,
            *,
            columns: Sequence[str],
            row_index: tuple[str, int] | None,
            pre_slice: tuple[int, int] | None,
            filters: Sequence[FilterExpr],
        ) -> Iterator[pl.DataFrame]:
            return (slf.df.slice(x, 1) for x in range(slf.df.height))

        q = pl.scan_external_reader(
            ReaderBuilder(
                calls=calls,
                collect_batches_calls=collect_batches_calls,
                reader_capabilities=reader_capabilities,
                collect_batches_impl=collect_batches_impl,
            ),
            sources=[*data],
            schema=next(iter(data.values())).schema,
        ).filter(pl.col("a") != -1)

        # Hinted full filter, so multi-scan does not apply.
        assert_frame_equal(
            q.collect(),
            pl.DataFrame({"a": [0, 1, 2, -1], "b": [0, 1, 2, -1]}),
        )

        assert len(collect_batches_calls) == 2

        for call in collect_batches_calls:
            assert len(call["filters"]) == 1

            assert re.search(
                r"pyarrow\.compute\.Expression.*\(a != -1\)",
                repr(call["filters"][0].pyarrow_expr),
            )

            assert re.search(
                r"pa\.compute\.field.*\('a'\) != -1\)",
                call["filters"][0].pyarrow_str,
            )

    class _:  # type: ignore[no-redef]
        calls: Any = []  # noqa: RUF012
        collect_batches_calls: Any = []  # noqa: RUF012
        reader_capabilities = ReaderCapabilities()

        def collect_batches_impl(  # type: ignore[invalid-annotation, misc]
            slf: Reader,
            *,
            columns: Sequence[str],
            row_index: tuple[str, int] | None,
            pre_slice: tuple[int, int] | None,
            filters: Sequence[FilterExpr],
        ) -> Iterator[pl.DataFrame]:
            for i, df in enumerate(slf.df.slice(x, 1) for x in range(slf.df.height)):
                if row_index is not None:
                    df = df.with_row_index(row_index[0], row_index[1] + i)

                yield df

        q = pl.scan_external_reader(
            ReaderBuilder(
                calls=calls,
                collect_batches_calls=collect_batches_calls,
                reader_capabilities=reader_capabilities,
                collect_batches_impl=collect_batches_impl,
            ),
            sources=[*data],
            schema=next(iter(data.values())).schema,
        ).with_row_index()

        assert_frame_equal(
            q.collect(),
            pl.DataFrame(
                {"index": [0, 1, 2, 3], "a": [0, 1, 2, -1], "b": [0, 1, 2, -1]},
                schema_overrides={"index": pl.get_index_type()},
            ),
        )

        assert len(collect_batches_calls) == 2

        for call in collect_batches_calls:
            assert call["row_index"] is None

    class _:  # type: ignore[no-redef]
        calls: Any = []  # noqa: RUF012
        collect_batches_calls: Any = []  # noqa: RUF012
        reader_capabilities = ReaderCapabilities(
            row_index=True, supported_filter="partial"
        )

        def collect_batches_impl(  # type: ignore[invalid-annotation, misc]
            slf: Reader,
            *,
            columns: Sequence[str],
            row_index: tuple[str, int] | None,
            pre_slice: tuple[int, int] | None,
            filters: Sequence[FilterExpr],
        ) -> Iterator[pl.DataFrame]:
            for i, df in enumerate(slf.df.slice(x, 1) for x in range(slf.df.height)):
                if row_index is not None:
                    df = df.with_row_index(row_index[0], row_index[1] + i)

                yield df

        q = (
            pl.scan_external_reader(
                ReaderBuilder(
                    calls=calls,
                    collect_batches_calls=collect_batches_calls,
                    reader_capabilities=reader_capabilities,
                    collect_batches_impl=collect_batches_impl,
                ),
                sources=[*data],
                schema=next(iter(data.values())).schema,
            )
            .with_row_index()
            .filter(pl.col("a") != -1)
        )

        assert_frame_equal(
            q.collect(),
            pl.DataFrame(
                {"index": [0, 1, 2], "a": [0, 1, 2], "b": [0, 1, 2]},
                schema_overrides={"index": pl.get_index_type()},
            ),
        )

        assert len(collect_batches_calls) == 2

        assert {call["row_index"] for call in collect_batches_calls} == {
            ("index", 0),
            ("index", 3),
        }

    class _:  # type: ignore[no-redef]
        calls: Any = []  # noqa: RUF012
        collect_batches_calls: Any = []  # noqa: RUF012
        reader_capabilities = ReaderCapabilities(
            row_index=True, negative_pre_slice=True
        )

        def collect_batches_impl(  # type: ignore[invalid-annotation, misc]
            slf: Reader,
            *,
            columns: Sequence[str],
            row_index: tuple[str, int] | None,
            pre_slice: tuple[int, int] | None,
            filters: Sequence[FilterExpr],
        ) -> Iterator[pl.DataFrame]:
            df = slf.df

            if row_index is not None:
                df = df.with_row_index(*row_index)

            if pre_slice is not None:
                df = df.slice(*pre_slice)

            yield df

        q = (
            pl.scan_external_reader(
                ReaderBuilder(
                    calls=calls,
                    collect_batches_calls=collect_batches_calls,
                    reader_capabilities=reader_capabilities,
                    collect_batches_impl=collect_batches_impl,
                ),
                sources=["0"],
                schema=next(iter(data.values())).schema,
            )
            .with_row_index()
            .tail(1)
        )

        assert_frame_equal(
            q.collect(),
            pl.DataFrame(
                {"index": [2], "a": [2], "b": [2]},
                schema_overrides={"index": pl.get_index_type()},
            ),
        )

        assert len(collect_batches_calls) == 1

        for call in collect_batches_calls:
            assert call["row_index"] == ("index", 0)
            assert call["pre_slice"] == (-1, 1)


@pytest.mark.write_disk
def test_external_reader_expand_paths(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"a")
    (tmp_path / "b").write_bytes(b"b")
    (tmp_path / "c").write_bytes(b"c")

    schema = {"path": pl.String}

    class Reader(FileReader):
        def __init__(
            self,
            source: str | bytes,
        ) -> None:
            self.source = source

        def n_rows_in_file(self, limit: int | None = None) -> int:
            return 1

        def schema(self) -> SchemaDict:
            return schema

        def collect_batches(
            self,
            *,
            columns: Sequence[str] = (),
            row_index: tuple[str, int] | None = None,
            pre_slice: tuple[int, int] | None = None,
            filters: Sequence[FilterExpr] = (),
        ) -> Iterator[pl.DataFrame]:
            yield pl.DataFrame({"path": self.source})

    class ReaderBuilder(FileReaderBuilder):
        def build_file_reader(self, source: str | bytes) -> FileReader:
            return Reader(source)

    assert (
        pl.scan_external_reader(ReaderBuilder(), sources=[str(tmp_path)], schema=schema)
        .select(pl.len())
        .collect()
        .item()
        == 1
    )

    assert (
        pl.scan_external_reader(
            ReaderBuilder(),
            sources=[str(tmp_path)],
            schema=schema,
            expand_paths=True,
        )
        .select(pl.len())
        .collect()
        .item()
        == 3
    )


def test_external_reader_with_resolver_use_case() -> None:
    data_dict = {
        1: pl.DataFrame({"x": [1]}),
        2: pl.DataFrame({"x": [2, 2]}),
        3: pl.DataFrame({"x": [3, 3, 3]}),
        4: pl.DataFrame({"x": [4, 4, 4, 4]}),
        5: pl.DataFrame({"x": [5, 5, 5, 5, 5]}),
    }

    schema = {"x": pl.Int64}

    class Reader(FileReader):
        def __init__(
            self,
            source: str | bytes,
        ) -> None:
            assert isinstance(source, bytes)
            self.source = source

        def n_rows_in_file(self, limit: int | None = None) -> int:
            return int.from_bytes(self.source, byteorder="little")

        def schema(self) -> SchemaDict:
            return schema

        def collect_batches(
            self,
            *,
            columns: Sequence[str] = (),
            row_index: tuple[str, int] | None = None,
            pre_slice: tuple[int, int] | None = None,
            filters: Sequence[FilterExpr] = (),
        ) -> Iterator[pl.DataFrame]:
            yield data_dict[int.from_bytes(self.source, byteorder="little")]

    class ReaderBuilder(FileReaderBuilder):
        def build_file_reader(self, source: str | bytes) -> FileReader:
            return Reader(source)

    class Resolver(LazyFrameResolver):
        def schema(self) -> SchemaDict:
            return schema

        def resolve_lazyframe(
            self,
            *,
            projection: list[str] | None,
            limit: int | None,
            filters: list[FilterExpr],
            filter_columns: list[str],
            filter_drop_columns_idx: int | None,
            existing_resolved_version_key: str | None,
        ) -> pl.LazyFrame | tuple[pl.LazyFrame | None, ResolvedLazyFrameProps]:
            sources = pl.Series("x", [*data_dict])

            if filters:
                assert len(filters) == 1
                sources = sources.to_frame().filter(filters[0].expr).to_series()

            return pl.scan_external_reader(
                ReaderBuilder(),
                sources=[
                    int.to_bytes(x, length=1, byteorder="little") for x in sources
                ],
                schema=schema,
            )

    q = pl.LazyFrame.from_lazyframe_resolver(Resolver())
    plan = q.explain()

    assert re.search(r"ExternalReaderBuilder.*4 other sources", plan) is not None
    assert_frame_equal(
        q.collect(), pl.DataFrame({"x": [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]})
    )

    q = pl.LazyFrame.from_lazyframe_resolver(Resolver()).filter(pl.col("x") >= 3)
    plan = q.explain()

    # Source pruned at resolver with filter.
    assert re.search(r"ExternalReaderBuilder.*2 other sources", plan) is not None
    assert_frame_equal(
        q.collect(), pl.DataFrame({"x": [3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]})
    )


def test_external_reader_explain() -> None:
    class ReaderBuilder(FileReaderBuilder):
        def build_file_reader(self, source: str | bytes) -> FileReader:
            msg = "unreachable"
            raise NotImplementedError(msg)

        def explain_properties(self) -> dict[str, str]:
            return {
                "explain_property_key": "explain_property_value",
            }

    q = pl.scan_external_reader(ReaderBuilder(), sources=[], schema={})
    plan = q.explain()

    assert (
        plan
        == """\
ExternalReaderBuilder SCAN []
  explain_property_key: explain_property_value
PROJECT */0 COLUMNS"""
    )

    phys_graph = q.show_graph(raw_output=True)
    assert "explain_property_key: explain_property_value" in phys_graph

    assert q.collect().shape == (0, 0)
