from __future__ import annotations

import contextlib
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import polars._utils.logging
from polars._utils.logging import eprint
from polars.io.external_reader import (
    FileReader,
    FileReaderBuilder,
)
from polars.io.external_reader._external_reader import ReaderCapabilities

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from concurrent.futures import Future

    import lance
    import pyarrow as pa

    import polars as pl
    from polars._typing import SchemaDict
    from polars.io.external_reader import (
        FilterExpr,
    )

CX_LOCK: threading.RLock = threading.RLock()
CX_PREV_READER_START_EVENT_KEY: object = object()
CX_THREADPOOL_KEY: object = object()
CX_ERROR_KEY: object = object()

CX_INFLIGHT_ROWS_KEY: object = object()
CX_CONSUMED_ROWS_KEY: object = object()
CX_TARGET_INFLIGHT_ROWS_KEY: object = object()

TARGET_MORSEL_SIZE: int = 100_000


class LanceReaderBuilder(FileReaderBuilder):
    def __init__(
        self,
        *,
        dataset: lance.LanceDataset,
        projected_schema: SchemaDict,
    ) -> None:
        self.dataset = dataset
        self.projected_schema = projected_schema

    def explain_properties(self) -> dict[str, str]:
        return {
            "name": f"{LanceReaderBuilder.__module__}.LanceReaderBuilder",
            "dataset_location": self.dataset.uri,
            "dataset_version": f"{self.dataset.version}",
        }

    def reader_capabilities(self) -> ReaderCapabilities:
        return ReaderCapabilities(row_index=True, pre_slice=True)

    def build_file_reader(
        self,
        source: str | bytes,
        multi_scan_context: dict[Any, Any],
    ) -> LanceFragmentReader:
        assert isinstance(source, str)

        fragment_ids = [int(source)]

        return LanceFragmentReader(
            dataset=self.dataset,
            fragment_ids=fragment_ids,
            projected_schema=self.projected_schema,
            multi_scan_context=multi_scan_context,
        )


class LanceFragmentReader(FileReader):
    def __init__(
        self,
        *,
        dataset: lance.LanceDataset,
        fragment_ids: Sequence[int],
        projected_schema: SchemaDict,
        multi_scan_context: dict[Any, Any],
    ) -> None:
        self.dataset = dataset
        self.fragment_ids = fragment_ids
        self.fragments_cached = None
        self.projected_schema = projected_schema
        self.multi_scan_context = multi_scan_context
        self.prev_reader_start_event = None
        self.this_reader_start_event = threading.Event()
        self.n_rows_in_file_cached = None
        self.consumed_rows_queue: queue.Queue[int] | None = None
        self.this_reader_inflight_rows = 0

    def initialize_serial(self) -> None:
        with CX_LOCK:
            if CX_INFLIGHT_ROWS_KEY not in self.multi_scan_context:
                self.multi_scan_context[CX_INFLIGHT_ROWS_KEY] = 0

            if CX_CONSUMED_ROWS_KEY not in self.multi_scan_context:
                self.multi_scan_context[CX_CONSUMED_ROWS_KEY] = queue.Queue()

            self.prev_reader_start_event = (
                x
                if (x := self.multi_scan_context.get(CX_PREV_READER_START_EVENT_KEY))
                is not self.this_reader_start_event
                else None
            )
            self.consumed_rows_queue = self.multi_scan_context[CX_CONSUMED_ROWS_KEY]

            self.multi_scan_context[CX_PREV_READER_START_EVENT_KEY] = (
                self.this_reader_start_event
            )

    def on_drop(self) -> None:
        self.this_reader_start_event.set()  # set here as safety precaution. should already be set

        if self.consumed_rows_queue is not None:
            self.consumed_rows_queue.put(self.this_reader_inflight_rows)
            self.this_reader_inflight_rows = 0

    def schema(self) -> SchemaDict:
        return self.projected_schema

    def collect_batches(
        self,
        *,
        columns: Sequence[str] = (),
        row_index: tuple[str, int] | None = None,
        pre_slice: tuple[int, int] | None = None,
        filters: Sequence[FilterExpr] = (),
        num_pipelines: int,
    ) -> Iterator[pl.DataFrame]:
        assert not filters

        slice_limit = None
        slice_offset = 0
        verbose = polars._utils.logging.verbose()

        if pre_slice is not None:
            slice_offset, slice_len = pre_slice
            assert slice_offset >= 0
            slice_limit = slice_offset + slice_len

        if self.prev_reader_start_event is not None:
            self.prev_reader_start_event.wait()

        n_rows_this_reader = self.n_rows_in_file()

        with CX_LOCK:
            if CX_THREADPOOL_KEY not in self.multi_scan_context:
                self.multi_scan_context[CX_THREADPOOL_KEY] = ThreadPoolExecutor(
                    num_pipelines
                )

            if CX_TARGET_INFLIGHT_ROWS_KEY not in self.multi_scan_context:
                self.multi_scan_context[CX_TARGET_INFLIGHT_ROWS_KEY] = (
                    TARGET_MORSEL_SIZE * num_pipelines
                )

            conversion_threadpool = self.multi_scan_context[CX_THREADPOOL_KEY]
            inflight_rows = self.multi_scan_context[CX_INFLIGHT_ROWS_KEY]
            target_inflight_rows = self.multi_scan_context[CX_TARGET_INFLIGHT_ROWS_KEY]
            consumed_rows_queue = self.multi_scan_context[CX_CONSUMED_ROWS_KEY]

        while (
            inflight_rows > 0
            and target_inflight_rows - inflight_rows < n_rows_this_reader
        ):
            if verbose:
                eprint(
                    f"LanceFragmentReader wait {n_rows_this_reader = }, {inflight_rows = }"
                )
            inflight_rows -= consumed_rows_queue.get()

        self.this_reader_inflight_rows = n_rows_this_reader

        with CX_LOCK:
            self.multi_scan_context[CX_INFLIGHT_ROWS_KEY] = (
                inflight_rows + n_rows_this_reader
            )

        if verbose:
            eprint(
                f"LanceFragmentReader start {n_rows_this_reader = }, {inflight_rows = }"
            )

        self.this_reader_start_event.set()

        # Future note: Lance operation order is filter->limit.
        batches_iter = self.dataset.scanner(
            fragments=self.get_fragments(),
            columns=columns,  # type: ignore[arg-type]
            limit=slice_limit,
            batch_readahead=num_pipelines,
        ).to_batches()

        fut_queue: queue.Queue[Future[pl.DataFrame] | None] = queue.Queue(num_pipelines)
        rx_closed = threading.Event()

        threading.Thread(
            target=send_arrow_batches,
            kwargs=dict(  # noqa: C408
                fut_queue=fut_queue,
                batches_iter=batches_iter,
                conversion_threadpool=conversion_threadpool,
                slice_offset=slice_offset,
                row_index=row_index,
                rx_closed=rx_closed,
                multi_scan_context=self.multi_scan_context,
            ),
            daemon=True,
        ).start()

        try:
            while (fut := fut_queue.get()) is not None:
                yield fut.result()

            with CX_LOCK:
                if (exc := self.multi_scan_context.pop(CX_ERROR_KEY, None)) is not None:
                    raise exc  # noqa: TRY301

        except GeneratorExit:
            conversion_threadpool.shutdown(wait=False, cancel_futures=True)
            return

        except BaseException:
            conversion_threadpool.shutdown(wait=False, cancel_futures=True)
            raise

        finally:
            rx_closed.set()

            with contextlib.suppress(queue.Empty):
                fut_queue.get_nowait()

    def n_rows_in_file(
        self,
        limit: int | None = None,  # noqa: ARG002
    ) -> int:
        if self.n_rows_in_file_cached is None:
            self.n_rows_in_file_cached = self.dataset.scanner(  # type: ignore[assignment]
                fragments=self.get_fragments()
            ).count_rows()  # type: ignore[no-untyped-call]

        return self.n_rows_in_file_cached  # type: ignore[return-value]

    def get_fragments(self) -> list[lance.LanceFragment]:
        if self.fragments_cached is None:
            self.fragments_cached = [
                self.dataset.get_fragment(x) for x in self.fragment_ids
            ]  # type: ignore[assignment]

        return self.fragments_cached  # type: ignore[return-value]


def send_arrow_batches(
    *,
    fut_queue: queue.Queue[Future[pl.DataFrame] | None],
    batches_iter: Iterator[pa.RecordBatch],
    conversion_threadpool: ThreadPoolExecutor,
    slice_offset: int,
    row_index: tuple[str, int] | None,
    rx_closed: threading.Event,
    multi_scan_context: dict[Any, Any],
) -> None:
    try:
        phys_position = 0
        batch_phys_rows = 0

        for pyarrow_batch in batches_iter:
            if rx_closed.is_set():
                break

            phys_position += batch_phys_rows
            batch_phys_rows = pyarrow_batch.shape[0]
            batch_offset = 0

            if phys_position < slice_offset:
                batch_offset = slice_offset - phys_position

                if batch_offset >= batch_phys_rows:
                    continue

            try:
                fut = conversion_threadpool.submit(
                    process_arrow_batch_to_polars_df,
                    pyarrow_batch=pyarrow_batch,
                    phys_position=phys_position,
                    batch_slice_offset=batch_offset,
                    row_index=row_index,
                )
            except RuntimeError as e:
                raise ThreadPoolSubmitError() from e

            fut_queue.put(fut)

    except ThreadPoolSubmitError:
        pass

    except BaseException as exc:
        with CX_LOCK:
            multi_scan_context[CX_ERROR_KEY] = exc

        conversion_threadpool.shutdown(wait=False, cancel_futures=True)

    finally:
        if not rx_closed.is_set():
            fut_queue.put(None)


def process_arrow_batch_to_polars_df(
    *,
    pyarrow_batch: pa.RecordBatch,
    phys_position: int,
    batch_slice_offset: int,
    row_index: tuple[str, int] | None,
) -> pl.DataFrame:
    import polars as pl

    if batch_slice_offset != 0:
        pyarrow_batch = pyarrow_batch.slice(batch_slice_offset)

    df = pl.DataFrame(pyarrow_batch)

    if row_index is not None:
        ri_name, ri_offset = row_index
        df = df.with_row_index(
            name=ri_name,
            offset=ri_offset + phys_position + batch_slice_offset,
        )

    return df


class ThreadPoolSubmitError(Exception): ...
