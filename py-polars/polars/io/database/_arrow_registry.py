from __future__ import annotations

from typing import TypedDict


class ArrowDriverProperties(TypedDict):
    # name of the method that fetches all arrow data; tuple form
    # calls the fetch_all method with the given chunk size (int)
    fetch_all: str | tuple[str, int]
    # name of the method that fetches arrow data in batches
    fetch_batches: str | None
    # indicate whether the given batch size is respected exactly
    exact_batch_size: bool | None
    # repeat batch calls (if False, the batch call is a generator)
    repeat_batch_calls: bool


ARROW_DRIVER_REGISTRY: dict[str, ArrowDriverProperties] = {
    "adbc_.*": {
        "fetch_all": "fetch_arrow_table",
        "fetch_batches": None,
        "exact_batch_size": None,
        "repeat_batch_calls": False,
    },
    "arrow_odbc_proxy": {
        "fetch_all": "fetch_arrow_table",
        "fetch_batches": "fetch_record_batches",
        "exact_batch_size": True,
        "repeat_batch_calls": False,
    },
    "databricks": {
        "fetch_all": "fetchall_arrow",
        "fetch_batches": "fetchmany_arrow",
        "exact_batch_size": True,
        "repeat_batch_calls": True,
    },
    "duckdb": {
        "fetch_all": "fetch_arrow_table",
        "fetch_batches": "fetch_record_batch",
        "exact_batch_size": True,
        "repeat_batch_calls": False,
    },
    "kuzu": {
        # 'get_as_arrow' currently takes a mandatory chunk size
        "fetch_all": ("get_as_arrow", 10_000),
        "fetch_batches": None,
        "exact_batch_size": None,
        "repeat_batch_calls": False,
    },
    "snowflake": {
        "fetch_all": "fetch_arrow_all",
        "fetch_batches": "fetch_arrow_batches",
        "exact_batch_size": False,
        "repeat_batch_calls": False,
    },
    "turbodbc": {
        "fetch_all": "fetchallarrow",
        "fetch_batches": "fetcharrowbatches",
        "exact_batch_size": False,
        "repeat_batch_calls": False,
    },
}
