"""Run the two-file Delta plugin smoke gate in a fresh `python -I -S -B` process."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars import DataFrame

FILES = {
    "part-a.parquet": [(47, "a-first"), (29, "a-second")],
    "part-z.parquet": [(13, "z-first"), (91, "z-second")],
}
COMMIT = "_delta_log/00000000000000000000.json"


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def prepare_fixture(
    root: Path, write_parquet: Callable[[Path, list[tuple[int, str]]], None]
) -> None:
    root.mkdir()
    (root / "_delta_log").mkdir()
    fields = [
        {"name": name, "type": dtype, "nullable": True, "metadata": {}}
        for name, dtype in [("id", "long"), ("label", "string")]
    ]
    actions = [
        {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}},
        {
            "metaData": {
                "id": "314f71cb-07ee-45be-b2dc-6315cc83abd4",
                "format": {"provider": "parquet", "options": {}},
                "schemaString": json.dumps({"type": "struct", "fields": fields}),
                "partitionColumns": [],
                "configuration": {},
                "createdTime": 0,
            }
        },
    ]
    # Reverse commit order so the first scan rows demonstrate raw path ordering.
    for name, rows in reversed(FILES.items()):
        path = root / name
        write_parquet(path, rows)
        actions.append(
            {
                "add": {
                    "path": name,
                    "partitionValues": {},
                    "size": path.stat().st_size,
                    "modificationTime": 0,
                    "dataChange": True,
                    "stats": json.dumps({"numRecords": len(rows)}),
                }
            }
        )
    (root / COMMIT).write_text(
        "".join(f"{json.dumps(action)}\n" for action in actions), encoding="utf-8"
    )


def require_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        msg = f"{label}: expected {expected!r}, got {actual!r}"
        raise AssertionError(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--polars-python", type=Path, required=True, help="polars package directory"
    )
    parser.add_argument(
        "--runtime", type=Path, required=True, help="host runtime extension file"
    )
    parser.add_argument(
        "--plugin", type=Path, required=True, help="standalone Delta extension file"
    )
    parser.add_argument(
        "--fixture", type=Path, required=True, help="two-file Delta table directory"
    )
    parser.add_argument(
        "--prepare", action="store_true", help="create the fixture before scanning"
    )
    args = parser.parse_args()
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        parser.error("run with python -I -S -B")

    polars_path = args.polars_python.resolve(strict=True)
    runtime = args.runtime.resolve(strict=True)
    plugin = args.plugin.resolve(strict=True)
    fixture = args.fixture.resolve()
    sys.path[:0] = [str(polars_path.parent), str(runtime.parent.parent)]
    os.environ["POLARS_DELTA_PLANNER"] = "plugin"

    spec = importlib.util.spec_from_file_location("polars_delta", plugin)
    if spec is None or spec.loader is None:
        parser.error(f"cannot load Delta extension {plugin}")
    delta = importlib.util.module_from_spec(spec)
    sys.modules["polars_delta"] = delta
    spec.loader.exec_module(delta)

    import polars as pl
    from polars import _plr
    from polars.io.delta import _plugin, _scan_resolver, functions

    require_equal(
        "polars Python path", Path(pl.__file__).resolve(), polars_path / "__init__.py"
    )
    require_equal("host runtime path", Path(_plr.__file__).resolve(), runtime)
    require_equal("Delta plugin path", Path(delta.__file__).resolve(), plugin)
    require_equal("forced Delta planner", _plugin.plugin_planner_required(), True)

    def write_parquet(path: Path, rows: list[tuple[int, str]]) -> None:
        frame: DataFrame = pl.DataFrame(
            rows, schema={"id": pl.Int64, "label": pl.String}, orient="row"
        )
        frame.write_parquet(path)

    if args.prepare:
        prepare_fixture(fixture, write_parquet)

    identity = {
        "python": {"path": sys.executable, "version": sys.version},
        "polars": {"path": str(polars_path), "version": pl.__version__},
        "runtime": {
            "path": str(runtime),
            "sha256": sha256(runtime),
            "commit": _plr._BUILD_COMMIT,
        },
        "plugin": {
            "path": str(plugin),
            "sha256": sha256(plugin),
            "commit": delta.__build_commit__,
            "version": delta.__version__,
        },
        "resolver": {
            m.__file__: sha256(Path(m.__file__))
            for m in (_plugin, _scan_resolver, functions)
        },
        "fixture": {name: sha256(fixture / name) for name in (COMMIT, *FILES)},
        "fixture_path": str(fixture),
        "planner": os.environ["POLARS_DELTA_PLANNER"],
    }
    print(json.dumps(identity, sort_keys=True), flush=True)

    expected = [row for rows in FILES.values() for row in rows]
    schema = pl.scan_delta(fixture).collect_schema()
    require_equal(
        "schema", list(schema.items()), [("id", pl.Int64), ("label", pl.String)]
    )
    head = pl.scan_delta(fixture).head(3).collect().rows()
    require_equal("head in raw path order", head, expected[:3])
    count = pl.scan_delta(fixture).select(pl.len()).collect().item()
    require_equal("count", count, len(expected))
    rows = pl.scan_delta(fixture).collect().rows()
    require_equal("collection in raw path order", rows, expected)
    require_equal("deltalake loaded", "deltalake" in sys.modules, False)
    print(
        json.dumps(
            {"result": "pass", "schema": str(schema), "head": head, "count": count, "rows": rows}
        )
    )


if __name__ == "__main__":
    main()
