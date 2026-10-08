"""
Iceberg scan planning with the `polars_iceberg` I/O plugin.

Used by `IcebergScanResolver` in place of PyIceberg's `plan_files()`. Selected by
`POLARS_ICEBERG_PLANNER`: `plugin` requires the plugin, `pyiceberg` never uses it, and
unset uses it when available, otherwise falls back to PyIceberg with a
`PerformanceWarning`. Unless the plugin is required, tables with features it does not
support (e.g. equality deletes) are planned with PyIceberg. The plugin is called
through the minimal FFI contract of the newest plugin ID shared by this Polars
(`plr._IO_PLUGIN_IDS`) and the plugin (`polars_iceberg._polars_io_plugin_ids`); see
`polars-io-ext-ffi`.

The handshake (`_polars_io_plugin_ids`, `_capsule(id)`) never changes. Plugin IDs end
in `.v<N>`, which only orders them for error messages.
"""

from __future__ import annotations

import importlib.metadata
import os
from typing import TYPE_CHECKING, Any

from polars._utils.logging import verbose
from polars._utils.wrap import wrap_ldf
from polars._warnings import issue_warning
from polars.exceptions import ComputeError, PerformanceWarning
from polars.io.cloud.credential_provider._builder import (
    _init_credential_provider_builder,
)
from polars.io.iceberg._cache import get_metadata_file_cache, plugin_storage_scope
from polars.io.scan_options.cast_options import ScanCastOptions

if TYPE_CHECKING:
    import pyiceberg.expressions
    import pyiceberg.table

    from polars import LazyFrame
    from polars._typing import StorageOptionsDict

# Python distribution name of the plugin, as used with pip.
PLUGIN_PACKAGE_NAME = "polars_iceberg"

# Oldest `polars_iceberg` release providing each plugin ID this Polars supports, for
# install hints.
_MIN_PLUGIN_VERSION: dict[str, str] = {
    "polars.io_plugin.iceberg.v1": "0.1.0",
}

PLANNER_ENV_VAR = "POLARS_ICEBERG_PLANNER"
# Testing only: `io` or `panic`, passed to the plugin as `testing_fail`.
_TESTING_FAIL_ENV_VAR = "POLARS_ICEBERG_PLUGIN_TESTING_FAIL"


def use_plugin_planner() -> bool:
    """
    Whether scans are planned with the plugin.

    `POLARS_ICEBERG_PLANNER=plugin` always uses it (`plugin_scan` raises if it is
    unavailable), `pyiceberg` never does. When unset, the plugin is used if it is
    installed and compatible; otherwise a `PerformanceWarning` is issued and PyIceberg
    plans the scan.
    """
    planner = _planner()

    if planner:
        return planner == "plugin"

    import polars._plr as plr

    # Without plugin support in this build, installing the plugin would not help.
    if not hasattr(plr, "_iceberg_plugin_scan"):
        return False

    try:
        _plugin_capsule(plr._IO_PLUGIN_IDS)
    except (ModuleNotFoundError, ComputeError) as e:
        issue_warning(
            f"{e}. Planning the Iceberg scan with PyIceberg instead, which is slower. "
            f"Set {PLANNER_ENV_VAR}=pyiceberg to silence this warning.",
            PerformanceWarning,
        )
        return False

    return True


def plugin_planner_required() -> bool:
    """
    Whether `POLARS_ICEBERG_PLANNER=plugin` is set.

    If not, scans of tables with features the plugin does not support (it raises
    `NotImplementedError`) fall back to PyIceberg.
    """
    return _planner() == "plugin"


def _planner() -> str | None:
    planner = os.getenv(PLANNER_ENV_VAR)

    if planner not in (None, "", "plugin", "pyiceberg"):
        msg = (
            f"iceberg: unknown value for {PLANNER_ENV_VAR}: "
            f"'{planner}', expected one of ('plugin', 'pyiceberg')"
        )
        raise ValueError(msg)

    return planner or None


def plugin_scan(
    tbl: pyiceberg.table.Table,
    *,
    snapshot_id: int | None,
    from_snapshot_id_exclusive: int | None,
    to_snapshot_id_inclusive: int | None,
    projection: list[str] | None,
    filter_columns: list[str] | None,
    iceberg_table_filter: pyiceberg.expressions.BooleanExpression | None,
    limit: int | None,
    use_metadata_statistics: bool,
    fast_deletion_count: bool,
    user_storage_options: StorageOptionsDict | None,
) -> LazyFrame:
    """Plan the scan of `tbl` with the plugin; returns the native parquet scan."""
    from polars.io.iceberg._dataset import (
        ICEBERG_TO_OBJECT_STORE_CONFIG_KEY_MAP,
        _convert_iceberg_to_object_store_storage_options,
    )

    plr = _plr()
    capsule = _plugin_capsule(plr._IO_PLUGIN_IDS)

    metadata_location = tbl.metadata_location

    # Catalog-provided IO properties (e.g. vended credentials) with known object store
    # equivalents, overridden by the user's storage options.
    storage_options: dict[str, Any] = {
        ICEBERG_TO_OBJECT_STORE_CONFIG_KEY_MAP[k]: v
        for k, v in tbl.io.properties.items()
        if k in ICEBERG_TO_OBJECT_STORE_CONFIG_KEY_MAP
    }
    if user_storage_options is not None:
        storage_options.update(
            _convert_iceberg_to_object_store_storage_options(user_storage_options)
        )

    credential_provider = _init_credential_provider_builder(
        "auto", metadata_location, storage_options, "scan_iceberg"
    )

    # Immutable metadata files are cached across scans with the same storage
    # configuration, as with the PyIceberg planner.
    metadata_cache = get_metadata_file_cache()
    metadata_cache_scope = (
        plugin_storage_scope(tbl.io, user_storage_options)
        if metadata_cache.enabled
        else None
    )

    return wrap_ldf(
        plr._iceberg_plugin_scan(
            capsule,
            # The `iceberg.v1` request (`polars_io_ext_ffi::iceberg_v1::Request`).
            metadata_location=metadata_location,
            snapshot_id=snapshot_id,
            from_snapshot_id_exclusive=from_snapshot_id_exclusive,
            to_snapshot_id_inclusive=to_snapshot_id_inclusive,
            projection=projection,
            filter_columns=filter_columns,
            row_filter=(
                iceberg_table_filter.model_dump_json()
                if iceberg_table_filter is not None
                else None
            ),
            limit=limit,
            use_metadata_statistics=use_metadata_statistics,
            fast_deletion_count=fast_deletion_count,
            verbose=verbose(),
            testing_fail=os.getenv(_TESTING_FAIL_ENV_VAR) or None,
            source_url=metadata_location,
            storage_options=storage_options or None,
            credential_provider=credential_provider,
            cast_options=ScanCastOptions._default_iceberg(),
            metadata_cache=(
                metadata_cache.plugin_cache()
                if metadata_cache_scope is not None
                else None
            ),
            metadata_cache_scope=metadata_cache_scope,
        )
    )


def _plr() -> Any:
    import polars._plr as plr

    if not hasattr(plr, "_iceberg_plugin_scan"):
        msg = (
            f"{_package('polars')} was built without I/O plugin support "
            "(requires the 'cloud' and 'parquet' features)"
        )
        raise NotImplementedError(msg)

    return plr


def _plugin_capsule(supported_ids: list[str]) -> Any:
    """Capsule of the newest plugin ID shared by Polars and the installed plugin."""
    try:
        import polars_iceberg
    except ModuleNotFoundError as e:
        msg = (
            f"Iceberg scan planning with the {PLUGIN_PACKAGE_NAME} plugin requires the "
            f"{PLUGIN_PACKAGE_NAME} package; install it with "
            f"`{_install_command(supported_ids)}`"
        )
        raise ModuleNotFoundError(msg) from e

    plugin_ids = tuple(getattr(polars_iceberg, "_polars_io_plugin_ids", ()))
    shared = [id for id in supported_ids if id in plugin_ids]

    if not shared:
        raise ComputeError(
            _incompatible_message(supported_ids, plugin_ids, polars_iceberg)
        )

    return polars_iceberg._capsule(max(shared, key=_id_version))


def _id_version(id: str) -> int:
    """`N` of a plugin ID ending in `.v<N>`; -1 if it has another form."""
    _, _, version = id.rpartition(".v")
    return int(version) if version.isdigit() else -1


def _incompatible_message(
    supported_ids: list[str], plugin_ids: tuple[str, ...], module: Any
) -> str:
    plugin, host = _package(PLUGIN_PACKAGE_NAME, module), _package("polars")
    msg = (
        f"{plugin} is incompatible with {host}: the plugin provides plugin IDs "
        f"{list(plugin_ids)}, Polars supports {supported_ids}"
    )

    plugin_versions = [_id_version(id) for id in plugin_ids]
    if plugin_versions and min(plugin_versions) > max(map(_id_version, supported_ids)):
        return f"{msg}; upgrade polars, or install an older {PLUGIN_PACKAGE_NAME}"

    return (
        f"{msg}; upgrade {PLUGIN_PACKAGE_NAME} with `{_install_command(supported_ids)}`"
    )


def _install_command(supported_ids: list[str]) -> str:
    """Pip command installing a plugin providing an ID this Polars supports."""
    versions = [
        _MIN_PLUGIN_VERSION[id] for id in supported_ids if id in _MIN_PLUGIN_VERSION
    ]
    if not versions:
        return f"pip install --upgrade {PLUGIN_PACKAGE_NAME}"
    oldest = min(versions, key=lambda v: tuple(int(x) for x in v.split(".")))
    return f"pip install --upgrade '{PLUGIN_PACKAGE_NAME}>={oldest}'"


def _package(distribution: str, module: Any = None) -> str:
    """Pip requirement pinning an installed distribution, e.g. `polars==2.0.0rc2`."""
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = getattr(module, "__version__", None) or "<unknown>"
    return f"{distribution}=={version}"
