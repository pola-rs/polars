"""Native Delta planning and the versioned I/O plugin handshake."""

from __future__ import annotations

import importlib
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlsplit, urlunsplit

from polars._utils.logging import verbose
from polars._utils.wrap import wrap_ldf
from polars._warnings import issue_warning
from polars.exceptions import ComputeError, PerformanceWarning, SchemaError

if TYPE_CHECKING:
    from polars import LazyFrame
    from polars.io.delta._scan_resolver import DeltaScanResolver

PLANNER_ENV_VAR = "POLARS_DELTA_PLANNER"


def plugin_planner_required() -> bool:
    return _planner() == "plugin"


def _planner() -> str | None:
    value = os.getenv(PLANNER_ENV_VAR) or None
    if value not in (None, "plugin", "deltalake"):
        msg = f"delta: unknown {PLANNER_ENV_VAR}={value!r}; expected 'plugin' or 'deltalake'"
        raise ValueError(msg)
    return value


class IncompatiblePluginError(ComputeError):
    """The installed `polars_delta` shares no I/O plugin contract with this Polars."""


def _plugin_capsule(supported_ids: list[str]) -> Any:
    plugin = importlib.import_module("polars_delta")
    shared = set(supported_ids).intersection(plugin._polars_io_plugin_ids)
    if not shared:
        msg = "polars_delta has no compatible Delta I/O plugin contract; upgrade polars and polars-delta"
        raise IncompatiblePluginError(msg)
    return plugin._capsule(max(shared, key=lambda s: int(s.rsplit(".v", 1)[1])))


def use_plugin_planner() -> bool:
    planner = _planner()
    if planner is not None:
        return planner == "plugin"
    import polars._plr as plr

    if not hasattr(plr, "_delta_plugin_scan"):
        return False
    try:
        _plugin_capsule(plr._IO_PLUGIN_IDS)
    except ModuleNotFoundError as e:
        if e.name != "polars_delta":
            _warn_plugin_unusable(f"failed to load: {e}")
        else:
            issue_warning(
                "Install polars-delta to plan Delta scans natively. Using deltalake; "
                f"set {PLANNER_ENV_VAR}=deltalake to use deltalake without this warning.",
                PerformanceWarning,
            )
        return False
    except IncompatiblePluginError:
        _warn_plugin_unusable("it is incompatible with this version of Polars")
        return False
    except Exception as e:
        _warn_plugin_unusable(f"failed to load: {e}")
        return False
    return True


def _warn_plugin_unusable(why: str) -> None:
    issue_warning(
        f"polars-delta cannot be used ({why}); upgrade polars-delta to a version "
        f"compatible with this Polars. Using deltalake; set {PLANNER_ENV_VAR}=deltalake "
        "to use deltalake without this warning.",
        PerformanceWarning,
    )


def dv_decoder_supported_build() -> bool:
    """Whether this Polars build can use a deletion-vector decoder plugin at all."""
    import polars._plr as plr

    return hasattr(plr, "_delta_plugin_register_decoder")


def dv_decoder_load_error() -> str | None:
    """
    Register the `polars-delta` deletion-vector decoder; the reason if it cannot be.

    Not installed, incompatible with this Polars, or failing to load all mean
    unavailable, never an exception.
    """
    import polars._plr as plr

    if not dv_decoder_supported_build():
        return "this Polars build cannot use a deletion vector decoder plugin"
    try:
        if not plr._delta_plugin_decoder_registered():
            plr._delta_plugin_register_decoder(_plugin_capsule(plr._IO_PLUGIN_IDS))
    except ModuleNotFoundError as e:
        if e.name == "polars_delta":
            return "it is not installed"
        return f"failed to load: {e}"
    except IncompatiblePluginError:
        return "the installed version is incompatible with this version of Polars, upgrade it"
    except Exception as e:
        return f"failed to load: {e}"
    return None


def dv_decoder_available() -> bool:
    """
    Whether the `polars-delta` plugin's deletion-vector decoder is registered.

    Registers it if the plugin is installed and compatible. Scans with native
    deletion vector descriptors need it, whichever planner produced them. Any
    failure to load or register the plugin means unavailable.
    """
    return dv_decoder_load_error() is None


def plugin_scan(
    dataset: DeltaScanResolver, **options: Any
) -> tuple[LazyFrame | None, int]:
    import polars._plr as plr

    if not hasattr(plr, "_delta_plugin_scan"):
        msg = "this Polars build has no Delta I/O plugin support (requires cloud and parquet)"
        raise NotImplementedError(msg)
    if dataset.delta_table_options:
        msg = "Delta plugin does not support delta_table_options"
        raise NotImplementedError(msg)
    table = dataset.table_.get()
    if table is not None and not hasattr(table, "_storage_options"):
        msg = "DeltaTable storage options are unavailable to the native planner"
        raise NotImplementedError(msg)
    if table is not None and dataset.version is None:
        table_schema = table.schema().to_json()
        if dataset._plugin_table_schema is None:
            dataset._plugin_table_schema = table_schema
        elif dataset._plugin_table_schema != table_schema:
            msg = "DeltaTable schema changed after scan creation; create a new scan"
            raise SchemaError(msg)
    version: int | str | datetime | None = dataset.version
    if version is None:
        # A caller-supplied DeltaTable can advance between collections. Capture its
        # current version once; URI scans retain the snapshot resolved with schema.
        version = table.version() if table is not None else dataset._resolved_version
    elif isinstance(version, (str, datetime)) and dataset._resolved_version is not None:
        # Time travel selects one snapshot, just like a URI scan of latest. Retain
        # the first successfully resolved version when planning another request.
        version = dataset._resolved_version
    if isinstance(version, int) and options.get("existing_resolved_version_key") == str(
        version
    ):
        # The host only supplies a key when its cached expansion covers this
        # request, including projection/filter columns, limit and statistics.
        # Keep this distinct from None, which signals planner fallback.
        return None, version
    timestamp_ms = None
    if isinstance(version, str):
        version = datetime.fromisoformat(version.replace("Z", "+00:00"))
    if isinstance(version, datetime):
        if version.tzinfo is None:
            version = version.replace(tzinfo=UTC)
        timestamp_ms = int(version.timestamp() * 1000)
        version = None
    from polars.io.delta._utils import _resolve_delta_lake_uri

    uri = _resolve_delta_lake_uri(dataset.table_uri(), strict=False)
    if table is not None and not uri.startswith("file://"):
        # DeltaTable exposes an escaped URL; native cloud roots are raw object paths.
        parts = urlsplit(uri)
        uri = urlunsplit(parts._replace(path=unquote(parts.path)))
    if dataset._plugin_metadata_cache is None:
        dataset._plugin_metadata_cache = plr.PyIcebergMetadataFileCache(
            64 * 1024 * 1024
        )
    if uri.startswith("lakefs://"):
        msg = "Delta plugin does not support lakefs table storage"
        raise NotImplementedError(msg)
    result, resolved_version = plr._delta_plugin_scan(
        _plugin_capsule(plr._IO_PLUGIN_IDS),
        table_uri=uri,
        version=version,
        timestamp_ms=timestamp_ms,
        projection=options.get("projection"),
        filter_columns=options.get("filter_columns"),
        statistics_columns=options.get("statistics_columns"),
        limit=options.get("limit"),
        use_metadata_statistics=True,
        verbose=verbose(),
        testing_fail=os.getenv("POLARS_DELTA_PLUGIN_TESTING_FAIL") or None,
        storage_options=dataset.storage_options,
        credential_provider=dataset.credential_provider_builder,
        metadata_cache=dataset._plugin_metadata_cache,
    )
    dataset._resolved_version = resolved_version
    return wrap_ldf(result), resolved_version
