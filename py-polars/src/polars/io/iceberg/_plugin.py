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
from polars.io.iceberg._cache import (
    _BUILTIN_FILE_IO_CLASSES,
    get_metadata_file_cache,
    plugin_storage_scope,
)
from polars.io.scan_options.cast_options import ScanCastOptions

if TYPE_CHECKING:
    from collections.abc import Mapping

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
    # Not only a missing or incompatible plugin: a broken install (e.g. an extension
    # module that fails to load) must not fail scans that PyIceberg can plan.
    except Exception as e:
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
    statistics_columns: list[str] | None,
    iceberg_table_filter: pyiceberg.expressions.BooleanExpression | None,
    limit: int | None,
    use_metadata_statistics: bool,
    fast_deletion_count: bool,
    user_storage_options: StorageOptionsDict | None,
) -> LazyFrame:
    """Plan the scan of `tbl` with the plugin; returns the native parquet scan."""
    from polars.io.iceberg._dataset import (
        _convert_iceberg_to_object_store_storage_options,
    )

    plr = _plr()
    capsule = _plugin_capsule(plr._IO_PLUGIN_IDS)

    # The plugin reads metadata with Polars' storage, configured from the FileIO's
    # properties; a custom FileIO may read from storage that Polars cannot access.
    file_io_class = f"{type(tbl.io).__module__}.{type(tbl.io).__qualname__}"
    if file_io_class not in _BUILTIN_FILE_IO_CLASSES:
        msg = f"iceberg: unsupported: custom PyIceberg FileIO ({file_io_class})"
        raise NotImplementedError(msg)

    metadata_location = tbl.metadata_location
    if metadata_location is None:
        # E.g. loaded from a REST catalog that does not return it.
        msg = "iceberg: unsupported: table without a metadata location"
        raise NotImplementedError(msg)

    # Catalog-provided IO properties (e.g. vended credentials) with known object store
    # equivalents, overridden by the user's storage options.
    storage_options = _catalog_storage_options(tbl.io.properties, metadata_location)
    if user_storage_options is not None:
        user_options = _convert_iceberg_to_object_store_storage_options(
            user_storage_options
        )
        # The user's credentials replace the catalog's as a whole: mixing them
        # (e.g. a profile with vended keys) is ambiguous, and rejected by the
        # credential provider.
        if any(_is_credential_key(k) for k in user_options):
            storage_options = {
                k: v
                for k, v in storage_options.items()
                if k not in _CATALOG_CREDENTIAL_KEYS
            }
        storage_options.update(user_options)

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
            statistics_columns=statistics_columns,
            row_filter=(
                _row_filter_json(iceberg_table_filter)
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


_AZURE_SCHEMES = ("abfs", "abfss", "adl", "az", "azure", "wasb", "wasbs")

# FileIO properties that PyIceberg honours and that have no object store equivalent,
# by URL scheme.
_UNSUPPORTED_IO_PROPERTIES: dict[tuple[str, ...], tuple[str, ...]] = {
    ("s3", "s3a", "s3n"): (
        # Remote signing (PyIceberg only signs with `s3.signer` set).
        "s3.signer",
        "s3.role-arn",
        "s3.role-session-name",
        "s3.profile-name",
        "client.role-arn",
        "client.role-session-name",
        "client.profile-name",
    ),
    _AZURE_SCHEMES: (
        "adls.connection-string",
        "adls.credential",
        "adls.blob-storage-authority",
        "adls.dfs-storage-authority",
        "adls.blob-storage-scheme",
        "adls.dfs-storage-scheme",
    ),
    ("gs", "gcs"): ("gcs.service.host", "gcs.requester-pays"),
    ("hf",): ("hf.endpoint",),
}

# PyIceberg's fallbacks for S3 properties.
_S3_CLIENT_PROPERTIES = {
    "client.access-key-id": "s3.access-key-id",
    "client.secret-access-key": "s3.secret-access-key",
    "client.session-token": "s3.session-token",
    "client.region": "s3.region",
}


def _catalog_storage_options(
    properties: Mapping[str, Any], location: str
) -> dict[str, Any]:
    """
    Object store options from a table's FileIO properties.

    Raises `NotImplementedError` for properties that change how storage is accessed
    but have no equivalent, so that PyIceberg plans the scan.
    """
    from polars.io.iceberg._dataset import (
        ICEBERG_TO_OBJECT_STORE_CONFIG_KEY_MAP,
        _convert_iceberg_property_value,
    )

    scheme = location.split("://", 1)[0].lower() if "://" in location else ""
    # Empty values are unset, as in PyIceberg.
    properties = {k: v for k, v in properties.items() if v not in ("", None)}
    for schemes, keys in _UNSUPPORTED_IO_PROPERTIES.items():
        if scheme in schemes and (
            unsupported := sorted(k for k in keys if k in properties)
        ):
            msg = f"iceberg: unsupported: FileIO properties: {unsupported}"
            raise NotImplementedError(msg)
    if scheme in ("s3", "s3a", "s3n"):
        if _property_is_true(properties, "s3.remote-signing-enabled"):
            msg = "iceberg: unsupported: S3 remote signing"
            raise NotImplementedError(msg)
        properties = {
            **{
                s3_key: properties[k]
                for k, s3_key in _S3_CLIENT_PROPERTIES.items()
                if k in properties
            },
            **properties,
        }

    storage_options = {
        (key := ICEBERG_TO_OBJECT_STORE_CONFIG_KEY_MAP[k]): (
            _convert_iceberg_property_value(key, v)
        )
        for k, v in properties.items()
        if k in ICEBERG_TO_OBJECT_STORE_CONFIG_KEY_MAP
    }
    # Anonymous access.
    for schemes, anonymous_key, skip_signature_key in (
        (("s3", "s3a", "s3n"), "s3.anonymous", "aws_skip_signature"),
        (_AZURE_SCHEMES, "adls.anon", "azure_skip_signature"),
    ):
        if scheme in schemes and _property_is_true(properties, anonymous_key):
            storage_options[skip_signature_key] = "true"
    return storage_options


def _property_is_true(properties: Mapping[str, Any], key: str) -> bool:
    # As PyIceberg's `property_as_bool` (`strtobool`); invalid values are false.
    value = properties.get(key)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("y", "yes", "t", "true", "on", "1")


def _row_filter_json(expr: pyiceberg.expressions.BooleanExpression) -> str:
    from pydantic_core import PydanticSerializationError

    try:
        return _balance_filter(expr).model_dump_json()
    except PydanticSerializationError as e:
        # E.g. exceeds Pydantic's recursion limit.
        msg = f"iceberg: unsupported: row filter: {e}"
        raise NotImplementedError(msg) from e


def _balance_filter(
    expr: pyiceberg.expressions.BooleanExpression,
) -> pyiceberg.expressions.BooleanExpression:
    """
    Rebalance chains of `And` / `Or`.

    Filters are converted from Polars as left-deep trees, whose JSON nesting
    exceeds the recursion limits of Pydantic and of JSON parsers.
    """
    from pyiceberg.expressions import And, Not, Or

    if isinstance(expr, Not):
        return Not(_balance_filter(expr.child))
    if not isinstance(expr, (And, Or)):
        return expr

    op = type(expr)
    operands = []
    stack: list[pyiceberg.expressions.BooleanExpression] = [expr]
    while stack:
        e = stack.pop()
        if type(e) is op:
            stack.extend((e.right, e.left))  # type: ignore[attr-defined]
        else:
            operands.append(_balance_filter(e))

    def build(
        operands: list[pyiceberg.expressions.BooleanExpression],
    ) -> pyiceberg.expressions.BooleanExpression:
        if len(operands) == 1:
            return operands[0]
        mid = len(operands) // 2
        return op(build(operands[:mid]), build(operands[mid:]))

    return build(operands)


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


# Object store keys of catalog-provided credentials
# (`_catalog_storage_options` values).
_CATALOG_CREDENTIAL_KEYS = frozenset(
    [
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "azure_storage_account_key",
        "azure_storage_sas_key",
        "azure_storage_tenant_id",
        "azure_storage_client_id",
        "azure_storage_client_secret",
        "azure_storage_token",
        "bearer_token",
        "token",
        "aws_skip_signature",
        "azure_skip_signature",
    ]
)

# Storage option keys (without the `aws_` / `azure_storage_` / `azure_` / `google_`
# prefixes) of user-provided credentials, which replace the catalog's. Other keys
# configure the client or location.
_USER_CREDENTIAL_KEYS = frozenset(
    [
        "access_key_id",
        "secret_access_key",
        "session_token",
        "token",
        "bearer_token",
        "profile",
        "skip_signature",
        "role_arn",
        "role_session_name",
        "web_identity_token_file",
        "container_credentials_relative_uri",
        "container_credentials_full_uri",
        "access_key",
        "account_key",
        "master_key",
        "sas_key",
        "sas_token",
        "client_id",
        "client_secret",
        "tenant_id",
        "authority_id",
        "federated_token_file",
        "msi_endpoint",
        "identity_endpoint",
        "fabric_session_token",
        "fabric_token_service_url",
        "credential_type",
        "use_emulator",
        "metadata_endpoint",
        "imdsv1_fallback",
        "msi_resource_id",
        "object_id",
        "use_azure_cli",
        "service_account",
        "service_account_key",
        "service_account_path",
        "application_credentials",
    ]
)


def _is_credential_key(key: str) -> bool:
    key = key.lower()
    for prefix in ("aws_", "azure_storage_", "azure_", "google_"):
        key = key.removeprefix(prefix)
    return key in _USER_CREDENTIAL_KEYS


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
