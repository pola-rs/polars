"""Storage credentials vended by the table's catalog, renewed before they expire."""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any, Final, Literal

from polars._utils.logging import eprint, verbose
from polars.exceptions import ComputeError
from polars.io.cloud._utils import NoPickleOption
from polars.io.cloud.credential_provider._builder import CredentialProviderBuilder
from polars.io.cloud.credential_provider._providers import CachingCredentialProvider
from polars.io.iceberg._cache import _credential_env_fingerprint
from polars.io.iceberg._dataset import (
    IcebergCatalogConfig,
    IcebergCatalogTableDescriptor,
    _reusable_catalog,
)
from polars.io.iceberg._plugin import (
    _PROPERTY_SCHEMES,
    _catalog_storage_options,
    _is_credential_key,
    _table_location,
    _url_scheme,
    _user_storage_options,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pyiceberg.table

    from polars.io.cloud.credential_provider._providers import (
        CredentialProviderFunctionReturn,
    )

# Credentials are renewed this long before they expire, so that requests in flight do
# not use expired ones.
_RENEW_BEFORE_EXPIRY_SECONDS = 60
# The catalog is not asked again before this after a renewal, a failed one, or one
# that gave expired credentials.
_RETRY_SECONDS = 10

# Vended credentials that the catalog renews, by FileIO property prefix: object
# store key → the key the credential provider returns it under
# (`credential_provider.rs`). Other credentials (e.g. a client secret) are passed as
# fixed storage options.
_RENEWABLE_CREDENTIAL_KEYS: Final[dict[str, dict[str, str]]] = {
    "s3.": {
        "aws_access_key_id": "aws_access_key_id",
        "aws_secret_access_key": "aws_secret_access_key",
        "aws_session_token": "aws_session_token",
    },
    "adls.": {
        "azure_storage_account_key": "account_key",
        "azure_storage_sas_key": "sas_token",
        "azure_storage_token": "bearer_token",
    },
    "gcs.": {"bearer_token": "bearer_token"},
}


def renewing_credential_provider(
    tbl: pyiceberg.table.Table,
    storage_options: dict[str, Any] | None,
    user_storage_options: Mapping[str, Any] | None,
    location: str,
    *,
    catalog_name: str | None = None,
) -> tuple[dict[str, Any], CredentialProviderBuilder | Literal["auto"]]:
    """
    Move the vended credentials that the catalog can renew out of `storage_options`.

    `storage_options` are those of `tbl`'s files at `location` (`_storage_options`).
    Vended credentials with an expiry are returned as a credential provider instead,
    which renews them from the catalog before they expire. Credentials given by the
    user, and vended ones without an expiry, stay where they are: the provider is
    then "auto", as for other scans and sinks.

    `catalog_name` is the catalog's name if `tbl` was loaded by it: renewals then
    resolve the name too, rather than carry the catalog's properties.
    """
    if not storage_options or _has_credentials(user_storage_options):
        return storage_options or {}, "auto"
    keys = _renewable_keys(tbl, storage_options, location)
    if not keys:
        return storage_options, "auto"

    provider = IcebergCatalogCredentialProvider(
        _table_descriptor(tbl, catalog_name), location, tbl.io.properties
    )
    return (
        {k: v for k, v in storage_options.items() if k not in keys},
        CredentialProviderBuilder.from_initialized_provider(provider),
    )


def _has_credentials(user_storage_options: Mapping[str, Any] | None) -> bool:
    """Whether the user's storage options give credentials (replacing the catalog's)."""
    if user_storage_options is None:
        return False
    return any(
        _is_credential_key(k) for k in _user_storage_options(user_storage_options)
    )


def _renewable_keys(
    tbl: pyiceberg.table.Table, storage_options: Mapping[str, Any], location: str
) -> list[str]:
    """Object store keys of the vended credentials that the catalog can renew."""
    if not _reloadable(tbl) or _expires_at(tbl.io.properties, location) is None:
        return []
    renewable = _RENEWABLE_CREDENTIAL_KEYS.get(_credential_prefix(location), {})
    return [k for k in storage_options if k in renewable]


def _reloadable(tbl: pyiceberg.table.Table) -> bool:
    from pyiceberg.catalog.noop import NoopCatalog
    from pyiceberg.table import StagedTable

    # A staged table (`CREATE TABLE AS`) cannot be loaded before its commit.
    return not isinstance(tbl, StagedTable) and not isinstance(tbl.catalog, NoopCatalog)


def _credential_prefix(location: str) -> str:
    """FileIO property prefix of the credentials of `location`'s cloud; "" if none."""
    scheme = _url_scheme(location)
    return next(
        (
            prefix
            for prefix, schemes in _PROPERTY_SCHEMES.items()
            if scheme in schemes and prefix in _RENEWABLE_CREDENTIAL_KEYS
        ),
        "",
    )


def _table_descriptor(
    tbl: pyiceberg.table.Table, catalog_name: str | None
) -> IcebergCatalogTableDescriptor:
    if catalog_name is not None:
        return IcebergCatalogTableDescriptor(
            table_identifier=tbl.name(), catalog_config=catalog_name
        )
    return IcebergCatalogTableDescriptor(
        table_identifier=tbl.name(),
        catalog_config=IcebergCatalogConfig.from_catalog(tbl.catalog),
        catalog_=NoPickleOption(_reusable_catalog(tbl.catalog)),
    )


class IcebergCatalogCredentialProvider(CachingCredentialProvider):
    """Storage credentials vended by an Iceberg catalog, renewed before they expire."""

    def __init__(
        self,
        table: IcebergCatalogTableDescriptor,
        location: str,
        properties: Mapping[str, Any],
    ) -> None:
        self._table = table
        self._location = location
        # The loaded table's credentials and when they expire. Pickled, so that
        # other processes start with them too.
        self._initial = _vended_credentials(properties, location)
        # Credentials from the environment (e.g. the catalog's) differ per caller,
        # who must not share an object store: this is part of its cache key.
        self._env = _credential_env_fingerprint()
        self._start()

    def _start(self) -> None:
        super().__init__()
        credentials, expires_at = self._initial
        # The last credentials vended, and when a renewal was last tried.
        self._latest = self._initial
        self._renewed_at = 0.0
        self._cached_credentials.set((credentials, _renew_at(expires_at)))

    def __getstate__(self) -> dict[str, Any]:
        # Also the object store's cache key, so only what does not change.
        return {
            k: getattr(self, k) for k in ("_table", "_location", "_initial", "_env")
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._start()

    def retrieve_credentials_impl(self) -> CredentialProviderFunctionReturn:
        """Fetch new credentials from the catalog."""
        try:
            credentials, expires_at = _vended_credentials(
                self._load_properties(), self._location
            )
        except Exception as e:
            # Read after the attempt, which may have taken a while.
            now = self._renewed_at = _now()
            credentials, expires_at = self._latest
            if expires_at is not None and expires_at <= now:
                raise
            # The last ones are used until they expire.
            if verbose():
                eprint(
                    "IcebergCatalogCredentialProvider: renewal failed, retried in "
                    f"{_RETRY_SECONDS}s: {e!r}"
                )
            retry_at = int(now) + _RETRY_SECONDS
            return credentials, min(retry_at, expires_at or retry_at)

        now = self._renewed_at = _now()
        self._latest = credentials, expires_at
        renew_at = _renew_at(expires_at)
        # Vended expired: retried after a while rather than on every request.
        if renew_at is not None and renew_at <= now:
            renew_at = int(now) + _RETRY_SECONDS
        return credentials, renew_at

    def clear_cached_credentials(self) -> None:
        # Polars clears them on any object store error (throttling, timeouts, ...):
        # the catalog is not asked again right after a renewal, or a failed one.
        if _now() - self._renewed_at >= _RETRY_SECONDS:
            super().clear_cached_credentials()

    def _load_properties(self) -> Mapping[str, Any]:
        catalog = self._table.catalog()
        identifier = self._table.table_identifier

        # REST catalogs vend credentials without loading the table, if the server
        # supports it (and tells when they expire).
        if hasattr(catalog, "load_credentials"):
            with contextlib.suppress(NotImplementedError):
                properties = catalog.load_credentials(identifier, self._location)
                if _expires_at(properties, self._location) is not None:
                    return properties

        return catalog.load_table(identifier).io.properties


def _vended_credentials(
    properties: Mapping[str, Any], location: str
) -> tuple[dict[str, str], int | None]:
    """The credentials vended for `location`, and when they expire."""
    storage_options = _catalog_storage_options(properties, location)
    if credentials := {
        provider_key: storage_options[k]
        for k, provider_key in _RENEWABLE_CREDENTIAL_KEYS.get(
            _credential_prefix(location), {}
        ).items()
        if k in storage_options
    }:
        return credentials, _expires_at(properties, location)

    msg = f"iceberg: the catalog did not vend storage credentials for {location}"
    raise ComputeError(msg)


def _expires_at(properties: Mapping[str, Any], location: str) -> int | None:
    """Earliest expiry of the credentials vended for `location`, in epoch seconds."""
    # Iceberg names these `...expires-at[-ms]`, in milliseconds since the epoch, e.g.
    # `s3.session-token-expires-at-ms` or `gcs.oauth2.token-expires-at`. Per-account
    # ones (`adls.sas-token-expires-at-ms.<account host>`) are not translated.
    prefix = _credential_prefix(location)
    if not prefix:
        return None
    expiries = [
        int(v)
        for k, v in properties.items()
        if k.startswith(prefix)
        and k.endswith(("expires-at", "expires-at-ms"))
        and str(v).isdigit()
    ]
    return min(expiries) // 1000 if expiries else None


def _renew_at(expires_at: int | None) -> int | None:
    if expires_at is None:
        return None
    # Credentials expiring sooner than that are used until they expire.
    renew_at = expires_at - _RENEW_BEFORE_EXPIRY_SECONDS
    return renew_at if renew_at > _now() else expires_at


def _now() -> float:
    return time.time()


def refresh_file_io(
    tbl: pyiceberg.table.Table, current: pyiceberg.table.Table | None = None
) -> None:
    """
    Give `tbl` a FileIO with newly vended credentials, if its own are expiring.

    For the PyIceberg side of a sink (manifests, cleanup), which may run long after
    the table was loaded. Best effort: on failure, the FileIO is left as it is.
    `current` is the table as loaded just now, if the caller has it.
    """
    expires_at = _expires_at(tbl.io.properties, _table_location(tbl))
    if (
        not _reloadable(tbl)
        or expires_at is None
        or expires_at - _RENEW_BEFORE_EXPIRY_SECONDS > _now()
    ):
        return
    if current is None:
        try:
            current = tbl.catalog.load_table(tbl.name())
        except Exception as e:
            if verbose():
                eprint(
                    f"iceberg: could not renew the table's storage credentials: {e!r}"
                )
            return
    # Not those of a table dropped and created again.
    if current.metadata.table_uuid == tbl.metadata.table_uuid:
        tbl.io = current.io
