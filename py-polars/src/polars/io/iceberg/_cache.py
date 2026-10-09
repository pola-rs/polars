"""Process-wide cache for immutable Iceberg metadata files.

Manifest lists and manifests are immutable at a given path, so their bytes can
be reused across scans of the same table within a process. The cache sits at
the PyIceberg ``FileIO`` boundary: PyIceberg reads a metadata file by calling
``io.new_input(path).open().read()``, and the wrapping ``FileIO`` here serves
that read from memory when the path was seen before.

Scans planned by the ``polars_iceberg`` plugin read metadata files through
Polars' storage instead; those reads are cached on the Rust side, in a cache of
the same size owned by the process-wide cache here (see ``plugin_cache``).
"""

from __future__ import annotations

import hashlib
import io
import itertools
import os
import re
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from polars._utils.various import qualified_type_name

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pyiceberg.io import FileIO, InputFile, InputStream

ENV_CACHE_MB = "POLARS_ICEBERG_METADATA_CACHE_MB"
DEFAULT_CACHE_MB = 64
_BYTES_PER_MB = 1_000_000

_UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def _is_cacheable(location: str) -> bool:
    # Manifest lists and manifests: Avro files with a write-time UUID in the name.
    name = location.rsplit("/", 1)[-1]
    return name.endswith(".avro") and _UUID_PATTERN.search(name) is not None


# Set by REST catalogs, changes on every commit to the table.
_FINGERPRINT_EXCLUDED_KEYS = frozenset(
    ("metadata_location", "previous_metadata_location")
)


_PLAIN_TYPES = (str, bytes, int, float, bool, type(None))

# Custom FileIO classes can hold identity outside their properties.
_BUILTIN_FILE_IO_CLASSES = frozenset(
    ("pyiceberg.io.pyarrow.PyArrowFileIO", "pyiceberg.io.fsspec.FsspecFileIO")
)


# Set by the REST catalog: the auth manager for catalog requests and remote signing.
_AUTH_MANAGER_KEY = "auth.manager"

# PyIceberg's own auth managers, built from the `auth`, `token` and `credential`
# properties or from credentials taken from the environment. Other auth managers
# can hold identity outside the properties.
_BUILTIN_AUTH_MANAGER_CLASSES = frozenset(
    (
        "pyiceberg.catalog.rest.auth.BasicAuthManager",
        "pyiceberg.catalog.rest.auth.EntraAuthManager",
        "pyiceberg.catalog.rest.auth.GoogleAuthManager",
        "pyiceberg.catalog.rest.auth.LegacyOAuth2AuthManager",
        "pyiceberg.catalog.rest.auth.NoopAuthManager",
        "pyiceberg.catalog.rest.auth.OAuth2AuthManager",
    )
)


_MAX_FINGERPRINT_DEPTH = 8
_MAX_FINGERPRINT_VALUES = 10_000


def _is_plain(value: Any) -> bool:
    # Exact plain types only: objects and subclasses, such as secret wrappers, can
    # carry identity that their repr does not show. Containers nested deeper than
    # `_MAX_FINGERPRINT_DEPTH`, which includes self-referencing ones, or holding more
    # than `_MAX_FINGERPRINT_VALUES` values in total are not plain.
    pending = [(value, 0)]
    budget = _MAX_FINGERPRINT_VALUES
    while pending:
        v, depth = pending.pop()
        t = type(v)
        if t in _PLAIN_TYPES:
            continue
        if t is not dict and t is not list and t is not tuple:
            return False
        if depth >= _MAX_FINGERPRINT_DEPTH or len(v) > budget:
            return False
        budget -= len(v)
        if t is dict:
            if any(type(k) is not str for k in v):
                return False
            pending.extend((x, depth + 1) for x in v.values())
        else:
            pending.extend((x, depth + 1) for x in v)
    return True


def _properties_fingerprint(properties: Mapping[str, Any]) -> str | None:
    # None when a property is not plain. A built-in REST auth manager is left out:
    # the properties it is built from are fingerprinted.
    kept = {
        k: v
        for k, v in properties.items()
        if k not in _FINGERPRINT_EXCLUDED_KEYS
        and not (
            k == _AUTH_MANAGER_KEY
            and qualified_type_name(type(v)) in _BUILTIN_AUTH_MANAGER_CLASSES
        )
    }
    if not _is_plain(kept):
        return None
    return hashlib.sha256(repr(sorted(kept.items())).encode()).hexdigest()


# Environment variables from which object store clients take credentials.
_CREDENTIAL_ENV_PREFIXES = ("AWS_", "AZURE_", "GOOGLE_", "GCS_", "GCP_")
_CREDENTIAL_ENV_VARS = frozenset(
    ("IDENTITY_ENDPOINT", "IDENTITY_HEADER", "MSI_ENDPOINT")
)


def _credential_env_fingerprint() -> str:
    env = sorted(
        (k, v)
        for k, v in os.environ.items()
        if k.startswith(_CREDENTIAL_ENV_PREFIXES) or k in _CREDENTIAL_ENV_VARS
    )
    return hashlib.sha256(repr(env).encode()).hexdigest()


def _file_io_scope(file_io: FileIO) -> str | None:
    # `type()`, not `__class__`, which mocks and proxies can override.
    if qualified_type_name(type(file_io)) not in _BUILTIN_FILE_IO_CLASSES:
        return None
    try:
        properties_scope = _properties_fingerprint(file_io.properties)
    except Exception:
        # Properties that cannot be fingerprinted bypass the cache.
        return None
    if properties_scope is None:
        return None
    # Without credentials in the properties, FileIOs take them from the environment.
    return hashlib.sha256(
        f"{properties_scope}:{_credential_env_fingerprint()}".encode()
    ).hexdigest()


# Unique tokens of credential provider objects. Not `id()`, which is reused after an
# object is freed.
_PROVIDER_TOKENS: weakref.WeakKeyDictionary[Any, int] = weakref.WeakKeyDictionary()
_PROVIDER_TOKENS_LOCK = threading.Lock()
_next_provider_token = itertools.count()


def _default_credential_provider_scope() -> str | None:
    # The plugin's storage uses Polars' default credential provider (`pl.Config`).
    # None when it cannot be identified.
    import polars.io.cloud.credential_provider._builder as builder

    provider = builder.DEFAULT_CREDENTIAL_PROVIDER
    if provider is None or isinstance(provider, str):
        return repr(provider)
    try:
        with _PROVIDER_TOKENS_LOCK:
            token = _PROVIDER_TOKENS.get(provider)
            if token is None:
                token = _PROVIDER_TOKENS[provider] = next(_next_provider_token)
    except TypeError:
        # Not weakly referenceable.
        return None
    return f"provider:{token}"


def plugin_storage_scope(
    file_io: FileIO, storage_options: Mapping[str, Any] | None
) -> str | None:
    """Cache scope of a scan planned by the plugin.

    The plugin reads with storage configured from the FileIO's properties, the
    user's storage options and Polars' default credential provider, so all are
    fingerprinted. None when one cannot be.
    """
    if (io_scope := _file_io_scope(file_io)) is None:
        return None
    if (provider_scope := _default_credential_provider_scope()) is None:
        return None
    try:
        options_scope = _properties_fingerprint(storage_options or {})
    except Exception:
        return None
    if options_scope is None:
        return None
    return hashlib.sha256(
        f"{io_scope}:{options_scope}:{provider_scope}".encode()
    ).hexdigest()


@dataclass
class CacheStats:
    """Hit and miss counts."""

    hits: int = 0
    misses: int = 0


class IcebergMetadataFileCache:
    """Byte cache with LRU eviction bounded by total size."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, bytes] = OrderedDict()
        self._total_bytes = 0
        # Per-path locks so concurrent misses on one path fetch once.
        self._fetch_locks: dict[str, threading.Lock] = {}
        self._plugin_cache: Any = None

    @property
    def enabled(self) -> bool:
        return self.max_bytes > 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def plugin_cache(self) -> Any:
        """Rust-side cache of the reads of scans planned by the plugin.

        Of the same size as this cache, and dropped with it.
        """
        with self._lock:
            if self._plugin_cache is None:
                import polars._plr as plr

                self._plugin_cache = plr.PyIcebergMetadataFileCache(self.max_bytes)
            return self._plugin_cache

    def _get_locked(self, location: str) -> bytes | None:
        data = self._entries.get(location)
        if data is not None:
            self._entries.move_to_end(location)
        return data

    def get(self, location: str) -> bytes | None:
        with self._lock:
            return self._get_locked(location)

    def put(self, location: str, data: bytes) -> None:
        # Keys count towards the budget, so empty entries are bounded too.
        size = len(location) + len(data)
        if size > self.max_bytes:
            return

        with self._lock:
            if location in self._entries:
                return

            self._entries[location] = data
            self._total_bytes += size

            while self._total_bytes > self.max_bytes:
                key, evicted = self._entries.popitem(last=False)
                self._total_bytes -= len(key) + len(evicted)

    def get_or_fetch(
        self, location: str, fetch: Callable[[], bytes], stats: CacheStats
    ) -> bytes:
        """Return the cached bytes, fetching on a miss; `stats` counts it."""
        with self._lock:
            if (data := self._get_locked(location)) is not None:
                stats.hits += 1
                return data

            fetch_lock = self._fetch_locks.setdefault(location, threading.Lock())

        try:
            with fetch_lock:
                with self._lock:
                    if (data := self._get_locked(location)) is not None:
                        stats.hits += 1
                        return data

                    stats.misses += 1

                data = fetch()
                self.put(location, data)
        finally:
            with self._lock:
                # A newer lock for the same path may have replaced this one.
                if self._fetch_locks.get(location) is fetch_lock:
                    del self._fetch_locks[location]

        return data


_global_cache: IcebergMetadataFileCache | None = None
_global_cache_lock = threading.Lock()


def _configured_size() -> int:
    value = os.getenv(ENV_CACHE_MB)
    if value is None:
        return DEFAULT_CACHE_MB * _BYTES_PER_MB

    try:
        size_mb = int(value)
    except ValueError:
        size_mb = -1

    if size_mb < 0:
        msg = (
            f"invalid value for {ENV_CACHE_MB}: {value!r}, expected a "
            "non-negative number of megabytes"
        )
        raise ValueError(msg)

    return size_mb * _BYTES_PER_MB


def get_metadata_file_cache() -> IcebergMetadataFileCache:
    """Return the process-wide cache, constructing it from the environment."""
    global _global_cache

    if _global_cache is None:
        with _global_cache_lock:
            if _global_cache is None:
                _global_cache = IcebergMetadataFileCache(_configured_size())

    return _global_cache


def reset_metadata_file_cache() -> None:
    """Drop the process-wide cache; the next use rebuilds it from the environment."""
    global _global_cache

    with _global_cache_lock:
        _global_cache = None


class CachedInputFile:
    """InputFile that serves reads from the cache, fetching on a miss."""

    def __init__(
        self,
        inner: FileIO,
        location: str,
        cache: IcebergMetadataFileCache,
        scope: str,
        stats: CacheStats,
    ) -> None:
        self._inner = inner
        self._location = location
        self._cache = cache
        self._key = f"{scope}:{location}"
        self._stats = stats

    @property
    def location(self) -> str:
        return self._location

    def _fetch(self) -> bytes:
        with self._inner.new_input(self._location).open(seekable=False) as f:
            return f.read()

    def _bytes(self) -> bytes:
        return self._cache.get_or_fetch(self._key, self._fetch, self._stats)

    def __len__(self) -> int:
        if (data := self._cache.get(self._key)) is not None:
            return len(data)
        return len(self._inner.new_input(self._location))

    def exists(self) -> bool:
        if self._cache.get(self._key) is not None:
            return True
        return self._inner.new_input(self._location).exists()

    def open(self, seekable: bool = True) -> InputStream:  # noqa: ARG002, FBT001
        return io.BytesIO(self._bytes())


class CachingFileIO:
    """FileIO wrapper that caches immutable metadata files.

    Reads of cacheable paths go through the cache. Everything else is
    forwarded to the wrapped FileIO. Cache entries are scoped to the
    properties of the wrapped FileIO. Nothing is cached when a property holds a
    value that is not of a plain type, other than one of PyIceberg's built-in
    REST auth managers, or when the wrapped FileIO is not one of PyIceberg's
    built-in classes. `stats` counts the cache reads through this wrapper.
    """

    def __init__(self, inner: FileIO, cache: IcebergMetadataFileCache) -> None:
        self._inner = inner
        self._cache = cache
        self.properties = inner.properties
        self._scope = _file_io_scope(inner)
        self.stats = CacheStats()

    @property
    def cache(self) -> IcebergMetadataFileCache:
        return self._cache

    def new_input(self, location: str) -> InputFile:
        scope = self._scope
        if scope is not None and self._cache.enabled and _is_cacheable(location):
            return CachedInputFile(  # type: ignore[return-value]
                self._inner, location, self._cache, scope, self.stats
            )
        return self._inner.new_input(location)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def __reduce__(self) -> tuple[Any, ...]:
        # The cache is process-local, a copy attaches the process-wide cache.
        return (_wrap_file_io, (self._inner,))


def _wrap_file_io(inner: FileIO) -> CachingFileIO:
    return CachingFileIO(inner, get_metadata_file_cache())


def with_metadata_file_cache(scan: Any) -> Any:
    """Route the metadata reads of a PyIceberg scan through the cache.

    Replaces the scan's FileIO with a caching wrapper. The table the scan was
    created from is left untouched. Returns the scan unchanged when the cache
    is disabled or the scan is already wrapped.
    """
    inner = getattr(scan, "io", None)

    # A nested wrapper deadlocks: both levels take the same per-file fetch lock.
    if inner is None or isinstance(inner, CachingFileIO):
        return scan

    cache = get_metadata_file_cache()

    if not cache.enabled:
        return scan

    scan.io = CachingFileIO(inner, cache)
    return scan
