"""Process-wide caches for immutable Iceberg metadata.

Manifest lists, manifests and table metadata files with a write-time UUID in
their name are immutable at a given path, so they can be reused across scans of
the same table within a process. Manifest lists and manifests are cached as
bytes at the PyIceberg ``FileIO`` boundary: PyIceberg reads a metadata file by
calling ``io.new_input(path).open().read()``, and the wrapping ``FileIO`` here
serves that read from memory when the path was seen before. Table metadata files
are cached as the loaded ``StaticTable``, in a cache of their own; scans of the
same metadata path share that table, including its ``FileIO``.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from polars._utils.various import qualified_type_name

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pyiceberg.io import FileIO, InputFile, InputStream
    from pyiceberg.table import StaticTable

ENV_METADATA_FILE_CACHE_MB = "POLARS_ICEBERG_METADATA_FILE_CACHE_MB"
ENV_TABLE_CACHE_MB = "POLARS_ICEBERG_TABLE_CACHE_MB"
DEFAULT_METADATA_FILE_CACHE_MB = 64
DEFAULT_TABLE_CACHE_MB = 64
_BYTES_PER_MB = 2**20

T = TypeVar("T")

_MAX_TOO_LARGE_KEYS = 256

_UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def _has_uuid_name(location: str, suffix: str) -> bool:
    # A Windows path can have a UUID in a directory name.
    name = re.split(r"[/\\]", location)[-1]
    return name.endswith(suffix) and _UUID_PATTERN.search(name) is not None


def _is_cacheable_metadata_file(location: str) -> bool:
    # Manifest lists and manifests with a write-time UUID in the name.
    return _has_uuid_name(location, ".avro")


def _is_cacheable_table_metadata(location: str) -> bool:
    return _has_uuid_name(location, ".metadata.json")


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


def _file_io_scope(file_io: FileIO) -> str | None:
    # `type()`, not `__class__`, which mocks and proxies can override.
    if qualified_type_name(type(file_io)) not in _BUILTIN_FILE_IO_CLASSES:
        return None
    try:
        return _properties_fingerprint(file_io.properties)
    except Exception:
        # Properties that cannot be fingerprinted bypass the cache.
        return None


@dataclass
class CacheStats:
    """Reads through a cache, or why the cache was bypassed."""

    hits: int = 0
    misses: int = 0
    # Misses not stored because they are larger than the cache.
    too_large: int = 0
    # Entries evicted to store the misses.
    evicted: int = 0
    bypass: str | None = None

    def describe(self, cache: _LruCache[Any]) -> str:
        """Summary for verbose output."""
        if self.bypass is not None:
            return f"bypassed: {self.bypass}"
        return (
            f"hits: {self.hits}, misses: {self.misses}, "
            f"too large: {self.too_large}, evicted: {self.evicted}, "
            f"cached bytes: {cache.total_bytes}"
        )


class _LruCache(Generic[T]):
    """Cache with LRU eviction bounded by the total of the entry sizes."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[T, int]] = OrderedDict()
        self._total_bytes = 0
        # Per-key locks so concurrent misses on one key fetch once.
        self._fetch_locks: dict[str, threading.Lock] = {}
        # Keys of values too large to store, most recent last.
        self._too_large: OrderedDict[str, None] = OrderedDict()

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

    def _get_locked(self, key: str) -> T | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry[0]

    def get(self, key: str) -> T | None:
        with self._lock:
            return self._get_locked(key)

    def is_too_large(self, key: str) -> bool:
        """Whether a value for `key` was too large to store."""
        with self._lock:
            return key in self._too_large

    def put(
        self, key: str, value: T, size: int, stats: CacheStats | None = None
    ) -> None:
        """Store `value`, accounted at `size` bytes plus the key length.

        `stats` counts a value too large to store and the entries evicted.
        """
        # Keys count towards the budget, so empty entries are bounded too.
        size += len(key)

        with self._lock:
            if size > self.max_bytes:
                if stats is not None:
                    stats.too_large += 1
                self._too_large[key] = None
                self._too_large.move_to_end(key)
                if len(self._too_large) > _MAX_TOO_LARGE_KEYS:
                    self._too_large.popitem(last=False)
                return

            if key in self._entries:
                return

            self._entries[key] = (value, size)
            self._total_bytes += size

            while self._total_bytes > self.max_bytes:
                _, (_, evicted_size) = self._entries.popitem(last=False)
                self._total_bytes -= evicted_size
                if stats is not None:
                    stats.evicted += 1

    def get_or_fetch(
        self, key: str, fetch: Callable[[], tuple[T, int]], stats: CacheStats
    ) -> T:
        """Return the cached value, fetching on a miss; `stats` counts it.

        `fetch` returns the value and the size it is accounted at.
        """
        with self._lock:
            if (cached := self._get_locked(key)) is not None:
                stats.hits += 1
                return cached

            fetch_lock = self._fetch_locks.setdefault(key, threading.Lock())

        try:
            with fetch_lock:
                with self._lock:
                    if (cached := self._get_locked(key)) is not None:
                        stats.hits += 1
                        return cached

                    stats.misses += 1

                value, size = fetch()
                self.put(key, value, size, stats)
        finally:
            with self._lock:
                # A newer lock for the same key may have replaced this one.
                if self._fetch_locks.get(key) is fetch_lock:
                    del self._fetch_locks[key]

        return value


class IcebergMetadataFileCache(_LruCache[bytes]):
    """Cache of manifest list and manifest bytes."""


class IcebergTableCache(_LruCache["StaticTable"]):
    """Cache of tables loaded from table metadata files."""


def _bypass_reason(
    cache: _LruCache[Any], file_io: FileIO, scope: str | None
) -> str | None:
    # `scope` is `_file_io_scope(file_io)`.
    if not cache.enabled:
        return "disabled"
    if scope is None:
        if qualified_type_name(type(file_io)) not in _BUILTIN_FILE_IO_CLASSES:
            return "custom FileIO"
        return "FileIO property that is not plain"
    return None


_global_metadata_file_cache: IcebergMetadataFileCache | None = None
_global_table_cache: IcebergTableCache | None = None
_global_cache_lock = threading.Lock()


def _configured_size(env_var: str, default_mb: int) -> int:
    value = os.getenv(env_var)
    if value is None:
        return default_mb * _BYTES_PER_MB

    try:
        size_mb = int(value)
    except ValueError:
        size_mb = -1

    if size_mb < 0:
        msg = (
            f"invalid value for {env_var}: {value!r}, expected a "
            "non-negative number of MiB"
        )
        raise ValueError(msg)

    return size_mb * _BYTES_PER_MB


def get_metadata_file_cache() -> IcebergMetadataFileCache:
    """Return the process-wide metadata file cache, built from the environment."""
    global _global_metadata_file_cache

    if _global_metadata_file_cache is None:
        with _global_cache_lock:
            if _global_metadata_file_cache is None:
                _global_metadata_file_cache = IcebergMetadataFileCache(
                    _configured_size(
                        ENV_METADATA_FILE_CACHE_MB, DEFAULT_METADATA_FILE_CACHE_MB
                    )
                )

    return _global_metadata_file_cache


def get_table_cache() -> IcebergTableCache:
    """Return the process-wide table cache, built from the environment."""
    global _global_table_cache

    if _global_table_cache is None:
        with _global_cache_lock:
            if _global_table_cache is None:
                _global_table_cache = IcebergTableCache(
                    _configured_size(ENV_TABLE_CACHE_MB, DEFAULT_TABLE_CACHE_MB)
                )

    return _global_table_cache


def reset_caches() -> None:
    """Drop the process-wide caches; the next use rebuilds them from the environment."""
    global _global_metadata_file_cache, _global_table_cache

    with _global_cache_lock:
        _global_metadata_file_cache = None
        _global_table_cache = None


# Measured on PyIceberg 0.12 across metadata files of 0.1 to 20 MB: the parsed
# metadata is 5.7x the size of its JSON, `model_dump_json()` 0.92x.
_PARSED_BYTES_PER_JSON_BYTE = 6

# Counted per cached table, which holds a FileIO and its storage clients.
_TABLE_OVERHEAD_BYTES = 2**20


def load_static_table(
    metadata_location: str, properties: dict[str, Any], stats: CacheStats
) -> StaticTable:
    """Load the table at `metadata_location` with `StaticTable.from_metadata`.

    Served from the table cache when the file name carries a UUID and
    `properties` can be scoped; `stats` counts that cache read, or records why the
    cache was bypassed. A cached table is shared by every scan of the same
    metadata file; a `StaticTable` cannot refresh, and nothing in Polars modifies
    it. A table too large for the cache bypasses it on later loads.
    """
    from pyiceberg.io import load_file_io
    from pyiceberg.table import StaticTable

    def load() -> tuple[StaticTable, int]:
        table = StaticTable.from_metadata(metadata_location, properties=properties)
        size = (
            len(table.metadata.model_dump_json()) * _PARSED_BYTES_PER_JSON_BYTE
            + _TABLE_OVERHEAD_BYTES
        )
        return table, size

    cache = get_table_cache()
    file_io = load_file_io(properties, location=metadata_location)
    scope = _file_io_scope(file_io)
    key = f"{scope}:{metadata_location}"
    stats.bypass = _bypass_reason(cache, file_io, scope)
    if stats.bypass is None and not _is_cacheable_table_metadata(metadata_location):
        stats.bypass = "no UUID in metadata file name"
    if stats.bypass is None and cache.is_too_large(key):
        stats.bypass = "too large for the cache"
    if stats.bypass is not None:
        return StaticTable.from_metadata(metadata_location, properties=properties)
    return cache.get_or_fetch(key, load, stats)


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

    def _fetch(self) -> tuple[bytes, int]:
        with self._inner.new_input(self._location).open(seekable=False) as f:
            data = f.read()
        return data, len(data)

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
        scope = _file_io_scope(inner)
        self.stats = CacheStats(bypass=_bypass_reason(cache, inner, scope))
        # None when reads bypass the cache.
        self._scope = scope if self.stats.bypass is None else None

    @property
    def cache(self) -> IcebergMetadataFileCache:
        return self._cache

    def new_input(self, location: str) -> InputFile:
        if (scope := self._scope) is not None and _is_cacheable_metadata_file(location):
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


def describe_metadata_file_cache(file_io: FileIO) -> str:
    """Verbose summary of the metadata file cache reads through `file_io`."""
    if isinstance(file_io, CachingFileIO):
        return file_io.stats.describe(file_io.cache)
    cache = get_metadata_file_cache()
    stats = CacheStats(bypass=_bypass_reason(cache, file_io, _file_io_scope(file_io)))
    return stats.describe(cache)


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
