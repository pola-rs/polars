"""Process-wide cache for immutable Iceberg metadata files.

Manifest lists and manifests are immutable at a given path, so their bytes can
be reused across scans of the same table within a process. The cache sits at
the PyIceberg ``FileIO`` boundary: PyIceberg reads a metadata file by calling
``io.new_input(path).open().read()``, and the wrapping ``FileIO`` here serves
that read from memory when the path was seen before.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

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


def _properties_fingerprint(properties: Mapping[str, Any]) -> str | None:
    # None when a key or value is not of an exact plain type: nested settings,
    # objects and subclasses, such as REST catalog auth or secret wrappers, can
    # carry identity that their repr does not show.
    items = []
    for k, v in properties.items():
        if type(k) is not str:
            return None
        if k in _FINGERPRINT_EXCLUDED_KEYS:
            continue
        if type(v) not in _PLAIN_TYPES:
            return None
        items.append((k, repr(v)))

    return hashlib.sha256(repr(sorted(items)).encode()).hexdigest()


def _file_io_scope(file_io: FileIO) -> str | None:
    cls = type(file_io)
    if f"{cls.__module__}.{cls.__qualname__}" not in _BUILTIN_FILE_IO_CLASSES:
        return None
    return _properties_fingerprint(file_io.properties)


class IcebergMetadataFileCache:
    """Byte cache with LRU eviction bounded by total size."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, bytes] = OrderedDict()
        self._total_bytes = 0
        # Per-path locks so concurrent misses on one path fetch once.
        self._fetch_locks: dict[str, threading.Lock] = {}
        self.hits = 0
        self.misses = 0

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

    def _get_locked(self, location: str) -> bytes | None:
        data = self._entries.get(location)
        if data is not None:
            self._entries.move_to_end(location)
        return data

    def get(self, location: str) -> bytes | None:
        with self._lock:
            return self._get_locked(location)

    def put(self, location: str, data: bytes) -> None:
        if len(data) > self.max_bytes:
            return

        with self._lock:
            if location in self._entries:
                return

            self._entries[location] = data
            self._total_bytes += len(data)

            while self._total_bytes > self.max_bytes:
                _, evicted = self._entries.popitem(last=False)
                self._total_bytes -= len(evicted)

    def get_or_fetch(self, location: str, fetch: Callable[[], bytes]) -> bytes:
        with self._lock:
            if (data := self._get_locked(location)) is not None:
                self.hits += 1
                return data

            fetch_lock = self._fetch_locks.setdefault(location, threading.Lock())

        try:
            with fetch_lock:
                with self._lock:
                    if (data := self._get_locked(location)) is not None:
                        self.hits += 1
                        return data

                    self.misses += 1

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
    ) -> None:
        self._inner = inner
        self._location = location
        self._cache = cache
        self._key = f"{scope}:{location}"

    @property
    def location(self) -> str:
        return self._location

    def _fetch(self) -> bytes:
        with self._inner.new_input(self._location).open(seekable=False) as f:
            return f.read()

    def _bytes(self) -> bytes:
        return self._cache.get_or_fetch(self._key, self._fetch)

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
    properties of the wrapped FileIO. Nothing is cached when a property is not
    of a plain type, or when the wrapped FileIO is not one of PyIceberg's
    built-in classes.
    """

    def __init__(self, inner: FileIO, cache: IcebergMetadataFileCache) -> None:
        self._inner = inner
        self._cache = cache
        self.properties = inner.properties
        self._scope = _file_io_scope(inner)

    def new_input(self, location: str) -> InputFile:
        scope = self._scope
        if scope is not None and self._cache.enabled and _is_cacheable(location):
            return CachedInputFile(  # type: ignore[return-value]
                self._inner, location, self._cache, scope
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
