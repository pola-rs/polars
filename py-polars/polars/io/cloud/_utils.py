from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from polars._utils.various import is_path_or_str_sequence
from polars.io.partition import PartitionMaxSize

if TYPE_CHECKING:
    from collections.abc import KeysView

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


class NoPickleOption(Generic[T]):
    """
    Wrapper that does not pickle the wrapped value.

    This wrapper will unpickle to contain a None. Used for cached values.
    """

    def __init__(self, opt_value: T | None = None) -> None:
        self._opt_value = opt_value

    def get(self) -> T | None:
        return self._opt_value

    def set(self, value: T | None) -> None:
        self._opt_value = value

    def __getstate__(self) -> tuple[()]:
        # Needs to return not-None for `__setstate__()` to be called
        return ()

    def __setstate__(self, _state: tuple[()]) -> None:
        NoPickleOption.__init__(self)


class LRUCache(Generic[K, V]):
    def __init__(self, max_items: int) -> None:
        self._max_items = 0
        self._dict: OrderedDict[K, V] = OrderedDict()

        self.set_max_items(max_items)

    def __len__(self) -> int:
        return len(self._dict)

    def get(self, key: K) -> V:
        """Raises KeyError if the key is not found."""
        self._dict.move_to_end(key)
        return self._dict[key]

    def keys(self) -> KeysView[K]:
        return self._dict.keys()

    def contains(self, key: K) -> bool:
        return key in self._dict

    def insert(self, key: K, value: V) -> None:
        """Insert a value into the cache."""
        if self.max_items() == 0:
            return

        while len(self) >= self.max_items():
            self.remove_lru()

        self._dict[key] = value

    def remove(self, key: K) -> V:
        """Raises KeyError if the key is not found."""
        return self._dict.pop(key)

    def max_items(self) -> int:
        return self._max_items

    def set_max_items(self, max_items: int) -> None:
        """
        Set a new maximum number of items.

        The cache is trimmed if its length exceeds the new maximum.
        """
        if max_items < 0:
            msg = f"max_items cannot be negative: {max_items}"
            raise ValueError(msg)

        while len(self) > max_items:
            self.remove_lru()

        self._max_items = max_items

    def remove_lru(self) -> tuple[K, V]:
        """
        Remove the least recently used value.

        Raises KeyError if the cache is empty.
        """
        return self._dict.popitem(last=False)


def _first_scan_path(
    source: Any,
) -> str | Path | None:
    if isinstance(source, (str, Path)):
        return source
    elif is_path_or_str_sequence(source) and source:
        return source[0]
    elif isinstance(source, PartitionMaxSize):
        return source._base_path

    return None


def _get_path_scheme(path: str | Path) -> str | None:
    path_str = str(path)
    i = path_str.find("://")

    return path_str[:i] if i >= 0 else None


def _is_aws_cloud(scheme: str) -> bool:
    return any(scheme == x for x in ["s3", "s3a"])


def _is_azure_cloud(scheme: str) -> bool:
    return any(scheme == x for x in ["az", "azure", "adl", "abfs", "abfss"])


def _is_gcp_cloud(scheme: str) -> bool:
    return any(scheme == x for x in ["gs", "gcp", "gcs"])
