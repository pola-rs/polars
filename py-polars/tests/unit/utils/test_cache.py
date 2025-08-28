from typing import Any

import pytest

from polars._utils.cache import LRUCache


def test_init() -> None:
    cache = LRUCache[str, float](maxsize=0)
    assert cache.maxsize == 0
    assert len(cache) == 0

    with pytest.raises(ValueError, match="`maxsize` cannot be negative; found -100"):
        LRUCache(maxsize=-100)


def test_init_maxsize_zero() -> None:
    # weird, but not actually invalid
    cache = LRUCache[str, list[int]](maxsize=0)
    cache["key"] = [1, 2, 3]
    assert len(cache) == 0


def test_setitem_getitem_delitem() -> None:
    cache = LRUCache[int, str](maxsize=3)
    cache[0] = "pear"
    cache[1] = "apple"
    cache[2] = "banana"
    cache[3] = "cherry"

    assert list(cache.keys()) == [1, 2, 3]
    assert list(cache.values()) == ["apple", "banana", "cherry"]
    assert list(cache.items()) == [(1, "apple"), (2, "banana"), (3, "cherry")]
    assert len(cache) == 3
    assert cache.maxsize == 3

    del cache[2]
    assert len(cache) == 2
    assert cache.maxsize == 3
    assert repr(cache) == "LRUCache({1: 'apple', 3: 'cherry'}, maxsize=3, currsize=2)"


def test_cache_access_updates_order() -> None:
    cache1 = LRUCache[int, str](maxsize=3)
    cache1[30] = "thirty"
    cache1[20] = "twenty"
    cache1[10] = "ten"
    assert list(cache1.keys()) == [30, 20, 10]

    cache1.get(20)
    cache1.get(30)
    assert list(cache1.keys()) == [10, 20, 30]

    cache2 = LRUCache[str, tuple[int, int]](maxsize=2)
    cache2["first"] = (1, 2)
    cache2["second"] = (3, 4)
    assert list(cache2.keys()) == ["first", "second"]

    _ = cache2["first"]
    assert list(cache2.keys()) == ["second", "first"]


def test_contains() -> None:
    cache = LRUCache[str, float](maxsize=3)
    cache["pi"] = 3.14159
    cache["e"] = 2.71828

    assert "pi" in cache
    assert "e" in cache
    assert "phi" not in cache


def test_getitem_keyerror() -> None:
    cache = LRUCache[int, str](maxsize=3)
    with pytest.raises(KeyError, match="999 not found in cache"):
        cache[999]


def test_get_with_default() -> None:
    cache = LRUCache[str, list[str]](maxsize=3)
    cache["fruits"] = ["apple", "banana"]

    assert cache.get("fruits") == ["apple", "banana"]
    assert cache.get("vegetables") is None
    assert cache.get("vegetables", ["carrot", "turnip"]) == ["carrot", "turnip"]


def test_iter_and_update() -> None:
    cache = LRUCache[int, str](maxsize=4)
    cache.update({100: "hundred", 200: "two hundred", 300: "three hundred"})
    assert list(cache) == [100, 200, 300]


def test_maxsize_eviction() -> None:
    cache = LRUCache[str, dict[str, Any]](maxsize=2)
    cache["user1"] = {"name": "Alice", "age": 30}
    cache["user2"] = {"name": "Bob", "age": 25}
    cache["user3"] = {"name": "Charlie", "age": 35}

    assert len(cache) == 2
    assert "user1" not in cache
    assert "user2" in cache
    assert "user3" in cache


def test_maxsize_increase() -> None:
    cache = LRUCache[str, int].fromkeys(4, keys=[f"k{n}" for n in range(6)], value=0)
    assert len(cache) == 4
    assert cache.maxsize == 4
    assert list(cache.keys()) == ["k2", "k3", "k4", "k5"]

    cache.maxsize = 6
    cache.update(k6=1, k7=2, k8=3)
    assert len(cache) == 6
    assert cache.maxsize == 6
    assert list(cache.keys()) == ["k3", "k4", "k5", "k6", "k7", "k8"]


def test_maxsize_decrease() -> None:
    cache = LRUCache[int, list[int]](maxsize=4)
    cache[1] = [1, 2]
    cache[2] = [3, 4]
    cache[3] = [5, 6]
    cache[4] = [7, 8]

    cache.maxsize = 2
    assert len(cache) == 2
    assert cache.maxsize == 2
    assert list(cache.items()) == [(3, [5, 6]), (4, [7, 8])]

    cache.maxsize = 0
    assert len(cache) == 0
    assert cache.maxsize == 0
    assert list(cache.values()) == []


def test_pop() -> None:
    cache = LRUCache[int, str](maxsize=4)
    cache[42] = "answer"
    cache[99] = "bottles"

    value = cache.pop(42)
    assert value == "answer"
    assert 42 not in cache
    assert len(cache) == 1

    with pytest.raises(KeyError, match="404"):
        cache.pop(404)


def test_popitem_setdefault() -> None:
    cache = LRUCache[str, set[int]](maxsize=3)
    cache["set1"] = {1, 2, 3}
    cache["set2"] = {4, 5, 6}
    cache["set3"] = {7, 8, 9}

    key, value = cache.popitem()
    assert key == "set1"
    assert value == {1, 2, 3}
    assert len(cache) == 2

    res = cache.setdefault("set2", {10, 11, 12})
    assert res == {4, 5, 6}
    assert list(cache) == ["set3", "set2"]

    res = cache.setdefault("set4", {10, 11, 12})
    assert res == {10, 11, 12}
    assert list(cache) == ["set3", "set2", "set4"]


def test_popitem_empty_cache() -> None:
    cache = LRUCache[str, bytes](maxsize=3)
    with pytest.raises(KeyError):
        cache.popitem()


def test_update_existing_key() -> None:
    cache = LRUCache[int, float](maxsize=3)
    cache.update([(1, 1.5), (2, 2.5)])
    assert list(cache.keys()) == [1, 2]

    cache[1] = 10.5
    assert list(cache.keys()) == [2, 1]
    assert cache[1] == 10.5


def test_update_existing_keys() -> None:
    cache = LRUCache[str, float](maxsize=3)
    cache["pi"] = 3.14
    cache["e"] = 2.71

    cache.update({"phi": 1.618, "pi": 3.14159})
    assert tuple(cache.items()) == (("e", 2.71), ("phi", 1.618), ("pi", 3.14159))


def test_update_check_eviction() -> None:
    cache = LRUCache[int, str](maxsize=2)
    cache[1] = "first"
    cache[2] = "second"

    cache.update({3: "third", 4: "fourth"})

    assert 1 not in cache
    assert 2 not in cache
    assert cache[3] == "third"
    assert cache[4] == "fourth"
    assert len(cache) == 2


def test_update_preserves_order() -> None:
    cache = LRUCache[str, int](maxsize=3)
    cache["a"] = 1
    cache["b"] = 2

    cache.update({"c": 3})
    assert list(cache.keys()) == ["a", "b", "c"]

    cache.update({"a": 10, "x": 4})
    assert list(cache.items()) == [("c", 3), ("a", 10), ("x", 4)]
