"""
Simple mapping cache implementation.
"""

import asyncio
import functools
import pickle
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Sequence, TypeVarTuple, TypeVar

from core.toolkit.hashing import stable_hash


T = TypeVarTuple("T")
U = TypeVar("U")
V = TypeVar("V")


def package_func_args(
    func: Callable[..., Any], args: tuple[*T], kwargs: dict[U, V]
) -> tuple[str, tuple[*T], tuple[tuple[U, V], ...]]:
    """
    Package function arguments into a tuple.
    """
    kwarg_items = sorted(kwargs.items())
    return func.__name__, args, tuple(kwarg_items)


class SimpleCache:
    """
    A simple dictionary-like persistent cache where the value of each key is stored in its own file.

    This cache was designed for infrequent but expensive operations that can persist across Python runs. It is NOT as fast as a true dictionary, as reading/setting values both require disk access.

    Parameters
    ----------
    cache_dir : Path
        The directory that the cache files will reside in. If it does not already exist, it will be created.
        Note that the directory will not be cleared when the Python program finishes running. Multiple `SimpleCache` objects created with the same directory may behave erraticly.

    Attributes
    ----------
    cache_dir : Path
        The directory that the cache files will reside in.
    hashes : list[str]
        The hashes of the keys in the cache.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hashes: list[str] = [path.name for path in cache_dir.glob("*")]

    def __getitem__(self, key: Hashable) -> Any:
        """Get the value associated with a key."""
        key_hash = stable_hash(key)
        if key_hash not in self.hashes:
            raise KeyError(key)
        with open(self.cache_dir / key_hash, "rb") as file:
            return pickle.load(file)

    # def __setitem__(self, key: "tuple[str, tuple, frozenset]", value: Any):
    def __setitem__(self, key: Hashable, value: Any):
        """Set the value associated with a key."""
        key_hash = stable_hash(key)
        if key_hash not in self.hashes:
            self.hashes.append(key_hash)
        with open(self.cache_dir / key_hash, "wb") as file:
            pickle.dump(value, file)

    def __contains__(self, key: Hashable):
        """Check if the cache has a saved value associated with a key."""
        return stable_hash(key) in self.hashes

    def __delitem__(self, key: Hashable):
        """Delete the value associated with a key."""
        key_hash = stable_hash(key)
        if key_hash not in self.hashes:
            raise KeyError(key)
        self.hashes.remove(key_hash)
        (self.cache_dir / key_hash).unlink()

    def __len__(self):
        """Get the number of keys in the cache."""
        return len(self.hashes)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.cache_dir})"

    def __str__(self):
        return f"{self.__class__.__name__}({self.cache_dir})"

    def clear(self):
        """Clear the cache."""
        for key_hash in self.hashes:
            (self.cache_dir / key_hash).unlink()
        self.hashes = []

    # def get(self, key: "tuple[str, tuple, frozenset]", default: Any = None) -> Any:
    def get(self, key: Hashable, default: Any = None) -> Any:
        """Get the value associated with a key, or a default value if the key is not in the cache."""
        try:
            return self[key]
        except KeyError:
            return default

    # def pop(self, key: "tuple[str, tuple, frozenset]", default: Any = None) -> Any:
    def pop(self, key: Hashable) -> Any:
        """Get the value associated with a key, and remove it from the cache."""
        raise NotImplementedError
        # try:
        #     value = self[key]
        #     del self[key]
        #     return value
        # except KeyError:
        #     return default

    # def popitem(self) -> "tuple[tuple[str, tuple, frozenset], Any]":
    def popitem(
        self,
    ) -> tuple[tuple[str, tuple[Any, ...], frozenset[tuple[Any, Any]]], Any]:
        """Remove and return a key-value pair from the cache."""
        raise NotImplementedError

    #     key_hash = self.hashes.pop()
    #     with open(self.cache_dir / key_hash, "rb") as file:
    #         key = pickle.load(file)
    #     value = self[key]
    #     del self[key]
    #     return key, value

    def values(self) -> Iterable[Any]:
        """Get all values in the cache."""
        raise NotImplementedError
        # return (self[key_hash] for key_hash in self.hashes)


def add_simple_cache_async(
    func: Callable[..., Any],
    cache: SimpleCache,
    use_cached_values: bool = True,
    write_to_cache: bool = True,
    override_existing: bool = False,
    excluded_kwargs: Sequence[str] = (),
) -> Callable[..., Any]:
    """
    Add a simple cache to a function.

    Parameters
    ----------
    func : Callable
        The function to add a cache to.
    cache : SimpleCache
        The cache to use. Multiple functions can share the same cache.
    use_cached_values : bool, optional
        Whether to use cached values if they exist, by default True
    write_to_cache : bool, optional
        Whether to write new values to the cache, by default True
    override_existing : bool, optional
        Whether to override existing values in the cache, by default False

    Returns
    -------
    Callable
        The function with a cache.
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any):
        packaged_kwargs = {k: v for k, v in kwargs.items() if k not in excluded_kwargs}
        key = package_func_args(func, args, packaged_kwargs)
        if use_cached_values and key in cache:
            return cache[key]
        value = await func(*args, **kwargs)
        if write_to_cache and (override_existing or key not in cache):
            cache[key] = value
        return value

    return wrapper


def test_simple_cache():
    """
    Test the SimpleCache class.
    """
    cache_dir = Path("test_cache")
    cache = SimpleCache(cache_dir)
    assert len(cache) == 0
    cache["a"] = 1
    assert len(cache) == 1
    assert cache["a"] == 1
    cache["b"] = 2
    assert len(cache) == 2
    assert cache["b"] == 2
    assert cache.get("c", 3) == 3
    # assert cache.pop("c", 3) == 3
    # assert cache.popitem() == (("b", (), frozenset()), 2)
    cache.clear()
    assert len(cache) == 0
    cache_dir.rmdir()


def test_simple_cache_async():
    """
    Test the add_simple_cache_async function.
    """
    cache_dir = Path("test_cache")
    cache = SimpleCache(cache_dir)
    assert len(cache) == 0

    async def test_func(a: int, b: int) -> int:
        return a + b

    test_func = add_simple_cache_async(test_func, cache)
    assert len(cache) == 0
    assert asyncio.run(test_func(1, 2)) == 3
    assert len(cache) == 1
    assert asyncio.run(test_func(1, 2)) == 3
    assert len(cache) == 1
    assert asyncio.run(test_func(1, 3)) == 4
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0
    cache_dir.rmdir()


if __name__ == "__main__":
    test_simple_cache()
    test_simple_cache_async()
