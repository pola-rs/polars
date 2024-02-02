from __future__ import annotations

from typing import List, Protocol, Tuple, TypeAlias, TypedDict, Union

# An integer indicating the version of the interface
PROTOCOL_VERSION = 3

Shape: TypeAlias = Tuple[int, ...]
TypeStr: TypeAlias = str
DescrName: TypeAlias = Union[str, Tuple[str, str]]
DescrType: TypeAlias = Union[TypeStr, List["Descr"]]
Descr: TypeAlias = Union[
    Tuple[DescrName, DescrType],
    Tuple[DescrName, DescrType, Shape],
]


class ArrayInterface(TypedDict):
    """Array interface."""

    # Tuple whose elements are the array size in each dimension
    shape: Shape

    # A string providing the basic type of the homogeneous array
    typestr: TypeStr

    # A list of tuples providing a more detailed description of the memory layout for
    # each item in the homogeneous array
    descr: list[Descr] | None

    # A 2-tuple whose first argument is a Python integer that points to the data-area
    # storing the array contents. The second entry in the tuple is a read-only flag.
    data: tuple[int, bool] | None

    # Either None to indicate a C-style contiguous array or a tuple of strides which
    # provides the number of bytes needed to jump to the next array element in the
    # corresponding dimension.
    strides: tuple[int, ...] | None

    # None or an object exposing the array interface. All elements of the mask array
    # should be interpreted only as true or not true indicating which elements of this
    # array are valid.
    mask: SupportsArrayInterface | None

    # An integer offset into the array data region.
    offset: int

    # An integer showing the version of the interface.
    version: int


class SupportsArrayInterface(Protocol):
    """Object that supports the NumPy array interface protocol."""

    @property
    def __array_interface__(self) -> ArrayInterface:
        """Convert to a dataframe object implementing the dataframe interchange protocol."""  # noqa: W505
