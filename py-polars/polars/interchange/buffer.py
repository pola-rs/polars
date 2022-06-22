from enum import IntEnum

# TODO: determine if we want to do this
_PYARROW_AVAILABLE = True
try:
    import pyarrow as pa
except ImportError:
    _PYARROW_AVAILABLE = False

# TODO
NotSupportedError = Exception


class _IXBuffer:
    """
    Data in the buffer is guaranteed to be contiguous in memory.

    Note that there is no dtype attribute present, a buffer can be thought of
    as simply a block of memory. However, if the column that the buffer is
    attached to has a dtype that's supported by DLPack and ``__dlpack__`` is
    implemented, then that dtype information will be contained in the return
    value from ``__dlpack__``.

    This distinction is useful to support both data exchange via DLPack on a
    buffer and (b) dtypes like variable-length strings which do not have a
    fixed number of bytes per element.
    """

    def __init__(self, arr: pa.Array, allow_copy: bool = True) -> None:
        if not isinstance(arr, pa.Array):
            raise ValueError("`arr` must be a pyarrow Array")

        # TODO: offsets are unhandled
        # if arr.offset != 0:
        #     raise NotImplementedError("`arr`s with offsets are not supported")

        if arr.num_chunks() > 1:
            if not allow_copy:
                raise NotSupportedError("A copy is needed to flatten the ChunkedArray")

            arr = arr.combine_chunks()

        self._arr = arr
        self._allow_copy = allow_copy

    @property
    def bufsize(self) -> int:
        """Buffer size in bytes."""
        # TODO: what about nbytes
        return self._arr.get_total_buffer_size()

    @property
    def ptr(self) -> int:
        """Po to start of the buffer as an integer."""
        ...

    def __dlpack__(self) -> None:
        """
        Produce DLPack capsule (see array API standard).

        Raises: (TODO)

            - TypeError : if the buffer contains unsupported dtypes.
            - NotImplementedError : if DLPack support is not implemented

        Useful to have to connect to array libraries. Support optional because
        it's not completely trivial to implement for a Python-only library.
        """
        ...

    def __dlpack_device__(self) -> tuple[IntEnum, int]:
        """
        Device type and device ID for where the data in the buffer resides.

        Uses device type codes matching DLPack. Enum members are::

            - CPU = 1
            - CUDA = 2
            - CPU_PINNED = 3
            - OPENCL = 4
            - VULKAN = 7
            - METAL = 8
            - VPI = 9
            - ROCM = 10

        Note: must be implemented even if ``__dlpack__`` is not.
        """
        ...
