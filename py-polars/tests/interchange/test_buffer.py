from pyarrow import Array

from polars.interchange.buffer import _IXBuffer

ARR = Array([None, 0, 1])


def test_buffer() -> None:
    buffer = _IXBuffer(ARR)

    assert buffer.bufsize == ARR.get_total_buffer_size()
    # TODO: assert buffer.ptr
