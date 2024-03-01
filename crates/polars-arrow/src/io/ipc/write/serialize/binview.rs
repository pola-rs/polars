use super::*;
use crate::array;

#[allow(clippy::too_many_arguments)]
pub(super) fn write_binview<T: ViewType + ?Sized>(
    array: &BinaryViewArrayGeneric<T>,
    buffers: &mut Vec<ipc::Buffer>,
    arrow_data: &mut Vec<u8>,
    offset: &mut i64,
    is_little_endian: bool,
    compression: Option<Compression>,
) {
    write_bitmap(
        array.validity(),
        array::Array::len(array),
        buffers,
        arrow_data,
        offset,
        compression,
    );

    write_buffer(
        array.views(),
        buffers,
        arrow_data,
        offset,
        is_little_endian,
        compression,
    );

    for data in array.data_buffers().as_ref() {
        write_bytes(data, buffers, arrow_data, offset, compression);
    }
}
