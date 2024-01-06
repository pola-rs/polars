use super::*;

pub(super) fn write_primitive<T: NativeType>(
    array: &PrimitiveArray<T>,
    buffers: &mut Vec<ipc::Buffer>,
    arrow_data: &mut Vec<u8>,
    offset: &mut i64,
    is_little_endian: bool,
    compression: Option<Compression>,
) {
    write_bitmap(
        array.validity(),
        array.len(),
        buffers,
        arrow_data,
        offset,
        compression,
    );

    write_buffer(
        array.values(),
        buffers,
        arrow_data,
        offset,
        is_little_endian,
        compression,
    )
}
