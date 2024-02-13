use super::*;

pub(super) fn write_fixed_size_binary(
    array: &FixedSizeBinaryArray,
    buffers: &mut Vec<ipc::Buffer>,
    arrow_data: &mut Vec<u8>,
    offset: &mut i64,
    _is_little_endian: bool,
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
    write_bytes(array.values(), buffers, arrow_data, offset, compression);
}
