use super::*;

#[allow(clippy::too_many_arguments)]
fn write_generic_binary<O: Offset>(
    validity: Option<&Bitmap>,
    offsets: &OffsetsBuffer<O>,
    values: &[u8],
    buffers: &mut Vec<ipc::Buffer>,
    arrow_data: &mut Vec<u8>,
    offset: &mut i64,
    is_little_endian: bool,
    compression: Option<Compression>,
) {
    let offsets = offsets.buffer();
    write_bitmap(
        validity,
        offsets.len() - 1,
        buffers,
        arrow_data,
        offset,
        compression,
    );

    let first = *offsets.first().unwrap();
    let last = *offsets.last().unwrap();
    if first == O::default() {
        write_buffer(
            offsets,
            buffers,
            arrow_data,
            offset,
            is_little_endian,
            compression,
        );
    } else {
        write_buffer_from_iter(
            offsets.iter().map(|x| *x - first),
            buffers,
            arrow_data,
            offset,
            is_little_endian,
            compression,
        );
    }

    write_bytes(
        &values[first.to_usize()..last.to_usize()],
        buffers,
        arrow_data,
        offset,
        compression,
    );
}

pub(super) fn write_binary<O: Offset>(
    array: &BinaryArray<O>,
    buffers: &mut Vec<ipc::Buffer>,
    arrow_data: &mut Vec<u8>,
    offset: &mut i64,
    is_little_endian: bool,
    compression: Option<Compression>,
) {
    write_generic_binary(
        array.validity(),
        array.offsets(),
        array.values(),
        buffers,
        arrow_data,
        offset,
        is_little_endian,
        compression,
    );
}

pub(super) fn write_utf8<O: Offset>(
    array: &Utf8Array<O>,
    buffers: &mut Vec<ipc::Buffer>,
    arrow_data: &mut Vec<u8>,
    offset: &mut i64,
    is_little_endian: bool,
    compression: Option<Compression>,
) {
    write_generic_binary(
        array.validity(),
        array.offsets(),
        array.values(),
        buffers,
        arrow_data,
        offset,
        is_little_endian,
        compression,
    );
}
