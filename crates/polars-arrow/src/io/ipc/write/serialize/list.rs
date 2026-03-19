use super::*;

pub(super) fn write_list<O: Offset>(
    array: &ListArray<O>,
    buffers: &mut Vec<ipc::Buffer>,
    arrow_data: &mut Vec<u8>,
    nodes: &mut Vec<ipc::FieldNode>,
    offset: &mut i64,
    is_little_endian: bool,
    compression: Option<Compression>,
) {
    let offsets = array.offsets().buffer();
    let validity = array.validity();

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
    if first == O::zero() {
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

    write(
        array
            .values()
            .sliced(first.to_usize(), last.to_usize() - first.to_usize())
            .as_ref(),
        buffers,
        arrow_data,
        nodes,
        offset,
        is_little_endian,
        compression,
    );
}
