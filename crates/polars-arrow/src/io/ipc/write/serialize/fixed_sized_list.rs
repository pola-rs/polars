use super::*;

pub(super) fn write_fixed_size_list(
    array: &FixedSizeListArray,
    buffers: &mut Vec<ipc::Buffer>,
    arrow_data: &mut Vec<u8>,
    nodes: &mut Vec<ipc::FieldNode>,
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
    write(
        array.values().as_ref(),
        buffers,
        arrow_data,
        nodes,
        offset,
        is_little_endian,
        compression,
    );
}
