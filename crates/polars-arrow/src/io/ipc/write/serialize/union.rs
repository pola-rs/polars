use super::*;

pub(super) fn write_union(
    array: &UnionArray,
    buffers: &mut Vec<ipc::Buffer>,
    arrow_data: &mut Vec<u8>,
    nodes: &mut Vec<ipc::FieldNode>,
    offset: &mut i64,
    is_little_endian: bool,
    compression: Option<Compression>,
) {
    write_buffer(
        array.types(),
        buffers,
        arrow_data,
        offset,
        is_little_endian,
        compression,
    );

    if let Some(offsets) = array.offsets() {
        write_buffer(
            offsets,
            buffers,
            arrow_data,
            offset,
            is_little_endian,
            compression,
        );
    }
    array.fields().iter().for_each(|array| {
        write(
            array.as_ref(),
            buffers,
            arrow_data,
            nodes,
            offset,
            is_little_endian,
            compression,
        )
    });
}
