use super::*;

pub(super) fn write_boolean(
    array: &BooleanArray,
    buffers: &mut Vec<ipc::Buffer>,
    arrow_data: &mut Vec<u8>,
    offset: &mut i64,
    _: bool,
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
    write_bitmap(
        Some(&array.values().clone()),
        array.len(),
        buffers,
        arrow_data,
        offset,
        compression,
    );
}
