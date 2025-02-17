use super::*;

// use `write_keys` to either write keys or values
#[allow(clippy::too_many_arguments)]
pub fn write_dictionary<K: DictionaryKey>(
    array: &DictionaryArray<K>,
    buffers: &mut Vec<ipc::Buffer>,
    arrow_data: &mut Vec<u8>,
    nodes: &mut Vec<ipc::FieldNode>,
    offset: &mut i64,
    is_little_endian: bool,
    compression: Option<Compression>,
    write_keys: bool,
) -> usize {
    if write_keys {
        write_primitive(
            array.keys(),
            buffers,
            arrow_data,
            offset,
            is_little_endian,
            compression,
        );
        array.keys().len()
    } else {
        write(
            array.values().as_ref(),
            buffers,
            arrow_data,
            nodes,
            offset,
            is_little_endian,
            compression,
        );
        array.values().len()
    }
}
