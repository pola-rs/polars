use std::collections::VecDeque;
use std::io::{Read, Seek};

use polars_error::{polars_bail, polars_err, PolarsResult};
use polars_utils::aliases::PlHashSet;

use super::super::{Compression, Dictionaries, IpcBuffer, Node};
use super::{read_primitive, skip_primitive};
use crate::array::{DictionaryArray, DictionaryKey};
use crate::datatypes::ArrowDataType;

#[allow(clippy::too_many_arguments)]
pub fn read_dictionary<T: DictionaryKey, R: Read + Seek>(
    field_nodes: &mut VecDeque<Node>,
    data_type: ArrowDataType,
    id: Option<i64>,
    buffers: &mut VecDeque<IpcBuffer>,
    reader: &mut R,
    dictionaries: &Dictionaries,
    block_offset: u64,
    compression: Option<Compression>,
    limit: Option<usize>,
    is_little_endian: bool,
    scratch: &mut Vec<u8>,
) -> PolarsResult<DictionaryArray<T>>
where
    Vec<u8>: TryInto<T::Bytes>,
{
    let id = if let Some(id) = id {
        id
    } else {
        polars_bail!(oos = "Dictionary has no id.");
    };
    let values = dictionaries
        .get(&id)
        .ok_or_else(|| {
            let valid_ids = dictionaries.keys().collect::<PlHashSet<_>>();
            polars_err!(ComputeError:
                "Dictionary id {id} not found. Valid ids: {valid_ids:?}"
            )
        })?
        .clone();

    let keys = read_primitive(
        field_nodes,
        T::PRIMITIVE.into(),
        buffers,
        reader,
        block_offset,
        is_little_endian,
        compression,
        limit,
        scratch,
    )?;

    DictionaryArray::<T>::try_new(data_type, keys, values)
}

pub fn skip_dictionary(
    field_nodes: &mut VecDeque<Node>,
    buffers: &mut VecDeque<IpcBuffer>,
) -> PolarsResult<()> {
    skip_primitive(field_nodes, buffers)
}
