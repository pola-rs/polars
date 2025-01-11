//! APIs to write to Arrow's IPC format.
pub(crate) mod common;
mod schema;
mod serialize;
mod stream;
pub(crate) mod writer;

pub use common::{
    dictionaries_to_encode, encode_dictionary, encode_new_dictionaries, encode_record_batch,
    Compression, DictionaryTracker, EncodedData, Record, WriteOptions,
};
pub use schema::schema_to_bytes;
pub use serialize::write;
use serialize::write_dictionary;
pub use stream::StreamWriter;
pub use writer::FileWriter;

pub(crate) mod common_sync;

use super::IpcField;
use crate::datatypes::{ArrowDataType, Field};

fn default_ipc_field(dtype: &ArrowDataType, current_id: &mut i64) -> IpcField {
    use crate::datatypes::ArrowDataType::*;
    match dtype.to_logical_type() {
        // single child => recurse
        Map(inner, ..) | FixedSizeList(inner, _) | LargeList(inner) | List(inner) => IpcField {
            fields: vec![default_ipc_field(inner.dtype(), current_id)],
            dictionary_id: None,
        },
        // multiple children => recurse
        Struct(fields) => IpcField {
            fields: fields
                .iter()
                .map(|f| default_ipc_field(f.dtype(), current_id))
                .collect(),
            dictionary_id: None,
        },
        // multiple children => recurse
        Union(u) => IpcField {
            fields: u
                .fields
                .iter()
                .map(|f| default_ipc_field(f.dtype(), current_id))
                .collect(),
            dictionary_id: None,
        },
        // dictionary => current_id
        Dictionary(_, dtype, _) => {
            let dictionary_id = Some(*current_id);
            *current_id += 1;
            IpcField {
                fields: vec![default_ipc_field(dtype, current_id)],
                dictionary_id,
            }
        },
        // no children => do nothing
        _ => IpcField {
            fields: vec![],
            dictionary_id: None,
        },
    }
}

/// Assigns every dictionary field a unique ID
pub fn default_ipc_fields<'a>(fields: impl ExactSizeIterator<Item = &'a Field>) -> Vec<IpcField> {
    let mut dictionary_id = 0i64;
    fields
        .map(|field| default_ipc_field(field.dtype().to_logical_type(), &mut dictionary_id))
        .collect()
}
