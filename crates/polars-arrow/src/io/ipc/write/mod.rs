//! APIs to write to Arrow's IPC format.
pub(crate) mod common;
mod schema;
mod serialize;
mod stream;
pub(crate) mod writer;

pub use common::{Compression, Record, WriteOptions};
pub use schema::schema_to_bytes;
pub use serialize::write;
use serialize::write_dictionary;
pub use stream::StreamWriter;
pub use writer::FileWriter;

pub(crate) mod common_sync;

#[cfg(feature = "io_ipc_write_async")]
mod common_async;
#[cfg(feature = "io_ipc_write_async")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_ipc_write_async")))]
pub mod stream_async;

#[cfg(feature = "io_ipc_write_async")]
#[cfg_attr(docsrs, doc(cfg(feature = "io_ipc_write_async")))]
pub mod file_async;

use super::IpcField;
use crate::datatypes::{ArrowDataType, Field};

fn default_ipc_field(data_type: &ArrowDataType, current_id: &mut i64) -> IpcField {
    use crate::datatypes::ArrowDataType::*;
    match data_type.to_logical_type() {
        // single child => recurse
        Map(inner, ..) | FixedSizeList(inner, _) | LargeList(inner) | List(inner) => IpcField {
            fields: vec![default_ipc_field(inner.data_type(), current_id)],
            dictionary_id: None,
        },
        // multiple children => recurse
        Union(fields, ..) | Struct(fields) => IpcField {
            fields: fields
                .iter()
                .map(|f| default_ipc_field(f.data_type(), current_id))
                .collect(),
            dictionary_id: None,
        },
        // dictionary => current_id
        Dictionary(_, data_type, _) => {
            let dictionary_id = Some(*current_id);
            *current_id += 1;
            IpcField {
                fields: vec![default_ipc_field(data_type, current_id)],
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
pub fn default_ipc_fields(fields: &[Field]) -> Vec<IpcField> {
    let mut dictionary_id = 0i64;
    fields
        .iter()
        .map(|field| default_ipc_field(field.data_type().to_logical_type(), &mut dictionary_id))
        .collect()
}
