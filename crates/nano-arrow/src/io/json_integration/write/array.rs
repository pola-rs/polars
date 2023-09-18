use crate::{
    array::{Array, PrimitiveArray},
    chunk::Chunk,
    datatypes::DataType,
};

use super::super::{ArrowJsonBatch, ArrowJsonColumn};

/// Serializes a [`Chunk`] to [`ArrowJsonBatch`].
pub fn serialize_chunk<A: ToString>(
    columns: &Chunk<Box<dyn Array>>,
    names: &[A],
) -> ArrowJsonBatch {
    let count = columns.len();

    let columns = columns
        .arrays()
        .iter()
        .zip(names.iter())
        .map(|(array, name)| match array.data_type() {
            DataType::Int8 => {
                let array = array.as_any().downcast_ref::<PrimitiveArray<i8>>().unwrap();

                let (validity, data) = array
                    .iter()
                    .map(|x| (x.is_some() as u8, x.copied().unwrap_or_default().into()))
                    .unzip();

                ArrowJsonColumn {
                    name: name.to_string(),
                    count: array.len(),
                    validity: Some(validity),
                    data: Some(data),
                    offset: None,
                    type_id: None,
                    children: None,
                }
            }
            _ => ArrowJsonColumn {
                name: name.to_string(),
                count: array.len(),
                validity: None,
                data: None,
                offset: None,
                type_id: None,
                children: None,
            },
        })
        .collect();

    ArrowJsonBatch { count, columns }
}
