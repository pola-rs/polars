//! APIs to read from ODBC
mod deserialize;
mod schema;

pub use deserialize::deserialize;
pub use schema::infer_schema;

use super::api;

/// Creates a [`api::buffers::ColumnarBuffer`] from the metadata.
/// # Errors
/// Iff the driver provides an incorrect [`api::ResultSetMetadata`]
pub fn buffer_from_metadata(
    resut_set_metadata: &impl api::ResultSetMetadata,
    max_batch_size: usize,
) -> std::result::Result<api::buffers::ColumnarBuffer<api::buffers::AnyColumnBuffer>, api::Error> {
    let num_cols: u16 = resut_set_metadata.num_result_cols()? as u16;

    let descs = (0..num_cols)
        .map(|index| {
            let mut column_description = api::ColumnDescription::default();

            resut_set_metadata.describe_col(index + 1, &mut column_description)?;

            Ok(api::buffers::BufferDescription {
                nullable: column_description.could_be_nullable(),
                kind: api::buffers::BufferKind::from_data_type(column_description.data_type)
                    .unwrap(),
            })
        })
        .collect::<std::result::Result<Vec<_>, api::Error>>()?;

    Ok(api::buffers::buffer_from_description(
        max_batch_size,
        descs.into_iter(),
    ))
}
