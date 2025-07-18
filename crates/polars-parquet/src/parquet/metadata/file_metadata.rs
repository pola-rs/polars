use polars_parquet_format::ColumnOrder as TColumnOrder;

use super::RowGroupMetadata;
use super::column_order::ColumnOrder;
use super::schema_descriptor::SchemaDescriptor;
use crate::parquet::error::ParquetError;
use crate::parquet::metadata::get_sort_order;
pub use crate::parquet::thrift_format::KeyValue;

/// Metadata for a Parquet file.
// This is almost equal to [`polars_parquet_format::FileMetaData`] but contains the descriptors,
// which are crucial to deserialize pages.
#[derive(Debug, Clone)]
pub struct FileMetadata {
    /// version of this file.
    pub version: i32,
    /// number of rows in the file.
    pub num_rows: usize,
    /// Max row group height, useful for sharing column materializations.
    pub max_row_group_height: usize,
    /// String message for application that wrote this file.
    ///
    /// This should have the following format:
    /// `<application> version <application version> (build <application build hash>)`.
    ///
    /// ```shell
    /// parquet-mr version 1.8.0 (build 0fda28af84b9746396014ad6a415b90592a98b3b)
    /// ```
    pub created_by: Option<String>,
    /// The row groups of this file
    pub row_groups: Vec<RowGroupMetadata>,
    /// key_value_metadata of this file.
    pub key_value_metadata: Option<Vec<KeyValue>>,
    /// schema descriptor.
    pub schema_descr: SchemaDescriptor,
    /// Column (sort) order used for `min` and `max` values of each column in this file.
    ///
    /// Each column order corresponds to one column, determined by its position in the
    /// list, matching the position of the column in the schema.
    ///
    /// When `None` is returned, there are no column orders available, and each column
    /// should be assumed to have undefined (legacy) column order.
    pub column_orders: Option<Vec<ColumnOrder>>,
}

impl FileMetadata {
    /// Returns the [`SchemaDescriptor`] that describes schema of this file.
    pub fn schema(&self) -> &SchemaDescriptor {
        &self.schema_descr
    }

    /// returns the metadata
    pub fn key_value_metadata(&self) -> &Option<Vec<KeyValue>> {
        &self.key_value_metadata
    }

    /// Returns column order for `i`th column in this file.
    /// If column orders are not available, returns undefined (legacy) column order.
    pub fn column_order(&self, i: usize) -> ColumnOrder {
        self.column_orders
            .as_ref()
            .map(|data| data[i])
            .unwrap_or(ColumnOrder::Undefined)
    }

    /// Deserializes [`crate::parquet::thrift_format::FileMetadata`] into this struct
    pub fn try_from_thrift(
        metadata: polars_parquet_format::FileMetaData,
    ) -> Result<Self, ParquetError> {
        let schema_descr = SchemaDescriptor::try_from_thrift(&metadata.schema)?;

        let mut max_row_group_height = 0;

        let row_groups = metadata
            .row_groups
            .into_iter()
            .map(|rg| {
                let md = RowGroupMetadata::try_from_thrift(&schema_descr, rg)?;
                max_row_group_height = max_row_group_height.max(md.num_rows());
                Ok(md)
            })
            .collect::<Result<_, ParquetError>>()?;

        let column_orders = metadata
            .column_orders
            .map(|orders| parse_column_orders(&orders, &schema_descr));

        Ok(FileMetadata {
            version: metadata.version,
            num_rows: metadata.num_rows.try_into()?,
            max_row_group_height,
            created_by: metadata.created_by,
            row_groups,
            key_value_metadata: metadata.key_value_metadata,
            schema_descr,
            column_orders,
        })
    }
}

/// Parses [`ColumnOrder`] from Thrift definition.
fn parse_column_orders(
    orders: &[TColumnOrder],
    schema_descr: &SchemaDescriptor,
) -> Vec<ColumnOrder> {
    schema_descr
        .columns()
        .iter()
        .zip(orders.iter())
        .map(|(column, order)| match order {
            TColumnOrder::TYPEORDER(_) => {
                let sort_order = get_sort_order(
                    &column.descriptor.primitive_type.logical_type,
                    &column.descriptor.primitive_type.converted_type,
                    &column.descriptor.primitive_type.physical_type,
                );
                ColumnOrder::TypeDefinedOrder(sort_order)
            },
        })
        .collect()
}
