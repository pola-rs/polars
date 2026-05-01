use polars_buffer::Buffer;
use polars_parquet_format::ColumnOrder as TColumnOrder;

use super::RowGroupMetadata;
use super::column_order::ColumnOrder;
use super::compact::CompactFileMetaData;
use super::schema_descriptor::SchemaDescriptor;
use crate::parquet::error::ParquetResult;
use crate::parquet::metadata::get_sort_order;
pub use crate::parquet::thrift_format::KeyValue;

/// Metadata for a Parquet file.
//
// Polars-side representation of a parsed Parquet file footer. Wraps the
// schema descriptor (with column descriptors needed for page deserialisation),
// per-row-group structures, and the footer buffer that backs lazily-resolved
// column-chunk statistics. Built from `CompactFileMetaData` (the hand-written
// decoder's output) via `Self::from_compact`.
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
    /// Footer bytes that back this file's column-chunk statistics. Stats
    /// `min_value` / `max_value` are stored as `(offset, len)` ranges into
    /// this buffer; pass `&self.footer_buf` to
    /// [`super::ColumnChunkMetadata::statistics`] to materialise them.
    pub footer_buf: Buffer<u8>,
}

impl FileMetadata {
    /// Returns the [`SchemaDescriptor`] that describes the schema of this file.
    pub fn schema(&self) -> &SchemaDescriptor {
        &self.schema_descr
    }

    /// Returns the file-level key-value metadata, if present.
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

    /// Build a `FileMetadata` from a [`CompactFileMetaData`], the output of
    /// the hand-written Thrift decoder. Parses the schema, attaches each
    /// row group's chunks to the schema's descriptors, and stores the
    /// footer buffer at the file level for stats resolution.
    ///
    /// Crate-internal: external callers go through
    /// [`crate::parquet::read::deserialize_metadata`] which combines the
    /// hand-written decoder with this constructor.
    pub(crate) fn from_compact(compact: CompactFileMetaData) -> ParquetResult<Self> {
        let CompactFileMetaData {
            version,
            schema,
            num_rows,
            row_groups,
            key_value_metadata,
            created_by,
            column_orders,
            footer_buf,
        } = compact;

        let schema_descr = SchemaDescriptor::try_from_thrift(&schema)?;

        let mut max_row_group_height = 0;
        let row_groups = row_groups
            .into_iter()
            .map(|rg| {
                let md = RowGroupMetadata::from_compact(&schema_descr, rg)?;
                max_row_group_height = max_row_group_height.max(md.num_rows());
                Ok(md)
            })
            .collect::<ParquetResult<_>>()?;

        let column_orders = column_orders.map(|orders| parse_column_orders(&orders, &schema_descr));

        Ok(FileMetadata {
            version,
            num_rows: num_rows.try_into()?,
            max_row_group_height,
            created_by,
            row_groups,
            key_value_metadata,
            schema_descr,
            column_orders,
            footer_buf,
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
