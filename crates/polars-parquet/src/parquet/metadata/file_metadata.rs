use polars_buffer::Buffer;
use polars_utils::aliases::{InitHashMaps, PlHashMap};

use super::RowGroupMetadata;
use super::column_order::{ColumnOrder, ColumnOrderTag};
use super::compact::{CompactColumnChunk, CompactFileMetaData, CompactRowGroup};
use super::schema_descriptor::SchemaDescriptor;
use crate::parquet::error::ParquetResult;
use crate::parquet::metadata::get_sort_order;
use crate::parquet::schema::types::ParquetType;
pub use crate::parquet::thrift_format::KeyValue;

/// Metadata for a Parquet file.
//
// Polars-side representation of a parsed Parquet file footer. Wraps the
// schema descriptor (with column descriptors needed for page deserialisation),
// per-row-group structures, and the footer buffer that backs lazily-resolved
// column-chunk statistics. Built from `CompactFileMetaData` (the hand-written
// decoder's output) via `Self::from_compact`.
//
// Custom `Serialize`/`Deserialize` (in `file_metadata_serde`) emit a pruned
// wire form: schema without `leaves`, stats materialised to owned bytes,
// `footer_buf` reconstructed on deserialize. Cheap enough to ship in IR
// plans for distributed execution.
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
    /// Column (sort) order used for `min_value` and `max_value` of each leaf column in
    /// this file, one per leaf of `schema_descr` in the same order.
    ///
    /// `None` when the file has no column orders, or not one per leaf; each column then
    /// has the undefined (legacy) column order.
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

    /// Returns column order for the `i`th leaf column in this file.
    /// If column orders are not available, returns undefined (legacy) column order.
    pub fn column_order(&self, i: usize) -> ColumnOrder {
        self.column_orders
            .as_ref()
            .and_then(|data| data.get(i).copied())
            .unwrap_or(ColumnOrder::Undefined)
    }

    /// Prune to projected columns, keeping statistics only for predicate
    /// columns.
    ///
    /// Returns a new [`FileMetadata`] containing only:
    /// - top-level schema fields whose name is in `keep_top_level_names`,
    /// - row-group chunks corresponding to those fields' leaves,
    /// - statistics on chunks whose column is in `predicate_top_level_names`.
    ///
    /// `predicate_top_level_names` is treated as a subset of
    /// `keep_top_level_names`; pass `&[]` to drop all stats. `created_by` and
    /// `key_value_metadata` are also dropped (not needed by the read hot
    /// path); `column_orders` is kept for the remaining leaves.
    ///
    /// Returns `Err` only when [`RowGroupMetadata::from_compact`] rejects
    /// the rebuilt row group (chunks-vs-leaves desync). Callers can fall
    /// back to unpruned metadata; the unpruned form is always valid.
    ///
    /// TODO: a planner-side pass could pre-evaluate static predicates
    /// against stats and drop fully-skipped row groups, removing stats
    /// from the wire for those cases.
    pub fn pruned(
        &self,
        keep_top_level_names: &[polars_utils::pl_str::PlSmallStr],
        predicate_top_level_names: &[polars_utils::pl_str::PlSmallStr],
    ) -> ParquetResult<Self> {
        // Column name → keep-stats flag. Names not in the map are pruned
        // entirely. O(1) lookup per chunk keeps this scalable to
        // wide-column workloads (10k+ columns × many row groups).
        let mut keep: PlHashMap<&str, bool> = PlHashMap::with_capacity(keep_top_level_names.len());
        keep.extend(keep_top_level_names.iter().map(|n| (n.as_str(), false)));
        // Promotes from false to true if already present.
        keep.extend(predicate_top_level_names.iter().map(|n| (n.as_str(), true)));

        // 1. Filter top-level fields, preserving order from the source schema.
        let pruned_fields: Vec<ParquetType> = self
            .schema_descr
            .fields()
            .iter()
            .filter(|f| keep.contains_key(f.get_field_info().name.as_str()))
            .cloned()
            .collect();

        // 2. Build the pruned SchemaDescriptor (DFS derives leaves Arc).
        let pruned_schema = SchemaDescriptor::new(self.schema_descr.name().into(), pruned_fields);

        // Leaves keep their relative order, so the kept leaves' orders are those of
        // the source leaves under a kept field.
        let column_orders = self.column_orders.as_ref().map(|orders| {
            self.schema_descr
                .columns()
                .iter()
                .zip(orders)
                .filter(|(column, _)| keep.contains_key(column.path_in_schema[0].as_str()))
                .map(|(_, order)| *order)
                .collect::<Vec<_>>()
        });
        debug_assert!(
            column_orders
                .as_ref()
                .is_none_or(|o| o.len() == pruned_schema.columns().len())
        );

        // 3. Per row group: pick chunks whose top-level field is in `keep`,
        //    drop stats from non-predicate columns.
        let mut max_row_group_height = 0;
        let row_groups: Vec<RowGroupMetadata> = self
            .row_groups
            .iter()
            .map(|rg| {
                let kept_chunks: Vec<CompactColumnChunk> = rg
                    .parquet_columns()
                    .iter()
                    .filter_map(|c| {
                        let keep_stats = *keep.get(c.descriptor().path_in_schema[0].as_str())?;
                        let mut chunk = c.compact_column_chunk().clone();
                        if !keep_stats {
                            chunk.meta_data.statistics = None;
                        }
                        Some(chunk)
                    })
                    .collect();

                let compact_rg = CompactRowGroup {
                    columns: kept_chunks,
                    total_byte_size: rg.total_byte_size() as i64,
                    num_rows: rg.num_rows() as i64,
                    sorting_columns: rg.sorting_columns().map(|sc| sc.to_vec()),
                };

                let md = RowGroupMetadata::from_compact(&pruned_schema, compact_rg)?;
                max_row_group_height = max_row_group_height.max(md.num_rows());
                Ok(md)
            })
            .collect::<ParquetResult<_>>()?;

        Ok(FileMetadata {
            version: self.version,
            num_rows: self.num_rows,
            max_row_group_height,
            created_by: None,
            row_groups,
            key_value_metadata: None,
            schema_descr: pruned_schema,
            column_orders,
            footer_buf: self.footer_buf.clone(),
        })
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

        let column_orders = column_orders.and_then(|o| parse_column_orders(&o, &schema_descr));

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

/// Parses [`ColumnOrder`] from Thrift definition. The list must hold one order
/// per leaf column; any other length says nothing about the leaves.
fn parse_column_orders(
    orders: &[ColumnOrderTag],
    schema_descr: &SchemaDescriptor,
) -> Option<Vec<ColumnOrder>> {
    if orders.len() != schema_descr.columns().len() {
        return None;
    }
    Some(
        schema_descr
            .columns()
            .iter()
            .zip(orders.iter())
            .map(|(column, order)| match order {
                ColumnOrderTag::TypeDefined => {
                    let sort_order = get_sort_order(
                        &column.descriptor.primitive_type.logical_type,
                        &column.descriptor.primitive_type.converted_type,
                        &column.descriptor.primitive_type.physical_type,
                    );
                    ColumnOrder::TypeDefinedOrder(sort_order)
                },
                ColumnOrderTag::IEEE754TotalOrder => ColumnOrder::IEEE754TotalOrder,
                ColumnOrderTag::Unsupported => ColumnOrder::Unsupported,
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::compression::Compression;
    use crate::parquet::metadata::SortOrder;
    use crate::parquet::metadata::compact::{CompactColumnMetaData, CompactStatistics};
    use crate::parquet::schema::Repetition;
    use crate::parquet::schema::types::{PhysicalType, PrimitiveLogicalType};

    fn leaf(
        name: &str,
        physical: PhysicalType,
        logical: Option<PrimitiveLogicalType>,
    ) -> ParquetType {
        ParquetType::try_from_primitive(
            name.into(),
            physical,
            Repetition::Optional,
            None,
            logical,
            None,
        )
        .unwrap()
    }

    /// `a: INT32, s: {x: INT32, y: BYTE_ARRAY (String)}, b: INT64`: four leaves.
    fn schema() -> SchemaDescriptor {
        SchemaDescriptor::new(
            "root".into(),
            vec![
                leaf("a", PhysicalType::Int32, None),
                ParquetType::from_group(
                    "s".into(),
                    Repetition::Optional,
                    None,
                    None,
                    vec![
                        leaf("x", PhysicalType::Int32, None),
                        leaf(
                            "y",
                            PhysicalType::ByteArray,
                            Some(PrimitiveLogicalType::String),
                        ),
                    ],
                    None,
                ),
                leaf("b", PhysicalType::Int64, None),
            ],
        )
    }

    fn chunk(statistics: Option<CompactStatistics>) -> CompactColumnChunk {
        CompactColumnChunk {
            meta_data: CompactColumnMetaData {
                codec: Compression::Uncompressed,
                num_values: 3,
                total_uncompressed_size: 0,
                total_compressed_size: 0,
                data_page_offset: 0,
                index_page_offset: None,
                dictionary_page_offset: None,
                statistics,
                bloom_filter_offset: None,
                bloom_filter_length: None,
            },
            offset_index_offset: None,
            offset_index_length: None,
            column_index_offset: None,
            column_index_length: None,
        }
    }

    const ORDERS: [ColumnOrderTag; 4] = [
        ColumnOrderTag::TypeDefined,
        ColumnOrderTag::IEEE754TotalOrder,
        ColumnOrderTag::Unsupported,
        ColumnOrderTag::TypeDefined,
    ];

    /// One row group whose first leaf has a min of `1i32` that is not exact and an
    /// exact max of `2i32`; the other leaves have no statistics.
    fn metadata(orders: Option<&[ColumnOrderTag]>) -> FileMetadata {
        let schema_descr = schema();
        let footer_buf = Buffer::from_vec(
            1i32.to_le_bytes()
                .into_iter()
                .chain(2i32.to_le_bytes())
                .collect(),
        );
        let stats = CompactStatistics {
            null_count: Some(1),
            distinct_count: None,
            max_value: Some(super::super::compact::ByteRange { offset: 4, len: 4 }),
            min_value: Some(super::super::compact::ByteRange { offset: 0, len: 4 }),
            is_max_value_exact: Some(true),
            is_min_value_exact: Some(false),
        };
        let rg = CompactRowGroup {
            columns: vec![chunk(Some(stats)), chunk(None), chunk(None), chunk(None)],
            total_byte_size: 0,
            num_rows: 3,
            sorting_columns: None,
        };
        let row_groups = vec![RowGroupMetadata::from_compact(&schema_descr, rg).unwrap()];
        FileMetadata {
            version: 2,
            num_rows: 3,
            max_row_group_height: 3,
            created_by: None,
            row_groups,
            key_value_metadata: None,
            column_orders: orders.and_then(|o| parse_column_orders(o, &schema_descr)),
            schema_descr,
            footer_buf,
        }
    }

    #[test]
    fn column_orders_are_resolved_per_leaf() {
        let md = metadata(Some(&ORDERS));
        assert_eq!(
            md.column_orders.as_deref(),
            Some(
                [
                    ColumnOrder::TypeDefinedOrder(SortOrder::Signed),
                    ColumnOrder::IEEE754TotalOrder,
                    ColumnOrder::Unsupported,
                    ColumnOrder::TypeDefinedOrder(SortOrder::Signed),
                ]
                .as_slice()
            )
        );
        assert_eq!(
            md.column_order(3),
            ColumnOrder::TypeDefinedOrder(SortOrder::Signed)
        );
        assert_eq!(md.column_order(4), ColumnOrder::Undefined);
    }

    #[test]
    fn column_orders_need_one_per_leaf() {
        assert!(metadata(None).column_orders.is_none());
        assert!(metadata(Some(&ORDERS[..3])).column_orders.is_none());
        assert_eq!(
            metadata(Some(&ORDERS[..3])).column_order(0),
            ColumnOrder::Undefined
        );
    }

    #[test]
    fn pruning_keeps_the_orders_of_the_kept_leaves() {
        let md = metadata(Some(&ORDERS));
        let pruned = md.pruned(&["s".into(), "b".into()], &["b".into()]).unwrap();
        let names: Vec<_> = pruned
            .schema_descr
            .columns()
            .iter()
            .map(|c| c.path_in_schema.join("."))
            .collect();
        assert_eq!(names, ["s.x", "s.y", "b"]);
        assert_eq!(
            pruned.column_orders.as_deref(),
            Some(
                [
                    ColumnOrder::IEEE754TotalOrder,
                    ColumnOrder::Unsupported,
                    ColumnOrder::TypeDefinedOrder(SortOrder::Signed),
                ]
                .as_slice()
            )
        );
        assert!(
            md.pruned(&["a".into()], &[])
                .unwrap()
                .column_orders
                .is_some()
        );
        assert!(
            metadata(None)
                .pruned(&["a".into()], &[])
                .unwrap()
                .column_orders
                .is_none()
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialization_keeps_orders_and_exactness() {
        use crate::parquet::statistics::Statistics;

        let md = metadata(Some(&ORDERS));
        let pruned = md.pruned(&["a".into(), "b".into()], &["a".into()]).unwrap();
        let bytes = polars_utils::pl_serialize::serialize_to_bytes::<_, false>(&pruned).unwrap();
        let back: FileMetadata =
            polars_utils::pl_serialize::deserialize_from_reader::<_, _, false>(bytes.as_slice())
                .unwrap();

        assert_eq!(back.column_orders, pruned.column_orders);
        let chunk = &back.row_groups[0].parquet_columns()[0];
        let Statistics::Int32(stats) = chunk.statistics(&back.footer_buf).unwrap().unwrap() else {
            panic!()
        };
        assert_eq!(stats.null_count, Some(1));
        assert_eq!(stats.min_value, None);
        assert_eq!(stats.max_value, Some(2));
    }
}
