//! Hand-written file-metadata decoder. Produces a [`CompactFileMetaData`]
//! using the [`super::parquet_thrift`] primitives (ported from arrow-rs 57.0).
//!
//! Two design wins over a format-struct decoder:
//! - Output type is a polars-controlled compact struct hierarchy that drops
//!   fields with no in-tree consumer (~110 bytes saved per `ColumnChunk`).
//! - `Statistics.min_value` / `max_value` payloads are recorded as `ByteRange`
//!   offsets into the input buffer rather than per-stat `Vec<u8>` allocations.
//!   The caller passes the footer as [`Buffer<u8>`] so those offsets stay
//!   resolvable for the lifetime of the resulting metadata.
//!
//!   See <https://github.com/apache/parquet-format/blob/96edf77704b60b6f3ca2232c218c64eff6c874d3/src/main/thrift/parquet.thrift> for spec

use polars_buffer::Buffer;
use polars_parquet_format::{ColumnOrder, KeyValue, SchemaElement, SortingColumn};

use super::parquet_thrift::{FieldType, ThriftCompactInputProtocol, ThriftSliceInputProtocol};
use crate::parquet::compression::Compression;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::metadata::{
    ByteRange, CompactColumnChunk, CompactColumnMetaData, CompactFileMetaData, CompactRowGroup,
    CompactStatistics,
};

trait RequireField<T> {
    /// Convert `None` into a uniform `ParquetError::oos("<name> missing")`.
    fn require(self, name: &str) -> ParquetResult<T>;
}

impl<T> RequireField<T> for Option<T> {
    #[inline]
    fn require(self, name: &str) -> ParquetResult<T> {
        self.ok_or_else(|| ParquetError::oos(format!("{name} missing")))
    }
}

/// Decode a Thrift list by reading the prefix and invoking `read_one` for
/// each element. Pre-sizes the `Vec` from the list header.
#[inline]
fn read_list<T>(
    prot: &mut ThriftSliceInputProtocol<'_>,
    mut read_one: impl FnMut(&mut ThriftSliceInputProtocol<'_>) -> ParquetResult<T>,
) -> ParquetResult<Vec<T>> {
    let list = prot.read_list_begin()?;
    let mut v = Vec::with_capacity(list.size.max(0) as usize);
    for _ in 0..list.size {
        v.push(read_one(prot)?);
    }
    Ok(v)
}

/// Decode a Parquet `FileMetaData` footer into [`CompactFileMetaData`].
///
/// `footer` holds the bytes `&buf` views. `ByteRange` offsets into stats
/// min/max are recorded relative to `footer.as_ptr()`, and the [`Buffer`]
/// is cloned (refcount-bump) into the output so they stay resolvable.
///
/// Crate-internal: external callers go through
/// [`crate::parquet::read::deserialize_metadata`] which combines this with
/// [`crate::parquet::metadata::FileMetadata::from_compact`] to produce the
/// public [`crate::parquet::metadata::FileMetadata`].
pub(crate) fn decode_file_metadata(footer: Buffer<u8>) -> ParquetResult<CompactFileMetaData> {
    let buf: &[u8] = footer.as_ref();
    let origin_ptr = buf.as_ptr();
    let mut prot = ThriftSliceInputProtocol::new(buf);
    read_file_metadata(&mut prot, origin_ptr, &footer)
}

fn read_file_metadata(
    prot: &mut ThriftSliceInputProtocol<'_>,
    origin_ptr: *const u8,
    footer: &Buffer<u8>,
) -> ParquetResult<CompactFileMetaData> {
    let mut version: Option<i32> = None;
    let mut schema: Option<Vec<SchemaElement>> = None;
    let mut num_rows: Option<i64> = None;
    let mut row_groups: Option<Vec<CompactRowGroup>> = None;
    let mut key_value_metadata: Option<Vec<KeyValue>> = None;
    let mut created_by: Option<String> = None;
    let mut column_orders: Option<Vec<ColumnOrder>> = None;

    let mut last_field_id = 0i16;
    loop {
        let f = prot.read_field_begin(last_field_id)?;
        if f.field_type == FieldType::Stop {
            break;
        }
        match f.id {
            1 => version = Some(prot.read_i32()?),
            2 => schema = Some(read_list(prot, read_schema_element)?),
            3 => num_rows = Some(prot.read_i64()?),
            4 => row_groups = Some(read_list(prot, |p| read_row_group(p, origin_ptr))?),
            5 => key_value_metadata = Some(read_list(prot, read_key_value)?),
            6 => created_by = Some(prot.read_string()?.to_owned()),
            7 => column_orders = Some(read_list(prot, read_column_order)?),
            // 8/9 (encryption): polars has no encryption support; skip.
            _ => prot.skip(f.field_type)?,
        }
        last_field_id = f.id;
    }

    Ok(CompactFileMetaData {
        version: version.require("FileMetaData.version")?,
        schema: schema.require("FileMetaData.schema")?,
        num_rows: num_rows.require("FileMetaData.num_rows")?,
        row_groups: row_groups.require("FileMetaData.row_groups")?,
        key_value_metadata,
        created_by,
        column_orders,
        footer_buf: footer.clone(),
    })
}

/// Decode a `SchemaElement`.
///
/// Same shape as `polars_parquet_format::SchemaElement`. We keep the format
/// type because polars's `SchemaDescriptor::try_from_thrift` consumes it
/// directly and refactoring that crosses the format-crate boundary.
fn read_schema_element(prot: &mut ThriftSliceInputProtocol<'_>) -> ParquetResult<SchemaElement> {
    use polars_parquet_format::{ConvertedType, FieldRepetitionType, LogicalType, Type};

    let mut type_: Option<Type> = None;
    let mut type_length: Option<i32> = None;
    let mut repetition_type: Option<FieldRepetitionType> = None;
    let mut name: Option<String> = None;
    let mut num_children: Option<i32> = None;
    let mut converted_type: Option<ConvertedType> = None;
    let mut scale: Option<i32> = None;
    let mut precision: Option<i32> = None;
    let mut field_id: Option<i32> = None;
    let mut logical_type: Option<LogicalType> = None;

    let mut last_field_id = 0i16;
    loop {
        let f = prot.read_field_begin(last_field_id)?;
        if f.field_type == FieldType::Stop {
            break;
        }
        match f.id {
            1 => type_ = Some(Type(prot.read_i32()?)),
            2 => type_length = Some(prot.read_i32()?),
            3 => repetition_type = Some(FieldRepetitionType(prot.read_i32()?)),
            4 => name = Some(prot.read_string()?.to_owned()),
            5 => num_children = Some(prot.read_i32()?),
            6 => converted_type = Some(ConvertedType(prot.read_i32()?)),
            7 => scale = Some(prot.read_i32()?),
            8 => precision = Some(prot.read_i32()?),
            9 => field_id = Some(prot.read_i32()?),
            10 => logical_type = Some(read_logical_type(prot)?),
            _ => prot.skip(f.field_type)?,
        }
        last_field_id = f.id;
    }

    Ok(SchemaElement {
        type_,
        type_length,
        repetition_type,
        name: name.require("SchemaElement.name")?,
        num_children,
        converted_type,
        scale,
        precision,
        field_id,
        logical_type,
    })
}

/// Decode a `RowGroup` into a [`CompactRowGroup`].
///
/// Skip-decode: `file_offset`, `total_compressed_size`, `ordinal`. None have
/// in-workspace consumers.
fn read_row_group(
    prot: &mut ThriftSliceInputProtocol<'_>,
    origin_ptr: *const u8,
) -> ParquetResult<CompactRowGroup> {
    let mut columns: Option<Vec<CompactColumnChunk>> = None;
    let mut total_byte_size: Option<i64> = None;
    let mut num_rows: Option<i64> = None;
    let mut sorting_columns: Option<Vec<SortingColumn>> = None;

    let mut last_field_id = 0i16;
    loop {
        let f = prot.read_field_begin(last_field_id)?;
        if f.field_type == FieldType::Stop {
            break;
        }
        match f.id {
            1 => columns = Some(read_list(prot, |p| read_column_chunk(p, origin_ptr))?),
            2 => total_byte_size = Some(prot.read_i64()?),
            3 => num_rows = Some(prot.read_i64()?),
            4 => sorting_columns = Some(read_list(prot, read_sorting_column)?),
            // Inlined skips: these fields have no in-tree consumer.
            5 => prot.skip_vlq()?, // file_offset (i64)
            6 => prot.skip_vlq()?, // total_compressed_size (i64)
            7 => prot.skip_vlq()?, // ordinal (i16)
            _ => prot.skip(f.field_type)?,
        }
        last_field_id = f.id;
    }

    Ok(CompactRowGroup {
        columns: columns.require("RowGroup.columns")?,
        total_byte_size: total_byte_size.require("RowGroup.total_byte_size")?,
        num_rows: num_rows.require("RowGroup.num_rows")?,
        sorting_columns,
    })
}

/// Decode a `ColumnChunk` into a [`CompactColumnChunk`].
///
/// Skip-decode: `file_path`, `file_offset`, encryption fields.
fn read_column_chunk(
    prot: &mut ThriftSliceInputProtocol<'_>,
    origin_ptr: *const u8,
) -> ParquetResult<CompactColumnChunk> {
    let mut meta_data: Option<CompactColumnMetaData> = None;
    let mut offset_index_offset: Option<i64> = None;
    let mut offset_index_length: Option<i32> = None;
    let mut column_index_offset: Option<i64> = None;
    let mut column_index_length: Option<i32> = None;

    let mut last_field_id = 0i16;
    loop {
        let f = prot.read_field_begin(last_field_id)?;
        if f.field_type == FieldType::Stop {
            break;
        }
        match f.id {
            // Inlined skips for fields polars doesn't consume (hot path: runs
            // once per column chunk × 200k chunks on wide fixtures).
            1 => prot.skip_binary()?, // file_path (optional binary)
            2 => prot.skip_vlq()?,    // file_offset (i64)
            3 => meta_data = Some(read_column_meta_data(prot, origin_ptr)?),
            4 => offset_index_offset = Some(prot.read_i64()?),
            5 => offset_index_length = Some(prot.read_i32()?),
            6 => column_index_offset = Some(prot.read_i64()?),
            7 => column_index_length = Some(prot.read_i32()?),
            // 8/9 (encryption_algorithm, encrypted_column_metadata): rare.
            _ => prot.skip(f.field_type)?,
        }
        last_field_id = f.id;
    }

    Ok(CompactColumnChunk {
        meta_data: meta_data.require("ColumnChunk.meta_data")?,
        offset_index_offset,
        offset_index_length,
        column_index_offset,
        column_index_length,
    })
}

/// Decode a `ColumnMetaData` into a [`CompactColumnMetaData`].
///
/// Skip-decode: `type_`, `encodings`, `path_in_schema`, `key_value_metadata`,
/// `encoding_stats`, `size_statistics`, `geospatial_statistics`. These have no
/// in-tree readers. See `metadata/compact.rs` for the audit.
fn read_column_meta_data(
    prot: &mut ThriftSliceInputProtocol<'_>,
    origin_ptr: *const u8,
) -> ParquetResult<CompactColumnMetaData> {
    let mut codec: Option<i32> = None;
    let mut num_values: Option<i64> = None;
    let mut total_uncompressed_size: Option<i64> = None;
    let mut total_compressed_size: Option<i64> = None;
    let mut data_page_offset: Option<i64> = None;
    let mut index_page_offset: Option<i64> = None;
    let mut dictionary_page_offset: Option<i64> = None;
    let mut statistics: Option<CompactStatistics> = None;
    let mut bloom_filter_offset: Option<i64> = None;
    let mut bloom_filter_length: Option<i32> = None;

    let mut last_field_id = 0i16;
    loop {
        let f = prot.read_field_begin(last_field_id)?;
        if f.field_type == FieldType::Stop {
            break;
        }
        match f.id {
            // Inlined skips for universally-present unused fields, bypasses
            // the generic recursive `skip_till_depth` on the hot path.
            1 => prot.skip_vlq()?,            // type_ (i32 enum)
            2 => prot.skip_list_of_varint()?, // encodings (list<i32>)
            3 => prot.skip_list_of_binary()?, // path_in_schema (list<binary>)
            4 => codec = Some(prot.read_i32()?),
            5 => num_values = Some(prot.read_i64()?),
            6 => total_uncompressed_size = Some(prot.read_i64()?),
            7 => total_compressed_size = Some(prot.read_i64()?),
            // 8 (key_value_metadata): column-level KV, falls through to the
            // generic `_ => skip(...)` arm below.
            9 => data_page_offset = Some(prot.read_i64()?),
            10 => index_page_offset = Some(prot.read_i64()?),
            11 => dictionary_page_offset = Some(prot.read_i64()?),
            12 => statistics = Some(read_statistics(prot, origin_ptr)?),
            14 => bloom_filter_offset = Some(prot.read_i64()?),
            15 => bloom_filter_length = Some(prot.read_i32()?),
            // 13 (encoding_stats), 16 (size_statistics), 17 (geospatial):
            // tried dedicated skip_simple_struct helpers, no measurable win
            // vs the generic recursive skip (same per-field dispatch cost).
            _ => prot.skip(f.field_type)?,
        }
        last_field_id = f.id;
    }

    let codec_i32 = codec.require("ColumnMetaData.codec")?;
    let codec = Compression::try_from(polars_parquet_format::CompressionCodec(codec_i32))?;

    Ok(CompactColumnMetaData {
        codec,
        num_values: num_values.require("ColumnMetaData.num_values")?,
        total_uncompressed_size: total_uncompressed_size
            .require("ColumnMetaData.total_uncompressed_size")?,
        total_compressed_size: total_compressed_size
            .require("ColumnMetaData.total_compressed_size")?,
        data_page_offset: data_page_offset.require("ColumnMetaData.data_page_offset")?,
        index_page_offset,
        dictionary_page_offset,
        statistics,
        bloom_filter_offset,
        bloom_filter_length,
    })
}

/// Decode a `Statistics` into a [`CompactStatistics`].
///
/// Skip-decode: deprecated `max` / `min` byte vecs (polars reads `max_value` /
/// `min_value`). Modern values land as [`ByteRange`]s into the shared footer
/// buffer instead of per-stat `Vec<u8>` allocations.
fn read_statistics(
    prot: &mut ThriftSliceInputProtocol<'_>,
    origin_ptr: *const u8,
) -> ParquetResult<CompactStatistics> {
    let mut null_count: Option<i64> = None;
    let mut distinct_count: Option<i64> = None;
    let mut max_value: Option<ByteRange> = None;
    let mut min_value: Option<ByteRange> = None;
    let mut is_max_value_exact: Option<bool> = None;
    let mut is_min_value_exact: Option<bool> = None;

    let mut last_field_id = 0i16;
    loop {
        let f = prot.read_field_begin(last_field_id)?;
        if f.field_type == FieldType::Stop {
            break;
        }
        match f.id {
            // Deprecated max/min (pre-2019 writers). Skip without allocation.
            1 => prot.skip_binary()?, // max (deprecated)
            2 => prot.skip_binary()?, // min (deprecated)
            3 => null_count = Some(prot.read_i64()?),
            4 => distinct_count = Some(prot.read_i64()?),
            5 => {
                // Record (offset, len) into the shared footer; skip the bytes.
                let len = prot.read_vlq()? as u32;
                let offset = prot.offset_from(origin_ptr);
                prot.skip_bytes(len as usize)?;
                max_value = Some(ByteRange { offset, len });
            },
            6 => {
                let len = prot.read_vlq()? as u32;
                let offset = prot.offset_from(origin_ptr);
                prot.skip_bytes(len as usize)?;
                min_value = Some(ByteRange { offset, len });
            },
            7 => is_max_value_exact = Some(f.bool_val.unwrap_or_default()),
            8 => is_min_value_exact = Some(f.bool_val.unwrap_or_default()),
            _ => prot.skip(f.field_type)?,
        }
        last_field_id = f.id;
    }

    Ok(CompactStatistics {
        null_count,
        distinct_count,
        max_value,
        min_value,
        is_max_value_exact,
        is_min_value_exact,
    })
}

/// Decode a `KeyValue` (kept as format-crate type, polars exposes it directly).
fn read_key_value(prot: &mut ThriftSliceInputProtocol<'_>) -> ParquetResult<KeyValue> {
    let mut key: Option<String> = None;
    let mut value: Option<String> = None;

    let mut last_field_id = 0i16;
    loop {
        let f = prot.read_field_begin(last_field_id)?;
        if f.field_type == FieldType::Stop {
            break;
        }
        match f.id {
            1 => key = Some(prot.read_string()?.to_owned()),
            2 => value = Some(prot.read_string()?.to_owned()),
            _ => prot.skip(f.field_type)?,
        }
        last_field_id = f.id;
    }

    Ok(KeyValue {
        key: key.require("KeyValue.key")?,
        value,
    })
}

/// Decode a `SortingColumn` (kept as format-crate type, polars exposes it directly).
fn read_sorting_column(prot: &mut ThriftSliceInputProtocol<'_>) -> ParquetResult<SortingColumn> {
    let mut column_idx: Option<i32> = None;
    let mut descending: Option<bool> = None;
    let mut nulls_first: Option<bool> = None;

    let mut last_field_id = 0i16;
    loop {
        let f = prot.read_field_begin(last_field_id)?;
        if f.field_type == FieldType::Stop {
            break;
        }
        match f.id {
            1 => column_idx = Some(prot.read_i32()?),
            2 => descending = Some(f.bool_val.unwrap_or_default()),
            3 => nulls_first = Some(f.bool_val.unwrap_or_default()),
            _ => prot.skip(f.field_type)?,
        }
        last_field_id = f.id;
    }

    Ok(SortingColumn {
        column_idx: column_idx.require("SortingColumn.column_idx")?,
        descending: descending.require("SortingColumn.descending")?,
        nulls_first: nulls_first.require("SortingColumn.nulls_first")?,
    })
}

/// Decode a `ColumnOrder` union (kept as format-crate type).
///
/// Thrift definition: `union ColumnOrder { 1: TypeDefinedOrder TYPE_ORDER }`.
fn read_column_order(prot: &mut ThriftSliceInputProtocol<'_>) -> ParquetResult<ColumnOrder> {
    use polars_parquet_format::TypeDefinedOrder;

    let mut ret: Option<ColumnOrder> = None;
    let mut last_field_id = 0i16;
    loop {
        let f = prot.read_field_begin(last_field_id)?;
        if f.field_type == FieldType::Stop {
            break;
        }
        match f.id {
            1 => {
                read_empty_struct(prot)?;
                ret.get_or_insert(ColumnOrder::TYPEORDER(TypeDefinedOrder {}));
            },
            _ => prot.skip(f.field_type)?,
        }
        last_field_id = f.id;
    }
    ret.ok_or_else(|| ParquetError::oos("ColumnOrder union has no variant set"))
}

/// Decode a `LogicalType` union.
///
/// Kept as `polars_parquet_format::LogicalType` so polars's `SchemaElement`
/// consumer (`SchemaDescriptor::try_from_thrift`) sees the same shape it
/// always has.
fn read_logical_type(
    prot: &mut ThriftSliceInputProtocol<'_>,
) -> ParquetResult<polars_parquet_format::LogicalType> {
    use polars_parquet_format::{
        BsonType, DateType, EnumType, Float16Type, JsonType, ListType, LogicalType, MapType,
        NullType, StringType, UUIDType,
    };

    let mut ret: Option<LogicalType> = None;
    let mut last_field_id = 0i16;
    loop {
        let f = prot.read_field_begin(last_field_id)?;
        if f.field_type == FieldType::Stop {
            break;
        }
        match f.id {
            1 => {
                read_empty_struct(prot)?;
                ret.get_or_insert(LogicalType::STRING(StringType {}));
            },
            2 => {
                read_empty_struct(prot)?;
                ret.get_or_insert(LogicalType::MAP(MapType {}));
            },
            3 => {
                read_empty_struct(prot)?;
                ret.get_or_insert(LogicalType::LIST(ListType {}));
            },
            4 => {
                read_empty_struct(prot)?;
                ret.get_or_insert(LogicalType::ENUM(EnumType {}));
            },
            5 => {
                let v = read_decimal_type(prot)?;
                ret.get_or_insert(LogicalType::DECIMAL(v));
            },
            6 => {
                read_empty_struct(prot)?;
                ret.get_or_insert(LogicalType::DATE(DateType {}));
            },
            7 => {
                let v = read_time_type(prot)?;
                ret.get_or_insert(LogicalType::TIME(v));
            },
            8 => {
                let v = read_timestamp_type(prot)?;
                ret.get_or_insert(LogicalType::TIMESTAMP(v));
            },
            10 => {
                let v = read_int_type(prot)?;
                ret.get_or_insert(LogicalType::INTEGER(v));
            },
            11 => {
                read_empty_struct(prot)?;
                ret.get_or_insert(LogicalType::UNKNOWN(NullType {}));
            },
            12 => {
                read_empty_struct(prot)?;
                ret.get_or_insert(LogicalType::JSON(JsonType {}));
            },
            13 => {
                read_empty_struct(prot)?;
                ret.get_or_insert(LogicalType::BSON(BsonType {}));
            },
            14 => {
                read_empty_struct(prot)?;
                ret.get_or_insert(LogicalType::UUID(UUIDType {}));
            },
            15 => {
                read_empty_struct(prot)?;
                ret.get_or_insert(LogicalType::FLOAT16(Float16Type {}));
            },
            _ => prot.skip(f.field_type)?,
        }
        last_field_id = f.id;
    }
    ret.ok_or_else(|| ParquetError::oos("LogicalType union has no variant set"))
}

fn read_empty_struct(prot: &mut ThriftSliceInputProtocol<'_>) -> ParquetResult<()> {
    let mut last_field_id = 0i16;
    loop {
        let f = prot.read_field_begin(last_field_id)?;
        if f.field_type == FieldType::Stop {
            return Ok(());
        }
        prot.skip(f.field_type)?;
        last_field_id = f.id;
    }
}

fn read_decimal_type(
    prot: &mut ThriftSliceInputProtocol<'_>,
) -> ParquetResult<polars_parquet_format::DecimalType> {
    use polars_parquet_format::DecimalType;
    let mut scale: Option<i32> = None;
    let mut precision: Option<i32> = None;
    let mut last_field_id = 0i16;
    loop {
        let f = prot.read_field_begin(last_field_id)?;
        if f.field_type == FieldType::Stop {
            break;
        }
        match f.id {
            1 => scale = Some(prot.read_i32()?),
            2 => precision = Some(prot.read_i32()?),
            _ => prot.skip(f.field_type)?,
        }
        last_field_id = f.id;
    }
    Ok(DecimalType {
        scale: scale.require("DecimalType.scale")?,
        precision: precision.require("DecimalType.precision")?,
    })
}

fn read_time_unit(
    prot: &mut ThriftSliceInputProtocol<'_>,
) -> ParquetResult<polars_parquet_format::TimeUnit> {
    use polars_parquet_format::{MicroSeconds, MilliSeconds, NanoSeconds, TimeUnit};
    let mut ret: Option<TimeUnit> = None;
    let mut last_field_id = 0i16;
    loop {
        let f = prot.read_field_begin(last_field_id)?;
        if f.field_type == FieldType::Stop {
            break;
        }
        match f.id {
            1 => {
                read_empty_struct(prot)?;
                ret.get_or_insert(TimeUnit::MILLIS(MilliSeconds {}));
            },
            2 => {
                read_empty_struct(prot)?;
                ret.get_or_insert(TimeUnit::MICROS(MicroSeconds {}));
            },
            3 => {
                read_empty_struct(prot)?;
                ret.get_or_insert(TimeUnit::NANOS(NanoSeconds {}));
            },
            _ => prot.skip(f.field_type)?,
        }
        last_field_id = f.id;
    }
    ret.ok_or_else(|| ParquetError::oos("TimeUnit union has no variant set"))
}

fn read_time_type(
    prot: &mut ThriftSliceInputProtocol<'_>,
) -> ParquetResult<polars_parquet_format::TimeType> {
    use polars_parquet_format::TimeType;
    let mut is_adjusted: Option<bool> = None;
    let mut unit: Option<polars_parquet_format::TimeUnit> = None;
    let mut last_field_id = 0i16;
    loop {
        let f = prot.read_field_begin(last_field_id)?;
        if f.field_type == FieldType::Stop {
            break;
        }
        match f.id {
            1 => is_adjusted = Some(f.bool_val.unwrap_or_default()),
            2 => unit = Some(read_time_unit(prot)?),
            _ => prot.skip(f.field_type)?,
        }
        last_field_id = f.id;
    }
    Ok(TimeType {
        is_adjusted_to_u_t_c: is_adjusted.require("TimeType.is_adjusted_to_u_t_c")?,
        unit: unit.require("TimeType.unit")?,
    })
}

fn read_timestamp_type(
    prot: &mut ThriftSliceInputProtocol<'_>,
) -> ParquetResult<polars_parquet_format::TimestampType> {
    use polars_parquet_format::TimestampType;
    let mut is_adjusted: Option<bool> = None;
    let mut unit: Option<polars_parquet_format::TimeUnit> = None;
    let mut last_field_id = 0i16;
    loop {
        let f = prot.read_field_begin(last_field_id)?;
        if f.field_type == FieldType::Stop {
            break;
        }
        match f.id {
            1 => is_adjusted = Some(f.bool_val.unwrap_or_default()),
            2 => unit = Some(read_time_unit(prot)?),
            _ => prot.skip(f.field_type)?,
        }
        last_field_id = f.id;
    }
    Ok(TimestampType {
        is_adjusted_to_u_t_c: is_adjusted.require("TimestampType.is_adjusted_to_u_t_c")?,
        unit: unit.require("TimestampType.unit")?,
    })
}

fn read_int_type(
    prot: &mut ThriftSliceInputProtocol<'_>,
) -> ParquetResult<polars_parquet_format::IntType> {
    use polars_parquet_format::IntType;
    let mut bit_width: Option<i8> = None;
    let mut is_signed: Option<bool> = None;
    let mut last_field_id = 0i16;
    loop {
        let f = prot.read_field_begin(last_field_id)?;
        if f.field_type == FieldType::Stop {
            break;
        }
        match f.id {
            1 => bit_width = Some(prot.read_i8()?),
            2 => is_signed = Some(f.bool_val.unwrap_or_default()),
            _ => prot.skip(f.field_type)?,
        }
        last_field_id = f.id;
    }
    Ok(IntType {
        bit_width: bit_width.require("IntType.bit_width")?,
        is_signed: is_signed.require("IntType.is_signed")?,
    })
}
