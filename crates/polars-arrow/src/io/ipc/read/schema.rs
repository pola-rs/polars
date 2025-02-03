use std::sync::Arc;

use arrow_format::ipc::planus::ReadAsRoot;
use arrow_format::ipc::{FieldRef, FixedSizeListRef, MapRef, TimeRef, TimestampRef, UnionRef};
use polars_error::{polars_bail, polars_err, PolarsResult};
use polars_utils::pl_str::PlSmallStr;

use super::super::{IpcField, IpcSchema};
use super::{OutOfSpecKind, StreamMetadata};
use crate::datatypes::{
    get_extension, ArrowDataType, ArrowSchema, Extension, ExtensionType, Field, IntegerType,
    IntervalUnit, Metadata, TimeUnit, UnionMode, UnionType,
};

fn try_unzip_vec<A, B, I: Iterator<Item = PolarsResult<(A, B)>>>(
    iter: I,
) -> PolarsResult<(Vec<A>, Vec<B>)> {
    let mut a = vec![];
    let mut b = vec![];
    for maybe_item in iter {
        let (a_i, b_i) = maybe_item?;
        a.push(a_i);
        b.push(b_i);
    }

    Ok((a, b))
}

fn deserialize_field(ipc_field: arrow_format::ipc::FieldRef) -> PolarsResult<(Field, IpcField)> {
    let metadata = read_metadata(&ipc_field)?;

    let extension = metadata.as_ref().and_then(get_extension);

    let (dtype, ipc_field_) = get_dtype(ipc_field, extension, true)?;

    let field = Field {
        name: PlSmallStr::from_str(
            ipc_field
                .name()?
                .ok_or_else(|| polars_err!(oos = "Every field in IPC must have a name"))?,
        ),
        dtype,
        is_nullable: ipc_field.nullable()?,
        metadata: metadata.map(Arc::new),
    };

    Ok((field, ipc_field_))
}

fn read_metadata(field: &arrow_format::ipc::FieldRef) -> PolarsResult<Option<Metadata>> {
    Ok(if let Some(list) = field.custom_metadata()? {
        let mut metadata_map = Metadata::new();
        for kv in list {
            let kv = kv?;
            if let (Some(k), Some(v)) = (kv.key()?, kv.value()?) {
                metadata_map.insert(PlSmallStr::from_str(k), PlSmallStr::from_str(v));
            }
        }
        Some(metadata_map)
    } else {
        None
    })
}

fn deserialize_integer(int: arrow_format::ipc::IntRef) -> PolarsResult<IntegerType> {
    Ok(match (int.bit_width()?, int.is_signed()?) {
        (8, true) => IntegerType::Int8,
        (8, false) => IntegerType::UInt8,
        (16, true) => IntegerType::Int16,
        (16, false) => IntegerType::UInt16,
        (32, true) => IntegerType::Int32,
        (32, false) => IntegerType::UInt32,
        (64, true) => IntegerType::Int64,
        (64, false) => IntegerType::UInt64,
        (128, true) => IntegerType::Int128,
        _ => polars_bail!(oos = "IPC: indexType can only be 8, 16, 32, 64 or 128."),
    })
}

fn deserialize_timeunit(time_unit: arrow_format::ipc::TimeUnit) -> PolarsResult<TimeUnit> {
    use arrow_format::ipc::TimeUnit::*;
    Ok(match time_unit {
        Second => TimeUnit::Second,
        Millisecond => TimeUnit::Millisecond,
        Microsecond => TimeUnit::Microsecond,
        Nanosecond => TimeUnit::Nanosecond,
    })
}

fn deserialize_time(time: TimeRef) -> PolarsResult<(ArrowDataType, IpcField)> {
    let unit = deserialize_timeunit(time.unit()?)?;

    let dtype = match (time.bit_width()?, unit) {
        (32, TimeUnit::Second) => ArrowDataType::Time32(TimeUnit::Second),
        (32, TimeUnit::Millisecond) => ArrowDataType::Time32(TimeUnit::Millisecond),
        (64, TimeUnit::Microsecond) => ArrowDataType::Time64(TimeUnit::Microsecond),
        (64, TimeUnit::Nanosecond) => ArrowDataType::Time64(TimeUnit::Nanosecond),
        (bits, precision) => {
            polars_bail!(ComputeError:
                "Time type with bit width of {bits} and unit of {precision:?}"
            )
        },
    };
    Ok((dtype, IpcField::default()))
}

fn deserialize_timestamp(timestamp: TimestampRef) -> PolarsResult<(ArrowDataType, IpcField)> {
    let timezone = timestamp.timezone()?;
    let time_unit = deserialize_timeunit(timestamp.unit()?)?;
    Ok((
        ArrowDataType::Timestamp(time_unit, timezone.map(PlSmallStr::from_str)),
        IpcField::default(),
    ))
}

fn deserialize_union(union_: UnionRef, field: FieldRef) -> PolarsResult<(ArrowDataType, IpcField)> {
    let mode = UnionMode::sparse(union_.mode()? == arrow_format::ipc::UnionMode::Sparse);
    let ids = union_.type_ids()?.map(|x| x.iter().collect());

    let fields = field
        .children()?
        .ok_or_else(|| polars_err!(oos = "IPC: Union must contain children"))?;
    if fields.is_empty() {
        polars_bail!(oos = "IPC: Union must contain at least one child");
    }

    let (fields, ipc_fields) = try_unzip_vec(fields.iter().map(|field| {
        let (field, fields) = deserialize_field(field?)?;
        Ok((field, fields))
    }))?;
    let ipc_field = IpcField {
        fields: ipc_fields,
        dictionary_id: None,
    };
    Ok((
        ArrowDataType::Union(Box::new(UnionType { fields, ids, mode })),
        ipc_field,
    ))
}

fn deserialize_map(map: MapRef, field: FieldRef) -> PolarsResult<(ArrowDataType, IpcField)> {
    let is_sorted = map.keys_sorted()?;

    let children = field
        .children()?
        .ok_or_else(|| polars_err!(oos = "IPC: Map must contain children"))?;
    let inner = children
        .get(0)
        .ok_or_else(|| polars_err!(oos = "IPC: Map must contain one child"))??;
    let (field, ipc_field) = deserialize_field(inner)?;

    let dtype = ArrowDataType::Map(Box::new(field), is_sorted);
    Ok((
        dtype,
        IpcField {
            fields: vec![ipc_field],
            dictionary_id: None,
        },
    ))
}

fn deserialize_struct(field: FieldRef) -> PolarsResult<(ArrowDataType, IpcField)> {
    let fields = field
        .children()?
        .ok_or_else(|| polars_err!(oos = "IPC: Struct must contain children"))?;
    let (fields, ipc_fields) = try_unzip_vec(fields.iter().map(|field| {
        let (field, fields) = deserialize_field(field?)?;
        Ok((field, fields))
    }))?;
    let ipc_field = IpcField {
        fields: ipc_fields,
        dictionary_id: None,
    };
    Ok((ArrowDataType::Struct(fields), ipc_field))
}

fn deserialize_list(field: FieldRef) -> PolarsResult<(ArrowDataType, IpcField)> {
    let children = field
        .children()?
        .ok_or_else(|| polars_err!(oos = "IPC: List must contain children"))?;
    let inner = children
        .get(0)
        .ok_or_else(|| polars_err!(oos = "IPC: List must contain one child"))??;
    let (field, ipc_field) = deserialize_field(inner)?;

    Ok((
        ArrowDataType::List(Box::new(field)),
        IpcField {
            fields: vec![ipc_field],
            dictionary_id: None,
        },
    ))
}

fn deserialize_large_list(field: FieldRef) -> PolarsResult<(ArrowDataType, IpcField)> {
    let children = field
        .children()?
        .ok_or_else(|| polars_err!(oos = "IPC: List must contain children"))?;
    let inner = children
        .get(0)
        .ok_or_else(|| polars_err!(oos = "IPC: List must contain one child"))??;
    let (field, ipc_field) = deserialize_field(inner)?;

    Ok((
        ArrowDataType::LargeList(Box::new(field)),
        IpcField {
            fields: vec![ipc_field],
            dictionary_id: None,
        },
    ))
}

fn deserialize_fixed_size_list(
    list: FixedSizeListRef,
    field: FieldRef,
) -> PolarsResult<(ArrowDataType, IpcField)> {
    let children = field
        .children()?
        .ok_or_else(|| polars_err!(oos = "IPC: FixedSizeList must contain children"))?;
    let inner = children
        .get(0)
        .ok_or_else(|| polars_err!(oos = "IPC: FixedSizeList must contain one child"))??;
    let (field, ipc_field) = deserialize_field(inner)?;

    let size = list
        .list_size()?
        .try_into()
        .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

    Ok((
        ArrowDataType::FixedSizeList(Box::new(field), size),
        IpcField {
            fields: vec![ipc_field],
            dictionary_id: None,
        },
    ))
}

/// Get the Arrow data type from the flatbuffer Field table
fn get_dtype(
    field: arrow_format::ipc::FieldRef,
    extension: Extension,
    may_be_dictionary: bool,
) -> PolarsResult<(ArrowDataType, IpcField)> {
    if let Some(dictionary) = field.dictionary()? {
        if may_be_dictionary {
            let int = dictionary
                .index_type()?
                .ok_or_else(|| polars_err!(oos = "indexType is mandatory in Dictionary."))?;
            let index_type = deserialize_integer(int)?;
            let (inner, mut ipc_field) = get_dtype(field, extension, false)?;
            ipc_field.dictionary_id = Some(dictionary.id()?);
            return Ok((
                ArrowDataType::Dictionary(index_type, Box::new(inner), dictionary.is_ordered()?),
                ipc_field,
            ));
        }
    }

    if let Some(extension) = extension {
        let (name, metadata) = extension;
        let (dtype, fields) = get_dtype(field, None, false)?;
        return Ok((
            ArrowDataType::Extension(Box::new(ExtensionType {
                name,
                inner: dtype,
                metadata,
            })),
            fields,
        ));
    }

    let type_ = field
        .type_()?
        .ok_or_else(|| polars_err!(oos = "IPC: field type is mandatory"))?;

    use arrow_format::ipc::TypeRef::*;
    Ok(match type_ {
        Null(_) => (ArrowDataType::Null, IpcField::default()),
        Bool(_) => (ArrowDataType::Boolean, IpcField::default()),
        Int(int) => {
            let dtype = deserialize_integer(int)?.into();
            (dtype, IpcField::default())
        },
        Binary(_) => (ArrowDataType::Binary, IpcField::default()),
        LargeBinary(_) => (ArrowDataType::LargeBinary, IpcField::default()),
        Utf8(_) => (ArrowDataType::Utf8, IpcField::default()),
        LargeUtf8(_) => (ArrowDataType::LargeUtf8, IpcField::default()),
        BinaryView(_) => (ArrowDataType::BinaryView, IpcField::default()),
        Utf8View(_) => (ArrowDataType::Utf8View, IpcField::default()),
        FixedSizeBinary(fixed) => (
            ArrowDataType::FixedSizeBinary(
                fixed
                    .byte_width()?
                    .try_into()
                    .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?,
            ),
            IpcField::default(),
        ),
        FloatingPoint(float) => {
            let dtype = match float.precision()? {
                arrow_format::ipc::Precision::Half => ArrowDataType::Float16,
                arrow_format::ipc::Precision::Single => ArrowDataType::Float32,
                arrow_format::ipc::Precision::Double => ArrowDataType::Float64,
            };
            (dtype, IpcField::default())
        },
        Date(date) => {
            let dtype = match date.unit()? {
                arrow_format::ipc::DateUnit::Day => ArrowDataType::Date32,
                arrow_format::ipc::DateUnit::Millisecond => ArrowDataType::Date64,
            };
            (dtype, IpcField::default())
        },
        Time(time) => deserialize_time(time)?,
        Timestamp(timestamp) => deserialize_timestamp(timestamp)?,
        Interval(interval) => {
            let dtype = match interval.unit()? {
                arrow_format::ipc::IntervalUnit::YearMonth => {
                    ArrowDataType::Interval(IntervalUnit::YearMonth)
                },
                arrow_format::ipc::IntervalUnit::DayTime => {
                    ArrowDataType::Interval(IntervalUnit::DayTime)
                },
                arrow_format::ipc::IntervalUnit::MonthDayNano => {
                    ArrowDataType::Interval(IntervalUnit::MonthDayNano)
                },
            };
            (dtype, IpcField::default())
        },
        Duration(duration) => {
            let time_unit = deserialize_timeunit(duration.unit()?)?;
            (ArrowDataType::Duration(time_unit), IpcField::default())
        },
        Decimal(decimal) => {
            let bit_width: usize = decimal
                .bit_width()?
                .try_into()
                .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;
            let precision: usize = decimal
                .precision()?
                .try_into()
                .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;
            let scale: usize = decimal
                .scale()?
                .try_into()
                .map_err(|_| polars_err!(oos = OutOfSpecKind::NegativeFooterLength))?;

            let dtype = match bit_width {
                128 => ArrowDataType::Decimal(precision, scale),
                256 => ArrowDataType::Decimal256(precision, scale),
                _ => return Err(polars_err!(oos = OutOfSpecKind::NegativeFooterLength)),
            };

            (dtype, IpcField::default())
        },
        List(_) => deserialize_list(field)?,
        LargeList(_) => deserialize_large_list(field)?,
        FixedSizeList(list) => deserialize_fixed_size_list(list, field)?,
        Struct(_) => deserialize_struct(field)?,
        Union(union_) => deserialize_union(union_, field)?,
        Map(map) => deserialize_map(map, field)?,
        RunEndEncoded(_) => todo!(),
        LargeListView(_) | ListView(_) => todo!(),
    })
}

/// Deserialize an flatbuffers-encoded Schema message into [`ArrowSchema`] and [`IpcSchema`].
pub fn deserialize_schema(
    message: &[u8],
) -> PolarsResult<(ArrowSchema, IpcSchema, Option<Metadata>)> {
    let message = arrow_format::ipc::MessageRef::read_as_root(message)
        .map_err(|err| polars_err!(oos = format!("Unable deserialize message: {err:?}")))?;

    let schema = match message
        .header()?
        .ok_or_else(|| polars_err!(oos = "Unable to convert header to a schema".to_string()))?
    {
        arrow_format::ipc::MessageHeaderRef::Schema(schema) => PolarsResult::Ok(schema),
        _ => polars_bail!(ComputeError: "The message is expected to be a Schema message"),
    }?;

    fb_to_schema(schema)
}

/// Deserialize the raw Schema table from IPC format to Schema data type
pub(super) fn fb_to_schema(
    schema: arrow_format::ipc::SchemaRef,
) -> PolarsResult<(ArrowSchema, IpcSchema, Option<Metadata>)> {
    let fields = schema
        .fields()?
        .ok_or_else(|| polars_err!(oos = OutOfSpecKind::MissingFields))?;

    let mut arrow_schema = ArrowSchema::with_capacity(fields.len());
    let mut ipc_fields = Vec::with_capacity(fields.len());

    for field in fields {
        let (field, ipc_field) = deserialize_field(field?)?;
        arrow_schema.insert(field.name.clone(), field);
        ipc_fields.push(ipc_field);
    }

    let is_little_endian = match schema.endianness()? {
        arrow_format::ipc::Endianness::Little => true,
        arrow_format::ipc::Endianness::Big => false,
    };

    let custom_schema_metadata = match schema.custom_metadata()? {
        None => None,
        Some(metadata) => {
            let metadata: Metadata = metadata
                .into_iter()
                .filter_map(|kv_result| {
                    // FIXME: silently hiding errors here
                    let kv_ref = kv_result.ok()?;
                    Some((kv_ref.key().ok()??.into(), kv_ref.value().ok()??.into()))
                })
                .collect();

            if metadata.is_empty() {
                None
            } else {
                Some(metadata)
            }
        },
    };

    Ok((
        arrow_schema,
        IpcSchema {
            fields: ipc_fields,
            is_little_endian,
        },
        custom_schema_metadata,
    ))
}

pub(super) fn deserialize_stream_metadata(meta: &[u8]) -> PolarsResult<StreamMetadata> {
    let message = arrow_format::ipc::MessageRef::read_as_root(meta)
        .map_err(|err| polars_err!(oos = format!("Unable to get root as message: {err:?}")))?;
    let version = message.version()?;
    // message header is a Schema, so read it
    let header = message
        .header()?
        .ok_or_else(|| polars_err!(oos = "Unable to read the first IPC message"))?;
    let schema = if let arrow_format::ipc::MessageHeaderRef::Schema(schema) = header {
        schema
    } else {
        polars_bail!(oos = "The first IPC message of the stream must be a schema")
    };
    let (schema, ipc_schema, custom_schema_metadata) = fb_to_schema(schema)?;

    Ok(StreamMetadata {
        schema,
        version,
        ipc_schema,
        custom_schema_metadata,
    })
}
