use arrow_format::ipc::planus::ReadAsRoot;
use arrow_format::ipc::{FieldRef, FixedSizeListRef, MapRef, TimeRef, TimestampRef, UnionRef};
use polars_error::{polars_bail, polars_err, PolarsResult};

use super::super::{IpcField, IpcSchema};
use super::{OutOfSpecKind, StreamMetadata};
use crate::datatypes::{
    get_extension, ArrowDataType, ArrowSchema, Extension, Field, IntegerType, IntervalUnit,
    Metadata, TimeUnit, UnionMode,
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

    let extension = get_extension(&metadata);

    let (data_type, ipc_field_) = get_data_type(ipc_field, extension, true)?;

    let field = Field {
        name: ipc_field
            .name()?
            .ok_or_else(|| polars_err!(oos = "Every field in IPC must have a name"))?
            .to_string(),
        data_type,
        is_nullable: ipc_field.nullable()?,
        metadata,
    };

    Ok((field, ipc_field_))
}

fn read_metadata(field: &arrow_format::ipc::FieldRef) -> PolarsResult<Metadata> {
    Ok(if let Some(list) = field.custom_metadata()? {
        let mut metadata_map = Metadata::new();
        for kv in list {
            let kv = kv?;
            if let (Some(k), Some(v)) = (kv.key()?, kv.value()?) {
                metadata_map.insert(k.to_string(), v.to_string());
            }
        }
        metadata_map
    } else {
        Metadata::default()
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
        _ => polars_bail!(oos = "IPC: indexType can only be 8, 16, 32 or 64."),
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

    let data_type = match (time.bit_width()?, unit) {
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
    Ok((data_type, IpcField::default()))
}

fn deserialize_timestamp(timestamp: TimestampRef) -> PolarsResult<(ArrowDataType, IpcField)> {
    let timezone = timestamp.timezone()?.map(|tz| tz.to_string());
    let time_unit = deserialize_timeunit(timestamp.unit()?)?;
    Ok((
        ArrowDataType::Timestamp(time_unit, timezone),
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
    Ok((ArrowDataType::Union(fields, ids, mode), ipc_field))
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

    let data_type = ArrowDataType::Map(Box::new(field), is_sorted);
    Ok((
        data_type,
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
    if fields.is_empty() {
        polars_bail!(oos = "IPC: Struct must contain at least one child");
    }
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
fn get_data_type(
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
            let (inner, mut ipc_field) = get_data_type(field, extension, false)?;
            ipc_field.dictionary_id = Some(dictionary.id()?);
            return Ok((
                ArrowDataType::Dictionary(index_type, Box::new(inner), dictionary.is_ordered()?),
                ipc_field,
            ));
        }
    }

    if let Some(extension) = extension {
        let (name, metadata) = extension;
        let (data_type, fields) = get_data_type(field, None, false)?;
        return Ok((
            ArrowDataType::Extension(name, Box::new(data_type), metadata),
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
            let data_type = deserialize_integer(int)?.into();
            (data_type, IpcField::default())
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
            let data_type = match float.precision()? {
                arrow_format::ipc::Precision::Half => ArrowDataType::Float16,
                arrow_format::ipc::Precision::Single => ArrowDataType::Float32,
                arrow_format::ipc::Precision::Double => ArrowDataType::Float64,
            };
            (data_type, IpcField::default())
        },
        Date(date) => {
            let data_type = match date.unit()? {
                arrow_format::ipc::DateUnit::Day => ArrowDataType::Date32,
                arrow_format::ipc::DateUnit::Millisecond => ArrowDataType::Date64,
            };
            (data_type, IpcField::default())
        },
        Time(time) => deserialize_time(time)?,
        Timestamp(timestamp) => deserialize_timestamp(timestamp)?,
        Interval(interval) => {
            let data_type = match interval.unit()? {
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
            (data_type, IpcField::default())
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

            let data_type = match bit_width {
                128 => ArrowDataType::Decimal(precision, scale),
                256 => ArrowDataType::Decimal256(precision, scale),
                _ => return Err(polars_err!(oos = OutOfSpecKind::NegativeFooterLength)),
            };

            (data_type, IpcField::default())
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
pub fn deserialize_schema(message: &[u8]) -> PolarsResult<(ArrowSchema, IpcSchema)> {
    let message = arrow_format::ipc::MessageRef::read_as_root(message)
        .map_err(|_err| polars_err!(oos = "Unable deserialize message: {err:?}"))?;

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
) -> PolarsResult<(ArrowSchema, IpcSchema)> {
    let fields = schema
        .fields()?
        .ok_or_else(|| polars_err!(oos = OutOfSpecKind::MissingFields))?;
    let (fields, ipc_fields) = try_unzip_vec(fields.iter().map(|field| {
        let (field, fields) = deserialize_field(field?)?;
        Ok((field, fields))
    }))?;

    let is_little_endian = match schema.endianness()? {
        arrow_format::ipc::Endianness::Little => true,
        arrow_format::ipc::Endianness::Big => false,
    };

    let mut metadata = Metadata::default();
    if let Some(md_fields) = schema.custom_metadata()? {
        for kv in md_fields {
            let kv = kv?;
            let k_str = kv.key()?;
            let v_str = kv.value()?;
            if let Some(k) = k_str {
                if let Some(v) = v_str {
                    metadata.insert(k.to_string(), v.to_string());
                }
            }
        }
    }

    Ok((
        ArrowSchema { fields, metadata },
        IpcSchema {
            fields: ipc_fields,
            is_little_endian,
        },
    ))
}

pub(super) fn deserialize_stream_metadata(meta: &[u8]) -> PolarsResult<StreamMetadata> {
    let message = arrow_format::ipc::MessageRef::read_as_root(meta)
        .map_err(|_err| polars_err!(oos = "Unable to get root as message: {err:?}"))?;
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
    let (schema, ipc_schema) = fb_to_schema(schema)?;

    Ok(StreamMetadata {
        schema,
        version,
        ipc_schema,
    })
}
