//! This module has entry points, [`parquet_to_arrow_schema`] and the more configurable [`parquet_to_arrow_schema_with_options`].
use std::sync::Arc;

use polars_arrow::datatypes::{
    ArrowDataType, ArrowSchema, Field, IntervalUnit, Metadata, TimeUnit,
};
use polars_error::{PolarsResult, polars_bail};
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;

use crate::arrow::read::schema::SchemaInferenceOptions;
use crate::parquet::schema::Repetition;
use crate::parquet::schema::types::{
    FieldInfo, GroupConvertedType, GroupLogicalType, IntegerType, ParquetType, PhysicalType,
    PrimitiveConvertedType, PrimitiveLogicalType, PrimitiveType, TimeUnit as ParquetTimeUnit,
};

/// Converts [`ParquetType`]s to a [`Field`], ignoring parquet fields that do not contain
/// any physical column.
pub fn parquet_to_arrow_schema(fields: &[ParquetType]) -> PolarsResult<ArrowSchema> {
    parquet_to_arrow_schema_with_options(fields, &SchemaInferenceOptions::default())
}

/// Like [`parquet_to_arrow_schema`] but with configurable options which affect the behavior of schema inference
pub fn parquet_to_arrow_schema_with_options(
    fields: &[ParquetType],
    options: &SchemaInferenceOptions,
) -> PolarsResult<ArrowSchema> {
    let fields = fields
        .iter()
        .map(|f| to_field(f, options))
        .collect::<PolarsResult<Vec<Option<Field>>>>()?;

    Ok(fields
        .into_iter()
        .flatten()
        .map(|x| (x.name.clone(), x))
        .collect())
}

fn from_int32(
    logical_type: Option<PrimitiveLogicalType>,
    converted_type: Option<PrimitiveConvertedType>,
) -> ArrowDataType {
    use PrimitiveLogicalType::*;
    match (logical_type, converted_type) {
        // handle logical types first
        (Some(Integer(t)), _) => match t {
            IntegerType::Int8 => ArrowDataType::Int8,
            IntegerType::Int16 => ArrowDataType::Int16,
            IntegerType::Int32 => ArrowDataType::Int32,
            IntegerType::UInt8 => ArrowDataType::UInt8,
            IntegerType::UInt16 => ArrowDataType::UInt16,
            IntegerType::UInt32 => ArrowDataType::UInt32,
            // The above are the only possible annotations for parquet's int32. Anything else
            // is a deviation to the parquet specification and we ignore
            _ => ArrowDataType::Int32,
        },
        (Some(Decimal(precision, scale)), _) => ArrowDataType::Decimal(precision, scale),
        (Some(Date), _) => ArrowDataType::Date32,
        (Some(Time { unit, .. }), _) => match unit {
            ParquetTimeUnit::Milliseconds => ArrowDataType::Time32(TimeUnit::Millisecond),
            // MILLIS is the only possible annotation for parquet's int32. Anything else
            // is a deviation to the parquet specification and we ignore
            _ => ArrowDataType::Int32,
        },
        // handle converted types:
        (_, Some(PrimitiveConvertedType::Uint8)) => ArrowDataType::UInt8,
        (_, Some(PrimitiveConvertedType::Uint16)) => ArrowDataType::UInt16,
        (_, Some(PrimitiveConvertedType::Uint32)) => ArrowDataType::UInt32,
        (_, Some(PrimitiveConvertedType::Int8)) => ArrowDataType::Int8,
        (_, Some(PrimitiveConvertedType::Int16)) => ArrowDataType::Int16,
        (_, Some(PrimitiveConvertedType::Int32)) => ArrowDataType::Int32,
        (_, Some(PrimitiveConvertedType::Date)) => ArrowDataType::Date32,
        (_, Some(PrimitiveConvertedType::TimeMillis)) => {
            ArrowDataType::Time32(TimeUnit::Millisecond)
        },
        (_, Some(PrimitiveConvertedType::Decimal(precision, scale))) => {
            ArrowDataType::Decimal(precision, scale)
        },
        (_, _) => ArrowDataType::Int32,
    }
}

fn from_int64(
    logical_type: Option<PrimitiveLogicalType>,
    converted_type: Option<PrimitiveConvertedType>,
) -> ArrowDataType {
    use PrimitiveLogicalType::*;
    match (logical_type, converted_type) {
        // handle logical types first
        (Some(Integer(integer)), _) => match integer {
            IntegerType::UInt64 => ArrowDataType::UInt64,
            IntegerType::Int64 => ArrowDataType::Int64,
            _ => ArrowDataType::Int64,
        },
        (
            Some(Timestamp {
                is_adjusted_to_utc,
                unit,
            }),
            _,
        ) => {
            let timezone = if is_adjusted_to_utc {
                // https://github.com/apache/parquet-format/blob/master/LogicalTypes.md
                // A TIMESTAMP with isAdjustedToUTC=true is defined as [...] elapsed since the Unix epoch
                Some(PlSmallStr::from_static("+00:00"))
            } else {
                // PARQUET:
                // https://github.com/apache/parquet-format/blob/master/LogicalTypes.md
                // A TIMESTAMP with isAdjustedToUTC=false represents [...] such
                // timestamps should always be displayed the same way, regardless of the local time zone in effect
                // ARROW:
                // https://github.com/apache/parquet-format/blob/master/LogicalTypes.md
                // If the time zone is null or equal to an empty string, the data is "time
                // zone naive" and shall be displayed *as is* to the user, not localized
                // to the locale of the user.
                None
            };

            match unit {
                ParquetTimeUnit::Milliseconds => {
                    ArrowDataType::Timestamp(TimeUnit::Millisecond, timezone)
                },
                ParquetTimeUnit::Microseconds => {
                    ArrowDataType::Timestamp(TimeUnit::Microsecond, timezone)
                },
                ParquetTimeUnit::Nanoseconds => {
                    ArrowDataType::Timestamp(TimeUnit::Nanosecond, timezone)
                },
            }
        },
        (Some(Time { unit, .. }), _) => match unit {
            ParquetTimeUnit::Microseconds => ArrowDataType::Time64(TimeUnit::Microsecond),
            ParquetTimeUnit::Nanoseconds => ArrowDataType::Time64(TimeUnit::Nanosecond),
            // MILLIS is only possible for int32. Appearing in int64 is a deviation
            // to parquet's spec, which we ignore
            _ => ArrowDataType::Int64,
        },
        (Some(Decimal(precision, scale)), _) => ArrowDataType::Decimal(precision, scale),
        // handle converted types:
        (_, Some(PrimitiveConvertedType::TimeMicros)) => {
            ArrowDataType::Time64(TimeUnit::Microsecond)
        },
        (_, Some(PrimitiveConvertedType::TimestampMillis)) => {
            ArrowDataType::Timestamp(TimeUnit::Millisecond, None)
        },
        (_, Some(PrimitiveConvertedType::TimestampMicros)) => {
            ArrowDataType::Timestamp(TimeUnit::Microsecond, None)
        },
        (_, Some(PrimitiveConvertedType::Int64)) => ArrowDataType::Int64,
        (_, Some(PrimitiveConvertedType::Uint64)) => ArrowDataType::UInt64,
        (_, Some(PrimitiveConvertedType::Decimal(precision, scale))) => {
            ArrowDataType::Decimal(precision, scale)
        },

        (_, _) => ArrowDataType::Int64,
    }
}

fn from_byte_array(
    logical_type: &Option<PrimitiveLogicalType>,
    converted_type: &Option<PrimitiveConvertedType>,
) -> ArrowDataType {
    match (logical_type, converted_type) {
        (Some(PrimitiveLogicalType::Decimal(precision, scale)), _) => {
            ArrowDataType::Decimal(*precision, *scale)
        },
        (None, Some(PrimitiveConvertedType::Decimal(precision, scale))) => {
            ArrowDataType::Decimal(*precision, *scale)
        },
        (Some(PrimitiveLogicalType::String), _) => ArrowDataType::Utf8View,
        (Some(PrimitiveLogicalType::Json), _) => ArrowDataType::BinaryView,
        (Some(PrimitiveLogicalType::Bson), _) => ArrowDataType::BinaryView,
        (Some(PrimitiveLogicalType::Enum), _) => ArrowDataType::Utf8View,
        (_, Some(PrimitiveConvertedType::Json)) => ArrowDataType::BinaryView,
        (_, Some(PrimitiveConvertedType::Bson)) => ArrowDataType::BinaryView,
        (_, Some(PrimitiveConvertedType::Enum)) => ArrowDataType::Utf8View,
        (_, Some(PrimitiveConvertedType::Utf8)) => ArrowDataType::Utf8View,
        (_, _) => ArrowDataType::BinaryView,
    }
}

fn from_fixed_len_byte_array(
    length: usize,
    logical_type: Option<PrimitiveLogicalType>,
    converted_type: Option<PrimitiveConvertedType>,
) -> ArrowDataType {
    match (logical_type, converted_type) {
        (Some(PrimitiveLogicalType::Decimal(precision, scale)), _) => {
            ArrowDataType::Decimal(precision, scale)
        },
        (None, Some(PrimitiveConvertedType::Decimal(precision, scale))) => {
            ArrowDataType::Decimal(precision, scale)
        },
        (None, Some(PrimitiveConvertedType::Interval)) => {
            ArrowDataType::Interval(IntervalUnit::MonthDayMillis)
        },
        _ => ArrowDataType::FixedSizeBinary(length),
    }
}

/// Maps a [`PhysicalType`] with optional metadata to a [`ArrowDataType`]
fn to_primitive_type_inner(
    primitive_type: &PrimitiveType,
    options: &SchemaInferenceOptions,
) -> ArrowDataType {
    match primitive_type.physical_type {
        PhysicalType::Boolean => ArrowDataType::Boolean,
        PhysicalType::Int32 => {
            from_int32(primitive_type.logical_type, primitive_type.converted_type)
        },
        PhysicalType::Int64 => {
            from_int64(primitive_type.logical_type, primitive_type.converted_type)
        },
        PhysicalType::Int96 => ArrowDataType::Timestamp(options.int96_coerce_to_timeunit, None),
        PhysicalType::Float => ArrowDataType::Float32,
        PhysicalType::Double => ArrowDataType::Float64,
        PhysicalType::ByteArray => {
            from_byte_array(&primitive_type.logical_type, &primitive_type.converted_type)
        },
        PhysicalType::FixedLenByteArray(length) => from_fixed_len_byte_array(
            length,
            primitive_type.logical_type,
            primitive_type.converted_type,
        ),
    }
}

/// Entry point for converting parquet primitive type to arrow type.
///
/// This function takes care of repetition.
fn to_primitive_type(
    primitive_type: &PrimitiveType,
    options: &SchemaInferenceOptions,
) -> ArrowDataType {
    let base_type = to_primitive_type_inner(primitive_type, options);

    if primitive_type.field_info.repetition == Repetition::Repeated {
        ArrowDataType::LargeList(Box::new(Field::new(
            primitive_type.field_info.name.clone(),
            base_type,
            is_optional(&primitive_type.field_info),
        )))
    } else {
        base_type
    }
}

fn non_repeated_group(
    field_info: &FieldInfo,
    logical_type: &Option<GroupLogicalType>,
    converted_type: &Option<GroupConvertedType>,
    fields: &[ParquetType],
    options: &SchemaInferenceOptions,
) -> PolarsResult<Option<ArrowDataType>> {
    debug_assert!(!fields.is_empty());
    match (logical_type, converted_type) {
        _ if is_list_annotated(logical_type, converted_type) => {
            to_list(field_info, fields, options)
        },
        (Some(GroupLogicalType::Map), _)
        | (None, Some(GroupConvertedType::Map) | Some(GroupConvertedType::MapKeyValue)) => {
            to_map(field_info, fields, options)
        },
        _ => to_struct(fields, options),
    }
}

fn is_list_annotated(
    logical_type: &Option<GroupLogicalType>,
    converted_type: &Option<GroupConvertedType>,
) -> bool {
    matches!(
        (logical_type, converted_type),
        (Some(GroupLogicalType::List), _) | (None, Some(GroupConvertedType::List))
    )
}

fn ensure_not_repeated_map(
    field_info: &FieldInfo,
    logical_type: &Option<GroupLogicalType>,
    converted_type: &Option<GroupConvertedType>,
) -> PolarsResult<()> {
    let is_map = matches!(logical_type, Some(GroupLogicalType::Map))
        || matches!(
            converted_type,
            Some(GroupConvertedType::Map | GroupConvertedType::MapKeyValue)
        );

    if is_map && field_info.repetition == Repetition::Repeated {
        polars_bail!(
            ComputeError:
            "parquet group '{}' is annotated as MAP, but it is repeated instead of optional or required",
            field_info.name,
        )
    }

    Ok(())
}

/// Converts a parquet group type to an arrow [`ArrowDataType::Struct`].
/// Returns [`None`] if all its fields are empty
fn to_struct(
    fields: &[ParquetType],
    options: &SchemaInferenceOptions,
) -> PolarsResult<Option<ArrowDataType>> {
    let fields = fields
        .iter()
        .map(|f| to_field(f, options))
        .collect::<PolarsResult<Vec<Option<Field>>>>()?;
    let fields = fields.into_iter().flatten().collect::<Vec<Field>>();

    if fields.is_empty() {
        Ok(None)
    } else {
        Ok(Some(ArrowDataType::Struct(fields)))
    }
}

/// Converts a parquet `MAP` / `MAP_KEY_VALUE` group to an arrow [`ArrowDataType::Map`].
///
/// We follow the spec (https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#maps)
/// to the letter, including the backwards-compatibility rules. On the happy path, this is
/// group (MAP) -> repeated group key_value -> {required key, optional value}.
fn to_map(
    map_info: &FieldInfo,
    fields: &[ParquetType],
    options: &SchemaInferenceOptions,
) -> PolarsResult<Option<ArrowDataType>> {
    macro_rules! invalid_map {
        ($($arg:tt)*) => {
            polars_bail!(
                ComputeError:
                "parquet group '{}' is annotated as MAP, but {}",
                map_info.name,
                format!($($arg)*),
            )
        };
    }

    let [entries] = fields else {
        invalid_map!(
            "it has {} children instead of a single `key_value`",
            fields.len()
        )
    };
    let ParquetType::GroupType {
        field_info,
        fields: kv_fields,
        ..
    } = entries
    else {
        invalid_map!("its `{}` child is not a group", entries.name())
    };
    if field_info.repetition != Repetition::Repeated {
        invalid_map!("its `{}` child is not repeated", field_info.name)
    }
    if kv_fields.is_empty() {
        invalid_map!("its `{}` child has no key field", field_info.name)
    }
    if kv_fields.len() > 2 {
        invalid_map!(
            "its `{}` child has {} fields instead of a key and an optional value",
            field_info.name,
            kv_fields.len(),
        )
    }

    // The key is identified by position, not by name.
    let key_info = kv_fields[0].get_field_info();
    if key_info.repetition != Repetition::Required {
        invalid_map!(
            "its map key `{}` is {:?} instead of required",
            key_info.name,
            key_info.repetition,
        )
    }

    // A `value` field is optional in parquet, but an arrow `Map` always has one. The spec allows
    // reading such a group as a set of keys, so fall through to the plain list conversion. This
    // enters the element rules directly: the entries group is a repeated `MAP_KEY_VALUE` group,
    // which `to_list` refuses.
    if kv_fields.len() == 1 {
        return list_elements(map_info, entries, options);
    }

    let Some(entries_dtype) = to_struct(kv_fields, options)? else {
        invalid_map!("its `{}` child has no columns", field_info.name)
    };
    // `to_struct` drops column-less children, so recheck that both survived.
    if !matches!(&entries_dtype, ArrowDataType::Struct(fields) if fields.len() == 2) {
        invalid_map!("its key or value field has no columns")
    }

    let entry = Field::new(field_info.name.clone(), entries_dtype, false);
    Ok(Some(ArrowDataType::Map(Box::new(entry), false)))
}

/// Entry point for converting parquet group type.
///
/// This function takes care of logical type and repetition.
fn to_group_type(
    field_info: &FieldInfo,
    logical_type: &Option<GroupLogicalType>,
    converted_type: &Option<GroupConvertedType>,
    fields: &[ParquetType],
    options: &SchemaInferenceOptions,
) -> PolarsResult<Option<ArrowDataType>> {
    debug_assert!(!fields.is_empty());
    ensure_not_repeated_map(field_info, logical_type, converted_type)?;

    if field_info.repetition == Repetition::Repeated {
        let Some(inner) = to_struct(fields, options)? else {
            return Ok(None);
        };
        Ok(Some(ArrowDataType::LargeList(Box::new(Field::new(
            field_info.name.clone(),
            inner,
            is_optional(field_info),
        )))))
    } else {
        non_repeated_group(field_info, logical_type, converted_type, fields, options)
    }
}

/// Checks whether this schema is nullable.
pub(crate) fn is_nullable(field_info: &FieldInfo) -> bool {
    match field_info.repetition {
        Repetition::Optional => true,
        Repetition::Repeated => true,
        Repetition::Required => false,
    }
}

/// Whether the value this field describes can be null.
///
/// Unlike [`is_nullable`], which the writer uses to decide whether a level is encoded, a repeated
/// field is not null here: it becomes a non-null list whose elements are non-null unless the
/// element itself is optional.
fn is_optional(field_info: &FieldInfo) -> bool {
    field_info.repetition == Repetition::Optional
}

/// Converts parquet schema to arrow field.
/// Returns `None` iff the parquet type has no associated primitive types,
/// i.e. if it is a column-less group type.
fn to_field(type_: &ParquetType, options: &SchemaInferenceOptions) -> PolarsResult<Option<Field>> {
    let field_info = type_.get_field_info();

    let options = options.enter_nested(field_info.name.as_str());

    let metadata: Option<Arc<Metadata>> = field_info.id.map(|x: i32| {
        Arc::new(
            [(
                PlSmallStr::from_static("PARQUET:field_id"),
                format_pl_smallstr!("{x}"),
            )]
            .into(),
        )
    });

    let Some(dtype) = to_dtype(type_, options)? else {
        return Ok(None);
    };

    let mut arrow_field = Field::new(field_info.name.clone(), dtype, is_optional(field_info));

    arrow_field.metadata = metadata;

    Ok(Some(arrow_field))
}

/// Converts a parquet `LIST` group to an arrow [`ArrowDataType::LargeList`].
///
/// We follow the spec (https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#lists)
/// to the letter, including the backwards-compatibility rules. On the happy path, this is
/// group (LIST) -> repeated group list -> element.
fn to_list(
    list_info: &FieldInfo,
    fields: &[ParquetType],
    options: &SchemaInferenceOptions,
) -> PolarsResult<Option<ArrowDataType>> {
    macro_rules! invalid_list {
        ($($arg:tt)*) => {
            polars_bail!(
                ComputeError:
                "parquet group '{}' is annotated as LIST, but {}",
                list_info.name,
                format!($($arg)*),
            )
        };
    }

    let [repeated] = fields else {
        invalid_list!(
            "it has {} children instead of a single repeated field",
            fields.len()
        )
    };
    let repetition = repeated.get_field_info().repetition;
    if repetition != Repetition::Repeated {
        invalid_list!(
            "its `{}` child is {:?} instead of repeated",
            repeated.name(),
            repetition,
        )
    }
    // The repeated level is converted here instead of being routed through `to_dtype`, so it never
    // reaches the check in `to_group_type`.
    if let ParquetType::GroupType {
        field_info,
        logical_type,
        converted_type,
        ..
    } = repeated
    {
        ensure_not_repeated_map(field_info, logical_type, converted_type)?;
    }

    list_elements(list_info, repeated, options)
}

/// Determines the element of a `LIST` group from its repeated level, by the spec's
/// backwards-compatibility rules. The rules are ordered: the first one that matches wins.
fn list_elements(
    list_info: &FieldInfo,
    repeated: &ParquetType,
    options: &SchemaInferenceOptions,
) -> PolarsResult<Option<ArrowDataType>> {
    let (name, is_nullable, dtype) = match repeated {
        // 1. "If the repeated field is not a group, then its type is the element type and
        //    elements are required."
        ParquetType::PrimitiveType(primitive) => (
            primitive.field_info.name.clone(),
            false,
            Some(to_primitive_type_inner(primitive, options)),
        ),
        ParquetType::GroupType {
            field_info,
            logical_type,
            converted_type,
            fields: inner,
        } => {
            // A group without children has no element.
            let Some(first) = inner.first() else {
                return Ok(None);
            };
            let single_repeated_field =
                inner.len() == 1 && first.get_field_info().repetition == Repetition::Repeated;

            // 2. "If the repeated field is a group with multiple fields, then its type is the
            //    element type and elements are required."
            // 3. "If the repeated field is a group with one field with `repeated` repetition, then
            //    its type is the element type and elements are required."
            // 4. "If the repeated field is a group with one field and is named either `array` or
            //    uses the `LIST`-annotated group's name with `_tuple` appended then the repeated
            //    type is the element type and elements are required."
            if single_repeated_field && is_list_annotated(logical_type, converted_type) {
                // The element is the repeated group's own type, so it must not be wrapped in
                // another list.
                (
                    field_info.name.clone(),
                    false,
                    to_list(field_info, inner, options)?,
                )
            } else if inner.len() > 1
                || single_repeated_field
                || field_info.name == "array"
                || field_info.name.strip_suffix("_tuple") == Some(list_info.name.as_str())
            {
                (field_info.name.clone(), false, to_struct(inner, options)?)
            } else {
                // 5. "Otherwise, the repeated field's type is the element type with the repeated
                //    field's repetition." The name of the repeated group is not enforced: it is
                //    `list` in the spec, but `bag` and `array` are common in the wild.
                let element_info = first.get_field_info();
                (
                    element_info.name.clone(),
                    is_optional(element_info),
                    to_dtype(first, options)?,
                )
            }
        },
    };

    let Some(dtype) = dtype else {
        return Ok(None);
    };
    Ok(Some(ArrowDataType::LargeList(Box::new(Field::new(
        name,
        dtype,
        is_nullable,
    )))))
}

/// Converts parquet schema to arrow data type.
///
/// This function discards schema name.
///
/// If this schema is a primitive type and not included in the leaves, the result is
/// Ok(None).
///
/// If this schema is a group type and none of its children is reserved in the
/// conversion, the result is Ok(None).
pub(crate) fn to_dtype(
    type_: &ParquetType,
    options: &SchemaInferenceOptions,
) -> PolarsResult<Option<ArrowDataType>> {
    match type_ {
        ParquetType::PrimitiveType(primitive) => Ok(Some(to_primitive_type(primitive, options))),
        ParquetType::GroupType {
            field_info,
            logical_type,
            converted_type,
            fields,
        } => {
            if fields.is_empty() {
                Ok(None)
            } else {
                to_group_type(field_info, logical_type, converted_type, fields, options)
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::metadata::SchemaDescriptor;

    #[test]
    fn test_flat_primitives() -> PolarsResult<()> {
        let message = "
        message test_schema {
            REQUIRED BOOLEAN boolean;
            REQUIRED INT32   int8  (INT_8);
            REQUIRED INT32   int16 (INT_16);
            REQUIRED INT32   uint8 (INTEGER(8,false));
            REQUIRED INT32   uint16 (INTEGER(16,false));
            REQUIRED INT32   int32;
            REQUIRED INT64   int64 ;
            OPTIONAL DOUBLE  double;
            OPTIONAL FLOAT   float;
            OPTIONAL BINARY  string (UTF8);
            OPTIONAL BINARY  string_2 (STRING);
        }
        ";
        let expected = &[
            Field::new("boolean".into(), ArrowDataType::Boolean, false),
            Field::new("int8".into(), ArrowDataType::Int8, false),
            Field::new("int16".into(), ArrowDataType::Int16, false),
            Field::new("uint8".into(), ArrowDataType::UInt8, false),
            Field::new("uint16".into(), ArrowDataType::UInt16, false),
            Field::new("int32".into(), ArrowDataType::Int32, false),
            Field::new("int64".into(), ArrowDataType::Int64, false),
            Field::new("double".into(), ArrowDataType::Float64, true),
            Field::new("float".into(), ArrowDataType::Float32, true),
            Field::new("string".into(), ArrowDataType::Utf8View, true),
            Field::new("string_2".into(), ArrowDataType::Utf8View, true),
        ];

        let parquet_schema = SchemaDescriptor::try_from_message(message)?;
        let fields = parquet_to_arrow_schema(parquet_schema.fields())?;
        let fields = fields.iter_values().cloned().collect::<Vec<_>>();

        assert_eq!(fields, expected);
        Ok(())
    }

    #[test]
    fn test_byte_array_fields() -> PolarsResult<()> {
        let message = "
        message test_schema {
            REQUIRED BYTE_ARRAY binary;
            REQUIRED FIXED_LEN_BYTE_ARRAY (20) fixed_binary;
        }
        ";
        let expected = vec![
            Field::new("binary".into(), ArrowDataType::BinaryView, false),
            Field::new(
                "fixed_binary".into(),
                ArrowDataType::FixedSizeBinary(20),
                false,
            ),
        ];

        let parquet_schema = SchemaDescriptor::try_from_message(message)?;
        let fields = parquet_to_arrow_schema(parquet_schema.fields())?;
        let fields = fields.iter_values().cloned().collect::<Vec<_>>();

        assert_eq!(fields, expected);
        Ok(())
    }

    #[test]
    fn test_duplicate_fields() -> PolarsResult<()> {
        let message = "
        message test_schema {
            REQUIRED BOOLEAN boolean;
            REQUIRED INT32 int8 (INT_8);
        }
        ";
        let expected = &[
            Field::new("boolean".into(), ArrowDataType::Boolean, false),
            Field::new("int8".into(), ArrowDataType::Int8, false),
        ];

        let parquet_schema = SchemaDescriptor::try_from_message(message)?;
        let fields = parquet_to_arrow_schema(parquet_schema.fields())?;
        let fields = fields.iter_values().cloned().collect::<Vec<_>>();

        assert_eq!(fields, expected);
        Ok(())
    }

    #[test]
    fn test_parquet_list_with_struct() -> PolarsResult<()> {
        let mut arrow_fields = Vec::new();

        let message_type = "
            message eventlog {
              REQUIRED group events (LIST) {
                REPEATED group array {
                  REQUIRED BYTE_ARRAY event_name (STRING);
                  REQUIRED INT64 event_time (TIMESTAMP(MILLIS,true));
                }
              }
            }
        ";

        {
            let struct_fields = vec![
                Field::new("event_name".into(), ArrowDataType::Utf8View, false),
                Field::new(
                    "event_time".into(),
                    ArrowDataType::Timestamp(TimeUnit::Millisecond, Some("+00:00".into())),
                    false,
                ),
            ];
            arrow_fields.push(Field::new(
                "events".into(),
                ArrowDataType::LargeList(Box::new(Field::new(
                    "array".into(),
                    ArrowDataType::Struct(struct_fields),
                    false,
                ))),
                false,
            ));
        }

        let parquet_schema = SchemaDescriptor::try_from_message(message_type)?;
        let fields = parquet_to_arrow_schema(parquet_schema.fields())?;
        let fields = fields.iter_values().cloned().collect::<Vec<_>>();

        assert_eq!(arrow_fields, fields);
        Ok(())
    }

    #[test]
    fn test_parquet_list_nullable() -> PolarsResult<()> {
        let mut arrow_fields = Vec::new();

        let message_type = "
        message test_schema {
          REQUIRED GROUP my_list1 (LIST) {
            REPEATED GROUP list {
              OPTIONAL BINARY element (UTF8);
            }
          }
          OPTIONAL GROUP my_list2 (LIST) {
            REPEATED GROUP list {
              REQUIRED BINARY element (UTF8);
            }
          }
          REQUIRED GROUP my_list3 (LIST) {
            REPEATED GROUP list {
              REQUIRED BINARY element (UTF8);
            }
          }
        }
        ";

        // // List<String> (list non-null, elements nullable)
        // required group my_list1 (LIST) {
        //   repeated group list {
        //     optional binary element (UTF8);
        //   }
        // }
        {
            arrow_fields.push(Field::new(
                "my_list1".into(),
                ArrowDataType::LargeList(Box::new(Field::new(
                    "element".into(),
                    ArrowDataType::Utf8View,
                    true,
                ))),
                false,
            ));
        }

        // // List<String> (list nullable, elements non-null)
        // optional group my_list2 (LIST) {
        //   repeated group list {
        //     required binary element (UTF8);
        //   }
        // }
        {
            arrow_fields.push(Field::new(
                "my_list2".into(),
                ArrowDataType::LargeList(Box::new(Field::new(
                    "element".into(),
                    ArrowDataType::Utf8View,
                    false,
                ))),
                true,
            ));
        }

        // // List<String> (list non-null, elements non-null)
        // repeated group my_list3 (LIST) {
        //   repeated group list {
        //     required binary element (UTF8);
        //   }
        // }
        {
            arrow_fields.push(Field::new(
                "my_list3".into(),
                ArrowDataType::LargeList(Box::new(Field::new(
                    "element".into(),
                    ArrowDataType::Utf8View,
                    false,
                ))),
                false,
            ));
        }

        let parquet_schema = SchemaDescriptor::try_from_message(message_type)?;
        let fields = parquet_to_arrow_schema(parquet_schema.fields())?;
        let fields = fields.iter_values().cloned().collect::<Vec<_>>();

        assert_eq!(arrow_fields, fields);
        Ok(())
    }

    #[test]
    fn test_nested_schema() -> PolarsResult<()> {
        let mut arrow_fields = Vec::new();
        {
            let group1_fields = vec![
                Field::new("leaf1".into(), ArrowDataType::Boolean, false),
                Field::new("leaf2".into(), ArrowDataType::Int32, false),
            ];
            let group1_struct =
                Field::new("group1".into(), ArrowDataType::Struct(group1_fields), false);
            arrow_fields.push(group1_struct);

            let leaf3_field = Field::new("leaf3".into(), ArrowDataType::Int64, false);
            arrow_fields.push(leaf3_field);
        }

        let message_type = "
        message test_schema {
          REQUIRED GROUP group1 {
            REQUIRED BOOLEAN leaf1;
            REQUIRED INT32 leaf2;
          }
          REQUIRED INT64 leaf3;
        }
        ";

        let parquet_schema = SchemaDescriptor::try_from_message(message_type)?;
        let fields = parquet_to_arrow_schema(parquet_schema.fields())?;
        let fields = fields.iter_values().cloned().collect::<Vec<_>>();

        assert_eq!(arrow_fields, fields);
        Ok(())
    }

    #[ignore]
    #[test]
    fn test_repeated_nested_schema() -> PolarsResult<()> {
        let mut arrow_fields = Vec::new();
        {
            arrow_fields.push(Field::new("leaf1".into(), ArrowDataType::Int32, true));

            let inner_group_list = Field::new(
                "innerGroup".into(),
                ArrowDataType::LargeList(Box::new(Field::new(
                    "innerGroup".into(),
                    ArrowDataType::Struct(vec![Field::new(
                        "leaf3".into(),
                        ArrowDataType::Int32,
                        true,
                    )]),
                    false,
                ))),
                false,
            );

            let outer_group_list = Field::new(
                "outerGroup".into(),
                ArrowDataType::LargeList(Box::new(Field::new(
                    "outerGroup".into(),
                    ArrowDataType::Struct(vec![
                        Field::new("leaf2".into(), ArrowDataType::Int32, true),
                        inner_group_list,
                    ]),
                    false,
                ))),
                false,
            );
            arrow_fields.push(outer_group_list);
        }

        let message_type = "
        message test_schema {
          OPTIONAL INT32 leaf1;
          REPEATED GROUP outerGroup {
            OPTIONAL INT32 leaf2;
            REPEATED GROUP innerGroup {
              OPTIONAL INT32 leaf3;
            }
          }
        }
        ";

        let parquet_schema = SchemaDescriptor::try_from_message(message_type)?;
        let fields = parquet_to_arrow_schema(parquet_schema.fields())?;
        let fields = fields.iter_values().cloned().collect::<Vec<_>>();

        assert_eq!(arrow_fields, fields);
        Ok(())
    }

    #[ignore]
    #[test]
    fn test_column_desc_to_field() -> PolarsResult<()> {
        let message_type = "
        message test_schema {
            REQUIRED BOOLEAN boolean;
            REQUIRED INT32   int8  (INT_8);
            REQUIRED INT32   uint8 (INTEGER(8,false));
            REQUIRED INT32   int16 (INT_16);
            REQUIRED INT32   uint16 (INTEGER(16,false));
            REQUIRED INT32   int32;
            REQUIRED INT64   int64;
            OPTIONAL DOUBLE  double;
            OPTIONAL FLOAT   float;
            OPTIONAL BINARY  string (UTF8);
            REPEATED BOOLEAN bools;
            OPTIONAL INT32   date       (DATE);
            OPTIONAL INT32   time_milli (TIME_MILLIS);
            OPTIONAL INT64   time_micro (TIME_MICROS);
            OPTIONAL INT64   time_nano (TIME(NANOS,false));
            OPTIONAL INT64   ts_milli (TIMESTAMP_MILLIS);
            REQUIRED INT64   ts_micro (TIMESTAMP_MICROS);
            REQUIRED INT64   ts_nano (TIMESTAMP(NANOS,true));
        }
        ";
        let arrow_fields = vec![
            Field::new("boolean".into(), ArrowDataType::Boolean, false),
            Field::new("int8".into(), ArrowDataType::Int8, false),
            Field::new("uint8".into(), ArrowDataType::UInt8, false),
            Field::new("int16".into(), ArrowDataType::Int16, false),
            Field::new("uint16".into(), ArrowDataType::UInt16, false),
            Field::new("int32".into(), ArrowDataType::Int32, false),
            Field::new("int64".into(), ArrowDataType::Int64, false),
            Field::new("double".into(), ArrowDataType::Float64, true),
            Field::new("float".into(), ArrowDataType::Float32, true),
            Field::new("string".into(), ArrowDataType::Utf8, true),
            Field::new(
                "bools".into(),
                ArrowDataType::LargeList(Box::new(Field::new(
                    "bools".into(),
                    ArrowDataType::Boolean,
                    false,
                ))),
                false,
            ),
            Field::new("date".into(), ArrowDataType::Date32, true),
            Field::new(
                "time_milli".into(),
                ArrowDataType::Time32(TimeUnit::Millisecond),
                true,
            ),
            Field::new(
                "time_micro".into(),
                ArrowDataType::Time64(TimeUnit::Microsecond),
                true,
            ),
            Field::new(
                "time_nano".into(),
                ArrowDataType::Time64(TimeUnit::Nanosecond),
                true,
            ),
            Field::new(
                "ts_milli".into(),
                ArrowDataType::Timestamp(TimeUnit::Millisecond, None),
                true,
            ),
            Field::new(
                "ts_micro".into(),
                ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new(
                "ts_nano".into(),
                ArrowDataType::Timestamp(TimeUnit::Nanosecond, Some("+00:00".into())),
                false,
            ),
        ];

        let parquet_schema = SchemaDescriptor::try_from_message(message_type)?;
        let fields = parquet_to_arrow_schema(parquet_schema.fields())?;
        let fields = fields.iter_values().cloned().collect::<Vec<_>>();

        assert_eq!(arrow_fields, fields);
        Ok(())
    }

    #[test]
    fn test_field_to_column_desc() -> PolarsResult<()> {
        let message_type = "
        message arrow_schema {
            REQUIRED BOOLEAN boolean;
            REQUIRED INT32   int8  (INT_8);
            REQUIRED INT32   int16 (INTEGER(16,true));
            REQUIRED INT32   int32;
            REQUIRED INT64   int64;
            OPTIONAL DOUBLE  double;
            OPTIONAL FLOAT   float;
            OPTIONAL BINARY  string (STRING);
            OPTIONAL GROUP   bools (LIST) {
                REPEATED GROUP list {
                    OPTIONAL BOOLEAN element;
                }
            }
            REQUIRED GROUP   bools_non_null (LIST) {
                REPEATED GROUP list {
                    REQUIRED BOOLEAN element;
                }
            }
            OPTIONAL INT32   date       (DATE);
            OPTIONAL INT32   time_milli (TIME(MILLIS,false));
            OPTIONAL INT64   time_micro (TIME_MICROS);
            OPTIONAL INT64   ts_milli (TIMESTAMP_MILLIS);
            REQUIRED INT64   ts_micro (TIMESTAMP(MICROS,false));
            REQUIRED GROUP struct {
                REQUIRED BOOLEAN bools;
                REQUIRED INT32 uint32 (INTEGER(32,false));
                REQUIRED GROUP   int32 (LIST) {
                    REPEATED GROUP list {
                        OPTIONAL INT32 element;
                    }
                }
            }
            REQUIRED BINARY  dictionary_strings (STRING);
        }
        ";

        let arrow_fields = vec![
            Field::new("boolean".into(), ArrowDataType::Boolean, false),
            Field::new("int8".into(), ArrowDataType::Int8, false),
            Field::new("int16".into(), ArrowDataType::Int16, false),
            Field::new("int32".into(), ArrowDataType::Int32, false),
            Field::new("int64".into(), ArrowDataType::Int64, false),
            Field::new("double".into(), ArrowDataType::Float64, true),
            Field::new("float".into(), ArrowDataType::Float32, true),
            Field::new("string".into(), ArrowDataType::Utf8View, true),
            Field::new(
                "bools".into(),
                ArrowDataType::LargeList(Box::new(Field::new(
                    "element".into(),
                    ArrowDataType::Boolean,
                    true,
                ))),
                true,
            ),
            Field::new(
                "bools_non_null".into(),
                ArrowDataType::LargeList(Box::new(Field::new(
                    "element".into(),
                    ArrowDataType::Boolean,
                    false,
                ))),
                false,
            ),
            Field::new("date".into(), ArrowDataType::Date32, true),
            Field::new(
                "time_milli".into(),
                ArrowDataType::Time32(TimeUnit::Millisecond),
                true,
            ),
            Field::new(
                "time_micro".into(),
                ArrowDataType::Time64(TimeUnit::Microsecond),
                true,
            ),
            Field::new(
                "ts_milli".into(),
                ArrowDataType::Timestamp(TimeUnit::Millisecond, None),
                true,
            ),
            Field::new(
                "ts_micro".into(),
                ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new(
                "struct".into(),
                ArrowDataType::Struct(vec![
                    Field::new("bools".into(), ArrowDataType::Boolean, false),
                    Field::new("uint32".into(), ArrowDataType::UInt32, false),
                    Field::new(
                        "int32".into(),
                        ArrowDataType::LargeList(Box::new(Field::new(
                            "element".into(),
                            ArrowDataType::Int32,
                            true,
                        ))),
                        false,
                    ),
                ]),
                false,
            ),
            Field::new("dictionary_strings".into(), ArrowDataType::Utf8View, false),
        ];

        let parquet_schema = SchemaDescriptor::try_from_message(message_type)?;
        let fields = parquet_to_arrow_schema(parquet_schema.fields())?;
        let fields = fields.iter_values().cloned().collect::<Vec<_>>();

        assert_eq!(arrow_fields, fields);
        Ok(())
    }

    /// The arrow dtype Polars infers for a parquet `MAP` group whose repeated child is named
    /// `entries_name` and holds `key` and `value`.
    fn map_of(entries_name: &str, key: Field, value: Field) -> ArrowDataType {
        ArrowDataType::Map(
            Box::new(Field::new(
                entries_name.into(),
                ArrowDataType::Struct(vec![key, value]),
                false,
            )),
            false,
        )
    }

    fn infer_one(message_type: &str) -> PolarsResult<Field> {
        Ok(infer_named(message_type, "my_map")?.unwrap())
    }

    fn infer_named(message_type: &str, name: &str) -> PolarsResult<Option<Field>> {
        let parquet_schema = SchemaDescriptor::try_from_message(message_type)?;
        let fields = parquet_to_arrow_schema(parquet_schema.fields())?;
        Ok(fields.get(name).cloned())
    }

    fn utf8_field(name: &str, is_nullable: bool) -> Field {
        Field::new(name.into(), ArrowDataType::Utf8View, is_nullable)
    }

    fn i32_field(name: &str, is_nullable: bool) -> Field {
        Field::new(name.into(), ArrowDataType::Int32, is_nullable)
    }

    fn list_of(name: &str, dtype: ArrowDataType, is_nullable: bool) -> ArrowDataType {
        ArrowDataType::LargeList(Box::new(Field::new(name.into(), dtype, is_nullable)))
    }

    /// Asserts that each `(expected error, message type)` fails the schema conversion.
    fn assert_rejects(cases: &[(&str, &str)]) {
        for (expected, message_type) in cases {
            let parquet_schema = SchemaDescriptor::try_from_message(message_type).unwrap();
            let err = parquet_to_arrow_schema(parquet_schema.fields())
                .unwrap_err()
                .to_string();
            assert!(
                err.contains(expected),
                "expected error to contain {expected:?}, got {err:?}",
            );
        }
    }

    /// The `LIST` shapes that [the spec] requires us to accept, including its
    /// backward-compatibility rules.
    ///
    /// [the spec]: https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#lists
    #[test]
    fn test_parquet_lists_backward_compat() -> PolarsResult<()> {
        let cases = [
            // The canonical form: `List<String>` with a nullable list and nullable elements.
            (
                "
                optional group f (LIST) {
                  repeated group list {
                    optional binary element (STRING);
                  }
                }",
                Some((list_of("element", ArrowDataType::Utf8View, true), true)),
            ),
            // 1. "If the repeated field is not a group, then its type is the element type and
            //    elements are required."
            (
                "
                optional group f (LIST) {
                  repeated int32 element;
                }",
                Some((list_of("element", ArrowDataType::Int32, false), true)),
            ),
            // 2. "If the repeated field is a group with multiple fields, then its type is the
            //    element type and elements are required."
            (
                "
                optional group f (LIST) {
                  repeated group element {
                    required binary str (STRING);
                    required int32 num;
                  }
                }",
                Some((
                    list_of(
                        "element",
                        ArrowDataType::Struct(vec![
                            utf8_field("str", false),
                            i32_field("num", false),
                        ]),
                        false,
                    ),
                    true,
                )),
            ),
            // 3. A repeated group holding a single repeated field is the element itself. This is
            //    the avro/thrift shape: `[[1, 2], [3, 4]]` is a list of lists, not of structs.
            (
                "
                required group f (LIST) {
                  repeated group array (LIST) {
                    repeated int32 array;
                  }
                }",
                Some((
                    list_of(
                        "array",
                        list_of("array", ArrowDataType::Int32, false),
                        false,
                    ),
                    false,
                )),
            ),
            // 3. The same, without the inner `LIST` annotation: the element is the group, whose
            //    own repeated field becomes a list.
            (
                "
                optional group f (LIST) {
                  repeated group inner {
                    repeated int32 num;
                  }
                }",
                Some((
                    list_of(
                        "inner",
                        ArrowDataType::Struct(vec![Field::new(
                            "num".into(),
                            list_of("num", ArrowDataType::Int32, false),
                            false,
                        )]),
                        false,
                    ),
                    true,
                )),
            ),
            // 4. "If the repeated field is a group with one field and is named either `array` or
            //    uses the `LIST`-annotated group's name with `_tuple` appended then the repeated
            //    type is the element type and elements are required."
            (
                "
                optional group f (LIST) {
                  repeated group array {
                    required binary str (STRING);
                  }
                }",
                Some((
                    list_of(
                        "array",
                        ArrowDataType::Struct(vec![utf8_field("str", false)]),
                        false,
                    ),
                    true,
                )),
            ),
            (
                "
                optional group f (LIST) {
                  repeated group f_tuple {
                    required binary str (STRING);
                  }
                }",
                Some((
                    list_of(
                        "f_tuple",
                        ArrowDataType::Struct(vec![utf8_field("str", false)]),
                        false,
                    ),
                    true,
                )),
            ),
            // 5. "Otherwise, the repeated field's type is the element type with the repeated
            //    field's repetition." The name of the repeated group is not part of the rule: hive
            //    and avro write `bag`, and the element name is taken from the file either way.
            (
                "
                optional group f (LIST) {
                  repeated group bag {
                    optional binary array_element (STRING);
                  }
                }",
                Some((
                    list_of("array_element", ArrowDataType::Utf8View, true),
                    true,
                )),
            ),
            // The `_tuple` suffix is compared as a string, not as bytes: `é_tupl` has the same
            // byte length as `f` + `_tuple`, but is not a match.
            (
                "
                optional group f (LIST) {
                  repeated group é_tupl {
                    required int32 x;
                  }
                }",
                Some((list_of("x", ArrowDataType::Int32, false), true)),
            ),
            // A repeated group without children holds no column, so the list is dropped like any
            // other column-less group.
            (
                "
                optional group f (LIST) {
                  repeated group list {
                  }
                }",
                None,
            ),
            // A repeated field that is not annotated as a `LIST` is a non-null list of non-null
            // elements: the file has no definition level for either.
            (
                "
                repeated int32 f;",
                Some((list_of("f", ArrowDataType::Int32, false), false)),
            ),
            (
                "
                optional group f {
                  repeated group phone {
                    required int64 number;
                    optional binary kind (STRING);
                  }
                }",
                Some((
                    ArrowDataType::Struct(vec![Field::new(
                        "phone".into(),
                        list_of(
                            "phone",
                            ArrowDataType::Struct(vec![
                                Field::new("number".into(), ArrowDataType::Int64, false),
                                utf8_field("kind", true),
                            ]),
                            false,
                        ),
                        false,
                    )]),
                    true,
                )),
            ),
        ];

        for (field, expected) in cases {
            let message_type = format!("message test_schema {{ {field} }}");
            let expected =
                expected.map(|(dtype, is_nullable)| Field::new("f".into(), dtype, is_nullable));
            assert_eq!(infer_named(&message_type, "f")?, expected, "{field}");
        }
        Ok(())
    }

    /// A group annotated as `LIST` that does not satisfy the spec is refused, rather than being
    /// silently decoded with levels the file does not have.
    #[test]
    fn test_parquet_lists_reject_nonconforming() {
        assert_rejects(&[
            // "The outer-most level must be a group annotated with `LIST` that contains a single
            // field named `list`."
            (
                "it has 2 children instead of a single repeated field",
                "
                message test_schema {
                  optional group my_list (LIST) {
                    repeated group list {
                      optional int32 element;
                    }
                    optional int32 stray;
                  }
                }",
            ),
            // "The middle level [...] must be a repeated group with a single field named
            // `element`."
            (
                "its `list` child is Optional instead of repeated",
                "
                message test_schema {
                  optional group my_list (LIST) {
                    optional group list {
                      optional int32 element;
                    }
                  }
                }",
            ),
            (
                "its `list` child is Required instead of repeated",
                "
                message test_schema {
                  optional group my_list (LIST) {
                    required group list {
                      optional int32 element;
                    }
                  }
                }",
            ),
            // A repeated `MAP` group is rejected wherever a list conversion reaches it, including
            // through the recursion of rule 3.
            (
                "parquet group 'm' is annotated as MAP, but it is repeated",
                "
                message test_schema {
                  optional group a (LIST) {
                    repeated group b (LIST) {
                      repeated group m (MAP) {
                        repeated group key_value {
                          required binary key (STRING);
                          optional int32 value;
                        }
                      }
                    }
                  }
                }",
            ),
        ]);
    }

    /// The `MAP` shapes that [the spec] requires us to accept, including its
    /// backward-compatibility rules.
    ///
    /// [the spec]: https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#maps
    #[test]
    fn test_parquet_maps() -> PolarsResult<()> {
        // The canonical form: `Map<String, Integer>` with a non-null map and nullable values.
        assert_eq!(
            infer_one(
                "
                message test_schema {
                  required group my_map (MAP) {
                    repeated group key_value {
                      required binary key (STRING);
                      optional int32 value;
                    }
                  }
                }"
            )?,
            Field::new(
                "my_map".into(),
                map_of(
                    "key_value",
                    utf8_field("key", false),
                    i32_field("value", true)
                ),
                false,
            ),
        );

        // Backward-compatibility: the `key_value`/`key`/`value` names "may not be used in
        // existing data and should not be enforced as errors when reading", so the fields are
        // identified by position.
        assert_eq!(
            infer_one(
                "
                message test_schema {
                  optional group my_map (MAP) {
                    repeated group map {
                      required binary str (STRING);
                      required int32 num;
                    }
                  }
                }"
            )?,
            Field::new(
                "my_map".into(),
                map_of("map", utf8_field("str", false), i32_field("num", false)),
                true,
            ),
        );

        // Backward-compatibility: a `MAP_KEY_VALUE` group that is not contained by a `MAP` group
        // "should be handled as a `MAP`-annotated group".
        assert_eq!(
            infer_one(
                "
                message test_schema {
                  optional group my_map (MAP_KEY_VALUE) {
                    repeated group map {
                      required binary key (STRING);
                      optional int32 value;
                    }
                  }
                }"
            )?,
            Field::new(
                "my_map".into(),
                map_of("map", utf8_field("key", false), i32_field("value", true)),
                true,
            ),
        );

        // A `MAP_KEY_VALUE` group that *is* contained by a `MAP` group is just the entries group.
        assert_eq!(
            infer_one(
                "
                message test_schema {
                  optional group my_map (MAP) {
                    repeated group key_value (MAP_KEY_VALUE) {
                      required binary key (STRING);
                      optional int32 value;
                    }
                  }
                }"
            )?,
            Field::new(
                "my_map".into(),
                map_of(
                    "key_value",
                    utf8_field("key", false),
                    i32_field("value", true)
                ),
                true,
            ),
        );

        // An array of maps annotates the LIST, and puts the MAP group inside its repeated level.
        assert_eq!(
            infer_one(
                "
                message test_schema {
                  optional group my_map (LIST) {
                    repeated group list {
                      optional group element (MAP) {
                        repeated group key_value {
                          required binary key (STRING);
                          optional int32 value;
                        }
                      }
                    }
                  }
                }"
            )?,
            Field::new(
                "my_map".into(),
                ArrowDataType::LargeList(Box::new(Field::new(
                    "element".into(),
                    map_of(
                        "key_value",
                        utf8_field("key", false),
                        i32_field("value", true)
                    ),
                    true,
                ))),
                true,
            ),
        );

        // The `value` field may be omitted. An arrow `Map` always has one, so we take the spec up
        // on its alternative and read the group "as a set of keys".
        assert_eq!(
            infer_one(
                "
                message test_schema {
                  optional group my_map (MAP) {
                    repeated group key_value {
                      required binary key (STRING);
                    }
                  }
                }"
            )?,
            Field::new(
                "my_map".into(),
                ArrowDataType::LargeList(Box::new(utf8_field("key", false))),
                true,
            ),
        );

        // A value-less map whose entries group also carries the legacy annotation. This reaches
        // `to_list` from `to_map`, where the repeated entries group is expected.
        assert_eq!(
            infer_one(
                "
                message test_schema {
                  optional group my_map (MAP) {
                    repeated group key_value (MAP_KEY_VALUE) {
                      required binary key (STRING);
                    }
                  }
                }"
            )?,
            Field::new(
                "my_map".into(),
                ArrowDataType::LargeList(Box::new(utf8_field("key", false))),
                true,
            ),
        );

        Ok(())
    }

    /// A group annotated as `MAP` that does not satisfy the spec is refused, rather than being
    /// silently reinterpreted as a list of its entries.
    #[test]
    fn test_parquet_maps_reject_nonconforming() {
        // "The `key` field [...] must have repetition `required`".
        let cases = [
            (
                "its map key `key` is Optional instead of required",
                "
                message test_schema {
                  optional group my_map (MAP) {
                    repeated group key_value {
                      optional binary key (STRING);
                      optional int32 value;
                    }
                  }
                }",
            ),
            // "It must not contain any other values."
            (
                "has 3 fields instead of a key and an optional value",
                "
                message test_schema {
                  optional group my_map (MAP) {
                    repeated group key_value {
                      required binary key (STRING);
                      optional int32 value;
                      optional int32 extra;
                    }
                  }
                }",
            ),
            // "The middle level [...] must be a repeated group".
            (
                "its `key_value` child is not repeated",
                "
                message test_schema {
                  optional group my_map (MAP) {
                    optional group key_value {
                      required binary key (STRING);
                      optional int32 value;
                    }
                  }
                }",
            ),
            // "[the outer-most level] contains a single field named `key_value`".
            (
                "it has 2 children instead of a single `key_value`",
                "
                message test_schema {
                  optional group my_map (MAP) {
                    repeated group key_value {
                      required binary key (STRING);
                      optional int32 value;
                    }
                    optional int32 stray;
                  }
                }",
            ),
            (
                "its `key_value` child is not a group",
                "
                message test_schema {
                  optional group my_map (MAP) {
                    repeated binary key_value (STRING);
                  }
                }",
            ),
            // "The repetition of this level must be either `optional` or `required`".
            (
                "it is repeated instead of optional or required",
                "
                message test_schema {
                  repeated group my_map (MAP) {
                    repeated group key_value {
                      required binary key (STRING);
                      optional int32 value;
                    }
                  }
                }",
            ),
            // The same rule, reached through a LIST whose repeated level is annotated as the map
            // instead of holding one.
            (
                "parquet group 'element' is annotated as MAP, but it is repeated",
                "
                message test_schema {
                  optional group my_map (LIST) {
                    repeated group element (MAP) {
                      repeated group key_value {
                        required binary key (STRING);
                        optional int32 value;
                      }
                    }
                  }
                }",
            ),
        ];

        assert_rejects(&cases);
    }

    #[test]
    fn test_int96_options() -> PolarsResult<()> {
        for tu in [
            TimeUnit::Second,
            TimeUnit::Microsecond,
            TimeUnit::Millisecond,
            TimeUnit::Nanosecond,
        ] {
            let message_type = "
            message arrow_schema {
                REQUIRED INT96   int96_field;
                OPTIONAL GROUP   int96_list (LIST) {
                    REPEATED GROUP list {
                        OPTIONAL INT96 element;
                    }
                }
                REQUIRED GROUP int96_struct {
                    REQUIRED INT96 int96_field;
                }
            }
            ";
            let coerced_to = ArrowDataType::Timestamp(tu, None);
            let arrow_fields = vec![
                Field::new("int96_field".into(), coerced_to.clone(), false),
                Field::new(
                    "int96_list".into(),
                    ArrowDataType::LargeList(Box::new(Field::new(
                        "element".into(),
                        coerced_to.clone(),
                        true,
                    ))),
                    true,
                ),
                Field::new(
                    "int96_struct".into(),
                    ArrowDataType::Struct(vec![Field::new(
                        "int96_field".into(),
                        coerced_to.clone(),
                        false,
                    )]),
                    false,
                ),
            ];

            let parquet_schema = SchemaDescriptor::try_from_message(message_type)?;
            let fields = parquet_to_arrow_schema_with_options(
                parquet_schema.fields(),
                &SchemaInferenceOptions {
                    int96_coerce_to_timeunit: tu,
                    ..Default::default()
                },
            )?;
            let fields = fields.iter_values().cloned().collect::<Vec<_>>();
            assert_eq!(arrow_fields, fields);
        }
        Ok(())
    }
}
