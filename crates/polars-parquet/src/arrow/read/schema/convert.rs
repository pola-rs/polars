//! This module has entry points, [`parquet_to_arrow_schema`] and the more configurable [`parquet_to_arrow_schema_with_options`].
use arrow::datatypes::{ArrowDataType, Field, IntervalUnit, TimeUnit};

use crate::arrow::read::schema::SchemaInferenceOptions;
use crate::parquet::schema::types::{
    FieldInfo, GroupConvertedType, GroupLogicalType, IntegerType, ParquetType, PhysicalType,
    PrimitiveConvertedType, PrimitiveLogicalType, PrimitiveType, TimeUnit as ParquetTimeUnit,
};
use crate::parquet::schema::Repetition;

/// Converts [`ParquetType`]s to a [`Field`], ignoring parquet fields that do not contain
/// any physical column.
pub fn parquet_to_arrow_schema(fields: &[ParquetType]) -> Vec<Field> {
    parquet_to_arrow_schema_with_options(fields, &None)
}

/// Like [`parquet_to_arrow_schema`] but with configurable options which affect the behavior of schema inference
pub fn parquet_to_arrow_schema_with_options(
    fields: &[ParquetType],
    options: &Option<SchemaInferenceOptions>,
) -> Vec<Field> {
    fields
        .iter()
        .filter_map(|f| to_field(f, options.as_ref().unwrap_or(&Default::default())))
        .collect::<Vec<_>>()
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
                Some("+00:00".to_string())
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
        (Some(PrimitiveLogicalType::String), _) => ArrowDataType::Utf8View,
        (Some(PrimitiveLogicalType::Json), _) => ArrowDataType::BinaryView,
        (Some(PrimitiveLogicalType::Bson), _) => ArrowDataType::BinaryView,
        (Some(PrimitiveLogicalType::Enum), _) => ArrowDataType::BinaryView,
        (_, Some(PrimitiveConvertedType::Json)) => ArrowDataType::BinaryView,
        (_, Some(PrimitiveConvertedType::Bson)) => ArrowDataType::BinaryView,
        (_, Some(PrimitiveConvertedType::Enum)) => ArrowDataType::BinaryView,
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
            // There is currently no reliable way of determining which IntervalUnit
            // to return. Thus without the original Arrow schema, the results
            // would be incorrect if all 12 bytes of the interval are populated
            ArrowDataType::Interval(IntervalUnit::DayTime)
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
            &primitive_type.field_info.name,
            base_type,
            is_nullable(&primitive_type.field_info),
        )))
    } else {
        base_type
    }
}

fn non_repeated_group(
    logical_type: &Option<GroupLogicalType>,
    converted_type: &Option<GroupConvertedType>,
    fields: &[ParquetType],
    parent_name: &str,
    options: &SchemaInferenceOptions,
) -> Option<ArrowDataType> {
    debug_assert!(!fields.is_empty());
    match (logical_type, converted_type) {
        (Some(GroupLogicalType::List), _) => to_list(fields, parent_name, options),
        (None, Some(GroupConvertedType::List)) => to_list(fields, parent_name, options),
        (Some(GroupLogicalType::Map), _) => to_list(fields, parent_name, options),
        (None, Some(GroupConvertedType::Map) | Some(GroupConvertedType::MapKeyValue)) => {
            to_map(fields, options)
        },
        _ => to_struct(fields, options),
    }
}

/// Converts a parquet group type to an arrow [`ArrowDataType::Struct`].
/// Returns [`None`] if all its fields are empty
fn to_struct(fields: &[ParquetType], options: &SchemaInferenceOptions) -> Option<ArrowDataType> {
    let fields = fields
        .iter()
        .filter_map(|f| to_field(f, options))
        .collect::<Vec<Field>>();
    if fields.is_empty() {
        None
    } else {
        Some(ArrowDataType::Struct(fields))
    }
}

/// Converts a parquet group type to an arrow [`ArrowDataType::Struct`].
/// Returns [`None`] if all its fields are empty
fn to_map(fields: &[ParquetType], options: &SchemaInferenceOptions) -> Option<ArrowDataType> {
    let inner = to_field(&fields[0], options)?;
    Some(ArrowDataType::Map(Box::new(inner), false))
}

/// Entry point for converting parquet group type.
///
/// This function takes care of logical type and repetition.
fn to_group_type(
    field_info: &FieldInfo,
    logical_type: &Option<GroupLogicalType>,
    converted_type: &Option<GroupConvertedType>,
    fields: &[ParquetType],
    parent_name: &str,
    options: &SchemaInferenceOptions,
) -> Option<ArrowDataType> {
    debug_assert!(!fields.is_empty());
    if field_info.repetition == Repetition::Repeated {
        Some(ArrowDataType::LargeList(Box::new(Field::new(
            &field_info.name,
            to_struct(fields, options)?,
            is_nullable(field_info),
        ))))
    } else {
        non_repeated_group(logical_type, converted_type, fields, parent_name, options)
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

/// Converts parquet schema to arrow field.
/// Returns `None` iff the parquet type has no associated primitive types,
/// i.e. if it is a column-less group type.
fn to_field(type_: &ParquetType, options: &SchemaInferenceOptions) -> Option<Field> {
    Some(Field::new(
        &type_.get_field_info().name,
        to_data_type(type_, options)?,
        is_nullable(type_.get_field_info()),
    ))
}

/// Converts a parquet list to arrow list.
///
/// To fully understand this algorithm, please refer to
/// [parquet doc](https://github.com/apache/parquet-format/blob/master/LogicalTypes.md).
fn to_list(
    fields: &[ParquetType],
    parent_name: &str,
    options: &SchemaInferenceOptions,
) -> Option<ArrowDataType> {
    let item = fields.first().unwrap();

    let item_type = match item {
        ParquetType::PrimitiveType(primitive) => Some(to_primitive_type_inner(primitive, options)),
        ParquetType::GroupType { fields, .. } => {
            if fields.len() == 1
                && item.name() != "array"
                && item.name() != format!("{parent_name}_tuple")
            {
                // extract the repetition field
                let nested_item = fields.first().unwrap();
                to_data_type(nested_item, options)
            } else {
                to_struct(fields, options)
            }
        },
    }?;

    // Check that the name of the list child is "list", in which case we
    // get the child nullability and name (normally "element") from the nested
    // group type.
    // Without this step, the child incorrectly inherits the parent's optionality
    let (list_item_name, item_is_optional) = match item {
        ParquetType::GroupType {
            field_info, fields, ..
        } if field_info.name == "list" && fields.len() == 1 => {
            let field = fields.first().unwrap();
            (
                &field.get_field_info().name,
                field.get_field_info().repetition == Repetition::Optional,
            )
        },
        _ => (
            &item.get_field_info().name,
            item.get_field_info().repetition == Repetition::Optional,
        ),
    };

    Some(ArrowDataType::LargeList(Box::new(Field::new(
        list_item_name,
        item_type,
        item_is_optional,
    ))))
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
pub(crate) fn to_data_type(
    type_: &ParquetType,
    options: &SchemaInferenceOptions,
) -> Option<ArrowDataType> {
    match type_ {
        ParquetType::PrimitiveType(primitive) => Some(to_primitive_type(primitive, options)),
        ParquetType::GroupType {
            field_info,
            logical_type,
            converted_type,
            fields,
        } => {
            if fields.is_empty() {
                None
            } else {
                to_group_type(
                    field_info,
                    logical_type,
                    converted_type,
                    fields,
                    &field_info.name,
                    options,
                )
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use polars_error::*;

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
            Field::new("boolean", ArrowDataType::Boolean, false),
            Field::new("int8", ArrowDataType::Int8, false),
            Field::new("int16", ArrowDataType::Int16, false),
            Field::new("uint8", ArrowDataType::UInt8, false),
            Field::new("uint16", ArrowDataType::UInt16, false),
            Field::new("int32", ArrowDataType::Int32, false),
            Field::new("int64", ArrowDataType::Int64, false),
            Field::new("double", ArrowDataType::Float64, true),
            Field::new("float", ArrowDataType::Float32, true),
            Field::new("string", ArrowDataType::Utf8View, true),
            Field::new("string_2", ArrowDataType::Utf8View, true),
        ];

        let parquet_schema = SchemaDescriptor::try_from_message(message)?;
        let fields = parquet_to_arrow_schema(parquet_schema.fields());

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
            Field::new("binary", ArrowDataType::BinaryView, false),
            Field::new("fixed_binary", ArrowDataType::FixedSizeBinary(20), false),
        ];

        let parquet_schema = SchemaDescriptor::try_from_message(message)?;
        let fields = parquet_to_arrow_schema(parquet_schema.fields());

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
            Field::new("boolean", ArrowDataType::Boolean, false),
            Field::new("int8", ArrowDataType::Int8, false),
        ];

        let parquet_schema = SchemaDescriptor::try_from_message(message)?;
        let fields = parquet_to_arrow_schema(parquet_schema.fields());

        assert_eq!(fields, expected);
        Ok(())
    }

    #[ignore]
    #[test]
    fn test_parquet_lists() -> PolarsResult<()> {
        let mut arrow_fields = Vec::new();

        // LIST encoding example taken from parquet-format/LogicalTypes.md
        let message_type = "
        message test_schema {
          REQUIRED GROUP my_list (LIST) {
            REPEATED GROUP list {
              OPTIONAL BINARY element (UTF8);
            }
          }
          OPTIONAL GROUP my_list (LIST) {
            REPEATED GROUP list {
              REQUIRED BINARY element (UTF8);
            }
          }
          OPTIONAL GROUP array_of_arrays (LIST) {
            REPEATED GROUP list {
              REQUIRED GROUP element (LIST) {
                REPEATED GROUP list {
                  REQUIRED INT32 element;
                }
              }
            }
          }
          OPTIONAL GROUP my_list (LIST) {
            REPEATED GROUP element {
              REQUIRED BINARY str (UTF8);
            }
          }
          OPTIONAL GROUP my_list (LIST) {
            REPEATED INT32 element;
          }
          OPTIONAL GROUP my_list (LIST) {
            REPEATED GROUP element {
              REQUIRED BINARY str (UTF8);
              REQUIRED INT32 num;
            }
          }
          OPTIONAL GROUP my_list (LIST) {
            REPEATED GROUP array {
              REQUIRED BINARY str (UTF8);
            }

          }
          OPTIONAL GROUP my_list (LIST) {
            REPEATED GROUP my_list_tuple {
              REQUIRED BINARY str (UTF8);
            }
          }
          REPEATED INT32 name;
        }
        ";

        // // List<String> (list non-null, elements nullable)
        // required group my_list (LIST) {
        //   repeated group list {
        //     optional binary element (UTF8);
        //   }
        // }
        {
            arrow_fields.push(Field::new(
                "my_list",
                ArrowDataType::LargeList(Box::new(Field::new(
                    "element",
                    ArrowDataType::Utf8,
                    true,
                ))),
                false,
            ));
        }

        // // List<String> (list nullable, elements non-null)
        // optional group my_list (LIST) {
        //   repeated group list {
        //     required binary element (UTF8);
        //   }
        // }
        {
            arrow_fields.push(Field::new(
                "my_list",
                ArrowDataType::LargeList(Box::new(Field::new(
                    "element",
                    ArrowDataType::Utf8,
                    false,
                ))),
                true,
            ));
        }

        // Element types can be nested structures. For example, a list of lists:
        //
        // // List<List<Integer>>
        // optional group array_of_arrays (LIST) {
        //   repeated group list {
        //     required group element (LIST) {
        //       repeated group list {
        //         required int32 element;
        //       }
        //     }
        //   }
        // }
        {
            let arrow_inner_list = ArrowDataType::LargeList(Box::new(Field::new(
                "element",
                ArrowDataType::Int32,
                false,
            )));
            arrow_fields.push(Field::new(
                "array_of_arrays",
                ArrowDataType::LargeList(Box::new(Field::new("element", arrow_inner_list, false))),
                true,
            ));
        }

        // // List<String> (list nullable, elements non-null)
        // optional group my_list (LIST) {
        //   repeated group element {
        //     required binary str (UTF8);
        //   };
        // }
        {
            arrow_fields.push(Field::new(
                "my_list",
                ArrowDataType::LargeList(Box::new(Field::new(
                    "element",
                    ArrowDataType::Utf8,
                    false,
                ))),
                true,
            ));
        }

        // // List<Integer> (nullable list, non-null elements)
        // optional group my_list (LIST) {
        //   repeated int32 element;
        // }
        {
            arrow_fields.push(Field::new(
                "my_list",
                ArrowDataType::LargeList(Box::new(Field::new(
                    "element",
                    ArrowDataType::Int32,
                    false,
                ))),
                true,
            ));
        }

        // // List<Tuple<String, Integer>> (nullable list, non-null elements)
        // optional group my_list (LIST) {
        //   repeated group element {
        //     required binary str (UTF8);
        //     required int32 num;
        //   };
        // }
        {
            let arrow_struct = ArrowDataType::Struct(vec![
                Field::new("str", ArrowDataType::Utf8, false),
                Field::new("num", ArrowDataType::Int32, false),
            ]);
            arrow_fields.push(Field::new(
                "my_list",
                ArrowDataType::LargeList(Box::new(Field::new("element", arrow_struct, false))),
                true,
            ));
        }

        // // List<OneTuple<String>> (nullable list, non-null elements)
        // optional group my_list (LIST) {
        //   repeated group array {
        //     required binary str (UTF8);
        //   };
        // }
        // Special case: group is named array
        {
            let arrow_struct =
                ArrowDataType::Struct(vec![Field::new("str", ArrowDataType::Utf8, false)]);
            arrow_fields.push(Field::new(
                "my_list",
                ArrowDataType::LargeList(Box::new(Field::new("array", arrow_struct, false))),
                true,
            ));
        }

        // // List<OneTuple<String>> (nullable list, non-null elements)
        // optional group my_list (LIST) {
        //   repeated group my_list_tuple {
        //     required binary str (UTF8);
        //   };
        // }
        // Special case: group named ends in _tuple
        {
            let arrow_struct =
                ArrowDataType::Struct(vec![Field::new("str", ArrowDataType::Utf8, false)]);
            arrow_fields.push(Field::new(
                "my_list",
                ArrowDataType::LargeList(Box::new(Field::new(
                    "my_list_tuple",
                    arrow_struct,
                    false,
                ))),
                true,
            ));
        }

        // One-level encoding: Only allows required lists with required cells
        //   repeated value_type name
        {
            arrow_fields.push(Field::new(
                "name",
                ArrowDataType::LargeList(Box::new(Field::new("name", ArrowDataType::Int32, false))),
                false,
            ));
        }

        let parquet_schema = SchemaDescriptor::try_from_message(message_type)?;
        let fields = parquet_to_arrow_schema(parquet_schema.fields());

        assert_eq!(arrow_fields, fields);
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
                Field::new("event_name", ArrowDataType::Utf8View, false),
                Field::new(
                    "event_time",
                    ArrowDataType::Timestamp(TimeUnit::Millisecond, Some("+00:00".into())),
                    false,
                ),
            ];
            arrow_fields.push(Field::new(
                "events",
                ArrowDataType::LargeList(Box::new(Field::new(
                    "array",
                    ArrowDataType::Struct(struct_fields),
                    false,
                ))),
                false,
            ));
        }

        let parquet_schema = SchemaDescriptor::try_from_message(message_type)?;
        let fields = parquet_to_arrow_schema(parquet_schema.fields());

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
                "my_list1",
                ArrowDataType::LargeList(Box::new(Field::new(
                    "element",
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
                "my_list2",
                ArrowDataType::LargeList(Box::new(Field::new(
                    "element",
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
                "my_list3",
                ArrowDataType::LargeList(Box::new(Field::new(
                    "element",
                    ArrowDataType::Utf8View,
                    false,
                ))),
                false,
            ));
        }

        let parquet_schema = SchemaDescriptor::try_from_message(message_type)?;
        let fields = parquet_to_arrow_schema(parquet_schema.fields());

        assert_eq!(arrow_fields, fields);
        Ok(())
    }

    #[test]
    fn test_nested_schema() -> PolarsResult<()> {
        let mut arrow_fields = Vec::new();
        {
            let group1_fields = vec![
                Field::new("leaf1", ArrowDataType::Boolean, false),
                Field::new("leaf2", ArrowDataType::Int32, false),
            ];
            let group1_struct = Field::new("group1", ArrowDataType::Struct(group1_fields), false);
            arrow_fields.push(group1_struct);

            let leaf3_field = Field::new("leaf3", ArrowDataType::Int64, false);
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
        let fields = parquet_to_arrow_schema(parquet_schema.fields());

        assert_eq!(arrow_fields, fields);
        Ok(())
    }

    #[ignore]
    #[test]
    fn test_repeated_nested_schema() -> PolarsResult<()> {
        let mut arrow_fields = Vec::new();
        {
            arrow_fields.push(Field::new("leaf1", ArrowDataType::Int32, true));

            let inner_group_list = Field::new(
                "innerGroup",
                ArrowDataType::LargeList(Box::new(Field::new(
                    "innerGroup",
                    ArrowDataType::Struct(vec![Field::new("leaf3", ArrowDataType::Int32, true)]),
                    false,
                ))),
                false,
            );

            let outer_group_list = Field::new(
                "outerGroup",
                ArrowDataType::LargeList(Box::new(Field::new(
                    "outerGroup",
                    ArrowDataType::Struct(vec![
                        Field::new("leaf2", ArrowDataType::Int32, true),
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
        let fields = parquet_to_arrow_schema(parquet_schema.fields());

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
            Field::new("boolean", ArrowDataType::Boolean, false),
            Field::new("int8", ArrowDataType::Int8, false),
            Field::new("uint8", ArrowDataType::UInt8, false),
            Field::new("int16", ArrowDataType::Int16, false),
            Field::new("uint16", ArrowDataType::UInt16, false),
            Field::new("int32", ArrowDataType::Int32, false),
            Field::new("int64", ArrowDataType::Int64, false),
            Field::new("double", ArrowDataType::Float64, true),
            Field::new("float", ArrowDataType::Float32, true),
            Field::new("string", ArrowDataType::Utf8, true),
            Field::new(
                "bools",
                ArrowDataType::LargeList(Box::new(Field::new(
                    "bools",
                    ArrowDataType::Boolean,
                    false,
                ))),
                false,
            ),
            Field::new("date", ArrowDataType::Date32, true),
            Field::new(
                "time_milli",
                ArrowDataType::Time32(TimeUnit::Millisecond),
                true,
            ),
            Field::new(
                "time_micro",
                ArrowDataType::Time64(TimeUnit::Microsecond),
                true,
            ),
            Field::new(
                "time_nano",
                ArrowDataType::Time64(TimeUnit::Nanosecond),
                true,
            ),
            Field::new(
                "ts_milli",
                ArrowDataType::Timestamp(TimeUnit::Millisecond, None),
                true,
            ),
            Field::new(
                "ts_micro",
                ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new(
                "ts_nano",
                ArrowDataType::Timestamp(TimeUnit::Nanosecond, Some("+00:00".to_string())),
                false,
            ),
        ];

        let parquet_schema = SchemaDescriptor::try_from_message(message_type)?;
        let fields = parquet_to_arrow_schema(parquet_schema.fields());

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
            Field::new("boolean", ArrowDataType::Boolean, false),
            Field::new("int8", ArrowDataType::Int8, false),
            Field::new("int16", ArrowDataType::Int16, false),
            Field::new("int32", ArrowDataType::Int32, false),
            Field::new("int64", ArrowDataType::Int64, false),
            Field::new("double", ArrowDataType::Float64, true),
            Field::new("float", ArrowDataType::Float32, true),
            Field::new("string", ArrowDataType::Utf8View, true),
            Field::new(
                "bools",
                ArrowDataType::LargeList(Box::new(Field::new(
                    "element",
                    ArrowDataType::Boolean,
                    true,
                ))),
                true,
            ),
            Field::new(
                "bools_non_null",
                ArrowDataType::LargeList(Box::new(Field::new(
                    "element",
                    ArrowDataType::Boolean,
                    false,
                ))),
                false,
            ),
            Field::new("date", ArrowDataType::Date32, true),
            Field::new(
                "time_milli",
                ArrowDataType::Time32(TimeUnit::Millisecond),
                true,
            ),
            Field::new(
                "time_micro",
                ArrowDataType::Time64(TimeUnit::Microsecond),
                true,
            ),
            Field::new(
                "ts_milli",
                ArrowDataType::Timestamp(TimeUnit::Millisecond, None),
                true,
            ),
            Field::new(
                "ts_micro",
                ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new(
                "struct",
                ArrowDataType::Struct(vec![
                    Field::new("bools", ArrowDataType::Boolean, false),
                    Field::new("uint32", ArrowDataType::UInt32, false),
                    Field::new(
                        "int32",
                        ArrowDataType::LargeList(Box::new(Field::new(
                            "element",
                            ArrowDataType::Int32,
                            true,
                        ))),
                        false,
                    ),
                ]),
                false,
            ),
            Field::new("dictionary_strings", ArrowDataType::Utf8View, false),
        ];

        let parquet_schema = SchemaDescriptor::try_from_message(message_type)?;
        let fields = parquet_to_arrow_schema(parquet_schema.fields());

        assert_eq!(arrow_fields, fields);
        Ok(())
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
                Field::new("int96_field", coerced_to.clone(), false),
                Field::new(
                    "int96_list",
                    ArrowDataType::LargeList(Box::new(Field::new(
                        "element",
                        coerced_to.clone(),
                        true,
                    ))),
                    true,
                ),
                Field::new(
                    "int96_struct",
                    ArrowDataType::Struct(vec![Field::new(
                        "int96_field",
                        coerced_to.clone(),
                        false,
                    )]),
                    false,
                ),
            ];

            let parquet_schema = SchemaDescriptor::try_from_message(message_type)?;
            let fields = parquet_to_arrow_schema_with_options(
                parquet_schema.fields(),
                &Some(SchemaInferenceOptions {
                    int96_coerce_to_timeunit: tu,
                }),
            );
            assert_eq!(arrow_fields, fields);
        }
        Ok(())
    }
}
