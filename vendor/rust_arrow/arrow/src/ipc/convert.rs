// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Utilities for converting between IPC types and native Arrow types

use crate::datatypes::{DataType, DateUnit, Field, IntervalUnit, Schema, TimeUnit};
use crate::ipc;

use flatbuffers::{
    FlatBufferBuilder, ForwardsUOffset, UnionWIPOffset, Vector, WIPOffset,
};
use std::collections::HashMap;
use std::sync::Arc;

use DataType::*;

/// Serialize a schema in IPC format
pub fn schema_to_fb(schema: &Schema) -> FlatBufferBuilder {
    let mut fbb = FlatBufferBuilder::new();

    let mut fields = vec![];
    for field in schema.fields() {
        let fb_field_name = fbb.create_string(field.name().as_str());
        let (ipc_type_type, ipc_type, ipc_children) =
            get_fb_field_type(field.data_type(), &mut fbb);
        let mut field_builder = ipc::FieldBuilder::new(&mut fbb);
        field_builder.add_name(fb_field_name);
        field_builder.add_type_type(ipc_type_type);
        field_builder.add_nullable(field.is_nullable());
        match ipc_children {
            None => {}
            Some(children) => field_builder.add_children(children),
        };
        field_builder.add_type_(ipc_type);
        fields.push(field_builder.finish());
    }

    let mut custom_metadata = vec![];
    for (k, v) in schema.metadata() {
        let fb_key_name = fbb.create_string(k.as_str());
        let fb_val_name = fbb.create_string(v.as_str());

        let mut kv_builder = ipc::KeyValueBuilder::new(&mut fbb);
        kv_builder.add_key(fb_key_name);
        kv_builder.add_value(fb_val_name);
        custom_metadata.push(kv_builder.finish());
    }

    let fb_field_list = fbb.create_vector(&fields);
    let fb_metadata_list = fbb.create_vector(&custom_metadata);

    let root = {
        let mut builder = ipc::SchemaBuilder::new(&mut fbb);
        builder.add_fields(fb_field_list);
        builder.add_custom_metadata(fb_metadata_list);
        builder.finish()
    };

    fbb.finish(root, None);

    fbb
}

pub fn schema_to_fb_offset<'a: 'b, 'b>(
    fbb: &'a mut FlatBufferBuilder,
    schema: &Schema,
) -> WIPOffset<ipc::Schema<'b>> {
    let mut fields = vec![];
    for field in schema.fields() {
        let fb_field_name = fbb.create_string(field.name().as_str());
        let (ipc_type_type, ipc_type, ipc_children) =
            get_fb_field_type(field.data_type(), fbb);
        let mut field_builder = ipc::FieldBuilder::new(fbb);
        field_builder.add_name(fb_field_name);
        field_builder.add_type_type(ipc_type_type);
        field_builder.add_nullable(field.is_nullable());
        match ipc_children {
            None => {}
            Some(children) => field_builder.add_children(children),
        };
        field_builder.add_type_(ipc_type);
        fields.push(field_builder.finish());
    }

    let mut custom_metadata = vec![];
    for (k, v) in schema.metadata() {
        let fb_key_name = fbb.create_string(k.as_str());
        let fb_val_name = fbb.create_string(v.as_str());

        let mut kv_builder = ipc::KeyValueBuilder::new(fbb);
        kv_builder.add_key(fb_key_name);
        kv_builder.add_value(fb_val_name);
        custom_metadata.push(kv_builder.finish());
    }

    let fb_field_list = fbb.create_vector(&fields);
    let fb_metadata_list = fbb.create_vector(&custom_metadata);

    let mut builder = ipc::SchemaBuilder::new(fbb);
    builder.add_fields(fb_field_list);
    builder.add_custom_metadata(fb_metadata_list);
    builder.finish()
}

/// Convert an IPC Field to Arrow Field
impl<'a> From<ipc::Field<'a>> for Field {
    fn from(field: ipc::Field) -> Field {
        if let Some(dictionary) = field.dictionary() {
            Field::new_dict(
                field.name().unwrap(),
                get_data_type(field, true),
                field.nullable(),
                dictionary.id(),
                dictionary.isOrdered(),
            )
        } else {
            Field::new(
                field.name().unwrap(),
                get_data_type(field, true),
                field.nullable(),
            )
        }
    }
}

/// Deserialize a Schema table from IPC format to Schema data type
pub fn fb_to_schema(fb: ipc::Schema) -> Schema {
    let mut fields: Vec<Field> = vec![];
    let c_fields = fb.fields().unwrap();
    let len = c_fields.len();
    for i in 0..len {
        let c_field: ipc::Field = c_fields.get(i);
        fields.push(c_field.into());
    }

    let mut metadata: HashMap<String, String> = HashMap::default();
    if let Some(md_fields) = fb.custom_metadata() {
        let len = md_fields.len();
        for i in 0..len {
            let kv = md_fields.get(i);
            let k_str = kv.key();
            let v_str = kv.value();
            if let Some(k) = k_str {
                if let Some(v) = v_str {
                    metadata.insert(k.to_string(), v.to_string());
                }
            }
        }
    }
    Schema::new_with_metadata(fields, metadata)
}

/// Deserialize an IPC message into a schema
pub fn schema_from_bytes(bytes: &[u8]) -> Option<Schema> {
    let ipc = ipc::get_root_as_message(bytes);
    ipc.header_as_schema().map(|schema| fb_to_schema(schema))
}

/// Get the Arrow data type from the flatbuffer Field table
pub(crate) fn get_data_type(field: ipc::Field, may_be_dictionary: bool) -> DataType {
    if let Some(dictionary) = field.dictionary() {
        if may_be_dictionary {
            let int = dictionary.indexType().unwrap();
            let index_type = match (int.bitWidth(), int.is_signed()) {
                (8, true) => DataType::Int8,
                (8, false) => DataType::UInt8,
                (16, true) => DataType::Int16,
                (16, false) => DataType::UInt16,
                (32, true) => DataType::Int32,
                (32, false) => DataType::UInt32,
                (64, true) => DataType::Int64,
                (64, false) => DataType::UInt64,
                _ => panic!("Unexpected bitwidth and signed"),
            };
            return DataType::Dictionary(
                Box::new(index_type),
                Box::new(get_data_type(field, false)),
            );
        }
    }

    match field.type_type() {
        ipc::Type::Null => DataType::Null,
        ipc::Type::Bool => DataType::Boolean,
        ipc::Type::Int => {
            let int = field.type_as_int().unwrap();
            match (int.bitWidth(), int.is_signed()) {
                (8, true) => DataType::Int8,
                (8, false) => DataType::UInt8,
                (16, true) => DataType::Int16,
                (16, false) => DataType::UInt16,
                (32, true) => DataType::Int32,
                (32, false) => DataType::UInt32,
                (64, true) => DataType::Int64,
                (64, false) => DataType::UInt64,
                _ => panic!("Unexpected bitwidth and signed"),
            }
        }
        ipc::Type::Binary => DataType::Binary,
        ipc::Type::Utf8 => DataType::Utf8,
        ipc::Type::FixedSizeBinary => {
            let fsb = field.type_as_fixed_size_binary().unwrap();
            DataType::FixedSizeBinary(fsb.byteWidth())
        }
        ipc::Type::FloatingPoint => {
            let float = field.type_as_floating_point().unwrap();
            match float.precision() {
                ipc::Precision::HALF => DataType::Float16,
                ipc::Precision::SINGLE => DataType::Float32,
                ipc::Precision::DOUBLE => DataType::Float64,
            }
        }
        ipc::Type::Date => {
            let date = field.type_as_date().unwrap();
            match date.unit() {
                ipc::DateUnit::DAY => DataType::Date32(DateUnit::Day),
                ipc::DateUnit::MILLISECOND => DataType::Date64(DateUnit::Millisecond),
            }
        }
        ipc::Type::Time => {
            let time = field.type_as_time().unwrap();
            match (time.bitWidth(), time.unit()) {
                (32, ipc::TimeUnit::SECOND) => DataType::Time32(TimeUnit::Second),
                (32, ipc::TimeUnit::MILLISECOND) => {
                    DataType::Time32(TimeUnit::Millisecond)
                }
                (64, ipc::TimeUnit::MICROSECOND) => {
                    DataType::Time64(TimeUnit::Microsecond)
                }
                (64, ipc::TimeUnit::NANOSECOND) => DataType::Time64(TimeUnit::Nanosecond),
                z => panic!(
                    "Time type with bit width of {} and unit of {:?} not supported",
                    z.0, z.1
                ),
            }
        }
        ipc::Type::Timestamp => {
            let timestamp = field.type_as_timestamp().unwrap();
            let timezone: Option<Arc<String>> =
                timestamp.timezone().map(|tz| Arc::new(tz.to_string()));
            match timestamp.unit() {
                ipc::TimeUnit::SECOND => DataType::Timestamp(TimeUnit::Second, timezone),
                ipc::TimeUnit::MILLISECOND => {
                    DataType::Timestamp(TimeUnit::Millisecond, timezone)
                }
                ipc::TimeUnit::MICROSECOND => {
                    DataType::Timestamp(TimeUnit::Microsecond, timezone)
                }
                ipc::TimeUnit::NANOSECOND => {
                    DataType::Timestamp(TimeUnit::Nanosecond, timezone)
                }
            }
        }
        ipc::Type::Interval => {
            let interval = field.type_as_interval().unwrap();
            match interval.unit() {
                ipc::IntervalUnit::YEAR_MONTH => {
                    DataType::Interval(IntervalUnit::YearMonth)
                }
                ipc::IntervalUnit::DAY_TIME => DataType::Interval(IntervalUnit::DayTime),
            }
        }
        ipc::Type::Duration => {
            let duration = field.type_as_duration().unwrap();
            match duration.unit() {
                ipc::TimeUnit::SECOND => DataType::Duration(TimeUnit::Second),
                ipc::TimeUnit::MILLISECOND => DataType::Duration(TimeUnit::Millisecond),
                ipc::TimeUnit::MICROSECOND => DataType::Duration(TimeUnit::Microsecond),
                ipc::TimeUnit::NANOSECOND => DataType::Duration(TimeUnit::Nanosecond),
            }
        }
        ipc::Type::List => {
            let children = field.children().unwrap();
            if children.len() != 1 {
                panic!("expect a list to have one child")
            }
            let child_field = children.get(0);
            // returning int16 for now, to test, not sure how to get data type
            DataType::List(Box::new(get_data_type(child_field, false)))
        }
        ipc::Type::FixedSizeList => {
            let children = field.children().unwrap();
            if children.len() != 1 {
                panic!("expect a list to have one child")
            }
            let child_field = children.get(0);
            let fsl = field.type_as_fixed_size_list().unwrap();
            DataType::FixedSizeList(
                Box::new(get_data_type(child_field, false)),
                fsl.listSize(),
            )
        }
        ipc::Type::Struct_ => {
            let mut fields = vec![];
            if let Some(children) = field.children() {
                for i in 0..children.len() {
                    fields.push(children.get(i).into());
                }
            };

            DataType::Struct(fields)
        }
        t => unimplemented!("Type {:?} not supported", t),
    }
}

/// Get the IPC type of a data type
pub(crate) fn get_fb_field_type<'a: 'b, 'b>(
    data_type: &DataType,
    fbb: &mut FlatBufferBuilder<'a>,
) -> (
    ipc::Type,
    WIPOffset<UnionWIPOffset>,
    Option<WIPOffset<Vector<'b, ForwardsUOffset<ipc::Field<'b>>>>>,
) {
    // some IPC implementations expect an empty list for child data, instead of a null value.
    // An empty field list is thus returned for primitive types
    let empty_fields: Vec<WIPOffset<ipc::Field>> = vec![];
    match data_type {
        Null => (
            ipc::Type::Null,
            ipc::NullBuilder::new(fbb).finish().as_union_value(),
            None,
        ),
        Boolean => {
            let children = fbb.create_vector(&empty_fields[..]);
            (
                ipc::Type::Bool,
                ipc::BoolBuilder::new(fbb).finish().as_union_value(),
                Some(children),
            )
        }
        UInt8 | UInt16 | UInt32 | UInt64 => {
            let children = fbb.create_vector(&empty_fields[..]);
            let mut builder = ipc::IntBuilder::new(fbb);
            builder.add_is_signed(false);
            match data_type {
                UInt8 => builder.add_bitWidth(8),
                UInt16 => builder.add_bitWidth(16),
                UInt32 => builder.add_bitWidth(32),
                UInt64 => builder.add_bitWidth(64),
                _ => {}
            };
            (
                ipc::Type::Int,
                builder.finish().as_union_value(),
                Some(children),
            )
        }
        Int8 | Int16 | Int32 | Int64 => {
            let children = fbb.create_vector(&empty_fields[..]);
            let mut builder = ipc::IntBuilder::new(fbb);
            builder.add_is_signed(true);
            match data_type {
                Int8 => builder.add_bitWidth(8),
                Int16 => builder.add_bitWidth(16),
                Int32 => builder.add_bitWidth(32),
                Int64 => builder.add_bitWidth(64),
                _ => {}
            };
            (
                ipc::Type::Int,
                builder.finish().as_union_value(),
                Some(children),
            )
        }
        Float16 | Float32 | Float64 => {
            let children = fbb.create_vector(&empty_fields[..]);
            let mut builder = ipc::FloatingPointBuilder::new(fbb);
            match data_type {
                Float16 => builder.add_precision(ipc::Precision::HALF),
                Float32 => builder.add_precision(ipc::Precision::SINGLE),
                Float64 => builder.add_precision(ipc::Precision::DOUBLE),
                _ => {}
            };
            (
                ipc::Type::FloatingPoint,
                builder.finish().as_union_value(),
                Some(children),
            )
        }
        Binary => {
            let children = fbb.create_vector(&empty_fields[..]);
            (
                ipc::Type::Binary,
                ipc::BinaryBuilder::new(fbb).finish().as_union_value(),
                Some(children),
            )
        }
        Utf8 => {
            let children = fbb.create_vector(&empty_fields[..]);
            (
                ipc::Type::Utf8,
                ipc::Utf8Builder::new(fbb).finish().as_union_value(),
                Some(children),
            )
        }
        FixedSizeBinary(len) => {
            let children = fbb.create_vector(&empty_fields[..]);
            let mut builder = ipc::FixedSizeBinaryBuilder::new(fbb);
            builder.add_byteWidth(*len as i32);
            (
                ipc::Type::FixedSizeBinary,
                builder.finish().as_union_value(),
                Some(children),
            )
        }
        Date32(_) => {
            let children = fbb.create_vector(&empty_fields[..]);
            let mut builder = ipc::DateBuilder::new(fbb);
            builder.add_unit(ipc::DateUnit::DAY);
            (
                ipc::Type::Date,
                builder.finish().as_union_value(),
                Some(children),
            )
        }
        Date64(_) => {
            let children = fbb.create_vector(&empty_fields[..]);
            let mut builder = ipc::DateBuilder::new(fbb);
            builder.add_unit(ipc::DateUnit::MILLISECOND);
            (
                ipc::Type::Date,
                builder.finish().as_union_value(),
                Some(children),
            )
        }
        Time32(unit) | Time64(unit) => {
            let children = fbb.create_vector(&empty_fields[..]);
            let mut builder = ipc::TimeBuilder::new(fbb);
            match unit {
                TimeUnit::Second => {
                    builder.add_bitWidth(32);
                    builder.add_unit(ipc::TimeUnit::SECOND);
                }
                TimeUnit::Millisecond => {
                    builder.add_bitWidth(32);
                    builder.add_unit(ipc::TimeUnit::MILLISECOND);
                }
                TimeUnit::Microsecond => {
                    builder.add_bitWidth(64);
                    builder.add_unit(ipc::TimeUnit::MICROSECOND);
                }
                TimeUnit::Nanosecond => {
                    builder.add_bitWidth(64);
                    builder.add_unit(ipc::TimeUnit::NANOSECOND);
                }
            }
            (
                ipc::Type::Time,
                builder.finish().as_union_value(),
                Some(children),
            )
        }
        Timestamp(unit, tz) => {
            let children = fbb.create_vector(&empty_fields[..]);
            let tz = tz.clone().unwrap_or_else(|| Arc::new(String::new()));
            let tz_str = fbb.create_string(tz.as_str());
            let mut builder = ipc::TimestampBuilder::new(fbb);
            let time_unit = match unit {
                TimeUnit::Second => ipc::TimeUnit::SECOND,
                TimeUnit::Millisecond => ipc::TimeUnit::MILLISECOND,
                TimeUnit::Microsecond => ipc::TimeUnit::MICROSECOND,
                TimeUnit::Nanosecond => ipc::TimeUnit::NANOSECOND,
            };
            builder.add_unit(time_unit);
            if !tz.is_empty() {
                builder.add_timezone(tz_str);
            }
            (
                ipc::Type::Timestamp,
                builder.finish().as_union_value(),
                Some(children),
            )
        }
        Interval(unit) => {
            let children = fbb.create_vector(&empty_fields[..]);
            let mut builder = ipc::IntervalBuilder::new(fbb);
            let interval_unit = match unit {
                IntervalUnit::YearMonth => ipc::IntervalUnit::YEAR_MONTH,
                IntervalUnit::DayTime => ipc::IntervalUnit::DAY_TIME,
            };
            builder.add_unit(interval_unit);
            (
                ipc::Type::Interval,
                builder.finish().as_union_value(),
                Some(children),
            )
        }
        Duration(unit) => {
            let children = fbb.create_vector(&empty_fields[..]);
            let mut builder = ipc::DurationBuilder::new(fbb);
            let time_unit = match unit {
                TimeUnit::Second => ipc::TimeUnit::SECOND,
                TimeUnit::Millisecond => ipc::TimeUnit::MILLISECOND,
                TimeUnit::Microsecond => ipc::TimeUnit::MICROSECOND,
                TimeUnit::Nanosecond => ipc::TimeUnit::NANOSECOND,
            };
            builder.add_unit(time_unit);
            (
                ipc::Type::Duration,
                builder.finish().as_union_value(),
                Some(children),
            )
        }
        List(ref list_type) => {
            let inner_types = get_fb_field_type(list_type, fbb);
            let child = ipc::Field::create(
                fbb,
                &ipc::FieldArgs {
                    name: None,
                    nullable: false,
                    type_type: inner_types.0,
                    type_: Some(inner_types.1),
                    dictionary: None,
                    children: inner_types.2,
                    custom_metadata: None,
                },
            );
            let children = fbb.create_vector(&[child]);
            (
                ipc::Type::List,
                ipc::ListBuilder::new(fbb).finish().as_union_value(),
                Some(children),
            )
        }
        FixedSizeList(ref list_type, len) => {
            let inner_types = get_fb_field_type(list_type, fbb);
            let child = ipc::Field::create(
                fbb,
                &ipc::FieldArgs {
                    name: None,
                    nullable: false,
                    type_type: inner_types.0,
                    type_: Some(inner_types.1),
                    dictionary: None,
                    children: inner_types.2,
                    custom_metadata: None,
                },
            );
            let children = fbb.create_vector(&[child]);
            let mut builder = ipc::FixedSizeListBuilder::new(fbb);
            builder.add_listSize(*len as i32);
            (
                ipc::Type::FixedSizeList,
                builder.finish().as_union_value(),
                Some(children),
            )
        }
        Struct(fields) => {
            // struct's fields are children
            let mut children = vec![];
            for field in fields {
                let inner_types = get_fb_field_type(field.data_type(), fbb);
                let field_name = fbb.create_string(field.name());
                children.push(ipc::Field::create(
                    fbb,
                    &ipc::FieldArgs {
                        name: Some(field_name),
                        nullable: field.is_nullable(),
                        type_type: inner_types.0,
                        type_: Some(inner_types.1),
                        dictionary: None,
                        children: inner_types.2,
                        custom_metadata: None,
                    },
                ));
            }
            let children = fbb.create_vector(&children[..]);
            (
                ipc::Type::Struct_,
                ipc::Struct_Builder::new(fbb).finish().as_union_value(),
                Some(children),
            )
        }
        t => unimplemented!("Type {:?} not supported", t),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatypes::{DataType, Field, Schema};

    #[test]
    fn convert_schema_round_trip() {
        let md: HashMap<String, String> = [("Key".to_string(), "value".to_string())]
            .iter()
            .cloned()
            .collect();
        let schema = Schema::new_with_metadata(
            vec![
                Field::new("uint8", DataType::UInt8, false),
                Field::new("uint16", DataType::UInt16, true),
                Field::new("uint32", DataType::UInt32, false),
                Field::new("uint64", DataType::UInt64, true),
                Field::new("int8", DataType::Int8, true),
                Field::new("int16", DataType::Int16, false),
                Field::new("int32", DataType::Int32, true),
                Field::new("int64", DataType::Int64, false),
                Field::new("float16", DataType::Float16, true),
                Field::new("float32", DataType::Float32, false),
                Field::new("float64", DataType::Float64, true),
                Field::new("null", DataType::Null, false),
                Field::new("bool", DataType::Boolean, false),
                Field::new("date32", DataType::Date32(DateUnit::Day), false),
                Field::new("date64", DataType::Date64(DateUnit::Millisecond), true),
                Field::new("time32[s]", DataType::Time32(TimeUnit::Second), true),
                Field::new("time32[ms]", DataType::Time32(TimeUnit::Millisecond), false),
                Field::new("time64[us]", DataType::Time64(TimeUnit::Microsecond), false),
                Field::new("time64[ns]", DataType::Time64(TimeUnit::Nanosecond), true),
                Field::new(
                    "timestamp[s]",
                    DataType::Timestamp(TimeUnit::Second, None),
                    false,
                ),
                Field::new(
                    "timestamp[ms]",
                    DataType::Timestamp(TimeUnit::Millisecond, None),
                    true,
                ),
                Field::new(
                    "timestamp[us]",
                    DataType::Timestamp(
                        TimeUnit::Microsecond,
                        Some(Arc::new("Africa/Johannesburg".to_string())),
                    ),
                    false,
                ),
                Field::new(
                    "timestamp[ns]",
                    DataType::Timestamp(TimeUnit::Nanosecond, None),
                    true,
                ),
                Field::new(
                    "interval[ym]",
                    DataType::Interval(IntervalUnit::YearMonth),
                    true,
                ),
                Field::new(
                    "interval[dt]",
                    DataType::Interval(IntervalUnit::DayTime),
                    true,
                ),
                Field::new("utf8", DataType::Utf8, false),
                Field::new("binary", DataType::Binary, false),
                Field::new("list[u8]", DataType::List(Box::new(DataType::UInt8)), true),
                Field::new(
                    "list[struct<float32, int32, bool>]",
                    DataType::List(Box::new(DataType::Struct(vec![
                        Field::new("float32", DataType::UInt8, false),
                        Field::new("int32", DataType::Int32, true),
                        Field::new("bool", DataType::Boolean, true),
                    ]))),
                    false,
                ),
                Field::new(
                    "struct<int64, list[struct<date32, list[struct<>]>]>",
                    DataType::Struct(vec![
                        Field::new("int64", DataType::Int64, true),
                        Field::new(
                            "list[struct<date32, list[struct<>]>]",
                            DataType::List(Box::new(DataType::Struct(vec![
                                Field::new(
                                    "date32",
                                    DataType::Date32(DateUnit::Day),
                                    true,
                                ),
                                Field::new(
                                    "list[struct<>]",
                                    DataType::List(Box::new(DataType::Struct(vec![]))),
                                    false,
                                ),
                            ]))),
                            false,
                        ),
                    ]),
                    false,
                ),
                Field::new("struct<>", DataType::Struct(vec![]), true),
            ],
            md,
        );

        let fb = schema_to_fb(&schema);

        // read back fields
        let ipc = ipc::get_root_as_schema(fb.finished_data());
        let schema2 = fb_to_schema(ipc);
        assert_eq!(schema, schema2);
    }

    #[test]
    fn schema_from_bytes() {
        // bytes of a schema generated from python (0.14.0), saved as an `ipc::Message`.
        // the schema is: Field("field1", DataType::UInt32, false)
        let bytes: Vec<u8> = vec![
            16, 0, 0, 0, 0, 0, 10, 0, 12, 0, 6, 0, 5, 0, 8, 0, 10, 0, 0, 0, 0, 1, 3, 0,
            12, 0, 0, 0, 8, 0, 8, 0, 0, 0, 4, 0, 8, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 20,
            0, 0, 0, 16, 0, 20, 0, 8, 0, 0, 0, 7, 0, 12, 0, 0, 0, 16, 0, 16, 0, 0, 0, 0,
            0, 0, 2, 32, 0, 0, 0, 20, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 8, 0,
            4, 0, 6, 0, 0, 0, 32, 0, 0, 0, 6, 0, 0, 0, 102, 105, 101, 108, 100, 49, 0, 0,
            0, 0, 0, 0,
        ];
        let ipc = ipc::get_root_as_message(&bytes[..]);
        let schema = ipc.header_as_schema().unwrap();

        // a message generated from Rust, same as the Python one
        let bytes: Vec<u8> = vec![
            16, 0, 0, 0, 0, 0, 10, 0, 14, 0, 12, 0, 11, 0, 4, 0, 10, 0, 0, 0, 20, 0, 0,
            0, 0, 0, 0, 1, 3, 0, 10, 0, 12, 0, 0, 0, 8, 0, 4, 0, 10, 0, 0, 0, 8, 0, 0, 0,
            8, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 16, 0, 0, 0, 12, 0, 18, 0, 12, 0, 0, 0,
            11, 0, 4, 0, 12, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 2, 20, 0, 0, 0, 0, 0, 6, 0,
            8, 0, 4, 0, 6, 0, 0, 0, 32, 0, 0, 0, 6, 0, 0, 0, 102, 105, 101, 108, 100, 49,
            0, 0,
        ];
        let ipc2 = ipc::get_root_as_message(&bytes[..]);
        let schema2 = ipc.header_as_schema().unwrap();

        assert_eq!(schema, schema2);
        assert_eq!(ipc.version(), ipc2.version());
        assert_eq!(ipc.header_type(), ipc2.header_type());
        assert_eq!(ipc.bodyLength(), ipc2.bodyLength());
        assert!(ipc.custom_metadata().is_none());
        assert!(ipc2.custom_metadata().is_none());
    }
}
