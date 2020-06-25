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

//! Utils for JSON integration testing
//!
//! These utilities define structs that read the integration JSON format for integration testing purposes.

use serde_derive::{Deserialize, Serialize};
use serde_json::{Number as VNumber, Value};

use crate::array::*;
use crate::datatypes::*;
use crate::record_batch::{RecordBatch, RecordBatchReader};

/// A struct that represents an Arrow file with a schema and record batches
#[derive(Deserialize, Serialize, Debug)]
pub struct ArrowJson {
    pub schema: ArrowJsonSchema,
    pub batches: Vec<ArrowJsonBatch>,
    pub dictionaries: Option<Vec<ArrowJsonDictionaryBatch>>,
}

/// A struct that partially reads the Arrow JSON schema.
///
/// Fields are left as JSON `Value` as they vary by `DataType`
#[derive(Deserialize, Serialize, Debug)]
pub struct ArrowJsonSchema {
    pub fields: Vec<Value>,
}

/// A struct that partially reads the Arrow JSON record batch
#[derive(Deserialize, Serialize, Debug)]
pub struct ArrowJsonBatch {
    count: usize,
    pub columns: Vec<ArrowJsonColumn>,
}

/// A struct that partially reads the Arrow JSON dictionary batch
#[derive(Deserialize, Serialize, Debug)]
#[allow(non_snake_case)]
pub struct ArrowJsonDictionaryBatch {
    id: i64,
    data: ArrowJsonBatch,
}

/// A struct that partially reads the Arrow JSON column/array
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ArrowJsonColumn {
    name: String,
    pub count: usize,
    #[serde(rename = "VALIDITY")]
    pub validity: Option<Vec<u8>>,
    #[serde(rename = "DATA")]
    pub data: Option<Vec<Value>>,
    #[serde(rename = "OFFSET")]
    pub offset: Option<Vec<Value>>, // leaving as Value as 64-bit offsets are strings
    pub children: Option<Vec<ArrowJsonColumn>>,
}

impl ArrowJson {
    /// Compare the Arrow JSON with a record batch reader
    pub fn equals_reader(&self, reader: &mut RecordBatchReader) -> bool {
        if !self.schema.equals_schema(&reader.schema()) {
            return false;
        }
        self.batches.iter().all(|col| {
            let batch = reader.next_batch();
            match batch {
                Ok(Some(batch)) => col.equals_batch(&batch),
                _ => false,
            }
        })
    }
}

impl ArrowJsonSchema {
    /// Compare the Arrow JSON schema with the Arrow `Schema`
    fn equals_schema(&self, schema: &Schema) -> bool {
        let field_len = self.fields.len();
        if field_len != schema.fields().len() {
            return false;
        }
        for i in 0..field_len {
            let json_field = &self.fields[i];
            let field = schema.field(i);
            assert_eq!(json_field, &field.to_json());
        }
        true
    }
}

impl ArrowJsonBatch {
    /// Compare the Arrow JSON record batch with a `RecordBatch`
    fn equals_batch(&self, batch: &RecordBatch) -> bool {
        if self.count != batch.num_rows() {
            return false;
        }
        let num_columns = self.columns.len();
        if num_columns != batch.num_columns() {
            return false;
        }
        let schema = batch.schema();
        self.columns
            .iter()
            .zip(batch.columns())
            .zip(schema.fields())
            .all(|((col, arr), field)| {
                // compare each column based on its type
                if &col.name != field.name() {
                    return false;
                }
                let json_array: Vec<Value> = json_from_col(&col, field.data_type());
                match field.data_type() {
                    DataType::Null => {
                        let arr = arr.as_any().downcast_ref::<NullArray>().unwrap();
                        arr.equals_json(&json_array.iter().collect::<Vec<&Value>>()[..])
                    }
                    DataType::Boolean => {
                        let arr = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
                        arr.equals_json(&json_array.iter().collect::<Vec<&Value>>()[..])
                    }
                    DataType::Int8 => {
                        let arr = arr.as_any().downcast_ref::<Int8Array>().unwrap();
                        arr.equals_json(&json_array.iter().collect::<Vec<&Value>>()[..])
                    }
                    DataType::Int16 => {
                        let arr = arr.as_any().downcast_ref::<Int16Array>().unwrap();
                        arr.equals_json(&json_array.iter().collect::<Vec<&Value>>()[..])
                    }
                    DataType::Int32 | DataType::Date32(_) | DataType::Time32(_) => {
                        let arr = Int32Array::from(arr.data());
                        arr.equals_json(&json_array.iter().collect::<Vec<&Value>>()[..])
                    }
                    DataType::Int64
                    | DataType::Date64(_)
                    | DataType::Time64(_)
                    | DataType::Timestamp(_, _)
                    | DataType::Duration(_) => {
                        let arr = Int64Array::from(arr.data());
                        arr.equals_json(&json_array.iter().collect::<Vec<&Value>>()[..])
                    }
                    DataType::Interval(IntervalUnit::YearMonth) => {
                        let arr = IntervalYearMonthArray::from(arr.data());
                        arr.equals_json(&json_array.iter().collect::<Vec<&Value>>()[..])
                    }
                    DataType::Interval(IntervalUnit::DayTime) => {
                        let arr = IntervalDayTimeArray::from(arr.data());
                        let x = json_array
                            .iter()
                            .map(|v| {
                                match v {
                                    Value::Null => Value::Null,
                                    Value::Object(v) => {
                                        // interval has days and milliseconds
                                        let days: i32 =
                                            v.get("days").unwrap().as_i64().unwrap()
                                                as i32;
                                        let milliseconds: i32 = v
                                            .get("milliseconds")
                                            .unwrap()
                                            .as_i64()
                                            .unwrap()
                                            as i32;
                                        let value: i64 = unsafe {
                                            std::mem::transmute::<[i32; 2], i64>([
                                                days,
                                                milliseconds,
                                            ])
                                        };
                                        Value::Number(VNumber::from(value))
                                    }
                                    // return null if Value is not an object
                                    _ => Value::Null,
                                }
                            })
                            .collect::<Vec<Value>>();
                        arr.equals_json(&x.iter().collect::<Vec<&Value>>()[..])
                    }
                    DataType::UInt8 => {
                        let arr = arr.as_any().downcast_ref::<UInt8Array>().unwrap();
                        arr.equals_json(&json_array.iter().collect::<Vec<&Value>>()[..])
                    }
                    DataType::UInt16 => {
                        let arr = arr.as_any().downcast_ref::<UInt16Array>().unwrap();
                        arr.equals_json(&json_array.iter().collect::<Vec<&Value>>()[..])
                    }
                    DataType::UInt32 => {
                        let arr = arr.as_any().downcast_ref::<UInt32Array>().unwrap();
                        arr.equals_json(&json_array.iter().collect::<Vec<&Value>>()[..])
                    }
                    DataType::UInt64 => {
                        let arr = arr.as_any().downcast_ref::<UInt64Array>().unwrap();
                        arr.equals_json(&json_array.iter().collect::<Vec<&Value>>()[..])
                    }
                    DataType::Float32 => {
                        let arr = arr.as_any().downcast_ref::<Float32Array>().unwrap();
                        arr.equals_json(&json_array.iter().collect::<Vec<&Value>>()[..])
                    }
                    DataType::Float64 => {
                        let arr = arr.as_any().downcast_ref::<Float64Array>().unwrap();
                        arr.equals_json(&json_array.iter().collect::<Vec<&Value>>()[..])
                    }
                    DataType::Binary => {
                        let arr = arr.as_any().downcast_ref::<BinaryArray>().unwrap();
                        arr.equals_json(&json_array.iter().collect::<Vec<&Value>>()[..])
                    }
                    DataType::Utf8 => {
                        let arr = arr.as_any().downcast_ref::<StringArray>().unwrap();
                        arr.equals_json(&json_array.iter().collect::<Vec<&Value>>()[..])
                    }
                    DataType::FixedSizeBinary(_) => {
                        let arr =
                            arr.as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap();
                        arr.equals_json(&json_array.iter().collect::<Vec<&Value>>()[..])
                    }
                    DataType::List(_) => {
                        let arr = arr.as_any().downcast_ref::<ListArray>().unwrap();
                        arr.equals_json(&json_array.iter().collect::<Vec<&Value>>()[..])
                    }
                    DataType::FixedSizeList(_, _) => {
                        let arr =
                            arr.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
                        arr.equals_json(&json_array.iter().collect::<Vec<&Value>>()[..])
                    }
                    DataType::Struct(_) => {
                        let arr = arr.as_any().downcast_ref::<StructArray>().unwrap();
                        arr.equals_json(&json_array.iter().collect::<Vec<&Value>>()[..])
                    }
                    DataType::Dictionary(ref key_type, _) => match key_type.as_ref() {
                        DataType::Int8 => {
                            let arr = arr
                                .as_any()
                                .downcast_ref::<Int8DictionaryArray>()
                                .unwrap();
                            arr.equals_json(
                                &json_array.iter().collect::<Vec<&Value>>()[..],
                            )
                        }
                        DataType::Int16 => {
                            let arr = arr
                                .as_any()
                                .downcast_ref::<Int16DictionaryArray>()
                                .unwrap();
                            arr.equals_json(
                                &json_array.iter().collect::<Vec<&Value>>()[..],
                            )
                        }
                        DataType::Int32 => {
                            let arr = arr
                                .as_any()
                                .downcast_ref::<Int32DictionaryArray>()
                                .unwrap();
                            arr.equals_json(
                                &json_array.iter().collect::<Vec<&Value>>()[..],
                            )
                        }
                        DataType::Int64 => {
                            let arr = arr
                                .as_any()
                                .downcast_ref::<Int64DictionaryArray>()
                                .unwrap();
                            arr.equals_json(
                                &json_array.iter().collect::<Vec<&Value>>()[..],
                            )
                        }
                        DataType::UInt8 => {
                            let arr = arr
                                .as_any()
                                .downcast_ref::<UInt8DictionaryArray>()
                                .unwrap();
                            arr.equals_json(
                                &json_array.iter().collect::<Vec<&Value>>()[..],
                            )
                        }
                        DataType::UInt16 => {
                            let arr = arr
                                .as_any()
                                .downcast_ref::<UInt16DictionaryArray>()
                                .unwrap();
                            arr.equals_json(
                                &json_array.iter().collect::<Vec<&Value>>()[..],
                            )
                        }
                        DataType::UInt32 => {
                            let arr = arr
                                .as_any()
                                .downcast_ref::<UInt32DictionaryArray>()
                                .unwrap();
                            arr.equals_json(
                                &json_array.iter().collect::<Vec<&Value>>()[..],
                            )
                        }
                        DataType::UInt64 => {
                            let arr = arr
                                .as_any()
                                .downcast_ref::<UInt64DictionaryArray>()
                                .unwrap();
                            arr.equals_json(
                                &json_array.iter().collect::<Vec<&Value>>()[..],
                            )
                        }
                        t => panic!("Unsupported dictionary comparison for {:?}", t),
                    },
                    t => panic!("Unsupported comparison for {:?}", t),
                }
            })
    }

    pub fn from_batch(batch: &RecordBatch) -> ArrowJsonBatch {
        let mut json_batch = ArrowJsonBatch {
            count: batch.num_rows(),
            columns: Vec::with_capacity(batch.num_columns()),
        };

        for (col, field) in batch.columns().iter().zip(batch.schema().fields.iter()) {
            let json_col = match field.data_type() {
                DataType::Int8 => {
                    let col = col.as_any().downcast_ref::<Int8Array>().unwrap();

                    let mut validity: Vec<u8> = Vec::with_capacity(col.len());
                    let mut data: Vec<Value> = Vec::with_capacity(col.len());

                    for i in 0..col.len() {
                        if col.is_null(i) {
                            validity.push(1);
                            data.push(
                                Int8Type::default_value().into_json_value().unwrap(),
                            );
                        } else {
                            validity.push(0);
                            data.push(col.value(i).into_json_value().unwrap());
                        }
                    }

                    ArrowJsonColumn {
                        name: field.name().clone(),
                        count: col.len(),
                        validity: Some(validity),
                        data: Some(data),
                        offset: None,
                        children: None,
                    }
                }
                _ => ArrowJsonColumn {
                    name: field.name().clone(),
                    count: col.len(),
                    validity: None,
                    data: None,
                    offset: None,
                    children: None,
                },
            };

            json_batch.columns.push(json_col);
        }

        json_batch
    }
}

/// Convert an Arrow JSON column/array into a vector of `Value`
fn json_from_col(col: &ArrowJsonColumn, data_type: &DataType) -> Vec<Value> {
    match data_type {
        DataType::List(dt) => json_from_list_col(col, &**dt),
        DataType::FixedSizeList(dt, list_size) => {
            json_from_fixed_size_list_col(col, &**dt, *list_size as usize)
        }
        DataType::Struct(fields) => json_from_struct_col(col, fields),
        _ => merge_json_array(
            col.validity.as_ref().unwrap().as_slice(),
            &col.data.clone().unwrap(),
        ),
    }
}

/// Merge VALIDITY and DATA vectors from a primitive data type into a `Value` vector with nulls
fn merge_json_array(validity: &[u8], data: &[Value]) -> Vec<Value> {
    validity
        .iter()
        .zip(data)
        .map(|(v, d)| match v {
            0 => Value::Null,
            1 => d.clone(),
            _ => panic!("Validity data should be 0 or 1"),
        })
        .collect()
}

/// Convert an Arrow JSON column/array of a `DataType::Struct` into a vector of `Value`
fn json_from_struct_col(col: &ArrowJsonColumn, fields: &[Field]) -> Vec<Value> {
    let mut values = Vec::with_capacity(col.count);

    let children: Vec<Vec<Value>> = col
        .children
        .clone()
        .unwrap()
        .iter()
        .zip(fields)
        .map(|(child, field)| json_from_col(child, field.data_type()))
        .collect();

    // create a struct from children
    for j in 0..col.count {
        let mut map = serde_json::map::Map::new();
        for i in 0..children.len() {
            map.insert(fields[i].name().to_string(), children[i][j].clone());
        }
        values.push(Value::Object(map));
    }

    values
}

/// Convert an Arrow JSON column/array of a `DataType::List` into a vector of `Value`
fn json_from_list_col(col: &ArrowJsonColumn, data_type: &DataType) -> Vec<Value> {
    let mut values = Vec::with_capacity(col.count);

    // get the inner array
    let child = &col.children.clone().expect("list type must have children")[0];
    let offsets: Vec<usize> = col
        .offset
        .clone()
        .unwrap()
        .iter()
        .map(|o| match o {
            Value::String(s) => s.parse::<usize>().unwrap(),
            Value::Number(n) => n.as_u64().unwrap() as usize,
            _ => panic!(
                "Offsets should be numbers or strings that are convertible to numbers"
            ),
        })
        .collect();
    let inner = match data_type {
        DataType::List(ref dt) => json_from_col(child, &**dt),
        DataType::Struct(fields) => json_from_struct_col(col, fields),
        _ => merge_json_array(
            child.validity.as_ref().unwrap().as_slice(),
            &child.data.clone().unwrap(),
        ),
    };

    for i in 0..col.count {
        match &col.validity {
            Some(validity) => match &validity[i] {
                0 => values.push(Value::Null),
                1 => {
                    values.push(Value::Array(inner[offsets[i]..offsets[i + 1]].to_vec()))
                }
                _ => panic!("Validity data should be 0 or 1"),
            },
            None => {
                // Null type does not have a validity vector
            }
        }
    }

    values
}

/// Convert an Arrow JSON column/array of a `DataType::List` into a vector of `Value`
fn json_from_fixed_size_list_col(
    col: &ArrowJsonColumn,
    data_type: &DataType,
    list_size: usize,
) -> Vec<Value> {
    let mut values = Vec::with_capacity(col.count);

    // get the inner array
    let child = &col.children.clone().expect("list type must have children")[0];
    let inner = match data_type {
        DataType::List(ref dt) => json_from_col(child, &**dt),
        DataType::FixedSizeList(ref dt, _) => json_from_col(child, &**dt),
        DataType::Struct(fields) => json_from_struct_col(col, fields),
        _ => merge_json_array(
            child.validity.as_ref().unwrap().as_slice(),
            &child.data.clone().unwrap(),
        ),
    };

    for i in 0..col.count {
        match &col.validity {
            Some(validity) => match &validity[i] {
                0 => values.push(Value::Null),
                1 => values.push(Value::Array(
                    inner[(list_size * i)..(list_size * (i + 1))].to_vec(),
                )),
                _ => panic!("Validity data should be 0 or 1"),
            },
            None => {}
        }
    }

    values
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::convert::TryFrom;
    use std::fs::File;
    use std::io::Read;
    use std::sync::Arc;

    use crate::buffer::Buffer;

    #[test]
    fn test_schema_equality() {
        let json = r#"
        {
            "fields": [
                {
                    "name": "c1",
                    "type": {"name": "int", "isSigned": true, "bitWidth": 32},
                    "nullable": true,
                    "children": []
                },
                {
                    "name": "c2",
                    "type": {"name": "floatingpoint", "precision": "DOUBLE"},
                    "nullable": true,
                    "children": []
                },
                {
                    "name": "c3",
                    "type": {"name": "utf8"},
                    "nullable": true,
                    "children": []
                },
                {
                    "name": "c4",
                    "type": {
                        "name": "list"
                    },
                    "nullable": true,
                    "children": [
                        {
                            "name": "item",
                            "type": {
                                "name": "int",
                                "isSigned": true,
                                "bitWidth": 32
                            },
                            "nullable": true,
                            "children": []
                        }
                    ]
                }
            ]
        }"#;
        let json_schema: ArrowJsonSchema = serde_json::from_str(json).unwrap();
        let schema = Schema::new(vec![
            Field::new("c1", DataType::Int32, true),
            Field::new("c2", DataType::Float64, true),
            Field::new("c3", DataType::Utf8, true),
            Field::new("c4", DataType::List(Box::new(DataType::Int32)), true),
        ]);
        assert!(json_schema.equals_schema(&schema));
    }

    #[test]
    fn test_arrow_data_equality() {
        let secs_tz = Some(Arc::new("Europe/Budapest".to_string()));
        let millis_tz = Some(Arc::new("America/New_York".to_string()));
        let micros_tz = Some(Arc::new("UTC".to_string()));
        let nanos_tz = Some(Arc::new("Africa/Johannesburg".to_string()));
        let schema = Schema::new(vec![
            Field::new("bools", DataType::Boolean, true),
            Field::new("int8s", DataType::Int8, true),
            Field::new("int16s", DataType::Int16, true),
            Field::new("int32s", DataType::Int32, true),
            Field::new("int64s", DataType::Int64, true),
            Field::new("uint8s", DataType::UInt8, true),
            Field::new("uint16s", DataType::UInt16, true),
            Field::new("uint32s", DataType::UInt32, true),
            Field::new("uint64s", DataType::UInt64, true),
            Field::new("float32s", DataType::Float32, true),
            Field::new("float64s", DataType::Float64, true),
            Field::new("date_days", DataType::Date32(DateUnit::Day), true),
            Field::new("date_millis", DataType::Date64(DateUnit::Millisecond), true),
            Field::new("time_secs", DataType::Time32(TimeUnit::Second), true),
            Field::new("time_millis", DataType::Time32(TimeUnit::Millisecond), true),
            Field::new("time_micros", DataType::Time64(TimeUnit::Microsecond), true),
            Field::new("time_nanos", DataType::Time64(TimeUnit::Nanosecond), true),
            Field::new("ts_secs", DataType::Timestamp(TimeUnit::Second, None), true),
            Field::new(
                "ts_millis",
                DataType::Timestamp(TimeUnit::Millisecond, None),
                true,
            ),
            Field::new(
                "ts_micros",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                true,
            ),
            Field::new(
                "ts_nanos",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                true,
            ),
            Field::new(
                "ts_secs_tz",
                DataType::Timestamp(TimeUnit::Second, secs_tz.clone()),
                true,
            ),
            Field::new(
                "ts_millis_tz",
                DataType::Timestamp(TimeUnit::Millisecond, millis_tz.clone()),
                true,
            ),
            Field::new(
                "ts_micros_tz",
                DataType::Timestamp(TimeUnit::Microsecond, micros_tz.clone()),
                true,
            ),
            Field::new(
                "ts_nanos_tz",
                DataType::Timestamp(TimeUnit::Nanosecond, nanos_tz.clone()),
                true,
            ),
            Field::new("utf8s", DataType::Utf8, true),
            Field::new("lists", DataType::List(Box::new(DataType::Int32)), true),
            Field::new(
                "structs",
                DataType::Struct(vec![
                    Field::new("int32s", DataType::Int32, true),
                    Field::new("utf8s", DataType::Utf8, true),
                ]),
                true,
            ),
        ]);

        let bools = BooleanArray::from(vec![Some(true), None, Some(false)]);
        let int8s = Int8Array::from(vec![Some(1), None, Some(3)]);
        let int16s = Int16Array::from(vec![Some(1), None, Some(3)]);
        let int32s = Int32Array::from(vec![Some(1), None, Some(3)]);
        let int64s = Int64Array::from(vec![Some(1), None, Some(3)]);
        let uint8s = UInt8Array::from(vec![Some(1), None, Some(3)]);
        let uint16s = UInt16Array::from(vec![Some(1), None, Some(3)]);
        let uint32s = UInt32Array::from(vec![Some(1), None, Some(3)]);
        let uint64s = UInt64Array::from(vec![Some(1), None, Some(3)]);
        let float32s = Float32Array::from(vec![Some(1.0), None, Some(3.0)]);
        let float64s = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
        let date_days = Date32Array::from(vec![Some(1196848), None, None]);
        let date_millis = Date64Array::from(vec![
            Some(167903550396207),
            Some(29923997007884),
            Some(30612271819236),
        ]);
        let time_secs =
            Time32SecondArray::from(vec![Some(27974), Some(78592), Some(43207)]);
        let time_millis = Time32MillisecondArray::from(vec![
            Some(6613125),
            Some(74667230),
            Some(52260079),
        ]);
        let time_micros =
            Time64MicrosecondArray::from(vec![Some(62522958593), None, None]);
        let time_nanos = Time64NanosecondArray::from(vec![
            Some(73380123595985),
            None,
            Some(16584393546415),
        ]);
        let ts_secs = TimestampSecondArray::from_opt_vec(
            vec![None, Some(193438817552), None],
            None,
        );
        let ts_millis = TimestampMillisecondArray::from_opt_vec(
            vec![None, Some(38606916383008), Some(58113709376587)],
            None,
        );
        let ts_micros =
            TimestampMicrosecondArray::from_opt_vec(vec![None, None, None], None);
        let ts_nanos = TimestampNanosecondArray::from_opt_vec(
            vec![None, None, Some(-6473623571954960143)],
            None,
        );
        let ts_secs_tz = TimestampSecondArray::from_opt_vec(
            vec![None, Some(193438817552), None],
            secs_tz,
        );
        let ts_millis_tz = TimestampMillisecondArray::from_opt_vec(
            vec![None, Some(38606916383008), Some(58113709376587)],
            millis_tz,
        );
        let ts_micros_tz =
            TimestampMicrosecondArray::from_opt_vec(vec![None, None, None], micros_tz);
        let ts_nanos_tz = TimestampNanosecondArray::from_opt_vec(
            vec![None, None, Some(-6473623571954960143)],
            nanos_tz,
        );
        let utf8s = StringArray::try_from(vec![Some("aa"), None, Some("bbb")]).unwrap();

        let value_data = Int32Array::from(vec![None, Some(2), None, None]);
        let value_offsets = Buffer::from(&[0, 3, 4, 4].to_byte_slice());
        let list_data_type = DataType::List(Box::new(DataType::Int32));
        let list_data = ArrayData::builder(list_data_type)
            .len(3)
            .add_buffer(value_offsets)
            .add_child_data(value_data.data())
            .build();
        let lists = ListArray::from(list_data);

        let structs_int32s = Int32Array::from(vec![None, Some(-2), None]);
        let structs_utf8s =
            StringArray::try_from(vec![None, None, Some("aaaaaa")]).unwrap();
        let structs = StructArray::from(vec![
            (
                Field::new("int32s", DataType::Int32, true),
                Arc::new(structs_int32s) as ArrayRef,
            ),
            (
                Field::new("utf8s", DataType::Utf8, true),
                Arc::new(structs_utf8s) as ArrayRef,
            ),
        ]);

        let record_batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(bools),
                Arc::new(int8s),
                Arc::new(int16s),
                Arc::new(int32s),
                Arc::new(int64s),
                Arc::new(uint8s),
                Arc::new(uint16s),
                Arc::new(uint32s),
                Arc::new(uint64s),
                Arc::new(float32s),
                Arc::new(float64s),
                Arc::new(date_days),
                Arc::new(date_millis),
                Arc::new(time_secs),
                Arc::new(time_millis),
                Arc::new(time_micros),
                Arc::new(time_nanos),
                Arc::new(ts_secs),
                Arc::new(ts_millis),
                Arc::new(ts_micros),
                Arc::new(ts_nanos),
                Arc::new(ts_secs_tz),
                Arc::new(ts_millis_tz),
                Arc::new(ts_micros_tz),
                Arc::new(ts_nanos_tz),
                Arc::new(utf8s),
                Arc::new(lists),
                Arc::new(structs),
            ],
        )
        .unwrap();
        let mut file = File::open("test/data/integration.json").unwrap();
        let mut json = String::new();
        file.read_to_string(&mut json).unwrap();
        let arrow_json: ArrowJson = serde_json::from_str(&json).unwrap();
        // test schemas
        assert!(arrow_json.schema.equals_schema(&schema));
        // test record batch
        assert!(arrow_json.batches[0].equals_batch(&record_batch));
    }
}
