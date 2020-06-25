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

///! Many builders are available to easily create different types of some arrays
extern crate arrow;

use std::sync::Arc;

use arrow::array::{
    Array, ArrayData, BooleanArray, Int32Array, Int32Builder, ListArray, PrimitiveArray,
    StringArray, StructArray,
};
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType, Date64Type, Field, Time64NanosecondType, ToByteSlice};

fn main() {
    // Primitive Arrays
    //
    // Primitive arrays are arrays of fixed-width primitive types (bool, u8, u16, u32,
    // u64, i8, i16, i32, i64, f32, f64)

    // Create a new builder with a capacity of 100
    let mut primitive_array_builder = Int32Builder::new(100);

    // Append an individual primitive value
    primitive_array_builder.append_value(55).unwrap();

    // Append a null value
    primitive_array_builder.append_null().unwrap();

    // Append a slice of primitive values
    primitive_array_builder.append_slice(&[39, 89, 12]).unwrap();

    // Append lots of values
    primitive_array_builder.append_null().unwrap();
    primitive_array_builder
        .append_slice(&(25..50).collect::<Vec<i32>>())
        .unwrap();

    // Build the `PrimitiveArray`
    let primitive_array = primitive_array_builder.finish();
    // Long arrays will have an ellipsis printed in the middle
    println!("{:?}", primitive_array);

    // Arrays can also be built from `Vec<Option<T>>`. `None`
    // represents a null value in the array.
    let date_array: PrimitiveArray<Date64Type> =
        vec![Some(1550902545147), None, Some(1550902545147)].into();
    println!("{:?}", date_array);

    let time_array: PrimitiveArray<Time64NanosecondType> =
        (0..100).collect::<Vec<i64>>().into();
    println!("{:?}", time_array);

    // We can build arrays directly from the underlying buffers.

    // BinaryArrays are arrays of byte arrays, where each byte array
    // is a slice of an underlying buffer.

    // Array data: ["hello", null, "parquet"]
    let values: [u8; 12] = [
        b'h', b'e', b'l', b'l', b'o', b'p', b'a', b'r', b'q', b'u', b'e', b't',
    ];
    let offsets: [i32; 4] = [0, 5, 5, 12];

    let array_data = ArrayData::builder(DataType::Utf8)
        .len(3)
        .add_buffer(Buffer::from(offsets.to_byte_slice()))
        .add_buffer(Buffer::from(&values[..]))
        .null_bit_buffer(Buffer::from([0b00000101]))
        .build();
    let binary_array = StringArray::from(array_data);
    println!("{:?}", binary_array);

    // ListArrays are similar to ByteArrays: they are arrays of other
    // arrays, where each child array is a slice of the underlying
    // buffer.
    let value_data = ArrayData::builder(DataType::Int32)
        .len(8)
        .add_buffer(Buffer::from(&[0, 1, 2, 3, 4, 5, 6, 7].to_byte_slice()))
        .build();

    // Construct a buffer for value offsets, for the nested array:
    //  [[0, 1, 2], [3, 4, 5], [6, 7]]
    let value_offsets = Buffer::from(&[0, 3, 6, 8].to_byte_slice());

    // Construct a list array from the above two
    let list_data_type = DataType::List(Box::new(DataType::Int32));
    let list_data = ArrayData::builder(list_data_type.clone())
        .len(3)
        .add_buffer(value_offsets.clone())
        .add_child_data(value_data.clone())
        .build();
    let list_array = ListArray::from(list_data);

    println!("{:?}", list_array);

    // StructArrays are arrays of tuples, where each tuple element is
    // from a child array. (In other words, they're like zipping
    // multiple columns into one and giving each subcolumn a label.)

    // StructArrays can be constructed using the StructArray::from
    // helper, which takes the underlying arrays and field types.
    let struct_array = StructArray::from(vec![
        (
            Field::new("b", DataType::Boolean, false),
            Arc::new(BooleanArray::from(vec![false, false, true, true]))
                as Arc<dyn Array>,
        ),
        (
            Field::new("c", DataType::Int32, false),
            Arc::new(Int32Array::from(vec![42, 28, 19, 31])),
        ),
    ]);
    println!("{:?}", struct_array);
}
