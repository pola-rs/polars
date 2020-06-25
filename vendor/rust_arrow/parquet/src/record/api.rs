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

//! Contains Row enum that is used to represent record in Rust.

use std::fmt;

use chrono::{Local, TimeZone};
use num_bigint::{BigInt, Sign};

use crate::basic::{LogicalType, Type as PhysicalType};
use crate::data_type::{ByteArray, Decimal, Int96};
use crate::errors::{ParquetError, Result};
use crate::schema::types::ColumnDescPtr;

/// Macro as a shortcut to generate 'not yet implemented' panic error.
macro_rules! nyi {
    ($column_descr:ident, $value:ident) => {{
        unimplemented!(
            "Conversion for physical type {}, logical type {}, value {:?}",
            $column_descr.physical_type(),
            $column_descr.logical_type(),
            $value
        );
    }};
}

/// `Row` represents a nested Parquet record.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    fields: Vec<(String, Field)>,
}

impl Row {
    /// Get the number of fields in this row.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Get an iterator to go through all columns in the row.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::fs::File;
    /// use parquet::record::Row;
    /// use parquet::file::reader::{FileReader, SerializedFileReader};
    ///
    /// let file = File::open("/path/to/file").unwrap();
    /// let reader = SerializedFileReader::new(file).unwrap();
    /// let row: Row = reader.get_row_iter(None).unwrap().next().unwrap();
    /// for (idx, (name, field)) in row.get_column_iter().enumerate() {
    ///     println!("column index: {}, column name: {}, column value: {}", idx, name, field);
    /// }
    /// ```
    pub fn get_column_iter(&self) -> RowColumnIter {
        RowColumnIter {
            fields: &self.fields,
            curr: 0,
            count: self.fields.len(),
        }
    }
}

pub struct RowColumnIter<'a> {
    fields: &'a Vec<(String, Field)>,
    curr: usize,
    count: usize,
}

impl<'a> Iterator for RowColumnIter<'a> {
    type Item = (&'a String, &'a Field);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.curr;
        if idx >= self.count {
            return None;
        }
        self.curr += 1;
        Some((&self.fields[idx].0, &self.fields[idx].1))
    }
}

/// Trait for type-safe convenient access to fields within a Row.
pub trait RowAccessor {
    fn get_bool(&self, i: usize) -> Result<bool>;
    fn get_byte(&self, i: usize) -> Result<i8>;
    fn get_short(&self, i: usize) -> Result<i16>;
    fn get_int(&self, i: usize) -> Result<i32>;
    fn get_long(&self, i: usize) -> Result<i64>;
    fn get_ubyte(&self, i: usize) -> Result<u8>;
    fn get_ushort(&self, i: usize) -> Result<u16>;
    fn get_uint(&self, i: usize) -> Result<u32>;
    fn get_ulong(&self, i: usize) -> Result<u64>;
    fn get_float(&self, i: usize) -> Result<f32>;
    fn get_double(&self, i: usize) -> Result<f64>;
    fn get_timestamp_millis(&self, i: usize) -> Result<u64>;
    fn get_timestamp_micros(&self, i: usize) -> Result<u64>;
    fn get_decimal(&self, i: usize) -> Result<&Decimal>;
    fn get_string(&self, i: usize) -> Result<&String>;
    fn get_bytes(&self, i: usize) -> Result<&ByteArray>;
    fn get_group(&self, i: usize) -> Result<&Row>;
    fn get_list(&self, i: usize) -> Result<&List>;
    fn get_map(&self, i: usize) -> Result<&Map>;
}

/// Trait for formating fields within a Row.
pub trait RowFormatter {
    fn fmt(&self, i: usize) -> &fmt::Display;
}

/// Macro to generate type-safe get_xxx methods for primitive types,
/// e.g. `get_bool`, `get_short`.
macro_rules! row_primitive_accessor {
    ($METHOD:ident, $VARIANT:ident, $TY:ty) => {
        fn $METHOD(&self, i: usize) -> Result<$TY> {
            match self.fields[i].1 {
                Field::$VARIANT(v) => Ok(v),
                _ => Err(general_err!(
                    "Cannot access {} as {}",
                    self.fields[i].1.get_type_name(),
                    stringify!($VARIANT)
                )),
            }
        }
    };
}

/// Macro to generate type-safe get_xxx methods for reference types,
/// e.g. `get_list`, `get_map`.
macro_rules! row_complex_accessor {
    ($METHOD:ident, $VARIANT:ident, $TY:ty) => {
        fn $METHOD(&self, i: usize) -> Result<&$TY> {
            match self.fields[i].1 {
                Field::$VARIANT(ref v) => Ok(v),
                _ => Err(general_err!(
                    "Cannot access {} as {}",
                    self.fields[i].1.get_type_name(),
                    stringify!($VARIANT)
                )),
            }
        }
    };
}

impl RowFormatter for Row {
    /// Get Display reference for a given field.
    fn fmt(&self, i: usize) -> &fmt::Display {
        &self.fields[i].1
    }
}

impl RowAccessor for Row {
    row_primitive_accessor!(get_bool, Bool, bool);

    row_primitive_accessor!(get_byte, Byte, i8);

    row_primitive_accessor!(get_short, Short, i16);

    row_primitive_accessor!(get_int, Int, i32);

    row_primitive_accessor!(get_long, Long, i64);

    row_primitive_accessor!(get_ubyte, UByte, u8);

    row_primitive_accessor!(get_ushort, UShort, u16);

    row_primitive_accessor!(get_uint, UInt, u32);

    row_primitive_accessor!(get_ulong, ULong, u64);

    row_primitive_accessor!(get_float, Float, f32);

    row_primitive_accessor!(get_double, Double, f64);

    row_primitive_accessor!(get_timestamp_millis, TimestampMillis, u64);

    row_primitive_accessor!(get_timestamp_micros, TimestampMicros, u64);

    row_complex_accessor!(get_decimal, Decimal, Decimal);

    row_complex_accessor!(get_string, Str, String);

    row_complex_accessor!(get_bytes, Bytes, ByteArray);

    row_complex_accessor!(get_group, Group, Row);

    row_complex_accessor!(get_list, ListInternal, List);

    row_complex_accessor!(get_map, MapInternal, Map);
}

/// Constructs a `Row` from the list of `fields` and returns it.
#[inline]
pub fn make_row(fields: Vec<(String, Field)>) -> Row {
    Row { fields }
}

impl fmt::Display for Row {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{")?;
        for (i, &(ref key, ref value)) in self.fields.iter().enumerate() {
            key.fmt(f)?;
            write!(f, ": ")?;
            value.fmt(f)?;
            if i < self.fields.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "}}")
    }
}

/// `List` represents a list which contains an array of elements.
#[derive(Clone, Debug, PartialEq)]
pub struct List {
    elements: Vec<Field>,
}

impl List {
    /// Get the number of fields in this row
    pub fn len(&self) -> usize {
        self.elements.len()
    }
}

/// Constructs a `List` from the list of `fields` and returns it.
#[inline]
pub fn make_list(elements: Vec<Field>) -> List {
    List { elements }
}

/// Trait for type-safe access of an index for a `List`.
/// Note that the get_XXX methods do not do bound checking.
pub trait ListAccessor {
    fn get_bool(&self, i: usize) -> Result<bool>;
    fn get_byte(&self, i: usize) -> Result<i8>;
    fn get_short(&self, i: usize) -> Result<i16>;
    fn get_int(&self, i: usize) -> Result<i32>;
    fn get_long(&self, i: usize) -> Result<i64>;
    fn get_ubyte(&self, i: usize) -> Result<u8>;
    fn get_ushort(&self, i: usize) -> Result<u16>;
    fn get_uint(&self, i: usize) -> Result<u32>;
    fn get_ulong(&self, i: usize) -> Result<u64>;
    fn get_float(&self, i: usize) -> Result<f32>;
    fn get_double(&self, i: usize) -> Result<f64>;
    fn get_timestamp_millis(&self, i: usize) -> Result<u64>;
    fn get_timestamp_micros(&self, i: usize) -> Result<u64>;
    fn get_decimal(&self, i: usize) -> Result<&Decimal>;
    fn get_string(&self, i: usize) -> Result<&String>;
    fn get_bytes(&self, i: usize) -> Result<&ByteArray>;
    fn get_group(&self, i: usize) -> Result<&Row>;
    fn get_list(&self, i: usize) -> Result<&List>;
    fn get_map(&self, i: usize) -> Result<&Map>;
}

/// Macro to generate type-safe get_xxx methods for primitive types,
/// e.g. get_bool, get_short
macro_rules! list_primitive_accessor {
    ($METHOD:ident, $VARIANT:ident, $TY:ty) => {
        fn $METHOD(&self, i: usize) -> Result<$TY> {
            match self.elements[i] {
                Field::$VARIANT(v) => Ok(v),
                _ => Err(general_err!(
                    "Cannot access {} as {}",
                    self.elements[i].get_type_name(),
                    stringify!($VARIANT)
                )),
            }
        }
    };
}

/// Macro to generate type-safe get_xxx methods for reference types
/// e.g. get_list, get_map
macro_rules! list_complex_accessor {
    ($METHOD:ident, $VARIANT:ident, $TY:ty) => {
        fn $METHOD(&self, i: usize) -> Result<&$TY> {
            match self.elements[i] {
                Field::$VARIANT(ref v) => Ok(v),
                _ => Err(general_err!(
                    "Cannot access {} as {}",
                    self.elements[i].get_type_name(),
                    stringify!($VARIANT)
                )),
            }
        }
    };
}

impl ListAccessor for List {
    list_primitive_accessor!(get_bool, Bool, bool);

    list_primitive_accessor!(get_byte, Byte, i8);

    list_primitive_accessor!(get_short, Short, i16);

    list_primitive_accessor!(get_int, Int, i32);

    list_primitive_accessor!(get_long, Long, i64);

    list_primitive_accessor!(get_ubyte, UByte, u8);

    list_primitive_accessor!(get_ushort, UShort, u16);

    list_primitive_accessor!(get_uint, UInt, u32);

    list_primitive_accessor!(get_ulong, ULong, u64);

    list_primitive_accessor!(get_float, Float, f32);

    list_primitive_accessor!(get_double, Double, f64);

    list_primitive_accessor!(get_timestamp_millis, TimestampMillis, u64);

    list_primitive_accessor!(get_timestamp_micros, TimestampMicros, u64);

    list_complex_accessor!(get_decimal, Decimal, Decimal);

    list_complex_accessor!(get_string, Str, String);

    list_complex_accessor!(get_bytes, Bytes, ByteArray);

    list_complex_accessor!(get_group, Group, Row);

    list_complex_accessor!(get_list, ListInternal, List);

    list_complex_accessor!(get_map, MapInternal, Map);
}

/// `Map` represents a map which contains a list of key->value pairs.
#[derive(Clone, Debug, PartialEq)]
pub struct Map {
    entries: Vec<(Field, Field)>,
}

impl Map {
    /// Get the number of fields in this row
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Constructs a `Map` from the list of `entries` and returns it.
#[inline]
pub fn make_map(entries: Vec<(Field, Field)>) -> Map {
    Map { entries }
}

/// Trait for type-safe access of an index for a `Map`
pub trait MapAccessor {
    fn get_keys<'a>(&'a self) -> Box<ListAccessor + 'a>;
    fn get_values<'a>(&'a self) -> Box<ListAccessor + 'a>;
}

struct MapList<'a> {
    elements: Vec<&'a Field>,
}

/// Macro to generate type-safe get_xxx methods for primitive types,
/// e.g. get_bool, get_short
macro_rules! map_list_primitive_accessor {
    ($METHOD:ident, $VARIANT:ident, $TY:ty) => {
        fn $METHOD(&self, i: usize) -> Result<$TY> {
            match self.elements[i] {
                Field::$VARIANT(v) => Ok(*v),
                _ => Err(general_err!(
                    "Cannot access {} as {}",
                    self.elements[i].get_type_name(),
                    stringify!($VARIANT)
                )),
            }
        }
    };
}

impl<'a> ListAccessor for MapList<'a> {
    map_list_primitive_accessor!(get_bool, Bool, bool);

    map_list_primitive_accessor!(get_byte, Byte, i8);

    map_list_primitive_accessor!(get_short, Short, i16);

    map_list_primitive_accessor!(get_int, Int, i32);

    map_list_primitive_accessor!(get_long, Long, i64);

    map_list_primitive_accessor!(get_ubyte, UByte, u8);

    map_list_primitive_accessor!(get_ushort, UShort, u16);

    map_list_primitive_accessor!(get_uint, UInt, u32);

    map_list_primitive_accessor!(get_ulong, ULong, u64);

    map_list_primitive_accessor!(get_float, Float, f32);

    map_list_primitive_accessor!(get_double, Double, f64);

    map_list_primitive_accessor!(get_timestamp_millis, TimestampMillis, u64);

    map_list_primitive_accessor!(get_timestamp_micros, TimestampMicros, u64);

    list_complex_accessor!(get_decimal, Decimal, Decimal);

    list_complex_accessor!(get_string, Str, String);

    list_complex_accessor!(get_bytes, Bytes, ByteArray);

    list_complex_accessor!(get_group, Group, Row);

    list_complex_accessor!(get_list, ListInternal, List);

    list_complex_accessor!(get_map, MapInternal, Map);
}

impl MapAccessor for Map {
    fn get_keys<'a>(&'a self) -> Box<ListAccessor + 'a> {
        let map_list = MapList {
            elements: self.entries.iter().map(|v| &v.0).collect(),
        };
        Box::new(map_list)
    }

    fn get_values<'a>(&'a self) -> Box<ListAccessor + 'a> {
        let map_list = MapList {
            elements: self.entries.iter().map(|v| &v.1).collect(),
        };
        Box::new(map_list)
    }
}

/// API to represent a single field in a `Row`.
#[derive(Clone, Debug, PartialEq)]
pub enum Field {
    // Primitive types
    /// Null value.
    Null,
    /// Boolean value (`true`, `false`).
    Bool(bool),
    /// Signed integer INT_8.
    Byte(i8),
    /// Signed integer INT_16.
    Short(i16),
    /// Signed integer INT_32.
    Int(i32),
    /// Signed integer INT_64.
    Long(i64),
    // Unsigned integer UINT_8.
    UByte(u8),
    // Unsigned integer UINT_16.
    UShort(u16),
    // Unsigned integer UINT_32.
    UInt(u32),
    // Unsigned integer UINT_64.
    ULong(u64),
    /// IEEE 32-bit floating point value.
    Float(f32),
    /// IEEE 64-bit floating point value.
    Double(f64),
    /// Decimal value.
    Decimal(Decimal),
    /// UTF-8 encoded character string.
    Str(String),
    /// General binary value.
    Bytes(ByteArray),
    /// Date without a time of day, stores the number of days from the
    /// Unix epoch, 1 January 1970.
    Date(u32),
    /// Milliseconds from the Unix epoch, 1 January 1970.
    TimestampMillis(u64),
    /// Microseconds from the Unix epoch, 1 Janiary 1970.
    TimestampMicros(u64),

    // ----------------------------------------------------------------------
    // Complex types
    /// Struct, child elements are tuples of field-value pairs.
    Group(Row),
    /// List of elements.
    ListInternal(List),
    /// List of key-value pairs.
    MapInternal(Map),
}

impl Field {
    /// Get the type name.
    fn get_type_name(&self) -> &'static str {
        match *self {
            Field::Null => "Null",
            Field::Bool(_) => "Bool",
            Field::Byte(_) => "Byte",
            Field::Short(_) => "Short",
            Field::Int(_) => "Int",
            Field::Long(_) => "Long",
            Field::UByte(_) => "UByte",
            Field::UShort(_) => "UShort",
            Field::UInt(_) => "UInt",
            Field::ULong(_) => "ULong",
            Field::Float(_) => "Float",
            Field::Double(_) => "Double",
            Field::Decimal(_) => "Decimal",
            Field::Date(_) => "Date",
            Field::Str(_) => "Str",
            Field::Bytes(_) => "Bytes",
            Field::TimestampMillis(_) => "TimestampMillis",
            Field::TimestampMicros(_) => "TimestampMicros",
            Field::Group(_) => "Group",
            Field::ListInternal(_) => "ListInternal",
            Field::MapInternal(_) => "MapInternal",
        }
    }

    /// Determines if this Row represents a primitive value.
    pub fn is_primitive(&self) -> bool {
        match *self {
            Field::Group(_) => false,
            Field::ListInternal(_) => false,
            Field::MapInternal(_) => false,
            _ => true,
        }
    }

    /// Converts Parquet BOOLEAN type with logical type into `bool` value.
    #[inline]
    pub fn convert_bool(_descr: &ColumnDescPtr, value: bool) -> Self {
        Field::Bool(value)
    }

    /// Converts Parquet INT32 type with logical type into `i32` value.
    #[inline]
    pub fn convert_int32(descr: &ColumnDescPtr, value: i32) -> Self {
        match descr.logical_type() {
            LogicalType::INT_8 => Field::Byte(value as i8),
            LogicalType::INT_16 => Field::Short(value as i16),
            LogicalType::INT_32 | LogicalType::NONE => Field::Int(value),
            LogicalType::UINT_8 => Field::UByte(value as u8),
            LogicalType::UINT_16 => Field::UShort(value as u16),
            LogicalType::UINT_32 => Field::UInt(value as u32),
            LogicalType::DATE => Field::Date(value as u32),
            LogicalType::DECIMAL => Field::Decimal(Decimal::from_i32(
                value,
                descr.type_precision(),
                descr.type_scale(),
            )),
            _ => nyi!(descr, value),
        }
    }

    /// Converts Parquet INT64 type with logical type into `i64` value.
    #[inline]
    pub fn convert_int64(descr: &ColumnDescPtr, value: i64) -> Self {
        match descr.logical_type() {
            LogicalType::INT_64 | LogicalType::NONE => Field::Long(value),
            LogicalType::UINT_64 => Field::ULong(value as u64),
            LogicalType::TIMESTAMP_MILLIS => Field::TimestampMillis(value as u64),
            LogicalType::TIMESTAMP_MICROS => Field::TimestampMicros(value as u64),
            LogicalType::DECIMAL => Field::Decimal(Decimal::from_i64(
                value,
                descr.type_precision(),
                descr.type_scale(),
            )),
            _ => nyi!(descr, value),
        }
    }

    /// Converts Parquet INT96 (nanosecond timestamps) type and logical type into
    /// `Timestamp` value.
    #[inline]
    pub fn convert_int96(_descr: &ColumnDescPtr, value: Int96) -> Self {
        Field::TimestampMillis(value.to_i64() as u64)
    }

    /// Converts Parquet FLOAT type with logical type into `f32` value.
    #[inline]
    pub fn convert_float(_descr: &ColumnDescPtr, value: f32) -> Self {
        Field::Float(value)
    }

    /// Converts Parquet DOUBLE type with logical type into `f64` value.
    #[inline]
    pub fn convert_double(_descr: &ColumnDescPtr, value: f64) -> Self {
        Field::Double(value)
    }

    /// Converts Parquet BYTE_ARRAY type with logical type into either UTF8 string or
    /// array of bytes.
    #[inline]
    pub fn convert_byte_array(descr: &ColumnDescPtr, value: ByteArray) -> Self {
        match descr.physical_type() {
            PhysicalType::BYTE_ARRAY => match descr.logical_type() {
                LogicalType::UTF8 | LogicalType::ENUM | LogicalType::JSON => {
                    let value = String::from_utf8(value.data().to_vec()).unwrap();
                    Field::Str(value)
                }
                LogicalType::BSON | LogicalType::NONE => Field::Bytes(value),
                LogicalType::DECIMAL => Field::Decimal(Decimal::from_bytes(
                    value,
                    descr.type_precision(),
                    descr.type_scale(),
                )),
                _ => nyi!(descr, value),
            },
            PhysicalType::FIXED_LEN_BYTE_ARRAY => match descr.logical_type() {
                LogicalType::DECIMAL => Field::Decimal(Decimal::from_bytes(
                    value,
                    descr.type_precision(),
                    descr.type_scale(),
                )),
                LogicalType::NONE => Field::Bytes(value),
                _ => nyi!(descr, value),
            },
            _ => nyi!(descr, value),
        }
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Field::Null => write!(f, "null"),
            Field::Bool(value) => write!(f, "{}", value),
            Field::Byte(value) => write!(f, "{}", value),
            Field::Short(value) => write!(f, "{}", value),
            Field::Int(value) => write!(f, "{}", value),
            Field::Long(value) => write!(f, "{}", value),
            Field::UByte(value) => write!(f, "{}", value),
            Field::UShort(value) => write!(f, "{}", value),
            Field::UInt(value) => write!(f, "{}", value),
            Field::ULong(value) => write!(f, "{}", value),
            Field::Float(value) => {
                if value > 1e19 || value < 1e-15 {
                    write!(f, "{:E}", value)
                } else {
                    write!(f, "{:?}", value)
                }
            }
            Field::Double(value) => {
                if value > 1e19 || value < 1e-15 {
                    write!(f, "{:E}", value)
                } else {
                    write!(f, "{:?}", value)
                }
            }
            Field::Decimal(ref value) => {
                write!(f, "{}", convert_decimal_to_string(value))
            }
            Field::Str(ref value) => write!(f, "\"{}\"", value),
            Field::Bytes(ref value) => write!(f, "{:?}", value.data()),
            Field::Date(value) => write!(f, "{}", convert_date_to_string(value)),
            Field::TimestampMillis(value) => {
                write!(f, "{}", convert_timestamp_millis_to_string(value))
            }
            Field::TimestampMicros(value) => {
                write!(f, "{}", convert_timestamp_micros_to_string(value))
            }
            Field::Group(ref fields) => write!(f, "{}", fields),
            Field::ListInternal(ref list) => {
                let elems = &list.elements;
                write!(f, "[")?;
                for (i, field) in elems.iter().enumerate() {
                    field.fmt(f)?;
                    if i < elems.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, "]")
            }
            Field::MapInternal(ref map) => {
                let entries = &map.entries;
                write!(f, "{{")?;
                for (i, &(ref key, ref value)) in entries.iter().enumerate() {
                    key.fmt(f)?;
                    write!(f, " -> ")?;
                    value.fmt(f)?;
                    if i < entries.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, "}}")
            }
        }
    }
}

/// Helper method to convert Parquet date into a string.
/// Input `value` is a number of days since the epoch in UTC.
/// Date is displayed in local timezone.
#[inline]
fn convert_date_to_string(value: u32) -> String {
    static NUM_SECONDS_IN_DAY: i64 = 60 * 60 * 24;
    let dt = Local.timestamp(value as i64 * NUM_SECONDS_IN_DAY, 0).date();
    format!("{}", dt.format("%Y-%m-%d %:z"))
}

/// Helper method to convert Parquet timestamp into a string.
/// Input `value` is a number of milliseconds since the epoch in UTC.
/// Datetime is displayed in local timezone.
#[inline]
fn convert_timestamp_millis_to_string(value: u64) -> String {
    let dt = Local.timestamp((value / 1000) as i64, 0);
    format!("{}", dt.format("%Y-%m-%d %H:%M:%S %:z"))
}

/// Helper method to convert Parquet timestamp into a string.
/// Input `value` is a number of microseconds since the epoch in UTC.
/// Datetime is displayed in local timezone.
#[inline]
fn convert_timestamp_micros_to_string(value: u64) -> String {
    convert_timestamp_millis_to_string(value / 1000)
}

/// Helper method to convert Parquet decimal into a string.
/// We assert that `scale >= 0` and `precision > scale`, but this will be enforced
/// when constructing Parquet schema.
#[inline]
fn convert_decimal_to_string(decimal: &Decimal) -> String {
    assert!(decimal.scale() >= 0 && decimal.precision() > decimal.scale());

    // Specify as signed bytes to resolve sign as part of conversion.
    let num = BigInt::from_signed_bytes_be(decimal.data());

    // Offset of the first digit in a string.
    let negative = if num.sign() == Sign::Minus { 1 } else { 0 };
    let mut num_str = num.to_string();
    let mut point = num_str.len() as i32 - decimal.scale() - negative;

    // Convert to string form without scientific notation.
    if point <= 0 {
        // Zeros need to be prepended to the unscaled value.
        while point < 0 {
            num_str.insert(negative as usize, '0');
            point += 1;
        }
        num_str.insert_str(negative as usize, "0.");
    } else {
        // No zeroes need to be prepended to the unscaled value, simply insert decimal
        // point.
        num_str.insert((point + negative) as usize, '.');
    }

    num_str
}

#[cfg(test)]
mod tests {
    use super::*;

    use chrono;
    use std::rc::Rc;

    use crate::schema::types::{ColumnDescriptor, ColumnPath, PrimitiveTypeBuilder};

    /// Creates test column descriptor based on provided type parameters.
    macro_rules! make_column_descr {
        ($physical_type:expr, $logical_type:expr) => {{
            let tpe = PrimitiveTypeBuilder::new("col", $physical_type)
                .with_logical_type($logical_type)
                .build()
                .unwrap();
            Rc::new(ColumnDescriptor::new(
                Rc::new(tpe),
                None,
                0,
                0,
                ColumnPath::from("col"),
            ))
        }};
        ($physical_type:expr, $logical_type:expr, $len:expr, $prec:expr, $scale:expr) => {{
            let tpe = PrimitiveTypeBuilder::new("col", $physical_type)
                .with_logical_type($logical_type)
                .with_length($len)
                .with_precision($prec)
                .with_scale($scale)
                .build()
                .unwrap();
            Rc::new(ColumnDescriptor::new(
                Rc::new(tpe),
                None,
                0,
                0,
                ColumnPath::from("col"),
            ))
        }};
    }

    #[test]
    fn test_row_convert_bool() {
        // BOOLEAN value does not depend on logical type
        let descr = make_column_descr![PhysicalType::BOOLEAN, LogicalType::NONE];

        let row = Field::convert_bool(&descr, true);
        assert_eq!(row, Field::Bool(true));

        let row = Field::convert_bool(&descr, false);
        assert_eq!(row, Field::Bool(false));
    }

    #[test]
    fn test_row_convert_int32() {
        let descr = make_column_descr![PhysicalType::INT32, LogicalType::INT_8];
        let row = Field::convert_int32(&descr, 111);
        assert_eq!(row, Field::Byte(111));

        let descr = make_column_descr![PhysicalType::INT32, LogicalType::INT_16];
        let row = Field::convert_int32(&descr, 222);
        assert_eq!(row, Field::Short(222));

        let descr = make_column_descr![PhysicalType::INT32, LogicalType::INT_32];
        let row = Field::convert_int32(&descr, 333);
        assert_eq!(row, Field::Int(333));

        let descr = make_column_descr![PhysicalType::INT32, LogicalType::UINT_8];
        let row = Field::convert_int32(&descr, -1);
        assert_eq!(row, Field::UByte(255));

        let descr = make_column_descr![PhysicalType::INT32, LogicalType::UINT_16];
        let row = Field::convert_int32(&descr, 256);
        assert_eq!(row, Field::UShort(256));

        let descr = make_column_descr![PhysicalType::INT32, LogicalType::UINT_32];
        let row = Field::convert_int32(&descr, 1234);
        assert_eq!(row, Field::UInt(1234));

        let descr = make_column_descr![PhysicalType::INT32, LogicalType::NONE];
        let row = Field::convert_int32(&descr, 444);
        assert_eq!(row, Field::Int(444));

        let descr = make_column_descr![PhysicalType::INT32, LogicalType::DATE];
        let row = Field::convert_int32(&descr, 14611);
        assert_eq!(row, Field::Date(14611));

        let descr =
            make_column_descr![PhysicalType::INT32, LogicalType::DECIMAL, 0, 8, 2];
        let row = Field::convert_int32(&descr, 444);
        assert_eq!(row, Field::Decimal(Decimal::from_i32(444, 8, 2)));
    }

    #[test]
    fn test_row_convert_int64() {
        let descr = make_column_descr![PhysicalType::INT64, LogicalType::INT_64];
        let row = Field::convert_int64(&descr, 1111);
        assert_eq!(row, Field::Long(1111));

        let descr = make_column_descr![PhysicalType::INT64, LogicalType::UINT_64];
        let row = Field::convert_int64(&descr, 78239823);
        assert_eq!(row, Field::ULong(78239823));

        let descr =
            make_column_descr![PhysicalType::INT64, LogicalType::TIMESTAMP_MILLIS];
        let row = Field::convert_int64(&descr, 1541186529153);
        assert_eq!(row, Field::TimestampMillis(1541186529153));

        let descr =
            make_column_descr![PhysicalType::INT64, LogicalType::TIMESTAMP_MICROS];
        let row = Field::convert_int64(&descr, 1541186529153123);
        assert_eq!(row, Field::TimestampMicros(1541186529153123));

        let descr = make_column_descr![PhysicalType::INT64, LogicalType::NONE];
        let row = Field::convert_int64(&descr, 2222);
        assert_eq!(row, Field::Long(2222));

        let descr =
            make_column_descr![PhysicalType::INT64, LogicalType::DECIMAL, 0, 8, 2];
        let row = Field::convert_int64(&descr, 3333);
        assert_eq!(row, Field::Decimal(Decimal::from_i64(3333, 8, 2)));
    }

    #[test]
    fn test_row_convert_int96() {
        // INT96 value does not depend on logical type
        let descr = make_column_descr![PhysicalType::INT96, LogicalType::NONE];

        let value = Int96::from(vec![0, 0, 2454923]);
        let row = Field::convert_int96(&descr, value);
        assert_eq!(row, Field::TimestampMillis(1238544000000));

        let value = Int96::from(vec![4165425152, 13, 2454923]);
        let row = Field::convert_int96(&descr, value);
        assert_eq!(row, Field::TimestampMillis(1238544060000));
    }

    #[test]
    #[should_panic(expected = "Expected non-negative milliseconds when converting Int96")]
    fn test_row_convert_int96_invalid() {
        // INT96 value does not depend on logical type
        let descr = make_column_descr![PhysicalType::INT96, LogicalType::NONE];

        let value = Int96::from(vec![0, 0, 0]);
        Field::convert_int96(&descr, value);
    }

    #[test]
    fn test_row_convert_float() {
        // FLOAT value does not depend on logical type
        let descr = make_column_descr![PhysicalType::FLOAT, LogicalType::NONE];
        let row = Field::convert_float(&descr, 2.31);
        assert_eq!(row, Field::Float(2.31));
    }

    #[test]
    fn test_row_convert_double() {
        // DOUBLE value does not depend on logical type
        let descr = make_column_descr![PhysicalType::DOUBLE, LogicalType::NONE];
        let row = Field::convert_double(&descr, 1.56);
        assert_eq!(row, Field::Double(1.56));
    }

    #[test]
    fn test_row_convert_byte_array() {
        // UTF8
        let descr = make_column_descr![PhysicalType::BYTE_ARRAY, LogicalType::UTF8];
        let value = ByteArray::from(vec![b'A', b'B', b'C', b'D']);
        let row = Field::convert_byte_array(&descr, value);
        assert_eq!(row, Field::Str("ABCD".to_string()));

        // ENUM
        let descr = make_column_descr![PhysicalType::BYTE_ARRAY, LogicalType::ENUM];
        let value = ByteArray::from(vec![b'1', b'2', b'3']);
        let row = Field::convert_byte_array(&descr, value);
        assert_eq!(row, Field::Str("123".to_string()));

        // JSON
        let descr = make_column_descr![PhysicalType::BYTE_ARRAY, LogicalType::JSON];
        let value = ByteArray::from(vec![b'{', b'"', b'a', b'"', b':', b'1', b'}']);
        let row = Field::convert_byte_array(&descr, value);
        assert_eq!(row, Field::Str("{\"a\":1}".to_string()));

        // NONE
        let descr = make_column_descr![PhysicalType::BYTE_ARRAY, LogicalType::NONE];
        let value = ByteArray::from(vec![1, 2, 3, 4, 5]);
        let row = Field::convert_byte_array(&descr, value.clone());
        assert_eq!(row, Field::Bytes(value));

        // BSON
        let descr = make_column_descr![PhysicalType::BYTE_ARRAY, LogicalType::BSON];
        let value = ByteArray::from(vec![1, 2, 3, 4, 5]);
        let row = Field::convert_byte_array(&descr, value.clone());
        assert_eq!(row, Field::Bytes(value));

        // DECIMAL
        let descr =
            make_column_descr![PhysicalType::BYTE_ARRAY, LogicalType::DECIMAL, 0, 8, 2];
        let value = ByteArray::from(vec![207, 200]);
        let row = Field::convert_byte_array(&descr, value.clone());
        assert_eq!(row, Field::Decimal(Decimal::from_bytes(value, 8, 2)));

        // DECIMAL (FIXED_LEN_BYTE_ARRAY)
        let descr = make_column_descr![
            PhysicalType::FIXED_LEN_BYTE_ARRAY,
            LogicalType::DECIMAL,
            8,
            17,
            5
        ];
        let value = ByteArray::from(vec![0, 0, 0, 0, 0, 4, 147, 224]);
        let row = Field::convert_byte_array(&descr, value.clone());
        assert_eq!(row, Field::Decimal(Decimal::from_bytes(value, 17, 5)));

        // NONE (FIXED_LEN_BYTE_ARRAY)
        let descr = make_column_descr![
            PhysicalType::FIXED_LEN_BYTE_ARRAY,
            LogicalType::NONE,
            6,
            0,
            0
        ];
        let value = ByteArray::from(vec![1, 2, 3, 4, 5, 6]);
        let row = Field::convert_byte_array(&descr, value.clone());
        assert_eq!(row, Field::Bytes(value));
    }

    #[test]
    fn test_convert_date_to_string() {
        fn check_date_conversion(y: u32, m: u32, d: u32) {
            let datetime = chrono::NaiveDate::from_ymd(y as i32, m, d).and_hms(0, 0, 0);
            let dt = Local.from_utc_datetime(&datetime);
            let res = convert_date_to_string((dt.timestamp() / 60 / 60 / 24) as u32);
            let exp = format!("{}", dt.format("%Y-%m-%d %:z"));
            assert_eq!(res, exp);
        }

        check_date_conversion(2010, 01, 02);
        check_date_conversion(2014, 05, 01);
        check_date_conversion(2016, 02, 29);
        check_date_conversion(2017, 09, 12);
        check_date_conversion(2018, 03, 31);
    }

    #[test]
    fn test_convert_timestamp_to_string() {
        fn check_datetime_conversion(y: u32, m: u32, d: u32, h: u32, mi: u32, s: u32) {
            let datetime = chrono::NaiveDate::from_ymd(y as i32, m, d).and_hms(h, mi, s);
            let dt = Local.from_utc_datetime(&datetime);
            let res = convert_timestamp_millis_to_string(dt.timestamp_millis() as u64);
            let exp = format!("{}", dt.format("%Y-%m-%d %H:%M:%S %:z"));
            assert_eq!(res, exp);
        }

        check_datetime_conversion(2010, 01, 02, 13, 12, 54);
        check_datetime_conversion(2011, 01, 03, 08, 23, 01);
        check_datetime_conversion(2012, 04, 05, 11, 06, 32);
        check_datetime_conversion(2013, 05, 12, 16, 38, 00);
        check_datetime_conversion(2014, 11, 28, 21, 15, 12);
    }

    #[test]
    fn test_convert_float_to_string() {
        assert_eq!(format!("{}", Field::Float(1.0)), "1.0");
        assert_eq!(format!("{}", Field::Float(9.63)), "9.63");
        assert_eq!(format!("{}", Field::Float(1e-15)), "0.000000000000001");
        assert_eq!(format!("{}", Field::Float(1e-16)), "1E-16");
        assert_eq!(format!("{}", Field::Float(1e19)), "10000000000000000000.0");
        assert_eq!(format!("{}", Field::Float(1e20)), "1E20");
        assert_eq!(format!("{}", Field::Float(1.7976931E30)), "1.7976931E30");
        assert_eq!(format!("{}", Field::Float(-1.7976931E30)), "-1.7976931E30");
    }

    #[test]
    fn test_convert_double_to_string() {
        assert_eq!(format!("{}", Field::Double(1.0)), "1.0");
        assert_eq!(format!("{}", Field::Double(9.63)), "9.63");
        assert_eq!(format!("{}", Field::Double(1e-15)), "0.000000000000001");
        assert_eq!(format!("{}", Field::Double(1e-16)), "1E-16");
        assert_eq!(format!("{}", Field::Double(1e19)), "10000000000000000000.0");
        assert_eq!(format!("{}", Field::Double(1e20)), "1E20");
        assert_eq!(
            format!("{}", Field::Double(1.79769313486E308)),
            "1.79769313486E308"
        );
        assert_eq!(
            format!("{}", Field::Double(-1.79769313486E308)),
            "-1.79769313486E308"
        );
    }

    #[test]
    fn test_convert_decimal_to_string() {
        // Helper method to compare decimal
        fn check_decimal(bytes: Vec<u8>, precision: i32, scale: i32, res: &str) {
            let decimal = Decimal::from_bytes(ByteArray::from(bytes), precision, scale);
            assert_eq!(convert_decimal_to_string(&decimal), res);
        }

        // This example previously used to fail in some engines
        check_decimal(
            vec![0, 0, 0, 0, 0, 0, 0, 0, 13, 224, 182, 179, 167, 100, 0, 0],
            38,
            18,
            "1.000000000000000000",
        );
        check_decimal(
            vec![
                249, 233, 247, 16, 185, 192, 202, 223, 215, 165, 192, 166, 67, 72,
            ],
            36,
            28,
            "-12344.0242342304923409234234293432",
        );
        check_decimal(vec![0, 0, 0, 0, 0, 4, 147, 224], 17, 5, "3.00000");
        check_decimal(vec![0, 0, 0, 0, 1, 201, 195, 140], 18, 2, "300000.12");
        check_decimal(vec![207, 200], 10, 2, "-123.44");
        check_decimal(vec![207, 200], 10, 8, "-0.00012344");
    }

    #[test]
    fn test_row_display() {
        // Primitive types
        assert_eq!(format!("{}", Field::Null), "null");
        assert_eq!(format!("{}", Field::Bool(true)), "true");
        assert_eq!(format!("{}", Field::Bool(false)), "false");
        assert_eq!(format!("{}", Field::Byte(1)), "1");
        assert_eq!(format!("{}", Field::Short(2)), "2");
        assert_eq!(format!("{}", Field::Int(3)), "3");
        assert_eq!(format!("{}", Field::Long(4)), "4");
        assert_eq!(format!("{}", Field::UByte(1)), "1");
        assert_eq!(format!("{}", Field::UShort(2)), "2");
        assert_eq!(format!("{}", Field::UInt(3)), "3");
        assert_eq!(format!("{}", Field::ULong(4)), "4");
        assert_eq!(format!("{}", Field::Float(5.0)), "5.0");
        assert_eq!(format!("{}", Field::Float(5.1234)), "5.1234");
        assert_eq!(format!("{}", Field::Double(6.0)), "6.0");
        assert_eq!(format!("{}", Field::Double(6.1234)), "6.1234");
        assert_eq!(format!("{}", Field::Str("abc".to_string())), "\"abc\"");
        assert_eq!(
            format!("{}", Field::Bytes(ByteArray::from(vec![1, 2, 3]))),
            "[1, 2, 3]"
        );
        assert_eq!(
            format!("{}", Field::Date(14611)),
            convert_date_to_string(14611)
        );
        assert_eq!(
            format!("{}", Field::TimestampMillis(1262391174000)),
            convert_timestamp_millis_to_string(1262391174000)
        );
        assert_eq!(
            format!("{}", Field::TimestampMicros(1262391174000000)),
            convert_timestamp_micros_to_string(1262391174000000)
        );
        assert_eq!(
            format!("{}", Field::Decimal(Decimal::from_i32(4, 8, 2))),
            convert_decimal_to_string(&Decimal::from_i32(4, 8, 2))
        );

        // Complex types
        let fields = vec![
            ("x".to_string(), Field::Null),
            ("Y".to_string(), Field::Int(2)),
            ("z".to_string(), Field::Float(3.1)),
            ("a".to_string(), Field::Str("abc".to_string())),
        ];
        let row = Field::Group(make_row(fields));
        assert_eq!(format!("{}", row), "{x: null, Y: 2, z: 3.1, a: \"abc\"}");

        let row = Field::ListInternal(make_list(vec![
            Field::Int(2),
            Field::Int(1),
            Field::Null,
            Field::Int(12),
        ]));
        assert_eq!(format!("{}", row), "[2, 1, null, 12]");

        let row = Field::MapInternal(make_map(vec![
            (Field::Int(1), Field::Float(1.2)),
            (Field::Int(2), Field::Float(4.5)),
            (Field::Int(3), Field::Float(2.3)),
        ]));
        assert_eq!(format!("{}", row), "{1 -> 1.2, 2 -> 4.5, 3 -> 2.3}");
    }

    #[test]
    fn test_is_primitive() {
        // primitives
        assert!(Field::Null.is_primitive());
        assert!(Field::Bool(true).is_primitive());
        assert!(Field::Bool(false).is_primitive());
        assert!(Field::Byte(1).is_primitive());
        assert!(Field::Short(2).is_primitive());
        assert!(Field::Int(3).is_primitive());
        assert!(Field::Long(4).is_primitive());
        assert!(Field::UByte(1).is_primitive());
        assert!(Field::UShort(2).is_primitive());
        assert!(Field::UInt(3).is_primitive());
        assert!(Field::ULong(4).is_primitive());
        assert!(Field::Float(5.0).is_primitive());
        assert!(Field::Float(5.1234).is_primitive());
        assert!(Field::Double(6.0).is_primitive());
        assert!(Field::Double(6.1234).is_primitive());
        assert!(Field::Str("abc".to_string()).is_primitive());
        assert!(Field::Bytes(ByteArray::from(vec![1, 2, 3])).is_primitive());
        assert!(Field::TimestampMillis(12345678).is_primitive());
        assert!(Field::TimestampMicros(12345678901).is_primitive());
        assert!(Field::Decimal(Decimal::from_i32(4, 8, 2)).is_primitive());

        // complex types
        assert_eq!(
            false,
            Field::Group(make_row(vec![
                ("x".to_string(), Field::Null),
                ("Y".to_string(), Field::Int(2)),
                ("z".to_string(), Field::Float(3.1)),
                ("a".to_string(), Field::Str("abc".to_string()))
            ]))
            .is_primitive()
        );

        assert_eq!(
            false,
            Field::ListInternal(make_list(vec![
                Field::Int(2),
                Field::Int(1),
                Field::Null,
                Field::Int(12)
            ]))
            .is_primitive()
        );

        assert_eq!(
            false,
            Field::MapInternal(make_map(vec![
                (Field::Int(1), Field::Float(1.2)),
                (Field::Int(2), Field::Float(4.5)),
                (Field::Int(3), Field::Float(2.3))
            ]))
            .is_primitive()
        );
    }

    #[test]
    fn test_row_primitive_field_fmt() {
        // Primitives types
        let row = make_row(vec![
            ("00".to_string(), Field::Null),
            ("01".to_string(), Field::Bool(false)),
            ("02".to_string(), Field::Byte(3)),
            ("03".to_string(), Field::Short(4)),
            ("04".to_string(), Field::Int(5)),
            ("05".to_string(), Field::Long(6)),
            ("06".to_string(), Field::UByte(7)),
            ("07".to_string(), Field::UShort(8)),
            ("08".to_string(), Field::UInt(9)),
            ("09".to_string(), Field::ULong(10)),
            ("10".to_string(), Field::Float(11.1)),
            ("11".to_string(), Field::Double(12.1)),
            ("12".to_string(), Field::Str("abc".to_string())),
            (
                "13".to_string(),
                Field::Bytes(ByteArray::from(vec![1, 2, 3, 4, 5])),
            ),
            ("14".to_string(), Field::Date(14611)),
            ("15".to_string(), Field::TimestampMillis(1262391174000)),
            ("16".to_string(), Field::TimestampMicros(1262391174000000)),
            ("17".to_string(), Field::Decimal(Decimal::from_i32(4, 7, 2))),
        ]);

        assert_eq!("null", format!("{}", row.fmt(0)));
        assert_eq!("false", format!("{}", row.fmt(1)));
        assert_eq!("3", format!("{}", row.fmt(2)));
        assert_eq!("4", format!("{}", row.fmt(3)));
        assert_eq!("5", format!("{}", row.fmt(4)));
        assert_eq!("6", format!("{}", row.fmt(5)));
        assert_eq!("7", format!("{}", row.fmt(6)));
        assert_eq!("8", format!("{}", row.fmt(7)));
        assert_eq!("9", format!("{}", row.fmt(8)));
        assert_eq!("10", format!("{}", row.fmt(9)));
        assert_eq!("11.1", format!("{}", row.fmt(10)));
        assert_eq!("12.1", format!("{}", row.fmt(11)));
        assert_eq!("\"abc\"", format!("{}", row.fmt(12)));
        assert_eq!("[1, 2, 3, 4, 5]", format!("{}", row.fmt(13)));
        assert_eq!(convert_date_to_string(14611), format!("{}", row.fmt(14)));
        assert_eq!(
            convert_timestamp_millis_to_string(1262391174000),
            format!("{}", row.fmt(15))
        );
        assert_eq!(
            convert_timestamp_micros_to_string(1262391174000000),
            format!("{}", row.fmt(16))
        );
        assert_eq!("0.04", format!("{}", row.fmt(17)));
    }

    #[test]
    fn test_row_complex_field_fmt() {
        // Complex types
        let row = make_row(vec![
            (
                "00".to_string(),
                Field::Group(make_row(vec![
                    ("x".to_string(), Field::Null),
                    ("Y".to_string(), Field::Int(2)),
                ])),
            ),
            (
                "01".to_string(),
                Field::ListInternal(make_list(vec![
                    Field::Int(2),
                    Field::Int(1),
                    Field::Null,
                    Field::Int(12),
                ])),
            ),
            (
                "02".to_string(),
                Field::MapInternal(make_map(vec![
                    (Field::Int(1), Field::Float(1.2)),
                    (Field::Int(2), Field::Float(4.5)),
                    (Field::Int(3), Field::Float(2.3)),
                ])),
            ),
        ]);

        assert_eq!("{x: null, Y: 2}", format!("{}", row.fmt(0)));
        assert_eq!("[2, 1, null, 12]", format!("{}", row.fmt(1)));
        assert_eq!("{1 -> 1.2, 2 -> 4.5, 3 -> 2.3}", format!("{}", row.fmt(2)));
    }

    #[test]
    fn test_row_primitive_accessors() {
        // primitives
        let row = make_row(vec![
            ("a".to_string(), Field::Null),
            ("b".to_string(), Field::Bool(false)),
            ("c".to_string(), Field::Byte(3)),
            ("d".to_string(), Field::Short(4)),
            ("e".to_string(), Field::Int(5)),
            ("f".to_string(), Field::Long(6)),
            ("g".to_string(), Field::UByte(3)),
            ("h".to_string(), Field::UShort(4)),
            ("i".to_string(), Field::UInt(5)),
            ("j".to_string(), Field::ULong(6)),
            ("k".to_string(), Field::Float(7.1)),
            ("l".to_string(), Field::Double(8.1)),
            ("m".to_string(), Field::Str("abc".to_string())),
            (
                "n".to_string(),
                Field::Bytes(ByteArray::from(vec![1, 2, 3, 4, 5])),
            ),
            ("o".to_string(), Field::Decimal(Decimal::from_i32(4, 7, 2))),
        ]);

        assert_eq!(false, row.get_bool(1).unwrap());
        assert_eq!(3, row.get_byte(2).unwrap());
        assert_eq!(4, row.get_short(3).unwrap());
        assert_eq!(5, row.get_int(4).unwrap());
        assert_eq!(6, row.get_long(5).unwrap());
        assert_eq!(3, row.get_ubyte(6).unwrap());
        assert_eq!(4, row.get_ushort(7).unwrap());
        assert_eq!(5, row.get_uint(8).unwrap());
        assert_eq!(6, row.get_ulong(9).unwrap());
        assert_eq!(7.1, row.get_float(10).unwrap());
        assert_eq!(8.1, row.get_double(11).unwrap());
        assert_eq!("abc", row.get_string(12).unwrap());
        assert_eq!(5, row.get_bytes(13).unwrap().len());
        assert_eq!(7, row.get_decimal(14).unwrap().precision());
    }

    #[test]
    fn test_row_primitive_invalid_accessors() {
        // primitives
        let row = make_row(vec![
            ("a".to_string(), Field::Null),
            ("b".to_string(), Field::Bool(false)),
            ("c".to_string(), Field::Byte(3)),
            ("d".to_string(), Field::Short(4)),
            ("e".to_string(), Field::Int(5)),
            ("f".to_string(), Field::Long(6)),
            ("g".to_string(), Field::UByte(3)),
            ("h".to_string(), Field::UShort(4)),
            ("i".to_string(), Field::UInt(5)),
            ("j".to_string(), Field::ULong(6)),
            ("k".to_string(), Field::Float(7.1)),
            ("l".to_string(), Field::Double(8.1)),
            ("m".to_string(), Field::Str("abc".to_string())),
            (
                "n".to_string(),
                Field::Bytes(ByteArray::from(vec![1, 2, 3, 4, 5])),
            ),
            ("o".to_string(), Field::Decimal(Decimal::from_i32(4, 7, 2))),
        ]);

        for i in 0..row.len() {
            assert!(row.get_group(i).is_err());
        }
    }

    #[test]
    fn test_row_complex_accessors() {
        let row = make_row(vec![
            (
                "a".to_string(),
                Field::Group(make_row(vec![
                    ("x".to_string(), Field::Null),
                    ("Y".to_string(), Field::Int(2)),
                ])),
            ),
            (
                "b".to_string(),
                Field::ListInternal(make_list(vec![
                    Field::Int(2),
                    Field::Int(1),
                    Field::Null,
                    Field::Int(12),
                ])),
            ),
            (
                "c".to_string(),
                Field::MapInternal(make_map(vec![
                    (Field::Int(1), Field::Float(1.2)),
                    (Field::Int(2), Field::Float(4.5)),
                    (Field::Int(3), Field::Float(2.3)),
                ])),
            ),
        ]);

        assert_eq!(2, row.get_group(0).unwrap().len());
        assert_eq!(4, row.get_list(1).unwrap().len());
        assert_eq!(3, row.get_map(2).unwrap().len());
    }

    #[test]
    fn test_row_complex_invalid_accessors() {
        let row = make_row(vec![
            (
                "a".to_string(),
                Field::Group(make_row(vec![
                    ("x".to_string(), Field::Null),
                    ("Y".to_string(), Field::Int(2)),
                ])),
            ),
            (
                "b".to_string(),
                Field::ListInternal(make_list(vec![
                    Field::Int(2),
                    Field::Int(1),
                    Field::Null,
                    Field::Int(12),
                ])),
            ),
            (
                "c".to_string(),
                Field::MapInternal(make_map(vec![
                    (Field::Int(1), Field::Float(1.2)),
                    (Field::Int(2), Field::Float(4.5)),
                    (Field::Int(3), Field::Float(2.3)),
                ])),
            ),
        ]);

        assert_eq!(
            ParquetError::General("Cannot access Group as Float".to_string()),
            row.get_float(0).unwrap_err()
        );
        assert_eq!(
            ParquetError::General("Cannot access ListInternal as Float".to_string()),
            row.get_float(1).unwrap_err()
        );
        assert_eq!(
            ParquetError::General("Cannot access MapInternal as Float".to_string()),
            row.get_float(2).unwrap_err()
        );
    }

    #[test]
    fn test_list_primitive_accessors() {
        // primitives
        let list = make_list(vec![Field::Bool(false)]);
        assert_eq!(false, list.get_bool(0).unwrap());

        let list = make_list(vec![Field::Byte(3), Field::Byte(4)]);
        assert_eq!(4, list.get_byte(1).unwrap());

        let list = make_list(vec![Field::Short(4), Field::Short(5), Field::Short(6)]);
        assert_eq!(6, list.get_short(2).unwrap());

        let list = make_list(vec![Field::Int(5)]);
        assert_eq!(5, list.get_int(0).unwrap());

        let list = make_list(vec![Field::Long(6), Field::Long(7)]);
        assert_eq!(7, list.get_long(1).unwrap());

        let list = make_list(vec![Field::UByte(3), Field::UByte(4)]);
        assert_eq!(4, list.get_ubyte(1).unwrap());

        let list = make_list(vec![Field::UShort(4), Field::UShort(5), Field::UShort(6)]);
        assert_eq!(6, list.get_ushort(2).unwrap());

        let list = make_list(vec![Field::UInt(5)]);
        assert_eq!(5, list.get_uint(0).unwrap());

        let list = make_list(vec![Field::ULong(6), Field::ULong(7)]);
        assert_eq!(7, list.get_ulong(1).unwrap());

        let list = make_list(vec![
            Field::Float(8.1),
            Field::Float(9.2),
            Field::Float(10.3),
        ]);
        assert_eq!(10.3, list.get_float(2).unwrap());

        let list = make_list(vec![Field::Double(3.1415)]);
        assert_eq!(3.1415, list.get_double(0).unwrap());

        let list = make_list(vec![Field::Str("abc".to_string())]);
        assert_eq!(&"abc".to_string(), list.get_string(0).unwrap());

        let list = make_list(vec![Field::Bytes(ByteArray::from(vec![1, 2, 3, 4, 5]))]);
        assert_eq!(&[1, 2, 3, 4, 5], list.get_bytes(0).unwrap().data());

        let list = make_list(vec![Field::Decimal(Decimal::from_i32(4, 5, 2))]);
        assert_eq!(&[0, 0, 0, 4], list.get_decimal(0).unwrap().data());
    }

    #[test]
    fn test_list_primitive_invalid_accessors() {
        // primitives
        let list = make_list(vec![Field::Bool(false)]);
        assert!(list.get_byte(0).is_err());

        let list = make_list(vec![Field::Byte(3), Field::Byte(4)]);
        assert!(list.get_short(1).is_err());

        let list = make_list(vec![Field::Short(4), Field::Short(5), Field::Short(6)]);
        assert!(list.get_int(2).is_err());

        let list = make_list(vec![Field::Int(5)]);
        assert!(list.get_long(0).is_err());

        let list = make_list(vec![Field::Long(6), Field::Long(7)]);
        assert!(list.get_float(1).is_err());

        let list = make_list(vec![Field::UByte(3), Field::UByte(4)]);
        assert!(list.get_short(1).is_err());

        let list = make_list(vec![Field::UShort(4), Field::UShort(5), Field::UShort(6)]);
        assert!(list.get_int(2).is_err());

        let list = make_list(vec![Field::UInt(5)]);
        assert!(list.get_long(0).is_err());

        let list = make_list(vec![Field::ULong(6), Field::ULong(7)]);
        assert!(list.get_float(1).is_err());

        let list = make_list(vec![
            Field::Float(8.1),
            Field::Float(9.2),
            Field::Float(10.3),
        ]);
        assert!(list.get_double(2).is_err());

        let list = make_list(vec![Field::Double(3.1415)]);
        assert!(list.get_string(0).is_err());

        let list = make_list(vec![Field::Str("abc".to_string())]);
        assert!(list.get_bytes(0).is_err());

        let list = make_list(vec![Field::Bytes(ByteArray::from(vec![1, 2, 3, 4, 5]))]);
        assert!(list.get_bool(0).is_err());

        let list = make_list(vec![Field::Decimal(Decimal::from_i32(4, 5, 2))]);
        assert!(list.get_bool(0).is_err());
    }

    #[test]
    fn test_list_complex_accessors() {
        let list = make_list(vec![Field::Group(make_row(vec![
            ("x".to_string(), Field::Null),
            ("Y".to_string(), Field::Int(2)),
        ]))]);
        assert_eq!(2, list.get_group(0).unwrap().len());

        let list = make_list(vec![Field::ListInternal(make_list(vec![
            Field::Int(2),
            Field::Int(1),
            Field::Null,
            Field::Int(12),
        ]))]);
        assert_eq!(4, list.get_list(0).unwrap().len());

        let list = make_list(vec![Field::MapInternal(make_map(vec![
            (Field::Int(1), Field::Float(1.2)),
            (Field::Int(2), Field::Float(4.5)),
            (Field::Int(3), Field::Float(2.3)),
        ]))]);
        assert_eq!(3, list.get_map(0).unwrap().len());
    }

    #[test]
    fn test_list_complex_invalid_accessors() {
        let list = make_list(vec![Field::Group(make_row(vec![
            ("x".to_string(), Field::Null),
            ("Y".to_string(), Field::Int(2)),
        ]))]);
        assert_eq!(
            general_err!("Cannot access Group as Float".to_string()),
            list.get_float(0).unwrap_err()
        );

        let list = make_list(vec![Field::ListInternal(make_list(vec![
            Field::Int(2),
            Field::Int(1),
            Field::Null,
            Field::Int(12),
        ]))]);
        assert_eq!(
            general_err!("Cannot access ListInternal as Float".to_string()),
            list.get_float(0).unwrap_err()
        );

        let list = make_list(vec![Field::MapInternal(make_map(vec![
            (Field::Int(1), Field::Float(1.2)),
            (Field::Int(2), Field::Float(4.5)),
            (Field::Int(3), Field::Float(2.3)),
        ]))]);
        assert_eq!(
            general_err!("Cannot access MapInternal as Float".to_string()),
            list.get_float(0).unwrap_err()
        );
    }

    #[test]
    fn test_map_accessors() {
        // a map from int to string
        let map = make_map(vec![
            (Field::Int(1), Field::Str("a".to_string())),
            (Field::Int(2), Field::Str("b".to_string())),
            (Field::Int(3), Field::Str("c".to_string())),
            (Field::Int(4), Field::Str("d".to_string())),
            (Field::Int(5), Field::Str("e".to_string())),
        ]);

        assert_eq!(5, map.len());
        for i in 0..5 {
            assert_eq!((i + 1) as i32, map.get_keys().get_int(i).unwrap());
            assert_eq!(
                &((i as u8 + 'a' as u8) as char).to_string(),
                map.get_values().get_string(i).unwrap()
            );
        }
    }
}
