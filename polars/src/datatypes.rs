pub use arrow::datatypes::{
    BooleanType, Date32Type, Date64Type, DateUnit, DurationNanosecondType, Float32Type,
    Float64Type, Int32Type, Int64Type, Time64NanosecondType, TimeUnit, UInt32Type,
};

use crate::chunked_array::ChunkedArray;
pub use arrow::datatypes::DataType as ArrowDataType;
use arrow::datatypes::{ArrowNumericType, ArrowPrimitiveType};
use std::ops::{Deref, DerefMut};

pub struct Utf8Type {
    data: String,
}

impl Deref for Utf8Type {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for Utf8Type {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

pub trait PolarsDataType {
    fn get_data_type() -> ArrowDataType;
}

impl<T> PolarsDataType for T
where
    T: ArrowPrimitiveType,
{
    fn get_data_type() -> ArrowDataType {
        T::get_data_type()
    }
}

impl<'a> PolarsDataType for Utf8Type {
    fn get_data_type() -> ArrowDataType {
        ArrowDataType::Utf8
    }
}

pub type BooleanChunked = ChunkedArray<BooleanType>;
pub type UInt32Chunked = ChunkedArray<UInt32Type>;
pub type Int32Chunked = ChunkedArray<Int32Type>;
pub type Int64Chunked = ChunkedArray<Int64Type>;
pub type Float32Chunked = ChunkedArray<Float32Type>;
pub type Float64Chunked = ChunkedArray<Float64Type>;
pub type Utf8Chunked = ChunkedArray<Utf8Type>;
pub type Date32Chunked = ChunkedArray<Date32Type>;
pub type Date64Chunked = ChunkedArray<Date64Type>;
pub type DurationNsChunked = ChunkedArray<DurationNanosecondType>;
pub type Time64NsChunked = ChunkedArray<Time64NanosecondType>;

pub trait PolarsNumericType: ArrowNumericType {}

impl PolarsNumericType for UInt32Type {}
impl PolarsNumericType for Int32Type {}
impl PolarsNumericType for Int64Type {}
impl PolarsNumericType for Float32Type {}
impl PolarsNumericType for Float64Type {}
impl PolarsNumericType for Date32Type {}
impl PolarsNumericType for Date64Type {}
impl PolarsNumericType for Time64NanosecondType {}
impl PolarsNumericType for DurationNanosecondType {}

pub trait PolarsIntegerType: PolarsNumericType {}
impl PolarsIntegerType for UInt32Type {}
impl PolarsIntegerType for Int32Type {}
impl PolarsIntegerType for Int64Type {}
impl PolarsIntegerType for Date32Type {}
impl PolarsIntegerType for Date64Type {}
impl PolarsIntegerType for Time64NanosecondType {}
impl PolarsIntegerType for DurationNanosecondType {}

#[derive(Debug, PartialEq)]
pub enum AnyType<'a> {
    Null,
    Bool(bool),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    U32(u32),
    Str(&'a str),
    Date64(i64),
    Date32(i32),
    Time64(i64, TimeUnit),
    Duration(i64, TimeUnit),
}

pub trait ToStr {
    fn to_str(&self) -> &'static str;
}

impl ToStr for ArrowDataType {
    fn to_str(&self) -> &'static str {
        match self {
            ArrowDataType::Null => "null",
            ArrowDataType::Boolean => "bool",
            ArrowDataType::UInt32 => "u32",
            ArrowDataType::Int32 => "i32",
            ArrowDataType::Int64 => "i64",
            ArrowDataType::Float32 => "f32",
            ArrowDataType::Float64 => "f64",
            ArrowDataType::Utf8 => "str",
            ArrowDataType::Date32(DateUnit::Millisecond) => "date32",
            ArrowDataType::Date64(DateUnit::Millisecond) => "date64",
            ArrowDataType::Time64(TimeUnit::Nanosecond) => "time64(ns)",
            ArrowDataType::Duration(TimeUnit::Nanosecond) => "duration(ns)",
            _ => unimplemented!(),
        }
    }
}
