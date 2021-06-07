//! # Data types supported by Polars.
//!
//! At the moment Polars doesn't include all data types available by Arrow. The goal is to
//! incrementally support more data types and prioritize these by usability.
//!
//! [See the AnyValue variants](enum.AnyValue.html#variants) for the data types that
//! are currently supported.
//!
use crate::prelude::*;
use ahash::RandomState;
pub use arrow::datatypes::DataType as ArrowDataType;
pub use arrow::datatypes::{
    ArrowNumericType, ArrowPrimitiveType, BooleanType, Date32Type, Date64Type,
    DurationMicrosecondType, DurationMillisecondType, DurationNanosecondType, DurationSecondType,
    Field as ArrowField, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type,
    IntervalDayTimeType, IntervalUnit, IntervalYearMonthType, Schema as ArrowSchema,
    Time32MillisecondType, Time32SecondType, Time64MicrosecondType, Time64NanosecondType, TimeUnit,
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt::{Display, Formatter};

pub struct Utf8Type {}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ListType {}

pub struct CategoricalType {}

pub trait PolarsDataType: Send + Sync {
    fn get_dtype() -> DataType;
}

macro_rules! impl_polars_datatype {
    ($ca:ident, $variant:ident) => {
        impl PolarsDataType for $ca {
            fn get_dtype() -> DataType {
                DataType::$variant
            }
        }
    };
}

impl_polars_datatype!(UInt8Type, UInt8);
impl_polars_datatype!(UInt16Type, UInt16);
impl_polars_datatype!(UInt32Type, UInt32);
impl_polars_datatype!(UInt64Type, UInt64);
impl_polars_datatype!(Int8Type, Int8);
impl_polars_datatype!(Int16Type, Int16);
impl_polars_datatype!(Int32Type, Int32);
impl_polars_datatype!(Int64Type, Int64);
impl_polars_datatype!(Float32Type, Float32);
impl_polars_datatype!(Float64Type, Float64);
impl_polars_datatype!(BooleanType, Boolean);
impl_polars_datatype!(Date32Type, Date32);
impl_polars_datatype!(Date64Type, Date64);

impl PolarsDataType for Time64NanosecondType {
    fn get_dtype() -> DataType {
        DataType::Time64(TimeUnit::Nanosecond)
    }
}

impl PolarsDataType for DurationNanosecondType {
    fn get_dtype() -> DataType {
        DataType::Duration(TimeUnit::Nanosecond)
    }
}

impl PolarsDataType for DurationMillisecondType {
    fn get_dtype() -> DataType {
        DataType::Duration(TimeUnit::Millisecond)
    }
}

impl PolarsDataType for Utf8Type {
    fn get_dtype() -> DataType {
        DataType::Utf8
    }
}

impl PolarsDataType for ListType {
    fn get_dtype() -> DataType {
        // null as we cannot no anything without self.
        DataType::List(ArrowDataType::Null)
    }
}

impl PolarsDataType for CategoricalType {
    fn get_dtype() -> DataType {
        DataType::Categorical
    }
}

#[cfg(feature = "object")]
#[cfg_attr(docsrs, doc(cfg(feature = "object")))]
pub struct ObjectType<T>(T);
#[cfg(feature = "object")]
pub type ObjectChunked<T> = ChunkedArray<ObjectType<T>>;

#[cfg(feature = "object")]
#[cfg_attr(docsrs, doc(cfg(feature = "object")))]
impl<T: PolarsObject> PolarsDataType for ObjectType<T> {
    fn get_dtype() -> DataType {
        DataType::Object(T::type_name())
    }
}

/// Any type that is not nested
pub trait PolarsSingleType: PolarsDataType {}

impl<T> PolarsSingleType for T where T: ArrowPrimitiveType + PolarsDataType {}

impl PolarsSingleType for Utf8Type {}

pub type ListChunked = ChunkedArray<ListType>;
pub type BooleanChunked = ChunkedArray<BooleanType>;
pub type UInt8Chunked = ChunkedArray<UInt8Type>;
pub type UInt16Chunked = ChunkedArray<UInt16Type>;
pub type UInt32Chunked = ChunkedArray<UInt32Type>;
pub type UInt64Chunked = ChunkedArray<UInt64Type>;
pub type Int8Chunked = ChunkedArray<Int8Type>;
pub type Int16Chunked = ChunkedArray<Int16Type>;
pub type Int32Chunked = ChunkedArray<Int32Type>;
pub type Int64Chunked = ChunkedArray<Int64Type>;
pub type Float32Chunked = ChunkedArray<Float32Type>;
pub type Float64Chunked = ChunkedArray<Float64Type>;
pub type Utf8Chunked = ChunkedArray<Utf8Type>;
pub type Date32Chunked = ChunkedArray<Date32Type>;
pub type Date64Chunked = ChunkedArray<Date64Type>;
pub type DurationNanosecondChunked = ChunkedArray<DurationNanosecondType>;
pub type DurationMillisecondChunked = ChunkedArray<DurationMillisecondType>;
pub type Time64NanosecondChunked = ChunkedArray<Time64NanosecondType>;
pub type CategoricalChunked = ChunkedArray<CategoricalType>;

pub trait PolarsPrimitiveType: ArrowPrimitiveType + Send + Sync + PolarsDataType {}
// impl PolarsPrimitiveType for BooleanType {}
impl PolarsPrimitiveType for UInt8Type {}
impl PolarsPrimitiveType for UInt16Type {}
impl PolarsPrimitiveType for UInt32Type {}
impl PolarsPrimitiveType for UInt64Type {}
impl PolarsPrimitiveType for Int8Type {}
impl PolarsPrimitiveType for Int16Type {}
impl PolarsPrimitiveType for Int32Type {}
impl PolarsPrimitiveType for Int64Type {}
impl PolarsPrimitiveType for Float32Type {}
impl PolarsPrimitiveType for Float64Type {}
impl PolarsPrimitiveType for Date32Type {}
impl PolarsPrimitiveType for Date64Type {}
impl PolarsPrimitiveType for Time64NanosecondType {}
impl PolarsPrimitiveType for DurationNanosecondType {}
impl PolarsPrimitiveType for DurationMillisecondType {}

pub trait PolarsNumericType: PolarsPrimitiveType + ArrowNumericType {}
impl PolarsNumericType for UInt8Type {}
impl PolarsNumericType for UInt16Type {}
impl PolarsNumericType for UInt32Type {}
impl PolarsNumericType for UInt64Type {}
impl PolarsNumericType for Int8Type {}
impl PolarsNumericType for Int16Type {}
impl PolarsNumericType for Int32Type {}
impl PolarsNumericType for Int64Type {}
impl PolarsNumericType for Float32Type {}
impl PolarsNumericType for Float64Type {}
impl PolarsNumericType for Date32Type {}
impl PolarsNumericType for Date64Type {}
impl PolarsNumericType for Time64NanosecondType {}
impl PolarsNumericType for DurationNanosecondType {}
impl PolarsNumericType for DurationMillisecondType {}

pub trait PolarsIntegerType: PolarsNumericType {}
impl PolarsIntegerType for UInt8Type {}
impl PolarsIntegerType for UInt16Type {}
impl PolarsIntegerType for UInt32Type {}
impl PolarsIntegerType for UInt64Type {}
impl PolarsIntegerType for Int8Type {}
impl PolarsIntegerType for Int16Type {}
impl PolarsIntegerType for Int32Type {}
impl PolarsIntegerType for Int64Type {}
impl PolarsIntegerType for Date32Type {}
impl PolarsIntegerType for Date64Type {}
impl PolarsIntegerType for Time64NanosecondType {}
impl PolarsIntegerType for DurationNanosecondType {}
impl PolarsIntegerType for DurationMillisecondType {}

pub trait PolarsFloatType: PolarsNumericType {}
impl PolarsFloatType for Float32Type {}
impl PolarsFloatType for Float64Type {}

#[derive(Debug, Clone)]
pub enum AnyValue<'a> {
    Null,
    /// A binary true or false.
    Boolean(bool),
    /// A UTF8 encoded string type.
    Utf8(&'a str),
    /// An unsigned 8-bit integer number.
    UInt8(u8),
    /// An unsigned 16-bit integer number.
    UInt16(u16),
    /// An unsigned 32-bit integer number.
    UInt32(u32),
    /// An unsigned 64-bit integer number.
    UInt64(u64),
    /// An 8-bit integer number.
    Int8(i8),
    /// A 16-bit integer number.
    Int16(i16),
    /// A 32-bit integer number.
    Int32(i32),
    /// A 64-bit integer number.
    Int64(i64),
    /// A 32-bit floating point number.
    Float32(f32),
    /// A 64-bit floating point number.
    Float64(f64),
    /// A 32-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in days (32 bits).
    Date32(i32),
    /// A 64-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in milliseconds (64 bits).
    Date64(i64),
    /// A 64-bit time representing the elapsed time since midnight in the unit of `TimeUnit`.
    Time64(i64, TimeUnit),
    /// A 32-bit time representing the elapsed time since midnight in the unit of `TimeUnit`.
    Duration(i64, TimeUnit),
    /// Naive Time elapsed from the Unix epoch, 00:00:00.000 on 1 January 1970, excluding leap seconds, as a 64-bit integer.
    /// Note that UNIX time does not include leap seconds.
    List(Series),
    #[cfg(feature = "object")]
    /// Use as_any to get a dyn Any
    Object(&'a str),
}

impl From<f64> for AnyValue<'_> {
    fn from(a: f64) -> Self {
        AnyValue::Float64(a)
    }
}
impl From<f32> for AnyValue<'_> {
    fn from(a: f32) -> Self {
        AnyValue::Float32(a)
    }
}
impl From<u32> for AnyValue<'_> {
    fn from(a: u32) -> Self {
        AnyValue::UInt32(a)
    }
}
impl From<u64> for AnyValue<'_> {
    fn from(a: u64) -> Self {
        AnyValue::UInt64(a)
    }
}
impl From<i64> for AnyValue<'_> {
    fn from(a: i64) -> Self {
        AnyValue::Int64(a)
    }
}
impl From<i32> for AnyValue<'_> {
    fn from(a: i32) -> Self {
        AnyValue::Int32(a)
    }
}
impl From<i16> for AnyValue<'_> {
    fn from(a: i16) -> Self {
        AnyValue::Int16(a)
    }
}
impl From<u16> for AnyValue<'_> {
    fn from(a: u16) -> Self {
        AnyValue::UInt16(a)
    }
}

impl From<i8> for AnyValue<'_> {
    fn from(a: i8) -> Self {
        AnyValue::Int8(a)
    }
}
impl From<u8> for AnyValue<'_> {
    fn from(a: u8) -> Self {
        AnyValue::UInt8(a)
    }
}

impl<'a, T> From<Option<T>> for AnyValue<'a>
where
    T: Into<AnyValue<'a>>,
{
    fn from(a: Option<T>) -> Self {
        match a {
            None => AnyValue::Null,
            Some(v) => v.into(),
        }
    }
}

impl<'a> AnyValue<'a> {
    pub fn add<'b>(&self, rhs: &AnyValue<'b>) -> AnyValue<'a> {
        use AnyValue::*;
        match (self, rhs) {
            (Null, _) => Null,
            (_, Null) => Null,
            (Int32(l), Int32(r)) => Int32(l + r),
            (Int64(l), Int64(r)) => Int64(l + r),
            (UInt32(l), UInt32(r)) => UInt32(l + r),
            (UInt64(l), UInt64(r)) => UInt64(l + r),
            (Float32(l), Float32(r)) => Float32(l + r),
            (Float64(l), Float64(r)) => Float64(l + r),
            _ => todo!(),
        }
    }
}

impl<'a> From<AnyValue<'a>> for Option<i64> {
    fn from(val: AnyValue<'a>) -> Self {
        use AnyValue::*;
        match val {
            Null => None,
            Int32(v) => Some(v as i64),
            Int64(v) => Some(v as i64),
            UInt32(v) => Some(v as i64),
            _ => todo!(),
        }
    }
}

impl Display for DataType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            DataType::Null => "null",
            DataType::Boolean => "bool",
            DataType::UInt8 => "u8",
            DataType::UInt16 => "u16",
            DataType::UInt32 => "u32",
            DataType::UInt64 => "u64",
            DataType::Int8 => "i8",
            DataType::Int16 => "i16",
            DataType::Int32 => "i32",
            DataType::Int64 => "i64",
            DataType::Float32 => "f32",
            DataType::Float64 => "f64",
            DataType::Utf8 => "str",
            DataType::Date32 => "date32(days)",
            DataType::Date64 => "date64(ms)",
            DataType::Time64(TimeUnit::Nanosecond) => "time64(ns)",
            DataType::Duration(TimeUnit::Nanosecond) => "duration(ns)",
            DataType::Duration(TimeUnit::Millisecond) => "duration(ms)",
            DataType::List(tp) => return write!(f, "list [{}]", DataType::from(tp)),
            #[cfg(feature = "object")]
            DataType::Object(s) => s,
            DataType::Categorical => "cat",
            _ => panic!("{:?} not implemented", self),
        };
        f.write_str(s)
    }
}

impl PartialEq for AnyValue<'_> {
    // Everything of Any is slow. Don't use.
    fn eq(&self, other: &Self) -> bool {
        use AnyValue::*;
        match (self, other) {
            (Utf8(l), Utf8(r)) => l == r,
            (UInt8(l), UInt8(r)) => l == r,
            (UInt16(l), UInt16(r)) => l == r,
            (UInt32(l), UInt32(r)) => l == r,
            (UInt64(l), UInt64(r)) => l == r,
            (Int8(l), Int8(r)) => l == r,
            (Int16(l), Int16(r)) => l == r,
            (Int32(l), Int32(r)) => l == r,
            (Int64(l), Int64(r)) => l == r,
            (Float32(l), Float32(r)) => l == r,
            (Float64(l), Float64(r)) => l == r,
            (Date32(l), Date32(r)) => l == r,
            (Date64(l), Date64(r)) => l == r,
            (Time64(l, _), Time64(r, _)) => l == r,
            (Duration(l, _), Duration(r, _)) => l == r,
            (Boolean(l), Boolean(r)) => l == r,
            (List(_), List(_)) => panic!("eq between list series not supported"),
            #[cfg(feature = "object")]
            (Object(_), Object(_)) => panic!("eq between object not supported"),
            // should it?
            (Null, Null) => true,
            _ => false,
        }
    }
}

impl PartialOrd for AnyValue<'_> {
    /// Only implemented for the same types and physical types!
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use AnyValue::*;
        match (self, other) {
            (UInt8(l), UInt8(r)) => l.partial_cmp(r),
            (UInt16(l), UInt16(r)) => l.partial_cmp(r),
            (UInt32(l), UInt32(r)) => l.partial_cmp(r),
            (UInt64(l), UInt64(r)) => l.partial_cmp(r),
            (Int8(l), Int8(r)) => l.partial_cmp(r),
            (Int16(l), Int16(r)) => l.partial_cmp(r),
            (Int32(l), Int32(r)) => l.partial_cmp(r),
            (Int64(l), Int64(r)) => l.partial_cmp(r),
            (Float32(l), Float32(r)) => l.partial_cmp(r),
            (Float64(l), Float64(r)) => l.partial_cmp(r),
            _ => None,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum DataType {
    Boolean,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
    Utf8,
    /// A 32-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in days (32 bits).
    Date32,
    /// A 64-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in milliseconds (64 bits).
    Date64,
    Time64(TimeUnit),
    List(ArrowDataType),
    Duration(TimeUnit),
    #[cfg(feature = "object")]
    /// A generic type that can be used in a `Series`
    /// &'static str can be used to determine/set inner type
    Object(&'static str),
    Null,
    Categorical,
}

impl DataType {
    pub fn to_arrow(&self) -> ArrowDataType {
        use DataType::*;
        match self {
            Boolean => ArrowDataType::Boolean,
            UInt8 => ArrowDataType::UInt8,
            UInt16 => ArrowDataType::UInt16,
            UInt32 => ArrowDataType::UInt32,
            UInt64 => ArrowDataType::UInt64,
            Int8 => ArrowDataType::Int8,
            Int16 => ArrowDataType::Int16,
            Int32 => ArrowDataType::Int32,
            Int64 => ArrowDataType::Int64,
            Float32 => ArrowDataType::Float32,
            Float64 => ArrowDataType::Float64,
            Utf8 => ArrowDataType::LargeUtf8,
            Date32 => ArrowDataType::Date32,
            Date64 => ArrowDataType::Date64,
            Time64(tu) => ArrowDataType::Time64(tu.clone()),
            List(dt) => {
                ArrowDataType::List(Box::new(arrow::datatypes::Field::new("", dt.clone(), true)))
            }
            Duration(tu) => ArrowDataType::Duration(tu.clone()),
            Null => ArrowDataType::Null,
            #[cfg(feature = "object")]
            Object(_) => unimplemented!(),
            Categorical => ArrowDataType::UInt32,
        }
    }
}

impl PartialEq<ArrowDataType> for DataType {
    fn eq(&self, other: &ArrowDataType) -> bool {
        let dt: DataType = other.into();
        self == &dt
    }
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct Field {
    name: String,
    data_type: DataType,
}

impl Field {
    pub fn new(name: &str, data_type: DataType) -> Self {
        Field {
            name: name.to_string(),
            data_type,
        }
    }
    pub fn name(&self) -> &String {
        &self.name
    }

    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }

    pub fn coerce(&mut self, dtype: DataType) {
        self.data_type = dtype;
    }

    pub fn to_arrow(&self) -> ArrowField {
        ArrowField::new(&self.name, self.data_type.to_arrow(), true)
    }
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct Schema {
    fields: Vec<Field>,
}

impl Default for Schema {
    fn default() -> Self {
        Schema { fields: vec![] }
    }
}

impl Schema {
    pub fn rename<I, J, T, S>(&self, old_names: I, new_names: J) -> Result<Schema>
    where
        I: IntoIterator<Item = T>,
        J: IntoIterator<Item = S>,
        T: AsRef<str>,
        S: AsRef<str>,
    {
        let idx = old_names
            .into_iter()
            .map(|name| self.index_of(name.as_ref()))
            .collect::<Result<Vec<_>>>()?;
        let mut new_fields = self.fields.clone();

        for (i, name) in idx.into_iter().zip(new_names) {
            let dt = new_fields[i].data_type.clone();
            new_fields[i] = Field::new(name.as_ref(), dt)
        }
        Ok(Self::new(new_fields))
    }

    pub fn new(fields: Vec<Field>) -> Self {
        Schema { fields }
    }

    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Returns an immutable reference of the vector of `Field` instances
    pub fn fields(&self) -> &Vec<Field> {
        &self.fields
    }

    /// Returns a mutable reference of the vector of `Field` instances
    pub fn fields_mut(&mut self) -> &mut Vec<Field> {
        &mut self.fields
    }

    /// Returns an immutable reference of a specific `Field` instance selected using an
    /// offset within the internal `fields` vector
    pub fn field(&self, i: usize) -> Option<&Field> {
        self.fields.get(i)
    }

    /// Returns an immutable reference of a specific `Field` instance selected by name
    pub fn field_with_name(&self, name: &str) -> Result<&Field> {
        Ok(&self.fields[self.index_of(name)?])
    }

    /// Find the index of the column with the given name
    pub fn index_of(&self, name: &str) -> Result<usize> {
        for i in 0..self.fields.len() {
            if self.fields[i].name == name {
                return Ok(i);
            }
        }
        let valid_fields: Vec<String> = self.fields.iter().map(|f| f.name().clone()).collect();
        Err(PolarsError::NotFound(format!(
            "Unable to get field named \"{}\". Valid fields: {:?}",
            name, valid_fields
        )))
    }

    pub fn to_arrow(&self) -> ArrowSchema {
        let fields = self.fields.iter().map(|f| f.to_arrow()).collect();
        ArrowSchema::new(fields)
    }

    pub fn try_merge(schemas: &[Self]) -> Result<Self> {
        let mut merged = Self::default();

        for schema in schemas {
            // merge fields
            for field in &schema.fields {
                let mut new_field = true;
                for merged_field in &mut merged.fields {
                    if field.name != merged_field.name {
                        continue;
                    }
                    new_field = false;
                }
                // found a new field, add to field list
                if new_field {
                    merged.fields.push(field.clone());
                }
            }
        }

        Ok(merged)
    }

    pub fn column_with_name(&self, name: &str) -> Option<(usize, &Field)> {
        self.fields
            .iter()
            .enumerate()
            .find(|&(_, c)| c.name == name)
    }
}

pub type SchemaRef = Arc<Schema>;

impl From<&ArrowDataType> for DataType {
    fn from(dt: &ArrowDataType) -> Self {
        match dt {
            ArrowDataType::Null => DataType::Null,
            ArrowDataType::UInt8 => DataType::UInt8,
            ArrowDataType::UInt16 => DataType::UInt16,
            ArrowDataType::UInt32 => DataType::UInt32,
            ArrowDataType::UInt64 => DataType::UInt64,
            ArrowDataType::Int8 => DataType::Int8,
            ArrowDataType::Int16 => DataType::Int16,
            ArrowDataType::Int32 => DataType::Int32,
            ArrowDataType::Int64 => DataType::Int64,
            ArrowDataType::LargeUtf8 => DataType::Utf8,
            ArrowDataType::Boolean => DataType::Boolean,
            ArrowDataType::Float32 => DataType::Float32,
            ArrowDataType::Float64 => DataType::Float64,
            ArrowDataType::LargeList(f) => DataType::List(f.data_type().clone()),
            ArrowDataType::Date32 => DataType::Date32,
            ArrowDataType::Date64 => DataType::Date64,
            ArrowDataType::Time64(TimeUnit::Nanosecond) => DataType::Time64(TimeUnit::Nanosecond),
            ArrowDataType::Duration(TimeUnit::Nanosecond) => {
                DataType::Duration(TimeUnit::Nanosecond)
            }
            ArrowDataType::Duration(TimeUnit::Millisecond) => {
                DataType::Duration(TimeUnit::Millisecond)
            }
            ArrowDataType::Utf8 => DataType::Utf8,
            dt => panic!("Arrow datatype {:?} not supported by Polars", dt),
        }
    }
}

impl From<&ArrowField> for Field {
    fn from(f: &ArrowField) -> Self {
        Field::new(f.name(), f.data_type().into())
    }
}
impl From<&ArrowSchema> for Schema {
    fn from(a_schema: &ArrowSchema) -> Self {
        Schema::new(
            a_schema
                .fields()
                .iter()
                .map(|arrow_f| arrow_f.into())
                .collect(),
        )
    }
}
impl From<ArrowSchema> for Schema {
    fn from(a_schema: ArrowSchema) -> Self {
        (&a_schema).into()
    }
}

#[cfg(feature = "private")]
pub type PlHashMap<K, V> = hashbrown::HashMap<K, V, RandomState>;
#[cfg(feature = "private")]
pub type PlHashSet<V> = hashbrown::HashSet<V, RandomState>;
