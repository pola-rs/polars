//! Contains all metadata, such as [`PhysicalType`], [`DataType`], [`Field`] and [`ArrowSchema`].

mod field;
mod physical_type;
mod schema;

use std::collections::BTreeMap;
use std::sync::Arc;

pub use field::Field;
pub use physical_type::*;
pub use schema::{ArrowSchema, ArrowSchemaRef};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// typedef for [BTreeMap<String, String>] denoting [`Field`]'s and [`ArrowSchema`]'s metadata.
pub type Metadata = BTreeMap<String, String>;
/// typedef for [Option<(String, Option<String>)>] descr
pub(crate) type Extension = Option<(String, Option<String>)>;

/// The set of supported logical types in this crate.
///
/// Each variant uniquely identifies a logical type, which define specific semantics to the data
/// (e.g. how it should be represented).
/// Each variant has a corresponding [`PhysicalType`], obtained via [`DataType::to_physical_type`],
/// which declares the in-memory representation of data.
/// The [`DataType::Extension`] is special in that it augments a [`DataType`] with metadata to support custom types.
/// Use `to_logical_type` to desugar such type and return its corresponding logical type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DataType {
    /// Null type
    Null,
    /// `true` and `false`.
    Boolean,
    /// An [`i8`]
    Int8,
    /// An [`i16`]
    Int16,
    /// An [`i32`]
    Int32,
    /// An [`i64`]
    Int64,
    /// An [`u8`]
    UInt8,
    /// An [`u16`]
    UInt16,
    /// An [`u32`]
    UInt32,
    /// An [`u64`]
    UInt64,
    /// An 16-bit float
    Float16,
    /// A [`f32`]
    Float32,
    /// A [`f64`]
    Float64,
    /// A [`i64`] representing a timestamp measured in [`TimeUnit`] with an optional timezone.
    ///
    /// Time is measured as a Unix epoch, counting the seconds from
    /// 00:00:00.000 on 1 January 1970, excluding leap seconds,
    /// as a 64-bit signed integer.
    ///
    /// The time zone is a string indicating the name of a time zone, one of:
    ///
    /// * As used in the Olson time zone database (the "tz database" or
    ///   "tzdata"), such as "America/New_York"
    /// * An absolute time zone offset of the form +XX:XX or -XX:XX, such as +07:30
    /// When the timezone is not specified, the timestamp is considered to have no timezone
    /// and is represented _as is_
    Timestamp(TimeUnit, Option<String>),
    /// An [`i32`] representing the elapsed time since UNIX epoch (1970-01-01)
    /// in days.
    Date32,
    /// An [`i64`] representing the elapsed time since UNIX epoch (1970-01-01)
    /// in milliseconds. Values are evenly divisible by 86400000.
    Date64,
    /// A 32-bit time representing the elapsed time since midnight in the unit of `TimeUnit`.
    /// Only [`TimeUnit::Second`] and [`TimeUnit::Millisecond`] are supported on this variant.
    Time32(TimeUnit),
    /// A 64-bit time representing the elapsed time since midnight in the unit of `TimeUnit`.
    /// Only [`TimeUnit::Microsecond`] and [`TimeUnit::Nanosecond`] are supported on this variant.
    Time64(TimeUnit),
    /// Measure of elapsed time. This elapsed time is a physical duration (i.e. 1s as defined in S.I.)
    Duration(TimeUnit),
    /// A "calendar" interval modeling elapsed time that takes into account calendar shifts.
    /// For example an interval of 1 day may represent more than 24 hours.
    Interval(IntervalUnit),
    /// Opaque binary data of variable length whose offsets are represented as [`i32`].
    Binary,
    /// Opaque binary data of fixed size.
    /// Enum parameter specifies the number of bytes per value.
    FixedSizeBinary(usize),
    /// Opaque binary data of variable length whose offsets are represented as [`i64`].
    LargeBinary,
    /// A variable-length UTF-8 encoded string whose offsets are represented as [`i32`].
    Utf8,
    /// A variable-length UTF-8 encoded string whose offsets are represented as [`i64`].
    LargeUtf8,
    /// A list of some logical data type whose offsets are represented as [`i32`].
    List(Box<Field>),
    /// A list of some logical data type with a fixed number of elements.
    FixedSizeList(Box<Field>, usize),
    /// A list of some logical data type whose offsets are represented as [`i64`].
    LargeList(Box<Field>),
    /// A nested [`DataType`] with a given number of [`Field`]s.
    Struct(Vec<Field>),
    /// A nested datatype that can represent slots of differing types.
    /// Third argument represents mode
    #[cfg_attr(feature = "serde", serde(skip))]
    Union(Vec<Field>, Option<Vec<i32>>, UnionMode),
    /// A nested type that is represented as
    ///
    /// List<entries: Struct<key: K, value: V>>
    ///
    /// In this layout, the keys and values are each respectively contiguous. We do
    /// not constrain the key and value types, so the application is responsible
    /// for ensuring that the keys are hashable and unique. Whether the keys are sorted
    /// may be set in the metadata for this field.
    ///
    /// In a field with Map type, the field has a child Struct field, which then
    /// has two children: key type and the second the value type. The names of the
    /// child fields may be respectively "entries", "key", and "value", but this is
    /// not enforced.
    ///
    /// Map
    /// ```text
    ///   - child[0] entries: Struct
    ///     - child[0] key: K
    ///     - child[1] value: V
    /// ```
    /// Neither the "entries" field nor the "key" field may be nullable.
    ///
    /// The metadata is structured so that Arrow systems without special handling
    /// for Map can make Map an alias for List. The "layout" attribute for the Map
    /// field must have the same contents as a List.
    Map(Box<Field>, bool),
    /// A dictionary encoded array (`key_type`, `value_type`), where
    /// each array element is an index of `key_type` into an
    /// associated dictionary of `value_type`.
    ///
    /// Dictionary arrays are used to store columns of `value_type`
    /// that contain many repeated values using less memory, but with
    /// a higher CPU overhead for some operations.
    ///
    /// This type mostly used to represent low cardinality string
    /// arrays or a limited set of primitive types as integers.
    ///
    /// The `bool` value indicates the `Dictionary` is sorted if set to `true`.
    Dictionary(IntegerType, Box<DataType>, bool),
    /// Decimal value with precision and scale
    /// precision is the number of digits in the number and
    /// scale is the number of decimal places.
    /// The number 999.99 has a precision of 5 and scale of 2.
    Decimal(usize, usize),
    /// Decimal backed by 256 bits
    Decimal256(usize, usize),
    /// Extension type.
    Extension(String, Box<DataType>, Option<String>),
}

#[cfg(feature = "arrow_rs")]
impl From<DataType> for arrow_schema::DataType {
    fn from(value: DataType) -> Self {
        use arrow_schema::{Field as ArrowField, UnionFields};

        match value {
            DataType::Null => Self::Null,
            DataType::Boolean => Self::Boolean,
            DataType::Int8 => Self::Int8,
            DataType::Int16 => Self::Int16,
            DataType::Int32 => Self::Int32,
            DataType::Int64 => Self::Int64,
            DataType::UInt8 => Self::UInt8,
            DataType::UInt16 => Self::UInt16,
            DataType::UInt32 => Self::UInt32,
            DataType::UInt64 => Self::UInt64,
            DataType::Float16 => Self::Float16,
            DataType::Float32 => Self::Float32,
            DataType::Float64 => Self::Float64,
            DataType::Timestamp(unit, tz) => Self::Timestamp(unit.into(), tz.map(Into::into)),
            DataType::Date32 => Self::Date32,
            DataType::Date64 => Self::Date64,
            DataType::Time32(unit) => Self::Time32(unit.into()),
            DataType::Time64(unit) => Self::Time64(unit.into()),
            DataType::Duration(unit) => Self::Duration(unit.into()),
            DataType::Interval(unit) => Self::Interval(unit.into()),
            DataType::Binary => Self::Binary,
            DataType::FixedSizeBinary(size) => Self::FixedSizeBinary(size as _),
            DataType::LargeBinary => Self::LargeBinary,
            DataType::Utf8 => Self::Utf8,
            DataType::LargeUtf8 => Self::LargeUtf8,
            DataType::List(f) => Self::List(Arc::new((*f).into())),
            DataType::FixedSizeList(f, size) => {
                Self::FixedSizeList(Arc::new((*f).into()), size as _)
            },
            DataType::LargeList(f) => Self::LargeList(Arc::new((*f).into())),
            DataType::Struct(f) => Self::Struct(f.into_iter().map(ArrowField::from).collect()),
            DataType::Union(fields, Some(ids), mode) => {
                let ids = ids.into_iter().map(|x| x as _);
                let fields = fields.into_iter().map(ArrowField::from);
                Self::Union(UnionFields::new(ids, fields), mode.into())
            },
            DataType::Union(fields, None, mode) => {
                let ids = 0..fields.len() as i8;
                let fields = fields.into_iter().map(ArrowField::from);
                Self::Union(UnionFields::new(ids, fields), mode.into())
            },
            DataType::Map(f, ordered) => Self::Map(Arc::new((*f).into()), ordered),
            DataType::Dictionary(key, value, _) => Self::Dictionary(
                Box::new(DataType::from(key).into()),
                Box::new((*value).into()),
            ),
            DataType::Decimal(precision, scale) => Self::Decimal128(precision as _, scale as _),
            DataType::Decimal256(precision, scale) => Self::Decimal256(precision as _, scale as _),
            DataType::Extension(_, d, _) => (*d).into(),
        }
    }
}

#[cfg(feature = "arrow_rs")]
impl From<arrow_schema::DataType> for DataType {
    fn from(value: arrow_schema::DataType) -> Self {
        use arrow_schema::DataType;
        match value {
            DataType::Null => Self::Null,
            DataType::Boolean => Self::Boolean,
            DataType::Int8 => Self::Int8,
            DataType::Int16 => Self::Int16,
            DataType::Int32 => Self::Int32,
            DataType::Int64 => Self::Int64,
            DataType::UInt8 => Self::UInt8,
            DataType::UInt16 => Self::UInt16,
            DataType::UInt32 => Self::UInt32,
            DataType::UInt64 => Self::UInt64,
            DataType::Float16 => Self::Float16,
            DataType::Float32 => Self::Float32,
            DataType::Float64 => Self::Float64,
            DataType::Timestamp(unit, tz) => {
                Self::Timestamp(unit.into(), tz.map(|x| x.to_string()))
            },
            DataType::Date32 => Self::Date32,
            DataType::Date64 => Self::Date64,
            DataType::Time32(unit) => Self::Time32(unit.into()),
            DataType::Time64(unit) => Self::Time64(unit.into()),
            DataType::Duration(unit) => Self::Duration(unit.into()),
            DataType::Interval(unit) => Self::Interval(unit.into()),
            DataType::Binary => Self::Binary,
            DataType::FixedSizeBinary(size) => Self::FixedSizeBinary(size as _),
            DataType::LargeBinary => Self::LargeBinary,
            DataType::Utf8 => Self::Utf8,
            DataType::LargeUtf8 => Self::LargeUtf8,
            DataType::List(f) => Self::List(Box::new(f.into())),
            DataType::FixedSizeList(f, size) => Self::FixedSizeList(Box::new(f.into()), size as _),
            DataType::LargeList(f) => Self::LargeList(Box::new(f.into())),
            DataType::Struct(f) => Self::Struct(f.into_iter().map(Into::into).collect()),
            DataType::Union(fields, mode) => {
                let ids = fields.iter().map(|(x, _)| x as _).collect();
                let fields = fields.iter().map(|(_, f)| f.into()).collect();
                Self::Union(fields, Some(ids), mode.into())
            },
            DataType::Map(f, ordered) => Self::Map(Box::new(f.into()), ordered),
            DataType::Dictionary(key, value) => {
                let key = match *key {
                    DataType::Int8 => IntegerType::Int8,
                    DataType::Int16 => IntegerType::Int16,
                    DataType::Int32 => IntegerType::Int32,
                    DataType::Int64 => IntegerType::Int64,
                    DataType::UInt8 => IntegerType::UInt8,
                    DataType::UInt16 => IntegerType::UInt16,
                    DataType::UInt32 => IntegerType::UInt32,
                    DataType::UInt64 => IntegerType::UInt64,
                    d => panic!("illegal dictionary key type: {d}"),
                };
                Self::Dictionary(key, Box::new((*value).into()), false)
            },
            DataType::Decimal128(precision, scale) => Self::Decimal(precision as _, scale as _),
            DataType::Decimal256(precision, scale) => Self::Decimal256(precision as _, scale as _),
            DataType::RunEndEncoded(_, _) => {
                panic!("Run-end encoding not supported by polars_arrow")
            },
        }
    }
}

/// Mode of [`DataType::Union`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde_types", derive(Serialize, Deserialize))]
pub enum UnionMode {
    /// Dense union
    Dense,
    /// Sparse union
    Sparse,
}

#[cfg(feature = "arrow_rs")]
impl From<UnionMode> for arrow_schema::UnionMode {
    fn from(value: UnionMode) -> Self {
        match value {
            UnionMode::Dense => Self::Dense,
            UnionMode::Sparse => Self::Sparse,
        }
    }
}

#[cfg(feature = "arrow_rs")]
impl From<arrow_schema::UnionMode> for UnionMode {
    fn from(value: arrow_schema::UnionMode) -> Self {
        match value {
            arrow_schema::UnionMode::Dense => Self::Dense,
            arrow_schema::UnionMode::Sparse => Self::Sparse,
        }
    }
}

impl UnionMode {
    /// Constructs a [`UnionMode::Sparse`] if the input bool is true,
    /// or otherwise constructs a [`UnionMode::Dense`]
    pub fn sparse(is_sparse: bool) -> Self {
        if is_sparse {
            Self::Sparse
        } else {
            Self::Dense
        }
    }

    /// Returns whether the mode is sparse
    pub fn is_sparse(&self) -> bool {
        matches!(self, Self::Sparse)
    }

    /// Returns whether the mode is dense
    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense)
    }
}

/// The time units defined in Arrow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TimeUnit {
    /// Time in seconds.
    Second,
    /// Time in milliseconds.
    Millisecond,
    /// Time in microseconds.
    Microsecond,
    /// Time in nanoseconds.
    Nanosecond,
}

#[cfg(feature = "arrow_rs")]
impl From<TimeUnit> for arrow_schema::TimeUnit {
    fn from(value: TimeUnit) -> Self {
        match value {
            TimeUnit::Nanosecond => Self::Nanosecond,
            TimeUnit::Millisecond => Self::Millisecond,
            TimeUnit::Microsecond => Self::Microsecond,
            TimeUnit::Second => Self::Second,
        }
    }
}

#[cfg(feature = "arrow_rs")]
impl From<arrow_schema::TimeUnit> for TimeUnit {
    fn from(value: arrow_schema::TimeUnit) -> Self {
        match value {
            arrow_schema::TimeUnit::Nanosecond => Self::Nanosecond,
            arrow_schema::TimeUnit::Millisecond => Self::Millisecond,
            arrow_schema::TimeUnit::Microsecond => Self::Microsecond,
            arrow_schema::TimeUnit::Second => Self::Second,
        }
    }
}

/// Interval units defined in Arrow
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum IntervalUnit {
    /// The number of elapsed whole months.
    YearMonth,
    /// The number of elapsed days and milliseconds,
    /// stored as 2 contiguous `i32`
    DayTime,
    /// The number of elapsed months (i32), days (i32) and nanoseconds (i64).
    MonthDayNano,
}

#[cfg(feature = "arrow_rs")]
impl From<IntervalUnit> for arrow_schema::IntervalUnit {
    fn from(value: IntervalUnit) -> Self {
        match value {
            IntervalUnit::YearMonth => Self::YearMonth,
            IntervalUnit::DayTime => Self::DayTime,
            IntervalUnit::MonthDayNano => Self::MonthDayNano,
        }
    }
}

#[cfg(feature = "arrow_rs")]
impl From<arrow_schema::IntervalUnit> for IntervalUnit {
    fn from(value: arrow_schema::IntervalUnit) -> Self {
        match value {
            arrow_schema::IntervalUnit::YearMonth => Self::YearMonth,
            arrow_schema::IntervalUnit::DayTime => Self::DayTime,
            arrow_schema::IntervalUnit::MonthDayNano => Self::MonthDayNano,
        }
    }
}

impl DataType {
    /// the [`PhysicalType`] of this [`DataType`].
    pub fn to_physical_type(&self) -> PhysicalType {
        use DataType::*;
        match self {
            Null => PhysicalType::Null,
            Boolean => PhysicalType::Boolean,
            Int8 => PhysicalType::Primitive(PrimitiveType::Int8),
            Int16 => PhysicalType::Primitive(PrimitiveType::Int16),
            Int32 | Date32 | Time32(_) | Interval(IntervalUnit::YearMonth) => {
                PhysicalType::Primitive(PrimitiveType::Int32)
            },
            Int64 | Date64 | Timestamp(_, _) | Time64(_) | Duration(_) => {
                PhysicalType::Primitive(PrimitiveType::Int64)
            },
            Decimal(_, _) => PhysicalType::Primitive(PrimitiveType::Int128),
            Decimal256(_, _) => PhysicalType::Primitive(PrimitiveType::Int256),
            UInt8 => PhysicalType::Primitive(PrimitiveType::UInt8),
            UInt16 => PhysicalType::Primitive(PrimitiveType::UInt16),
            UInt32 => PhysicalType::Primitive(PrimitiveType::UInt32),
            UInt64 => PhysicalType::Primitive(PrimitiveType::UInt64),
            Float16 => PhysicalType::Primitive(PrimitiveType::Float16),
            Float32 => PhysicalType::Primitive(PrimitiveType::Float32),
            Float64 => PhysicalType::Primitive(PrimitiveType::Float64),
            Interval(IntervalUnit::DayTime) => PhysicalType::Primitive(PrimitiveType::DaysMs),
            Interval(IntervalUnit::MonthDayNano) => {
                PhysicalType::Primitive(PrimitiveType::MonthDayNano)
            },
            Binary => PhysicalType::Binary,
            FixedSizeBinary(_) => PhysicalType::FixedSizeBinary,
            LargeBinary => PhysicalType::LargeBinary,
            Utf8 => PhysicalType::Utf8,
            LargeUtf8 => PhysicalType::LargeUtf8,
            List(_) => PhysicalType::List,
            FixedSizeList(_, _) => PhysicalType::FixedSizeList,
            LargeList(_) => PhysicalType::LargeList,
            Struct(_) => PhysicalType::Struct,
            Union(_, _, _) => PhysicalType::Union,
            Map(_, _) => PhysicalType::Map,
            Dictionary(key, _, _) => PhysicalType::Dictionary(*key),
            Extension(_, key, _) => key.to_physical_type(),
        }
    }

    /// Returns `&self` for all but [`DataType::Extension`]. For [`DataType::Extension`],
    /// (recursively) returns the inner [`DataType`].
    /// Never returns the variant [`DataType::Extension`].
    pub fn to_logical_type(&self) -> &DataType {
        use DataType::*;
        match self {
            Extension(_, key, _) => key.to_logical_type(),
            _ => self,
        }
    }
}

impl From<IntegerType> for DataType {
    fn from(item: IntegerType) -> Self {
        match item {
            IntegerType::Int8 => DataType::Int8,
            IntegerType::Int16 => DataType::Int16,
            IntegerType::Int32 => DataType::Int32,
            IntegerType::Int64 => DataType::Int64,
            IntegerType::UInt8 => DataType::UInt8,
            IntegerType::UInt16 => DataType::UInt16,
            IntegerType::UInt32 => DataType::UInt32,
            IntegerType::UInt64 => DataType::UInt64,
        }
    }
}

impl From<PrimitiveType> for DataType {
    fn from(item: PrimitiveType) -> Self {
        match item {
            PrimitiveType::Int8 => DataType::Int8,
            PrimitiveType::Int16 => DataType::Int16,
            PrimitiveType::Int32 => DataType::Int32,
            PrimitiveType::Int64 => DataType::Int64,
            PrimitiveType::UInt8 => DataType::UInt8,
            PrimitiveType::UInt16 => DataType::UInt16,
            PrimitiveType::UInt32 => DataType::UInt32,
            PrimitiveType::UInt64 => DataType::UInt64,
            PrimitiveType::Int128 => DataType::Decimal(32, 32),
            PrimitiveType::Int256 => DataType::Decimal256(32, 32),
            PrimitiveType::Float16 => DataType::Float16,
            PrimitiveType::Float32 => DataType::Float32,
            PrimitiveType::Float64 => DataType::Float64,
            PrimitiveType::DaysMs => DataType::Interval(IntervalUnit::DayTime),
            PrimitiveType::MonthDayNano => DataType::Interval(IntervalUnit::MonthDayNano),
        }
    }
}

/// typedef for [`Arc<ArrowSchema>`].
pub type SchemaRef = Arc<ArrowSchema>;

/// support get extension for metadata
pub fn get_extension(metadata: &Metadata) -> Extension {
    if let Some(name) = metadata.get("ARROW:extension:name") {
        let metadata = metadata.get("ARROW:extension:metadata").cloned();
        Some((name.clone(), metadata))
    } else {
        None
    }
}
