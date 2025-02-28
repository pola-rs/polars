//! Contains all metadata, such as [`PhysicalType`], [`ArrowDataType`], [`Field`] and [`ArrowSchema`].

mod field;
mod physical_type;
pub mod reshape;
mod schema;

use std::collections::BTreeMap;
use std::sync::Arc;

pub use field::{Field, DTYPE_CATEGORICAL, DTYPE_ENUM_VALUES};
pub use physical_type::*;
use polars_utils::pl_str::PlSmallStr;
pub use schema::{ArrowSchema, ArrowSchemaRef};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// typedef for [BTreeMap<PlSmallStr, PlSmallStr>] denoting [`Field`]'s and [`ArrowSchema`]'s metadata.
pub type Metadata = BTreeMap<PlSmallStr, PlSmallStr>;
/// typedef for [Option<(PlSmallStr, Option<PlSmallStr>)>] descr
pub(crate) type Extension = Option<(PlSmallStr, Option<PlSmallStr>)>;

/// The set of supported logical types in this crate.
///
/// Each variant uniquely identifies a logical type, which define specific semantics to the data
/// (e.g. how it should be represented).
/// Each variant has a corresponding [`PhysicalType`], obtained via [`ArrowDataType::to_physical_type`],
/// which declares the in-memory representation of data.
/// The [`ArrowDataType::Extension`] is special in that it augments a [`ArrowDataType`] with metadata to support custom types.
/// Use `to_logical_type` to desugar such type and return its corresponding logical type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ArrowDataType {
    /// Null type
    #[default]
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
    /// An [`i128`]
    Int128,
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
    ///
    /// When the timezone is not specified, the timestamp is considered to have no timezone
    /// and is represented _as is_
    Timestamp(TimeUnit, Option<PlSmallStr>),
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
    /// A nested [`ArrowDataType`] with a given number of [`Field`]s.
    Struct(Vec<Field>),
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
    /// - Field
    /// - ordered
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
    Dictionary(IntegerType, Box<ArrowDataType>, bool),
    /// Decimal value with precision and scale
    /// precision is the number of digits in the number and
    /// scale is the number of decimal places.
    /// The number 999.99 has a precision of 5 and scale of 2.
    Decimal(usize, usize),
    /// Decimal backed by 256 bits
    Decimal256(usize, usize),
    /// Extension type.
    Extension(Box<ExtensionType>),
    /// A binary type that inlines small values
    /// and can intern bytes.
    BinaryView,
    /// A string type that inlines small values
    /// and can intern strings.
    Utf8View,
    /// A type unknown to Arrow.
    Unknown,
    /// A nested datatype that can represent slots of differing types.
    /// Third argument represents mode
    #[cfg_attr(feature = "serde", serde(skip))]
    Union(Box<UnionType>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExtensionType {
    pub name: PlSmallStr,
    pub inner: ArrowDataType,
    pub metadata: Option<PlSmallStr>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnionType {
    pub fields: Vec<Field>,
    pub ids: Option<Vec<i32>>,
    pub mode: UnionMode,
}

/// Mode of [`ArrowDataType::Union`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum UnionMode {
    /// Dense union
    Dense,
    /// Sparse union
    Sparse,
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

impl ArrowDataType {
    /// the [`PhysicalType`] of this [`ArrowDataType`].
    pub fn to_physical_type(&self) -> PhysicalType {
        use ArrowDataType::*;
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
            Int128 => PhysicalType::Primitive(PrimitiveType::Int128),
            Interval(IntervalUnit::DayTime) => PhysicalType::Primitive(PrimitiveType::DaysMs),
            Interval(IntervalUnit::MonthDayNano) => {
                PhysicalType::Primitive(PrimitiveType::MonthDayNano)
            },
            Binary => PhysicalType::Binary,
            FixedSizeBinary(_) => PhysicalType::FixedSizeBinary,
            LargeBinary => PhysicalType::LargeBinary,
            Utf8 => PhysicalType::Utf8,
            LargeUtf8 => PhysicalType::LargeUtf8,
            BinaryView => PhysicalType::BinaryView,
            Utf8View => PhysicalType::Utf8View,
            List(_) => PhysicalType::List,
            FixedSizeList(_, _) => PhysicalType::FixedSizeList,
            LargeList(_) => PhysicalType::LargeList,
            Struct(_) => PhysicalType::Struct,
            Union(_) => PhysicalType::Union,
            Map(_, _) => PhysicalType::Map,
            Dictionary(key, _, _) => PhysicalType::Dictionary(*key),
            Extension(ext) => ext.inner.to_physical_type(),
            Unknown => unimplemented!(),
        }
    }

    // The datatype underlying this (possibly logical) arrow data type.
    pub fn underlying_physical_type(&self) -> ArrowDataType {
        use ArrowDataType::*;
        match self {
            Date32 | Time32(_) | Interval(IntervalUnit::YearMonth) => Int32,
            Date64
            | Timestamp(_, _)
            | Time64(_)
            | Duration(_)
            | Interval(IntervalUnit::DayTime) => Int64,
            Interval(IntervalUnit::MonthDayNano) => unimplemented!(),
            Binary => Binary,
            List(field) => List(Box::new(Field {
                dtype: field.dtype.underlying_physical_type(),
                ..*field.clone()
            })),
            LargeList(field) => LargeList(Box::new(Field {
                dtype: field.dtype.underlying_physical_type(),
                ..*field.clone()
            })),
            FixedSizeList(field, width) => FixedSizeList(
                Box::new(Field {
                    dtype: field.dtype.underlying_physical_type(),
                    ..*field.clone()
                }),
                *width,
            ),
            Struct(fields) => Struct(
                fields
                    .iter()
                    .map(|field| Field {
                        dtype: field.dtype.underlying_physical_type(),
                        ..field.clone()
                    })
                    .collect(),
            ),
            Dictionary(keys, _, _) => (*keys).into(),
            Union(_) => unimplemented!(),
            Map(_, _) => unimplemented!(),
            Extension(ext) => ext.inner.underlying_physical_type(),
            _ => self.clone(),
        }
    }

    /// Returns `&self` for all but [`ArrowDataType::Extension`]. For [`ArrowDataType::Extension`],
    /// (recursively) returns the inner [`ArrowDataType`].
    /// Never returns the variant [`ArrowDataType::Extension`].
    pub fn to_logical_type(&self) -> &ArrowDataType {
        use ArrowDataType::*;
        match self {
            Extension(ext) => ext.inner.to_logical_type(),
            _ => self,
        }
    }

    pub fn inner_dtype(&self) -> Option<&ArrowDataType> {
        match self {
            ArrowDataType::List(inner) => Some(inner.dtype()),
            ArrowDataType::LargeList(inner) => Some(inner.dtype()),
            ArrowDataType::FixedSizeList(inner, _) => Some(inner.dtype()),
            _ => None,
        }
    }

    pub fn is_nested(&self) -> bool {
        use ArrowDataType as D;

        matches!(
            self,
            D::List(_)
                | D::LargeList(_)
                | D::FixedSizeList(_, _)
                | D::Struct(_)
                | D::Union(_)
                | D::Map(_, _)
                | D::Dictionary(_, _, _)
                | D::Extension(_)
        )
    }

    pub fn is_view(&self) -> bool {
        matches!(self, ArrowDataType::Utf8View | ArrowDataType::BinaryView)
    }

    pub fn is_numeric(&self) -> bool {
        use ArrowDataType as D;
        matches!(
            self,
            D::Int8
                | D::Int16
                | D::Int32
                | D::Int64
                | D::Int128
                | D::UInt8
                | D::UInt16
                | D::UInt32
                | D::UInt64
                | D::Float32
                | D::Float64
                | D::Decimal(_, _)
                | D::Decimal256(_, _)
        )
    }

    pub fn to_fixed_size_list(self, size: usize, is_nullable: bool) -> ArrowDataType {
        ArrowDataType::FixedSizeList(
            Box::new(Field::new(
                PlSmallStr::from_static("item"),
                self,
                is_nullable,
            )),
            size,
        )
    }

    /// Check (recursively) whether datatype contains an [`ArrowDataType::Dictionary`] type.
    pub fn contains_dictionary(&self) -> bool {
        use ArrowDataType as D;
        match self {
            D::Null
            | D::Boolean
            | D::Int8
            | D::Int16
            | D::Int32
            | D::Int64
            | D::UInt8
            | D::UInt16
            | D::UInt32
            | D::UInt64
            | D::Int128
            | D::Float16
            | D::Float32
            | D::Float64
            | D::Timestamp(_, _)
            | D::Date32
            | D::Date64
            | D::Time32(_)
            | D::Time64(_)
            | D::Duration(_)
            | D::Interval(_)
            | D::Binary
            | D::FixedSizeBinary(_)
            | D::LargeBinary
            | D::Utf8
            | D::LargeUtf8
            | D::Decimal(_, _)
            | D::Decimal256(_, _)
            | D::BinaryView
            | D::Utf8View
            | D::Unknown => false,
            D::List(field)
            | D::FixedSizeList(field, _)
            | D::Map(field, _)
            | D::LargeList(field) => field.dtype().contains_dictionary(),
            D::Struct(fields) => fields.iter().any(|f| f.dtype().contains_dictionary()),
            D::Union(union) => union.fields.iter().any(|f| f.dtype().contains_dictionary()),
            D::Dictionary(_, _, _) => true,
            D::Extension(ext) => ext.inner.contains_dictionary(),
        }
    }
}

impl From<IntegerType> for ArrowDataType {
    fn from(item: IntegerType) -> Self {
        match item {
            IntegerType::Int8 => ArrowDataType::Int8,
            IntegerType::Int16 => ArrowDataType::Int16,
            IntegerType::Int32 => ArrowDataType::Int32,
            IntegerType::Int64 => ArrowDataType::Int64,
            IntegerType::Int128 => ArrowDataType::Int128,
            IntegerType::UInt8 => ArrowDataType::UInt8,
            IntegerType::UInt16 => ArrowDataType::UInt16,
            IntegerType::UInt32 => ArrowDataType::UInt32,
            IntegerType::UInt64 => ArrowDataType::UInt64,
        }
    }
}

impl From<PrimitiveType> for ArrowDataType {
    fn from(item: PrimitiveType) -> Self {
        match item {
            PrimitiveType::Int8 => ArrowDataType::Int8,
            PrimitiveType::Int16 => ArrowDataType::Int16,
            PrimitiveType::Int32 => ArrowDataType::Int32,
            PrimitiveType::Int64 => ArrowDataType::Int64,
            PrimitiveType::UInt8 => ArrowDataType::UInt8,
            PrimitiveType::UInt16 => ArrowDataType::UInt16,
            PrimitiveType::UInt32 => ArrowDataType::UInt32,
            PrimitiveType::UInt64 => ArrowDataType::UInt64,
            PrimitiveType::Int128 => ArrowDataType::Int128,
            PrimitiveType::Int256 => ArrowDataType::Decimal256(32, 32),
            PrimitiveType::Float16 => ArrowDataType::Float16,
            PrimitiveType::Float32 => ArrowDataType::Float32,
            PrimitiveType::Float64 => ArrowDataType::Float64,
            PrimitiveType::DaysMs => ArrowDataType::Interval(IntervalUnit::DayTime),
            PrimitiveType::MonthDayNano => ArrowDataType::Interval(IntervalUnit::MonthDayNano),
            PrimitiveType::UInt128 => unimplemented!(),
        }
    }
}

/// typedef for [`Arc<ArrowSchema>`].
pub type SchemaRef = Arc<ArrowSchema>;

/// support get extension for metadata
pub fn get_extension(metadata: &Metadata) -> Extension {
    if let Some(name) = metadata.get(&PlSmallStr::from_static("ARROW:extension:name")) {
        let metadata = metadata
            .get(&PlSmallStr::from_static("ARROW:extension:metadata"))
            .cloned();
        Some((name.clone(), metadata))
    } else {
        None
    }
}

#[cfg(not(feature = "bigidx"))]
pub type IdxArr = super::array::UInt32Array;
#[cfg(feature = "bigidx")]
pub type IdxArr = super::array::UInt64Array;
