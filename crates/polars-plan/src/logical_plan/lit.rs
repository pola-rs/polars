use std::hash::{Hash, Hasher};

#[cfg(feature = "temporal")]
use polars_core::export::chrono::{Duration as ChronoDuration, NaiveDate, NaiveDateTime};
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;

#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LiteralValue {
    Null,
    /// A binary true or false.
    Boolean(bool),
    /// A UTF8 encoded string type.
    Utf8(String),
    /// A raw binary array
    Binary(Vec<u8>),
    /// An unsigned 8-bit integer number.
    #[cfg(feature = "dtype-u8")]
    UInt8(u8),
    /// An unsigned 16-bit integer number.
    #[cfg(feature = "dtype-u16")]
    UInt16(u16),
    /// An unsigned 32-bit integer number.
    UInt32(u32),
    /// An unsigned 64-bit integer number.
    UInt64(u64),
    /// An 8-bit integer number.
    #[cfg(feature = "dtype-i8")]
    Int8(i8),
    /// A 16-bit integer number.
    #[cfg(feature = "dtype-i16")]
    Int16(i16),
    /// A 32-bit integer number.
    Int32(i32),
    /// A 64-bit integer number.
    Int64(i64),
    /// A 32-bit floating point number.
    Float32(f32),
    /// A 64-bit floating point number.
    Float64(f64),
    Range {
        low: i64,
        high: i64,
        data_type: DataType,
    },
    #[cfg(feature = "dtype-date")]
    Date(i32),
    #[cfg(feature = "dtype-datetime")]
    DateTime(i64, TimeUnit, Option<TimeZone>),
    #[cfg(feature = "dtype-duration")]
    Duration(i64, TimeUnit),
    #[cfg(feature = "dtype-time")]
    Time(i64),
    Series(SpecialEq<Series>),
}

impl LiteralValue {
    pub(crate) fn is_float(&self) -> bool {
        matches!(self, LiteralValue::Float32(_) | LiteralValue::Float64(_))
    }

    pub fn to_anyvalue(&self) -> Option<AnyValue> {
        use LiteralValue::*;
        let av = match self {
            Null => AnyValue::Null,
            Boolean(v) => AnyValue::Boolean(*v),
            #[cfg(feature = "dtype-u8")]
            UInt8(v) => AnyValue::UInt8(*v),
            #[cfg(feature = "dtype-u16")]
            UInt16(v) => AnyValue::UInt16(*v),
            UInt32(v) => AnyValue::UInt32(*v),
            UInt64(v) => AnyValue::UInt64(*v),
            #[cfg(feature = "dtype-i8")]
            Int8(v) => AnyValue::Int8(*v),
            #[cfg(feature = "dtype-i16")]
            Int16(v) => AnyValue::Int16(*v),
            Int32(v) => AnyValue::Int32(*v),
            Int64(v) => AnyValue::Int64(*v),
            Float32(v) => AnyValue::Float32(*v),
            Float64(v) => AnyValue::Float64(*v),
            Utf8(v) => AnyValue::Utf8(v),
            #[cfg(feature = "dtype-duration")]
            Duration(v, tu) => AnyValue::Duration(*v, *tu),
            #[cfg(feature = "dtype-date")]
            Date(v) => AnyValue::Date(*v),
            #[cfg(feature = "dtype-datetime")]
            DateTime(v, tu, tz) => AnyValue::Datetime(*v, *tu, tz),
            #[cfg(feature = "dtype-time")]
            Time(v) => AnyValue::Time(*v),
            _ => return None,
        };
        Some(av)
    }

    /// Getter for the `DataType` of the value
    pub fn get_datatype(&self) -> DataType {
        match self {
            LiteralValue::Boolean(_) => DataType::Boolean,
            #[cfg(feature = "dtype-u8")]
            LiteralValue::UInt8(_) => DataType::UInt8,
            #[cfg(feature = "dtype-u16")]
            LiteralValue::UInt16(_) => DataType::UInt16,
            LiteralValue::UInt32(_) => DataType::UInt32,
            LiteralValue::UInt64(_) => DataType::UInt64,
            #[cfg(feature = "dtype-i8")]
            LiteralValue::Int8(_) => DataType::Int8,
            #[cfg(feature = "dtype-i16")]
            LiteralValue::Int16(_) => DataType::Int16,
            LiteralValue::Int32(_) => DataType::Int32,
            LiteralValue::Int64(_) => DataType::Int64,
            LiteralValue::Float32(_) => DataType::Float32,
            LiteralValue::Float64(_) => DataType::Float64,
            LiteralValue::Utf8(_) => DataType::Utf8,
            LiteralValue::Binary(_) => DataType::Binary,
            LiteralValue::Range { data_type, .. } => data_type.clone(),
            #[cfg(all(feature = "temporal", feature = "dtype-date"))]
            LiteralValue::Date(_) => DataType::Date,
            #[cfg(all(feature = "temporal", feature = "dtype-datetime"))]
            LiteralValue::DateTime(_, tu, tz) => DataType::Datetime(*tu, tz.clone()),
            #[cfg(all(feature = "temporal", feature = "dtype-duration"))]
            LiteralValue::Duration(_, tu) => DataType::Duration(*tu),
            LiteralValue::Series(s) => s.dtype().clone(),
            LiteralValue::Null => DataType::Null,
            #[cfg(feature = "dtype-time")]
            LiteralValue::Time(_) => DataType::Time,
        }
    }
}

pub trait Literal {
    /// [Literal](Expr::Literal) expression.
    fn lit(self) -> Expr;
}

impl Literal for String {
    fn lit(self) -> Expr {
        Expr::Literal(LiteralValue::Utf8(self))
    }
}

impl<'a> Literal for &'a str {
    fn lit(self) -> Expr {
        Expr::Literal(LiteralValue::Utf8(self.to_owned()))
    }
}

impl Literal for Vec<u8> {
    fn lit(self) -> Expr {
        Expr::Literal(LiteralValue::Binary(self))
    }
}

impl<'a> Literal for &'a [u8] {
    fn lit(self) -> Expr {
        Expr::Literal(LiteralValue::Binary(self.to_vec()))
    }
}

impl TryFrom<AnyValue<'_>> for LiteralValue {
    type Error = PolarsError;
    fn try_from(value: AnyValue) -> PolarsResult<Self> {
        match value {
            AnyValue::Null => Ok(Self::Null),
            AnyValue::Boolean(b) => Ok(Self::Boolean(b)),
            AnyValue::Utf8(s) => Ok(Self::Utf8(s.to_string())),
            AnyValue::Binary(b) => Ok(Self::Binary(b.to_vec())),
            #[cfg(feature = "dtype-u8")]
            AnyValue::UInt8(u) => Ok(Self::UInt8(u)),
            #[cfg(feature = "dtype-u16")]
            AnyValue::UInt16(u) => Ok(Self::UInt16(u)),
            AnyValue::UInt32(u) => Ok(Self::UInt32(u)),
            AnyValue::UInt64(u) => Ok(Self::UInt64(u)),
            #[cfg(feature = "dtype-i8")]
            AnyValue::Int8(i) => Ok(Self::Int8(i)),
            #[cfg(feature = "dtype-i16")]
            AnyValue::Int16(i) => Ok(Self::Int16(i)),
            AnyValue::Int32(i) => Ok(Self::Int32(i)),
            AnyValue::Int64(i) => Ok(Self::Int64(i)),
            AnyValue::Float32(f) => Ok(Self::Float32(f)),
            AnyValue::Float64(f) => Ok(Self::Float64(f)),
            #[cfg(all(feature = "temporal", feature = "dtype-datetime"))]
            AnyValue::Date(v) => Ok(LiteralValue::Date(v)),
            #[cfg(all(feature = "temporal", feature = "dtype-datetime"))]
            AnyValue::Datetime(value, tu, tz) => Ok(LiteralValue::DateTime(value, tu, tz.clone())),
            #[cfg(all(feature = "temporal", feature = "dtype-duration"))]
            AnyValue::Duration(value, tu) => Ok(LiteralValue::Duration(value, tu)),
            #[cfg(all(feature = "temporal", feature = "dtype-datetime"))]
            AnyValue::Time(v) => Ok(LiteralValue::Time(v)),
            AnyValue::List(l) => Ok(Self::Series(SpecialEq::new(l))),
            AnyValue::Utf8Owned(o) => Ok(Self::Utf8(o.into())),
            #[cfg(feature = "dtype-categorical")]
            AnyValue::Categorical(c, rev_mapping, arr) => {
                if arr.is_null() {
                    Ok(Self::Utf8(rev_mapping.get(c).to_string()))
                } else {
                    unsafe {
                        Ok(Self::Utf8(
                            arr.deref_unchecked().value(c as usize).to_string(),
                        ))
                    }
                }
            },
            v => polars_bail!(
                ComputeError: "cannot convert any-value {:?} to literal", v
            ),
        }
    }
}

macro_rules! make_literal {
    ($TYPE:ty, $SCALAR:ident) => {
        impl Literal for $TYPE {
            fn lit(self) -> Expr {
                Expr::Literal(LiteralValue::$SCALAR(self))
            }
        }
    };
}

make_literal!(bool, Boolean);
make_literal!(f32, Float32);
make_literal!(f64, Float64);
#[cfg(feature = "dtype-i8")]
make_literal!(i8, Int8);
#[cfg(feature = "dtype-i16")]
make_literal!(i16, Int16);
make_literal!(i32, Int32);
make_literal!(i64, Int64);
#[cfg(feature = "dtype-u8")]
make_literal!(u8, UInt8);
#[cfg(feature = "dtype-u16")]
make_literal!(u16, UInt16);
make_literal!(u32, UInt32);
make_literal!(u64, UInt64);

/// The literal Null
pub struct Null {}
pub const NULL: Null = Null {};

impl Literal for Null {
    fn lit(self) -> Expr {
        Expr::Literal(LiteralValue::Null)
    }
}

#[cfg(all(feature = "temporal", feature = "dtype-datetime"))]
impl Literal for NaiveDateTime {
    fn lit(self) -> Expr {
        if in_nanoseconds_window(&self) {
            Expr::Literal(LiteralValue::DateTime(
                self.timestamp_nanos_opt().unwrap(),
                TimeUnit::Nanoseconds,
                None,
            ))
        } else {
            Expr::Literal(LiteralValue::DateTime(
                self.timestamp_micros(),
                TimeUnit::Microseconds,
                None,
            ))
        }
    }
}

#[cfg(all(feature = "temporal", feature = "dtype-duration"))]
impl Literal for ChronoDuration {
    fn lit(self) -> Expr {
        if let Some(value) = self.num_nanoseconds() {
            Expr::Literal(LiteralValue::Duration(value, TimeUnit::Nanoseconds))
        } else {
            Expr::Literal(LiteralValue::Duration(
                self.num_microseconds().unwrap(),
                TimeUnit::Microseconds,
            ))
        }
    }
}

#[cfg(all(feature = "temporal", feature = "dtype-datetime"))]
impl Literal for NaiveDate {
    fn lit(self) -> Expr {
        self.and_hms_opt(0, 0, 0).unwrap().lit()
    }
}

impl Literal for Series {
    fn lit(self) -> Expr {
        Expr::Literal(LiteralValue::Series(SpecialEq::new(self)))
    }
}

impl Literal for LiteralValue {
    fn lit(self) -> Expr {
        Expr::Literal(self)
    }
}

/// Create a Literal Expression from `L`. A literal expression behaves like a column that contains a single distinct
/// value.
///
/// The column is automatically of the "correct" length to make the operations work. Often this is determined by the
/// length of the `LazyFrame` it is being used with. For instance, `lazy_df.with_column(lit(5).alias("five"))` creates a
/// new column named "five" that is the length of the Dataframe (at the time `collect` is called), where every value in
/// the column is `5`.
pub fn lit<L: Literal>(t: L) -> Expr {
    t.lit()
}

impl Hash for LiteralValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            LiteralValue::Series(s) => {
                s.dtype().hash(state);
                s.len().hash(state);
            },
            LiteralValue::Range {
                low,
                high,
                data_type,
            } => {
                low.hash(state);
                high.hash(state);
                data_type.hash(state)
            },
            _ => {
                if let Some(v) = self.to_anyvalue() {
                    v.hash_impl(state, true)
                }
            },
        }
    }
}
