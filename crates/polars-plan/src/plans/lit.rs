use std::hash::{Hash, Hasher};

#[cfg(feature = "temporal")]
use polars_core::export::chrono::{Duration as ChronoDuration, NaiveDate, NaiveDateTime};
use polars_core::prelude::*;
use polars_core::utils::materialize_dyn_int;
use polars_utils::hashing::hash_to_partition;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::constants::{get_literal_name, LITERAL_NAME};
use crate::prelude::*;

#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LiteralValue {
    Null,
    /// A binary true or false.
    Boolean(bool),
    /// A UTF8 encoded string type.
    String(String),
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
    /// A 128-bit decimal number with a maximum scale of 38.
    #[cfg(feature = "dtype-decimal")]
    Decimal(i128, usize),
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
    // Used for dynamic languages
    Float(f64),
    // Used for dynamic languages
    Int(i128),
    // Dynamic string, still needs to be made concrete.
    StrCat(String),
}

impl LiteralValue {
    /// Get the output name as `&str`.
    pub(crate) fn output_name(&self) -> &str {
        match self {
            LiteralValue::Series(s) => s.name(),
            _ => LITERAL_NAME,
        }
    }

    /// Get the output name as [`ColumnName`].
    pub(crate) fn output_column_name(&self) -> ColumnName {
        match self {
            LiteralValue::Series(s) => ColumnName::from(s.name()),
            _ => get_literal_name(),
        }
    }

    pub fn materialize(self) -> Self {
        match self {
            LiteralValue::Int(_) | LiteralValue::Float(_) | LiteralValue::StrCat(_) => {
                let av = self.to_any_value().unwrap();
                av.try_into().unwrap()
            },
            lv => lv,
        }
    }

    pub(crate) fn projects_as_scalar(&self) -> bool {
        match self {
            LiteralValue::Range { low, high, .. } => high.saturating_sub(*low) == 1,
            LiteralValue::Series(s) => s.len() == 1,
            _ => true,
        }
    }

    pub fn to_any_value(&self) -> Option<AnyValue> {
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
            #[cfg(feature = "dtype-decimal")]
            Decimal(v, scale) => AnyValue::Decimal(*v, *scale),
            String(v) => AnyValue::String(v),
            #[cfg(feature = "dtype-duration")]
            Duration(v, tu) => AnyValue::Duration(*v, *tu),
            #[cfg(feature = "dtype-date")]
            Date(v) => AnyValue::Date(*v),
            #[cfg(feature = "dtype-datetime")]
            DateTime(v, tu, tz) => AnyValue::Datetime(*v, *tu, tz),
            #[cfg(feature = "dtype-time")]
            Time(v) => AnyValue::Time(*v),
            Series(s) => AnyValue::List(s.0.clone().into_series()),
            Int(v) => materialize_dyn_int(*v),
            Float(v) => AnyValue::Float64(*v),
            StrCat(v) => AnyValue::String(v),
            Range {
                low,
                high,
                data_type,
            } => {
                let opt_s = match data_type {
                    DataType::Int32 => {
                        if *low < i32::MIN as i64 || *high > i32::MAX as i64 {
                            return None;
                        }

                        let low = *low as i32;
                        let high = *high as i32;
                        new_int_range::<Int32Type>(low, high, 1, "range").ok()
                    },
                    DataType::Int64 => {
                        let low = *low;
                        let high = *high;
                        new_int_range::<Int64Type>(low, high, 1, "range").ok()
                    },
                    DataType::UInt32 => {
                        if *low < 0 || *high > u32::MAX as i64 {
                            return None;
                        }
                        let low = *low as u32;
                        let high = *high as u32;
                        new_int_range::<UInt32Type>(low, high, 1, "range").ok()
                    },
                    _ => return None,
                };
                match opt_s {
                    Some(s) => AnyValue::List(s),
                    None => return None,
                }
            },
            Binary(v) => AnyValue::Binary(v),
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
            #[cfg(feature = "dtype-decimal")]
            LiteralValue::Decimal(_, scale) => DataType::Decimal(None, Some(*scale)),
            LiteralValue::String(_) => DataType::String,
            LiteralValue::Binary(_) => DataType::Binary,
            LiteralValue::Range { data_type, .. } => data_type.clone(),
            #[cfg(feature = "dtype-date")]
            LiteralValue::Date(_) => DataType::Date,
            #[cfg(feature = "dtype-datetime")]
            LiteralValue::DateTime(_, tu, tz) => DataType::Datetime(*tu, tz.clone()),
            #[cfg(feature = "dtype-duration")]
            LiteralValue::Duration(_, tu) => DataType::Duration(*tu),
            LiteralValue::Series(s) => s.dtype().clone(),
            LiteralValue::Null => DataType::Null,
            #[cfg(feature = "dtype-time")]
            LiteralValue::Time(_) => DataType::Time,
            LiteralValue::Int(v) => DataType::Unknown(UnknownKind::Int(*v)),
            LiteralValue::Float(_) => DataType::Unknown(UnknownKind::Float),
            LiteralValue::StrCat(_) => DataType::Unknown(UnknownKind::Str),
        }
    }
}

pub trait Literal {
    /// [Literal](Expr::Literal) expression.
    fn lit(self) -> Expr;
}

pub trait TypedLiteral: Literal {
    /// [Literal](Expr::Literal) expression.
    fn typed_lit(self) -> Expr
    where
        Self: Sized,
    {
        self.lit()
    }
}

impl TypedLiteral for String {}
impl TypedLiteral for &str {}

impl Literal for String {
    fn lit(self) -> Expr {
        Expr::Literal(LiteralValue::String(self))
    }
}

impl<'a> Literal for &'a str {
    fn lit(self) -> Expr {
        Expr::Literal(LiteralValue::String(self.to_string()))
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
            AnyValue::String(s) => Ok(Self::String(s.to_string())),
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
            #[cfg(feature = "dtype-decimal")]
            AnyValue::Decimal(v, scale) => Ok(Self::Decimal(v, scale)),
            #[cfg(feature = "dtype-date")]
            AnyValue::Date(v) => Ok(LiteralValue::Date(v)),
            #[cfg(feature = "dtype-datetime")]
            AnyValue::Datetime(value, tu, tz) => Ok(LiteralValue::DateTime(value, tu, tz.clone())),
            #[cfg(feature = "dtype-duration")]
            AnyValue::Duration(value, tu) => Ok(LiteralValue::Duration(value, tu)),
            #[cfg(feature = "dtype-time")]
            AnyValue::Time(v) => Ok(LiteralValue::Time(v)),
            AnyValue::List(l) => Ok(Self::Series(SpecialEq::new(l))),
            AnyValue::StringOwned(o) => Ok(Self::String(o.into())),
            #[cfg(feature = "dtype-categorical")]
            AnyValue::Categorical(c, rev_mapping, arr) | AnyValue::Enum(c, rev_mapping, arr) => {
                if arr.is_null() {
                    Ok(Self::String(rev_mapping.get(c).to_string()))
                } else {
                    unsafe {
                        Ok(Self::String(
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

macro_rules! make_literal_typed {
    ($TYPE:ty, $SCALAR:ident) => {
        impl TypedLiteral for $TYPE {
            fn typed_lit(self) -> Expr {
                Expr::Literal(LiteralValue::$SCALAR(self))
            }
        }
    };
}

macro_rules! make_dyn_lit {
    ($TYPE:ty, $SCALAR:ident) => {
        impl Literal for $TYPE {
            fn lit(self) -> Expr {
                Expr::Literal(LiteralValue::$SCALAR(self.try_into().unwrap()))
            }
        }
    };
}

make_literal!(bool, Boolean);
make_literal_typed!(f32, Float32);
make_literal_typed!(f64, Float64);
#[cfg(feature = "dtype-i8")]
make_literal_typed!(i8, Int8);
#[cfg(feature = "dtype-i16")]
make_literal_typed!(i16, Int16);
make_literal_typed!(i32, Int32);
make_literal_typed!(i64, Int64);
#[cfg(feature = "dtype-u8")]
make_literal_typed!(u8, UInt8);
#[cfg(feature = "dtype-u16")]
make_literal_typed!(u16, UInt16);
make_literal_typed!(u32, UInt32);
make_literal_typed!(u64, UInt64);

make_dyn_lit!(f32, Float);
make_dyn_lit!(f64, Float);
#[cfg(feature = "dtype-i8")]
make_dyn_lit!(i8, Int);
#[cfg(feature = "dtype-i16")]
make_dyn_lit!(i16, Int);
make_dyn_lit!(i32, Int);
make_dyn_lit!(i64, Int);
#[cfg(feature = "dtype-u8")]
make_dyn_lit!(u8, Int);
#[cfg(feature = "dtype-u16")]
make_dyn_lit!(u16, Int);
make_dyn_lit!(u32, Int);
make_dyn_lit!(u64, Int);
make_dyn_lit!(i128, Int);

/// The literal Null
pub struct Null {}
pub const NULL: Null = Null {};

impl Literal for Null {
    fn lit(self) -> Expr {
        Expr::Literal(LiteralValue::Null)
    }
}

#[cfg(feature = "dtype-datetime")]
impl Literal for NaiveDateTime {
    fn lit(self) -> Expr {
        if in_nanoseconds_window(&self) {
            Expr::Literal(LiteralValue::DateTime(
                self.and_utc().timestamp_nanos_opt().unwrap(),
                TimeUnit::Nanoseconds,
                None,
            ))
        } else {
            Expr::Literal(LiteralValue::DateTime(
                self.and_utc().timestamp_micros(),
                TimeUnit::Microseconds,
                None,
            ))
        }
    }
}

#[cfg(feature = "dtype-duration")]
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

#[cfg(feature = "dtype-duration")]
impl Literal for Duration {
    fn lit(self) -> Expr {
        let ns = self.duration_ns();
        Expr::Literal(LiteralValue::Duration(
            if self.negative() { -ns } else { ns },
            TimeUnit::Nanoseconds,
        ))
    }
}

#[cfg(feature = "dtype-datetime")]
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

pub fn typed_lit<L: TypedLiteral>(t: L) -> Expr {
    t.typed_lit()
}

impl Hash for LiteralValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            LiteralValue::Series(s) => {
                // Free stats
                s.dtype().hash(state);
                let len = s.len();
                len.hash(state);
                s.null_count().hash(state);
                const RANDOM: u64 = 0x2c194fa5df32a367;
                let mut rng = (len as u64) ^ RANDOM;
                for _ in 0..std::cmp::min(5, len) {
                    let idx = hash_to_partition(rng, len);
                    s.get(idx).unwrap().hash(state);
                    rng = rng.rotate_right(17).wrapping_add(RANDOM);
                }
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
                if let Some(v) = self.to_any_value() {
                    v.hash_impl(state, true)
                }
            },
        }
    }
}
