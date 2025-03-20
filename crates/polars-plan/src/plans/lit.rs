use std::hash::{Hash, Hasher};

#[cfg(feature = "temporal")]
use chrono::{Duration as ChronoDuration, NaiveDate, NaiveDateTime};
use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
use polars_core::utils::materialize_dyn_int;
use polars_utils::hashing::hash_to_partition;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::constants::get_literal_name;
use crate::prelude::*;

#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DynLiteralValue {
    Str(PlSmallStr),
    Int(i128),
    Float(f64),
    List(DynListLiteralValue),
}
#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DynListLiteralValue {
    Str(Box<[Option<PlSmallStr>]>),
    Int(Box<[Option<i128>]>),
    Float(Box<[Option<f64>]>),
    List(Box<[Option<DynListLiteralValue>]>),
}

impl Hash for DynLiteralValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Str(i) => i.hash(state),
            Self::Int(i) => i.hash(state),
            Self::Float(i) => i.to_ne_bytes().hash(state),
            Self::List(i) => i.hash(state),
        }
    }
}

impl Hash for DynListLiteralValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Str(i) => i.hash(state),
            Self::Int(i) => i.hash(state),
            Self::Float(i) => i
                .iter()
                .for_each(|i| i.map(|i| i.to_ne_bytes()).hash(state)),
            Self::List(i) => i.hash(state),
        }
    }
}

#[derive(Clone, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RangeLiteralValue {
    pub low: i128,
    pub high: i128,
    pub dtype: DataType,
}
#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LiteralValue {
    /// A dynamically inferred literal value. This needs to be materialized into a specific type.
    Dyn(DynLiteralValue),
    Scalar(Scalar),
    Series(SpecialEq<Series>),
    Range(RangeLiteralValue),
}

pub enum MaterializedLiteralValue {
    Scalar(Scalar),
    Series(Series),
}

impl DynListLiteralValue {
    pub fn try_materialize_to_dtype(self, dtype: &DataType) -> PolarsResult<Scalar> {
        let Some(inner_dtype) = dtype.inner_dtype() else {
            polars_bail!(InvalidOperation: "conversion from list literal to `{dtype}` failed.");
        };

        let s = match self {
            DynListLiteralValue::Str(vs) => {
                StringChunked::from_iter_options(PlSmallStr::from_static("literal"), vs.into_iter())
                    .into_series()
            },
            DynListLiteralValue::Int(vs) => {
                #[cfg(feature = "dtype-i128")]
                {
                    Int128Chunked::from_iter_options(
                        PlSmallStr::from_static("literal"),
                        vs.into_iter(),
                    )
                    .into_series()
                }

                #[cfg(not(feature = "dtype-i128"))]
                {
                    Int64Chunked::from_iter_options(
                        PlSmallStr::from_static("literal"),
                        vs.into_iter().map(|v| v.map(|v| v as i64)),
                    )
                    .into_series()
                }
            },
            DynListLiteralValue::Float(vs) => Float64Chunked::from_iter_options(
                PlSmallStr::from_static("literal"),
                vs.into_iter(),
            )
            .into_series(),
            DynListLiteralValue::List(_) => todo!("nested lists"),
        };

        let s = s.cast_with_options(inner_dtype, CastOptions::Strict)?;
        let value = match dtype {
            DataType::List(_) => AnyValue::List(s),
            #[cfg(feature = "dtype-array")]
            DataType::Array(_, size) => AnyValue::Array(s, *size),
            _ => unreachable!(),
        };

        Ok(Scalar::new(dtype.clone(), value))
    }
}

impl DynLiteralValue {
    pub fn try_materialize_to_dtype(self, dtype: &DataType) -> PolarsResult<Scalar> {
        match self {
            DynLiteralValue::Str(s) => {
                Ok(Scalar::from(s).cast_with_options(dtype, CastOptions::Strict)?)
            },
            DynLiteralValue::Int(i) => {
                Ok(Scalar::from(i).cast_with_options(dtype, CastOptions::Strict)?)
            },
            DynLiteralValue::Float(f) => {
                Ok(Scalar::from(f).cast_with_options(dtype, CastOptions::Strict)?)
            },
            DynLiteralValue::List(dyn_list_value) => dyn_list_value.try_materialize_to_dtype(dtype),
        }
    }
}

impl RangeLiteralValue {
    pub fn try_materialize_to_series(self, dtype: &DataType) -> PolarsResult<Series> {
        fn handle_range_oob(range: &RangeLiteralValue, to_dtype: &DataType) -> PolarsResult<()> {
            polars_bail!(
                InvalidOperation:
                "conversion from `{}` to `{to_dtype}` failed for range({}, {})",
                range.dtype, range.low, range.high,
            )
        }

        let s = match dtype {
            DataType::Int32 => {
                if self.low < i32::MIN as i128 || self.high > i32::MAX as i128 {
                    handle_range_oob(&self, dtype)?;
                }

                new_int_range::<Int32Type>(
                    self.low as i32,
                    self.high as i32,
                    1,
                    PlSmallStr::from_static("range"),
                )
                .unwrap()
            },
            DataType::Int64 => {
                if self.low < i64::MIN as i128 || self.high > i64::MAX as i128 {
                    handle_range_oob(&self, dtype)?;
                }

                new_int_range::<Int64Type>(
                    self.low as i64,
                    self.high as i64,
                    1,
                    PlSmallStr::from_static("range"),
                )
                .unwrap()
            },
            DataType::UInt32 => {
                if self.low < u32::MIN as i128 || self.high > u32::MAX as i128 {
                    handle_range_oob(&self, dtype)?;
                }
                new_int_range::<UInt32Type>(
                    self.low as u32,
                    self.high as u32,
                    1,
                    PlSmallStr::from_static("range"),
                )
                .unwrap()
            },
            _ => polars_bail!(InvalidOperation: "unsupported range datatype `{dtype}`"),
        };

        Ok(s)
    }
}

impl LiteralValue {
    /// Get the output name as `&str`.
    pub(crate) fn output_name(&self) -> &PlSmallStr {
        match self {
            LiteralValue::Series(s) => s.name(),
            _ => get_literal_name(),
        }
    }

    /// Get the output name as [`PlSmallStr`].
    pub(crate) fn output_column_name(&self) -> &PlSmallStr {
        match self {
            LiteralValue::Series(s) => s.name(),
            _ => get_literal_name(),
        }
    }

    pub fn try_materialize_to_dtype(
        self,
        dtype: &DataType,
    ) -> PolarsResult<MaterializedLiteralValue> {
        use LiteralValue as L;
        match self {
            L::Dyn(dyn_value) => dyn_value
                .try_materialize_to_dtype(dtype)
                .map(MaterializedLiteralValue::Scalar),
            L::Scalar(sc) => Ok(MaterializedLiteralValue::Scalar(
                sc.cast_with_options(dtype, CastOptions::Strict)?,
            )),
            L::Range(range) => {
                let Some(inner_dtype) = dtype.inner_dtype() else {
                    polars_bail!(
                        InvalidOperation: "cannot turn `{}` range into `{dtype}`",
                        range.dtype
                    );
                };

                let s = range.try_materialize_to_series(inner_dtype)?;
                let value = match dtype {
                    DataType::List(_) => AnyValue::List(s),
                    #[cfg(feature = "dtype-array")]
                    DataType::Array(_, size) => AnyValue::Array(s, *size),
                    _ => unreachable!(),
                };
                Ok(MaterializedLiteralValue::Scalar(Scalar::new(
                    dtype.clone(),
                    value,
                )))
            },
            L::Series(s) => Ok(MaterializedLiteralValue::Series(
                s.cast_with_options(dtype, CastOptions::Strict)?,
            )),
        }
    }

    pub fn extract_usize(&self) -> PolarsResult<usize> {
        macro_rules! cast_usize {
            ($v:expr) => {
                usize::try_from($v).map_err(
                    |_| polars_err!(InvalidOperation: "cannot convert value {} to usize", $v)
                )
            }
        }
        match &self {
            Self::Dyn(DynLiteralValue::Int(v)) => cast_usize!(*v),
            Self::Scalar(sc) => match sc.as_any_value() {
                AnyValue::UInt8(v) => Ok(v as usize),
                AnyValue::UInt16(v) => Ok(v as usize),
                AnyValue::UInt32(v) => cast_usize!(v),
                AnyValue::UInt64(v) => cast_usize!(v),
                AnyValue::Int8(v) => cast_usize!(v),
                AnyValue::Int16(v) => cast_usize!(v),
                AnyValue::Int32(v) => cast_usize!(v),
                AnyValue::Int64(v) => cast_usize!(v),
                AnyValue::Int128(v) => cast_usize!(v),
                _ => {
                    polars_bail!(InvalidOperation: "expression must be constant literal to extract integer")
                },
            },
            _ => {
                polars_bail!(InvalidOperation: "expression must be constant literal to extract integer")
            },
        }
    }

    pub fn materialize(self) -> Self {
        match self {
            LiteralValue::Dyn(_) => {
                let av = self.to_any_value().unwrap();
                av.into()
            },
            lv => lv,
        }
    }

    pub fn is_scalar(&self) -> bool {
        !matches!(self, LiteralValue::Series(_) | LiteralValue::Range { .. })
    }

    pub fn to_any_value(&self) -> Option<AnyValue> {
        let av = match self {
            Self::Scalar(sc) => sc.value().clone(),
            Self::Range(range) => {
                let s = range.clone().try_materialize_to_series(&range.dtype).ok()?;
                AnyValue::List(s)
            },
            Self::Series(_) => return None,
            Self::Dyn(d) => match d {
                DynLiteralValue::Int(v) => materialize_dyn_int(*v),
                DynLiteralValue::Float(v) => AnyValue::Float64(*v),
                DynLiteralValue::Str(v) => AnyValue::String(v),
                DynLiteralValue::List(_) => todo!(),
            },
        };
        Some(av)
    }

    /// Getter for the `DataType` of the value
    pub fn get_datatype(&self) -> DataType {
        match self {
            Self::Dyn(d) => match d {
                DynLiteralValue::Int(v) => DataType::Unknown(UnknownKind::Int(*v)),
                DynLiteralValue::Float(_) => DataType::Unknown(UnknownKind::Float),
                DynLiteralValue::Str(_) => DataType::Unknown(UnknownKind::Str),
                DynLiteralValue::List(_) => todo!(),
            },
            Self::Scalar(sc) => sc.dtype().clone(),
            Self::Series(s) => s.dtype().clone(),
            Self::Range(s) => s.dtype.clone(),
        }
    }

    pub fn new_idxsize(value: IdxSize) -> Self {
        LiteralValue::Scalar(value.into())
    }

    pub fn extract_str(&self) -> Option<&str> {
        match self {
            LiteralValue::Dyn(DynLiteralValue::Str(s)) => Some(s.as_str()),
            LiteralValue::Scalar(sc) => match sc.value() {
                AnyValue::String(s) => Some(s),
                AnyValue::StringOwned(s) => Some(s),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn extract_binary(&self) -> Option<&[u8]> {
        match self {
            LiteralValue::Scalar(sc) => match sc.value() {
                AnyValue::Binary(s) => Some(s),
                AnyValue::BinaryOwned(s) => Some(s),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        match self {
            Self::Scalar(sc) => sc.is_null(),
            Self::Series(s) => s.len() == 1 && s.null_count() == 1,
            _ => false,
        }
    }

    pub fn bool(&self) -> Option<bool> {
        match self {
            LiteralValue::Scalar(s) => match s.as_any_value() {
                AnyValue::Boolean(b) => Some(b),
                _ => None,
            },
            _ => None,
        }
    }

    pub const fn untyped_null() -> Self {
        Self::Scalar(Scalar::null(DataType::Null))
    }
}

impl From<Scalar> for LiteralValue {
    fn from(value: Scalar) -> Self {
        Self::Scalar(value)
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

impl Literal for PlSmallStr {
    fn lit(self) -> Expr {
        Expr::Literal(Scalar::from(self).into())
    }
}

impl Literal for String {
    fn lit(self) -> Expr {
        Expr::Literal(Scalar::from(PlSmallStr::from_string(self)).into())
    }
}

impl Literal for &str {
    fn lit(self) -> Expr {
        Expr::Literal(Scalar::from(PlSmallStr::from_str(self)).into())
    }
}

impl Literal for Vec<u8> {
    fn lit(self) -> Expr {
        Expr::Literal(Scalar::from(self).into())
    }
}

impl Literal for &[u8] {
    fn lit(self) -> Expr {
        Expr::Literal(Scalar::from(self.to_vec()).into())
    }
}

impl From<AnyValue<'_>> for LiteralValue {
    fn from(value: AnyValue<'_>) -> Self {
        Self::Scalar(Scalar::new(value.dtype(), value.into_static()))
    }
}

macro_rules! make_literal {
    ($TYPE:ty, $SCALAR:ident) => {
        impl Literal for $TYPE {
            fn lit(self) -> Expr {
                Expr::Literal(Scalar::from(self).into())
            }
        }
    };
}

macro_rules! make_literal_typed {
    ($TYPE:ty, $SCALAR:ident) => {
        impl TypedLiteral for $TYPE {
            fn typed_lit(self) -> Expr {
                Expr::Literal(Scalar::from(self).into())
            }
        }
    };
}

macro_rules! make_dyn_lit {
    ($TYPE:ty, $SCALAR:ident) => {
        impl Literal for $TYPE {
            fn lit(self) -> Expr {
                Expr::Literal(LiteralValue::Dyn(DynLiteralValue::$SCALAR(
                    self.try_into().unwrap(),
                )))
            }
        }
    };
}

make_literal!(bool, Boolean);
make_literal_typed!(f32, Float32);
make_literal_typed!(f64, Float64);
make_literal_typed!(i8, Int8);
make_literal_typed!(i16, Int16);
make_literal_typed!(i32, Int32);
make_literal_typed!(i64, Int64);
make_literal_typed!(i128, Int128);
make_literal_typed!(u8, UInt8);
make_literal_typed!(u16, UInt16);
make_literal_typed!(u32, UInt32);
make_literal_typed!(u64, UInt64);

make_dyn_lit!(f32, Float);
make_dyn_lit!(f64, Float);
make_dyn_lit!(i8, Int);
make_dyn_lit!(i16, Int);
make_dyn_lit!(i32, Int);
make_dyn_lit!(i64, Int);
make_dyn_lit!(u8, Int);
make_dyn_lit!(u16, Int);
make_dyn_lit!(u32, Int);
make_dyn_lit!(u64, Int);
make_dyn_lit!(i128, Int);

/// The literal Null
pub struct Null {}
pub const NULL: Null = Null {};

impl Literal for Null {
    fn lit(self) -> Expr {
        Expr::Literal(LiteralValue::Scalar(Scalar::null(DataType::Null)))
    }
}

#[cfg(feature = "dtype-datetime")]
impl Literal for NaiveDateTime {
    fn lit(self) -> Expr {
        if in_nanoseconds_window(&self) {
            Expr::Literal(
                Scalar::new_datetime(
                    self.and_utc().timestamp_nanos_opt().unwrap(),
                    TimeUnit::Nanoseconds,
                    None,
                )
                .into(),
            )
        } else {
            Expr::Literal(
                Scalar::new_datetime(
                    self.and_utc().timestamp_micros(),
                    TimeUnit::Microseconds,
                    None,
                )
                .into(),
            )
        }
    }
}

#[cfg(feature = "dtype-duration")]
impl Literal for ChronoDuration {
    fn lit(self) -> Expr {
        if let Some(value) = self.num_nanoseconds() {
            Expr::Literal(Scalar::new_duration(value, TimeUnit::Nanoseconds).into())
        } else {
            Expr::Literal(
                Scalar::new_duration(self.num_microseconds().unwrap(), TimeUnit::Microseconds)
                    .into(),
            )
        }
    }
}

#[cfg(feature = "dtype-duration")]
impl Literal for Duration {
    fn lit(self) -> Expr {
        assert!(
            self.months() == 0,
            "Cannot create literal duration that is not of fixed length; found {}",
            self
        );
        let ns = self.duration_ns();
        Expr::Literal(
            Scalar::new_duration(
                if self.negative() { -ns } else { ns },
                TimeUnit::Nanoseconds,
            )
            .into(),
        )
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

impl Literal for Scalar {
    fn lit(self) -> Expr {
        Expr::Literal(self.into())
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
            LiteralValue::Range(range) => range.hash(state),
            LiteralValue::Scalar(sc) => sc.hash(state),
            LiteralValue::Dyn(d) => d.hash(state),
        }
    }
}
