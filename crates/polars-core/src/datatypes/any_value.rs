#![allow(unsafe_op_in_unsafe_fn)]
use std::borrow::Cow;

use arrow::types::PrimitiveType;
use polars_compute::cast::SerPrimitive;
use polars_error::feature_gated;
use polars_utils::total_ord::ToTotalOrd;

use super::*;
use crate::CHEAP_SERIES_HASH_LIMIT;
#[cfg(feature = "dtype-struct")]
use crate::prelude::any_value::arr_to_any_value;

#[cfg(feature = "object")]
#[derive(Debug)]
pub struct OwnedObject(pub Box<dyn PolarsObjectSafe>);

#[cfg(feature = "object")]
impl Clone for OwnedObject {
    fn clone(&self) -> Self {
        Self(self.0.to_boxed())
    }
}

#[derive(Debug, Clone, Default)]
pub enum AnyValue<'a> {
    #[default]
    Null,
    /// A binary true or false.
    Boolean(bool),
    /// A UTF8 encoded string type.
    String(&'a str),
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
    /// A 128-bit integer number.
    Int128(i128),
    /// A 32-bit floating point number.
    Float32(f32),
    /// A 64-bit floating point number.
    Float64(f64),
    /// A 32-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in days (32 bits).
    #[cfg(feature = "dtype-date")]
    Date(i32),
    /// A 64-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in nanoseconds (64 bits).
    #[cfg(feature = "dtype-datetime")]
    Datetime(i64, TimeUnit, Option<&'a TimeZone>),
    /// A 64-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in nanoseconds (64 bits).
    #[cfg(feature = "dtype-datetime")]
    DatetimeOwned(i64, TimeUnit, Option<Arc<TimeZone>>),
    /// A 64-bit integer representing difference between date-times in [`TimeUnit`]
    #[cfg(feature = "dtype-duration")]
    Duration(i64, TimeUnit),
    /// A 64-bit time representing the elapsed time since midnight in nanoseconds
    #[cfg(feature = "dtype-time")]
    Time(i64),
    #[cfg(feature = "dtype-categorical")]
    Categorical(CatSize, &'a Arc<CategoricalMapping>),
    #[cfg(feature = "dtype-categorical")]
    CategoricalOwned(CatSize, Arc<CategoricalMapping>),
    #[cfg(feature = "dtype-categorical")]
    Enum(CatSize, &'a Arc<CategoricalMapping>),
    #[cfg(feature = "dtype-categorical")]
    EnumOwned(CatSize, Arc<CategoricalMapping>),
    /// Nested type, contains arrays that are filled with one of the datatypes.
    List(Series),
    #[cfg(feature = "dtype-array")]
    Array(Series, usize),
    /// Can be used to fmt and implements Any, so can be downcasted to the proper value type.
    #[cfg(feature = "object")]
    Object(&'a dyn PolarsObjectSafe),
    #[cfg(feature = "object")]
    ObjectOwned(OwnedObject),
    // 3 pointers and thus not larger than string/vec
    // - idx in the `&StructArray`
    // - The array itself
    // - The fields
    #[cfg(feature = "dtype-struct")]
    Struct(usize, &'a StructArray, &'a [Field]),
    #[cfg(feature = "dtype-struct")]
    StructOwned(Box<(Vec<AnyValue<'a>>, Vec<Field>)>),
    /// An UTF8 encoded string type.
    StringOwned(PlSmallStr),
    Binary(&'a [u8]),
    BinaryOwned(Vec<u8>),
    /// A 128-bit fixed point decimal number with a scale.
    #[cfg(feature = "dtype-decimal")]
    Decimal(i128, usize),
}

impl AnyValue<'static> {
    pub fn zero_sum(dtype: &DataType) -> Self {
        match dtype {
            DataType::String => AnyValue::StringOwned(PlSmallStr::EMPTY),
            DataType::Binary => AnyValue::BinaryOwned(Vec::new()),
            DataType::Boolean => (0 as IdxSize).into(),
            // SAFETY: numeric values are static, inform the compiler of this.
            d if d.is_primitive_numeric() => unsafe {
                std::mem::transmute::<AnyValue<'_>, AnyValue<'static>>(
                    AnyValue::UInt8(0).cast(dtype),
                )
            },
            #[cfg(feature = "dtype-duration")]
            DataType::Duration(unit) => AnyValue::Duration(0, *unit),
            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(_p, s) => {
                AnyValue::Decimal(0, s.expect("unknown scale during execution"))
            },
            _ => AnyValue::Null,
        }
    }

    /// Can the [`AnyValue`] exist as having `dtype` as its `DataType`.
    pub fn can_have_dtype(&self, dtype: &DataType) -> bool {
        matches!(self, AnyValue::Null) || dtype == &self.dtype()
    }

    /// Generate a default dummy value for a given datatype.
    pub fn default_value(
        dtype: &DataType,
        numeric_to_one: bool,
        num_list_values: usize,
    ) -> AnyValue<'static> {
        use {AnyValue as AV, DataType as DT};
        match dtype {
            DT::Boolean => AV::Boolean(false),
            DT::UInt8 => AV::UInt8(numeric_to_one.into()),
            DT::UInt16 => AV::UInt16(numeric_to_one.into()),
            DT::UInt32 => AV::UInt32(numeric_to_one.into()),
            DT::UInt64 => AV::UInt64(numeric_to_one.into()),
            DT::Int8 => AV::Int8(numeric_to_one.into()),
            DT::Int16 => AV::Int16(numeric_to_one.into()),
            DT::Int32 => AV::Int32(numeric_to_one.into()),
            DT::Int64 => AV::Int64(numeric_to_one.into()),
            DT::Int128 => AV::Int128(numeric_to_one.into()),
            DT::Float32 => AV::Float32(numeric_to_one.into()),
            DT::Float64 => AV::Float64(numeric_to_one.into()),
            #[cfg(feature = "dtype-decimal")]
            DT::Decimal(_, scale) => AV::Decimal(0, scale.unwrap()),
            DT::String => AV::String(""),
            DT::Binary => AV::Binary(&[]),
            DT::BinaryOffset => AV::Binary(&[]),
            DT::Date => feature_gated!("dtype-date", AV::Date(0)),
            DT::Datetime(time_unit, time_zone) => feature_gated!(
                "dtype-datetime",
                AV::DatetimeOwned(0, *time_unit, time_zone.clone().map(Arc::new))
            ),
            DT::Duration(time_unit) => {
                feature_gated!("dtype-duration", AV::Duration(0, *time_unit))
            },
            DT::Time => feature_gated!("dtype-time", AV::Time(0)),
            #[cfg(feature = "dtype-array")]
            DT::Array(inner_dtype, width) => {
                let inner_value =
                    AnyValue::default_value(inner_dtype, numeric_to_one, num_list_values);
                AV::Array(
                    Scalar::new(inner_dtype.as_ref().clone(), inner_value)
                        .into_series(PlSmallStr::EMPTY)
                        .new_from_index(0, *width),
                    *width,
                )
            },
            DT::List(inner_dtype) => AV::List(if num_list_values == 0 {
                Series::new_empty(PlSmallStr::EMPTY, inner_dtype.as_ref())
            } else {
                let inner_value =
                    AnyValue::default_value(inner_dtype, numeric_to_one, num_list_values);

                Scalar::new(inner_dtype.as_ref().clone(), inner_value)
                    .into_series(PlSmallStr::EMPTY)
                    .new_from_index(0, num_list_values)
            }),
            #[cfg(feature = "object")]
            DT::Object(_) => AV::Null,
            DT::Null => AV::Null,
            #[cfg(feature = "dtype-categorical")]
            DT::Categorical(_, _) => AV::Null,
            #[cfg(feature = "dtype-categorical")]
            DT::Enum(categories, mapping) => match categories.categories().is_empty() {
                true => AV::Null,
                false => AV::EnumOwned(0, mapping.clone()),
            },
            #[cfg(feature = "dtype-struct")]
            DT::Struct(fields) => AV::StructOwned(Box::new((
                fields
                    .iter()
                    .map(|f| AnyValue::default_value(f.dtype(), numeric_to_one, num_list_values))
                    .collect(),
                fields.clone(),
            ))),
            DT::Unknown(_) => unreachable!(),
        }
    }
}

impl<'a> AnyValue<'a> {
    /// Get the matching [`DataType`] for this [`AnyValue`]`.
    ///
    /// Note: For `Categorical` and `Enum` values, the exact mapping information
    /// is not preserved in the result for performance reasons.
    pub fn dtype(&self) -> DataType {
        use AnyValue::*;
        match self {
            Null => DataType::Null,
            Boolean(_) => DataType::Boolean,
            Int8(_) => DataType::Int8,
            Int16(_) => DataType::Int16,
            Int32(_) => DataType::Int32,
            Int64(_) => DataType::Int64,
            Int128(_) => DataType::Int128,
            UInt8(_) => DataType::UInt8,
            UInt16(_) => DataType::UInt16,
            UInt32(_) => DataType::UInt32,
            UInt64(_) => DataType::UInt64,
            Float32(_) => DataType::Float32,
            Float64(_) => DataType::Float64,
            String(_) | StringOwned(_) => DataType::String,
            Binary(_) | BinaryOwned(_) => DataType::Binary,
            #[cfg(feature = "dtype-date")]
            Date(_) => DataType::Date,
            #[cfg(feature = "dtype-time")]
            Time(_) => DataType::Time,
            #[cfg(feature = "dtype-datetime")]
            Datetime(_, tu, tz) => DataType::Datetime(*tu, (*tz).cloned()),
            #[cfg(feature = "dtype-datetime")]
            DatetimeOwned(_, tu, tz) => {
                DataType::Datetime(*tu, tz.as_ref().map(|v| v.as_ref().clone()))
            },
            #[cfg(feature = "dtype-duration")]
            Duration(_, tu) => DataType::Duration(*tu),
            #[cfg(feature = "dtype-categorical")]
            Categorical(_, _) | CategoricalOwned(_, _) => {
                unimplemented!("can not get dtype of Categorical AnyValue")
            },
            #[cfg(feature = "dtype-categorical")]
            Enum(_, _) | EnumOwned(_, _) => unimplemented!("can not get dtype of Enum AnyValue"),
            List(s) => DataType::List(Box::new(s.dtype().clone())),
            #[cfg(feature = "dtype-array")]
            Array(s, size) => DataType::Array(Box::new(s.dtype().clone()), *size),
            #[cfg(feature = "dtype-struct")]
            Struct(_, _, fields) => DataType::Struct(fields.to_vec()),
            #[cfg(feature = "dtype-struct")]
            StructOwned(payload) => DataType::Struct(payload.1.clone()),
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, scale) => DataType::Decimal(None, Some(*scale)),
            #[cfg(feature = "object")]
            Object(o) => DataType::Object(o.type_name()),
            #[cfg(feature = "object")]
            ObjectOwned(o) => DataType::Object(o.0.type_name()),
        }
    }

    /// Extract a numerical value from the AnyValue
    #[doc(hidden)]
    #[inline]
    pub fn extract<T: NumCast>(&self) -> Option<T> {
        use AnyValue::*;
        match self {
            Int8(v) => NumCast::from(*v),
            Int16(v) => NumCast::from(*v),
            Int32(v) => NumCast::from(*v),
            Int64(v) => NumCast::from(*v),
            Int128(v) => NumCast::from(*v),
            UInt8(v) => NumCast::from(*v),
            UInt16(v) => NumCast::from(*v),
            UInt32(v) => NumCast::from(*v),
            UInt64(v) => NumCast::from(*v),
            Float32(v) => NumCast::from(*v),
            Float64(v) => NumCast::from(*v),
            #[cfg(feature = "dtype-date")]
            Date(v) => NumCast::from(*v),
            #[cfg(feature = "dtype-datetime")]
            Datetime(v, _, _) | DatetimeOwned(v, _, _) => NumCast::from(*v),
            #[cfg(feature = "dtype-time")]
            Time(v) => NumCast::from(*v),
            #[cfg(feature = "dtype-duration")]
            Duration(v, _) => NumCast::from(*v),
            #[cfg(feature = "dtype-decimal")]
            Decimal(v, scale) => {
                if *scale == 0 {
                    NumCast::from(*v)
                } else {
                    let f: Option<f64> = NumCast::from(*v);
                    NumCast::from(f? / 10f64.powi(*scale as _))
                }
            },
            Boolean(v) => NumCast::from(if *v { 1 } else { 0 }),
            String(v) => {
                if let Ok(val) = (*v).parse::<i128>() {
                    NumCast::from(val)
                } else {
                    NumCast::from((*v).parse::<f64>().ok()?)
                }
            },
            StringOwned(v) => String(v.as_str()).extract(),
            _ => None,
        }
    }

    #[inline]
    pub fn try_extract<T: NumCast>(&self) -> PolarsResult<T> {
        self.extract().ok_or_else(|| {
            polars_err!(
                ComputeError: "could not extract number from any-value of dtype: '{:?}'",
                self.dtype(),
            )
        })
    }

    pub fn is_boolean(&self) -> bool {
        matches!(self, AnyValue::Boolean(_))
    }

    pub fn is_primitive_numeric(&self) -> bool {
        self.is_integer() || self.is_float()
    }

    pub fn is_float(&self) -> bool {
        matches!(self, AnyValue::Float32(_) | AnyValue::Float64(_))
    }

    pub fn is_integer(&self) -> bool {
        self.is_signed_integer() || self.is_unsigned_integer()
    }

    pub fn is_signed_integer(&self) -> bool {
        matches!(
            self,
            AnyValue::Int8(_)
                | AnyValue::Int16(_)
                | AnyValue::Int32(_)
                | AnyValue::Int64(_)
                | AnyValue::Int128(_)
        )
    }

    pub fn is_unsigned_integer(&self) -> bool {
        matches!(
            self,
            AnyValue::UInt8(_) | AnyValue::UInt16(_) | AnyValue::UInt32(_) | AnyValue::UInt64(_)
        )
    }

    pub fn is_nan(&self) -> bool {
        match self {
            AnyValue::Float32(f) => f.is_nan(),
            AnyValue::Float64(f) => f.is_nan(),
            _ => false,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, AnyValue::Null)
    }

    pub fn is_nested_null(&self) -> bool {
        match self {
            AnyValue::Null => true,
            AnyValue::List(s) => s.null_count() == s.len(),
            #[cfg(feature = "dtype-array")]
            AnyValue::Array(s, _) => s.null_count() == s.len(),
            #[cfg(feature = "dtype-struct")]
            AnyValue::Struct(_, _, _) => self._iter_struct_av().all(|av| av.is_nested_null()),
            _ => false,
        }
    }

    /// Cast `AnyValue` to the provided data type and return a new `AnyValue` with type `dtype`,
    /// if possible.
    pub fn strict_cast(&self, dtype: &'a DataType) -> Option<AnyValue<'a>> {
        let new_av = match (self, dtype) {
            // to numeric
            (av, DataType::UInt8) => AnyValue::UInt8(av.extract::<u8>()?),
            (av, DataType::UInt16) => AnyValue::UInt16(av.extract::<u16>()?),
            (av, DataType::UInt32) => AnyValue::UInt32(av.extract::<u32>()?),
            (av, DataType::UInt64) => AnyValue::UInt64(av.extract::<u64>()?),
            (av, DataType::Int8) => AnyValue::Int8(av.extract::<i8>()?),
            (av, DataType::Int16) => AnyValue::Int16(av.extract::<i16>()?),
            (av, DataType::Int32) => AnyValue::Int32(av.extract::<i32>()?),
            (av, DataType::Int64) => AnyValue::Int64(av.extract::<i64>()?),
            (av, DataType::Int128) => AnyValue::Int128(av.extract::<i128>()?),
            (av, DataType::Float32) => AnyValue::Float32(av.extract::<f32>()?),
            (av, DataType::Float64) => AnyValue::Float64(av.extract::<f64>()?),

            // to boolean
            (AnyValue::UInt8(v), DataType::Boolean) => AnyValue::Boolean(*v != u8::default()),
            (AnyValue::UInt16(v), DataType::Boolean) => AnyValue::Boolean(*v != u16::default()),
            (AnyValue::UInt32(v), DataType::Boolean) => AnyValue::Boolean(*v != u32::default()),
            (AnyValue::UInt64(v), DataType::Boolean) => AnyValue::Boolean(*v != u64::default()),
            (AnyValue::Int8(v), DataType::Boolean) => AnyValue::Boolean(*v != i8::default()),
            (AnyValue::Int16(v), DataType::Boolean) => AnyValue::Boolean(*v != i16::default()),
            (AnyValue::Int32(v), DataType::Boolean) => AnyValue::Boolean(*v != i32::default()),
            (AnyValue::Int64(v), DataType::Boolean) => AnyValue::Boolean(*v != i64::default()),
            (AnyValue::Int128(v), DataType::Boolean) => AnyValue::Boolean(*v != i128::default()),
            (AnyValue::Float32(v), DataType::Boolean) => AnyValue::Boolean(*v != f32::default()),
            (AnyValue::Float64(v), DataType::Boolean) => AnyValue::Boolean(*v != f64::default()),

            // Categorical casts.
            #[cfg(feature = "dtype-categorical")]
            (
                &AnyValue::Categorical(cat, &ref lmap) | &AnyValue::CategoricalOwned(cat, ref lmap),
                DataType::Categorical(_, rmap),
            ) => {
                if Arc::ptr_eq(lmap, rmap) {
                    self.clone()
                } else {
                    let s = unsafe { lmap.cat_to_str_unchecked(cat) };
                    let new_cat = rmap.insert_cat(s).unwrap();
                    AnyValue::CategoricalOwned(new_cat, rmap.clone())
                }
            },

            #[cfg(feature = "dtype-categorical")]
            (
                &AnyValue::Enum(cat, &ref lmap) | &AnyValue::EnumOwned(cat, ref lmap),
                DataType::Enum(_, rmap),
            ) => {
                if Arc::ptr_eq(lmap, rmap) {
                    self.clone()
                } else {
                    let s = unsafe { lmap.cat_to_str_unchecked(cat) };
                    let new_cat = rmap.get_cat(s)?;
                    AnyValue::EnumOwned(new_cat, rmap.clone())
                }
            },

            #[cfg(feature = "dtype-categorical")]
            (
                &AnyValue::Categorical(cat, &ref map)
                | &AnyValue::CategoricalOwned(cat, ref map)
                | &AnyValue::Enum(cat, &ref map)
                | &AnyValue::EnumOwned(cat, ref map),
                DataType::String,
            ) => {
                let s = unsafe { map.cat_to_str_unchecked(cat) };
                AnyValue::StringOwned(PlSmallStr::from(s))
            },

            #[cfg(feature = "dtype-categorical")]
            (AnyValue::String(s), DataType::Categorical(_, map)) => {
                AnyValue::CategoricalOwned(map.insert_cat(s).unwrap(), map.clone())
            },

            #[cfg(feature = "dtype-categorical")]
            (AnyValue::StringOwned(s), DataType::Categorical(_, map)) => {
                AnyValue::CategoricalOwned(map.insert_cat(s).unwrap(), map.clone())
            },

            #[cfg(feature = "dtype-categorical")]
            (AnyValue::String(s), DataType::Enum(_, map)) => {
                AnyValue::CategoricalOwned(map.get_cat(s)?, map.clone())
            },

            #[cfg(feature = "dtype-categorical")]
            (AnyValue::StringOwned(s), DataType::Enum(_, map)) => {
                AnyValue::CategoricalOwned(map.get_cat(s)?, map.clone())
            },

            // to string
            (AnyValue::String(v), DataType::String) => AnyValue::String(v),
            (AnyValue::StringOwned(v), DataType::String) => AnyValue::StringOwned(v.clone()),

            (av, DataType::String) => {
                let mut tmp = vec![];
                if av.is_unsigned_integer() {
                    let val = av.extract::<u64>()?;
                    SerPrimitive::write(&mut tmp, val);
                } else if av.is_float() {
                    let val = av.extract::<f64>()?;
                    SerPrimitive::write(&mut tmp, val);
                } else {
                    let val = av.extract::<i64>()?;
                    SerPrimitive::write(&mut tmp, val);
                }
                AnyValue::StringOwned(PlSmallStr::from_str(std::str::from_utf8(&tmp).unwrap()))
            },

            // to binary
            (AnyValue::String(v), DataType::Binary) => AnyValue::Binary(v.as_bytes()),

            // to datetime
            #[cfg(feature = "dtype-datetime")]
            (av, DataType::Datetime(tu, tz)) if av.is_primitive_numeric() => {
                AnyValue::Datetime(av.extract::<i64>()?, *tu, tz.as_ref())
            },
            #[cfg(all(feature = "dtype-datetime", feature = "dtype-date"))]
            (AnyValue::Date(v), DataType::Datetime(tu, _)) => AnyValue::Datetime(
                match tu {
                    TimeUnit::Nanoseconds => (*v as i64) * NS_IN_DAY,
                    TimeUnit::Microseconds => (*v as i64) * US_IN_DAY,
                    TimeUnit::Milliseconds => (*v as i64) * MS_IN_DAY,
                },
                *tu,
                None,
            ),
            #[cfg(feature = "dtype-datetime")]
            (
                AnyValue::Datetime(v, tu, _) | AnyValue::DatetimeOwned(v, tu, _),
                DataType::Datetime(tu_r, tz_r),
            ) => AnyValue::Datetime(
                match (tu, tu_r) {
                    (TimeUnit::Nanoseconds, TimeUnit::Microseconds) => *v / 1_000i64,
                    (TimeUnit::Nanoseconds, TimeUnit::Milliseconds) => *v / 1_000_000i64,
                    (TimeUnit::Microseconds, TimeUnit::Nanoseconds) => *v * 1_000i64,
                    (TimeUnit::Microseconds, TimeUnit::Milliseconds) => *v / 1_000i64,
                    (TimeUnit::Milliseconds, TimeUnit::Microseconds) => *v * 1_000i64,
                    (TimeUnit::Milliseconds, TimeUnit::Nanoseconds) => *v * 1_000_000i64,
                    _ => *v,
                },
                *tu_r,
                tz_r.as_ref(),
            ),

            // to date
            #[cfg(feature = "dtype-date")]
            (av, DataType::Date) if av.is_primitive_numeric() => {
                AnyValue::Date(av.extract::<i32>()?)
            },
            #[cfg(all(feature = "dtype-date", feature = "dtype-datetime"))]
            (AnyValue::Datetime(v, tu, _) | AnyValue::DatetimeOwned(v, tu, _), DataType::Date) => {
                AnyValue::Date(match tu {
                    TimeUnit::Nanoseconds => *v / NS_IN_DAY,
                    TimeUnit::Microseconds => *v / US_IN_DAY,
                    TimeUnit::Milliseconds => *v / MS_IN_DAY,
                } as i32)
            },

            // to time
            #[cfg(feature = "dtype-time")]
            (av, DataType::Time) if av.is_primitive_numeric() => {
                AnyValue::Time(av.extract::<i64>()?)
            },
            #[cfg(all(feature = "dtype-time", feature = "dtype-datetime"))]
            (AnyValue::Datetime(v, tu, _) | AnyValue::DatetimeOwned(v, tu, _), DataType::Time) => {
                AnyValue::Time(match tu {
                    TimeUnit::Nanoseconds => *v % NS_IN_DAY,
                    TimeUnit::Microseconds => (*v % US_IN_DAY) * 1_000i64,
                    TimeUnit::Milliseconds => (*v % MS_IN_DAY) * 1_000_000i64,
                })
            },

            // to duration
            #[cfg(feature = "dtype-duration")]
            (av, DataType::Duration(tu)) if av.is_primitive_numeric() => {
                AnyValue::Duration(av.extract::<i64>()?, *tu)
            },
            #[cfg(all(feature = "dtype-duration", feature = "dtype-time"))]
            (AnyValue::Time(v), DataType::Duration(tu)) => AnyValue::Duration(
                match *tu {
                    TimeUnit::Nanoseconds => *v,
                    TimeUnit::Microseconds => *v / 1_000i64,
                    TimeUnit::Milliseconds => *v / 1_000_000i64,
                },
                *tu,
            ),
            #[cfg(feature = "dtype-duration")]
            (AnyValue::Duration(v, tu), DataType::Duration(tu_r)) => AnyValue::Duration(
                match (tu, tu_r) {
                    (_, _) if tu == tu_r => *v,
                    (TimeUnit::Nanoseconds, TimeUnit::Microseconds) => *v / 1_000i64,
                    (TimeUnit::Nanoseconds, TimeUnit::Milliseconds) => *v / 1_000_000i64,
                    (TimeUnit::Microseconds, TimeUnit::Nanoseconds) => *v * 1_000i64,
                    (TimeUnit::Microseconds, TimeUnit::Milliseconds) => *v / 1_000i64,
                    (TimeUnit::Milliseconds, TimeUnit::Microseconds) => *v * 1_000i64,
                    (TimeUnit::Milliseconds, TimeUnit::Nanoseconds) => *v * 1_000_000i64,
                    _ => *v,
                },
                *tu_r,
            ),

            // to decimal
            #[cfg(feature = "dtype-decimal")]
            (av, DataType::Decimal(prec, scale)) if av.is_integer() => {
                let value = av.try_extract::<i128>().unwrap();
                let scale = scale.unwrap_or(0);
                let factor = 10_i128.pow(scale as _); // Conversion to u32 is safe, max value is 38.
                let converted = value.checked_mul(factor)?;

                // Check if the converted value fits into the specified precision
                let prec = prec.unwrap_or(38) as u32;
                let num_digits = (converted.abs() as f64).log10().ceil() as u32;
                if num_digits > prec {
                    return None;
                }

                AnyValue::Decimal(converted, scale)
            },
            #[cfg(feature = "dtype-decimal")]
            (AnyValue::Decimal(value, scale_av), DataType::Decimal(_, scale)) => {
                let Some(scale) = scale else {
                    return Some(self.clone());
                };
                // TODO: Allow lossy conversion?
                let scale_diff = scale.checked_sub(*scale_av)?;
                let factor = 10_i128.pow(scale_diff as _); // Conversion is safe, max value is 38.
                let converted = value.checked_mul(factor)?;
                AnyValue::Decimal(converted, *scale)
            },

            // to self
            (av, dtype) if av.dtype() == *dtype => self.clone(),

            _ => return None,
        };
        Some(new_av)
    }

    /// Cast `AnyValue` to the provided data type and return a new `AnyValue` with type `dtype`,
    /// if possible.
    pub fn try_strict_cast(&self, dtype: &'a DataType) -> PolarsResult<AnyValue<'a>> {
        self.strict_cast(dtype).ok_or_else(
            || polars_err!(ComputeError: "cannot cast any-value {:?} to dtype '{}'", self, dtype),
        )
    }

    pub fn cast(&self, dtype: &'a DataType) -> AnyValue<'a> {
        match self.strict_cast(dtype) {
            Some(av) => av,
            None => AnyValue::Null,
        }
    }

    pub fn idx(&self) -> IdxSize {
        match self {
            #[cfg(not(feature = "bigidx"))]
            Self::UInt32(v) => *v,
            #[cfg(feature = "bigidx")]
            Self::UInt64(v) => *v,
            _ => panic!("expected index type found {self:?}"),
        }
    }

    pub fn str_value(&self) -> Cow<'a, str> {
        match self {
            Self::String(s) => Cow::Borrowed(s),
            Self::StringOwned(s) => Cow::Owned(s.to_string()),
            Self::Null => Cow::Borrowed("null"),
            #[cfg(feature = "dtype-categorical")]
            Self::Categorical(cat, map) | Self::Enum(cat, map) => {
                Cow::Borrowed(unsafe { map.cat_to_str_unchecked(*cat) })
            },
            #[cfg(feature = "dtype-categorical")]
            Self::CategoricalOwned(cat, map) | Self::EnumOwned(cat, map) => {
                Cow::Owned(unsafe { map.cat_to_str_unchecked(*cat) }.to_owned())
            },
            av => Cow::Owned(av.to_string()),
        }
    }

    pub fn to_physical(self) -> Self {
        match self {
            Self::Null
            | Self::Boolean(_)
            | Self::String(_)
            | Self::StringOwned(_)
            | Self::Binary(_)
            | Self::BinaryOwned(_)
            | Self::UInt8(_)
            | Self::UInt16(_)
            | Self::UInt32(_)
            | Self::UInt64(_)
            | Self::Int8(_)
            | Self::Int16(_)
            | Self::Int32(_)
            | Self::Int64(_)
            | Self::Int128(_)
            | Self::Float32(_)
            | Self::Float64(_) => self,

            #[cfg(feature = "object")]
            Self::Object(_) | Self::ObjectOwned(_) => self,

            #[cfg(feature = "dtype-date")]
            Self::Date(v) => Self::Int32(v),
            #[cfg(feature = "dtype-datetime")]
            Self::Datetime(v, _, _) | Self::DatetimeOwned(v, _, _) => Self::Int64(v),

            #[cfg(feature = "dtype-duration")]
            Self::Duration(v, _) => Self::Int64(v),
            #[cfg(feature = "dtype-time")]
            Self::Time(v) => Self::Int64(v),

            #[cfg(feature = "dtype-categorical")]
            Self::Categorical(v, _)
            | Self::CategoricalOwned(v, _)
            | Self::Enum(v, _)
            | Self::EnumOwned(v, _) => Self::UInt32(v),
            Self::List(series) => Self::List(series.to_physical_repr().into_owned()),

            #[cfg(feature = "dtype-array")]
            Self::Array(series, width) => {
                Self::Array(series.to_physical_repr().into_owned(), width)
            },

            #[cfg(feature = "dtype-struct")]
            Self::Struct(_, _, _) => todo!(),
            #[cfg(feature = "dtype-struct")]
            Self::StructOwned(values) => Self::StructOwned(Box::new((
                values.0.into_iter().map(|v| v.to_physical()).collect(),
                values
                    .1
                    .into_iter()
                    .map(|mut f| {
                        f.dtype = f.dtype.to_physical();
                        f
                    })
                    .collect(),
            ))),

            #[cfg(feature = "dtype-decimal")]
            Self::Decimal(v, _) => Self::Int128(v),
        }
    }

    #[inline]
    pub fn extract_bool(&self) -> Option<bool> {
        match self {
            AnyValue::Boolean(v) => Some(*v),
            _ => None,
        }
    }

    #[inline]
    pub fn extract_str(&self) -> Option<&str> {
        match self {
            AnyValue::String(v) => Some(v),
            AnyValue::StringOwned(v) => Some(v.as_str()),
            _ => None,
        }
    }

    #[inline]
    pub fn extract_bytes(&self) -> Option<&[u8]> {
        match self {
            AnyValue::Binary(v) => Some(v),
            AnyValue::BinaryOwned(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

impl From<AnyValue<'_>> for DataType {
    fn from(value: AnyValue<'_>) -> Self {
        value.dtype()
    }
}

impl<'a> From<&AnyValue<'a>> for DataType {
    fn from(value: &AnyValue<'a>) -> Self {
        value.dtype()
    }
}

impl AnyValue<'_> {
    pub fn hash_impl<H: Hasher>(&self, state: &mut H, cheap: bool) {
        use AnyValue::*;
        std::mem::discriminant(self).hash(state);
        match self {
            Int8(v) => v.hash(state),
            Int16(v) => v.hash(state),
            Int32(v) => v.hash(state),
            Int64(v) => v.hash(state),
            Int128(v) => feature_gated!("dtype-i128", v.hash(state)),
            UInt8(v) => v.hash(state),
            UInt16(v) => v.hash(state),
            UInt32(v) => v.hash(state),
            UInt64(v) => v.hash(state),
            String(v) => v.hash(state),
            StringOwned(v) => v.hash(state),
            Float32(v) => v.to_ne_bytes().hash(state),
            Float64(v) => v.to_ne_bytes().hash(state),
            Binary(v) => v.hash(state),
            BinaryOwned(v) => v.hash(state),
            Boolean(v) => v.hash(state),
            List(v) => {
                if !cheap || v.len() < CHEAP_SERIES_HASH_LIMIT {
                    Hash::hash(&Wrap(v.clone()), state)
                }
            },
            #[cfg(feature = "dtype-array")]
            Array(v, width) => {
                if !cheap || v.len() < CHEAP_SERIES_HASH_LIMIT {
                    Hash::hash(&Wrap(v.clone()), state)
                }
                width.hash(state)
            },
            #[cfg(feature = "dtype-date")]
            Date(v) => v.hash(state),
            #[cfg(feature = "dtype-datetime")]
            Datetime(v, tu, tz) => {
                v.hash(state);
                tu.hash(state);
                tz.hash(state);
            },
            #[cfg(feature = "dtype-datetime")]
            DatetimeOwned(v, tu, tz) => {
                v.hash(state);
                tu.hash(state);
                tz.hash(state);
            },
            #[cfg(feature = "dtype-duration")]
            Duration(v, tz) => {
                v.hash(state);
                tz.hash(state);
            },
            #[cfg(feature = "dtype-time")]
            Time(v) => v.hash(state),
            #[cfg(feature = "dtype-categorical")]
            Categorical(v, _) | CategoricalOwned(v, _) | Enum(v, _) | EnumOwned(v, _) => {
                v.hash(state)
            },
            #[cfg(feature = "object")]
            Object(_) => {},
            #[cfg(feature = "object")]
            ObjectOwned(_) => {},
            #[cfg(feature = "dtype-struct")]
            Struct(_, _, _) => {
                if !cheap {
                    let mut buf = vec![];
                    self._materialize_struct_av(&mut buf);
                    buf.hash(state)
                }
            },
            #[cfg(feature = "dtype-struct")]
            StructOwned(v) => v.0.hash(state),
            #[cfg(feature = "dtype-decimal")]
            Decimal(v, k) => {
                v.hash(state);
                k.hash(state);
            },
            Null => {},
        }
    }
}

impl Hash for AnyValue<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_impl(state, false)
    }
}

impl Eq for AnyValue<'_> {}

impl<'a, T> From<Option<T>> for AnyValue<'a>
where
    T: Into<AnyValue<'a>>,
{
    #[inline]
    fn from(a: Option<T>) -> Self {
        match a {
            None => AnyValue::Null,
            Some(v) => v.into(),
        }
    }
}

impl<'a> AnyValue<'a> {
    #[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
    pub(crate) fn as_date(&self) -> AnyValue<'static> {
        match self {
            #[cfg(feature = "dtype-date")]
            AnyValue::Int32(v) => AnyValue::Date(*v),
            AnyValue::Null => AnyValue::Null,
            dt => panic!("cannot create date from other type. dtype: {dt}"),
        }
    }
    #[cfg(feature = "dtype-datetime")]
    pub(crate) fn as_datetime(&self, tu: TimeUnit, tz: Option<&'a TimeZone>) -> AnyValue<'a> {
        match self {
            AnyValue::Int64(v) => AnyValue::Datetime(*v, tu, tz),
            AnyValue::Null => AnyValue::Null,
            dt => panic!("cannot create date from other type. dtype: {dt}"),
        }
    }

    #[cfg(feature = "dtype-duration")]
    pub(crate) fn as_duration(&self, tu: TimeUnit) -> AnyValue<'static> {
        match self {
            AnyValue::Int64(v) => AnyValue::Duration(*v, tu),
            AnyValue::Null => AnyValue::Null,
            dt => panic!("cannot create date from other type. dtype: {dt}"),
        }
    }

    #[cfg(feature = "dtype-time")]
    pub(crate) fn as_time(&self) -> AnyValue<'static> {
        match self {
            AnyValue::Int64(v) => AnyValue::Time(*v),
            AnyValue::Null => AnyValue::Null,
            dt => panic!("cannot create date from other type. dtype: {dt}"),
        }
    }

    pub(crate) fn to_i128(&self) -> Option<i128> {
        match self {
            AnyValue::UInt8(v) => Some((*v).into()),
            AnyValue::UInt16(v) => Some((*v).into()),
            AnyValue::UInt32(v) => Some((*v).into()),
            AnyValue::UInt64(v) => Some((*v).into()),
            AnyValue::Int8(v) => Some((*v).into()),
            AnyValue::Int16(v) => Some((*v).into()),
            AnyValue::Int32(v) => Some((*v).into()),
            AnyValue::Int64(v) => Some((*v).into()),
            AnyValue::Int128(v) => Some(*v),
            _ => None,
        }
    }

    pub(crate) fn to_f64(&self) -> Option<f64> {
        match self {
            AnyValue::Float32(v) => Some((*v).into()),
            AnyValue::Float64(v) => Some(*v),
            _ => None,
        }
    }

    #[must_use]
    pub fn add(&self, rhs: &AnyValue) -> AnyValue<'static> {
        use AnyValue::*;
        match (self, rhs) {
            (Null, r) => r.clone().into_static(),
            (l, Null) => l.clone().into_static(),
            (Int32(l), Int32(r)) => Int32(l + r),
            (Int64(l), Int64(r)) => Int64(l + r),
            (UInt32(l), UInt32(r)) => UInt32(l + r),
            (UInt64(l), UInt64(r)) => UInt64(l + r),
            (Float32(l), Float32(r)) => Float32(l + r),
            (Float64(l), Float64(r)) => Float64(l + r),
            #[cfg(feature = "dtype-duration")]
            (Duration(l, lu), Duration(r, ru)) => {
                if lu != ru {
                    unimplemented!("adding durations with different units is not supported here");
                }

                Duration(l + r, *lu)
            },
            #[cfg(feature = "dtype-decimal")]
            (Decimal(l, ls), Decimal(r, rs)) => {
                if ls != rs {
                    unimplemented!("adding decimals with different scales is not supported here");
                }

                Decimal(l + r, *ls)
            },
            _ => unimplemented!(),
        }
    }

    #[inline]
    pub fn as_borrowed(&self) -> AnyValue<'_> {
        match self {
            AnyValue::BinaryOwned(data) => AnyValue::Binary(data),
            AnyValue::StringOwned(data) => AnyValue::String(data.as_str()),
            #[cfg(feature = "dtype-datetime")]
            AnyValue::DatetimeOwned(v, tu, tz) => {
                AnyValue::Datetime(*v, *tu, tz.as_ref().map(AsRef::as_ref))
            },
            #[cfg(feature = "dtype-categorical")]
            AnyValue::CategoricalOwned(cat, map) => AnyValue::Categorical(*cat, map),
            #[cfg(feature = "dtype-categorical")]
            AnyValue::EnumOwned(cat, map) => AnyValue::Enum(*cat, map),
            av => av.clone(),
        }
    }

    /// Try to coerce to an AnyValue with static lifetime.
    /// This can be done if it does not borrow any values.
    #[inline]
    pub fn into_static(self) -> AnyValue<'static> {
        use AnyValue::*;
        match self {
            Null => Null,
            Int8(v) => Int8(v),
            Int16(v) => Int16(v),
            Int32(v) => Int32(v),
            Int64(v) => Int64(v),
            Int128(v) => Int128(v),
            UInt8(v) => UInt8(v),
            UInt16(v) => UInt16(v),
            UInt32(v) => UInt32(v),
            UInt64(v) => UInt64(v),
            Boolean(v) => Boolean(v),
            Float32(v) => Float32(v),
            Float64(v) => Float64(v),
            #[cfg(feature = "dtype-datetime")]
            Datetime(v, tu, tz) => DatetimeOwned(v, tu, tz.map(|v| Arc::new(v.clone()))),
            #[cfg(feature = "dtype-datetime")]
            DatetimeOwned(v, tu, tz) => DatetimeOwned(v, tu, tz),
            #[cfg(feature = "dtype-date")]
            Date(v) => Date(v),
            #[cfg(feature = "dtype-duration")]
            Duration(v, tu) => Duration(v, tu),
            #[cfg(feature = "dtype-time")]
            Time(v) => Time(v),
            List(v) => List(v),
            #[cfg(feature = "dtype-array")]
            Array(s, size) => Array(s, size),
            String(v) => StringOwned(PlSmallStr::from_str(v)),
            StringOwned(v) => StringOwned(v),
            Binary(v) => BinaryOwned(v.to_vec()),
            BinaryOwned(v) => BinaryOwned(v),
            #[cfg(feature = "object")]
            Object(v) => ObjectOwned(OwnedObject(v.to_boxed())),
            #[cfg(feature = "dtype-struct")]
            Struct(idx, arr, fields) => {
                let avs = struct_to_avs_static(idx, arr, fields);
                StructOwned(Box::new((avs, fields.to_vec())))
            },
            #[cfg(feature = "dtype-struct")]
            StructOwned(payload) => {
                let av = StructOwned(payload);
                // SAFETY: owned is already static
                unsafe { std::mem::transmute::<AnyValue<'a>, AnyValue<'static>>(av) }
            },
            #[cfg(feature = "object")]
            ObjectOwned(payload) => {
                let av = ObjectOwned(payload);
                // SAFETY: owned is already static
                unsafe { std::mem::transmute::<AnyValue<'a>, AnyValue<'static>>(av) }
            },
            #[cfg(feature = "dtype-decimal")]
            Decimal(val, scale) => Decimal(val, scale),
            #[cfg(feature = "dtype-categorical")]
            Categorical(cat, map) => CategoricalOwned(cat, map.clone()),
            #[cfg(feature = "dtype-categorical")]
            CategoricalOwned(cat, map) => CategoricalOwned(cat, map),
            #[cfg(feature = "dtype-categorical")]
            Enum(cat, map) => EnumOwned(cat, map.clone()),
            #[cfg(feature = "dtype-categorical")]
            EnumOwned(cat, map) => EnumOwned(cat, map),
        }
    }

    /// Get a reference to the `&str` contained within [`AnyValue`].
    pub fn get_str(&self) -> Option<&str> {
        match self {
            AnyValue::String(s) => Some(s),
            AnyValue::StringOwned(s) => Some(s.as_str()),
            #[cfg(feature = "dtype-categorical")]
            Self::Categorical(cat, map) | Self::Enum(cat, map) => {
                Some(unsafe { map.cat_to_str_unchecked(*cat) })
            },
            #[cfg(feature = "dtype-categorical")]
            Self::CategoricalOwned(cat, map) | Self::EnumOwned(cat, map) => {
                Some(unsafe { map.cat_to_str_unchecked(*cat) })
            },
            _ => None,
        }
    }
}

impl<'a> From<AnyValue<'a>> for Option<i64> {
    fn from(val: AnyValue<'a>) -> Self {
        use AnyValue::*;
        match val {
            Null => None,
            Int32(v) => Some(v as i64),
            Int64(v) => Some(v),
            UInt32(v) => Some(v as i64),
            _ => todo!(),
        }
    }
}

impl AnyValue<'_> {
    #[inline]
    pub fn eq_missing(&self, other: &Self, null_equal: bool) -> bool {
        fn struct_owned_value_iter<'a>(
            v: &'a (Vec<AnyValue<'_>>, Vec<Field>),
        ) -> impl ExactSizeIterator<Item = AnyValue<'a>> {
            v.0.iter().map(|v| v.as_borrowed())
        }
        fn struct_value_iter(
            idx: usize,
            arr: &StructArray,
        ) -> impl ExactSizeIterator<Item = AnyValue<'_>> {
            assert!(idx < arr.len());

            arr.values().iter().map(move |field_arr| unsafe {
                // SAFETY: We asserted before that idx is smaller than the array length. Since it
                // is an invariant of StructArray that all fields have the same length this is fine
                // to do.
                field_arr.get_unchecked(idx)
            })
        }

        fn struct_eq_missing<'a>(
            l: impl ExactSizeIterator<Item = AnyValue<'a>>,
            r: impl ExactSizeIterator<Item = AnyValue<'a>>,
            null_equal: bool,
        ) -> bool {
            if l.len() != r.len() {
                return false;
            }

            l.zip(r).all(|(lv, rv)| lv.eq_missing(&rv, null_equal))
        }

        use AnyValue::*;
        match (self, other) {
            // Map to borrowed.
            (StringOwned(l), r) => AnyValue::String(l.as_str()) == *r,
            (BinaryOwned(l), r) => AnyValue::Binary(l.as_slice()) == *r,
            #[cfg(feature = "object")]
            (ObjectOwned(l), r) => AnyValue::Object(&*l.0) == *r,
            (l, StringOwned(r)) => *l == AnyValue::String(r.as_str()),
            (l, BinaryOwned(r)) => *l == AnyValue::Binary(r.as_slice()),
            #[cfg(feature = "object")]
            (l, ObjectOwned(r)) => *l == AnyValue::Object(&*r.0),
            #[cfg(feature = "dtype-datetime")]
            (DatetimeOwned(lv, ltu, ltz), r) => {
                Datetime(*lv, *ltu, ltz.as_ref().map(|v| v.as_ref())) == *r
            },
            #[cfg(feature = "dtype-datetime")]
            (l, DatetimeOwned(rv, rtu, rtz)) => {
                *l == Datetime(*rv, *rtu, rtz.as_ref().map(|v| v.as_ref()))
            },
            #[cfg(feature = "dtype-categorical")]
            (CategoricalOwned(cat, map), r) => Categorical(*cat, map) == *r,
            #[cfg(feature = "dtype-categorical")]
            (l, CategoricalOwned(cat, map)) => *l == Categorical(*cat, map),
            #[cfg(feature = "dtype-categorical")]
            (EnumOwned(cat, map), r) => Enum(*cat, map) == *r,
            #[cfg(feature = "dtype-categorical")]
            (l, EnumOwned(cat, map)) => *l == Enum(*cat, map),

            // Comparison with null.
            (Null, Null) => null_equal,
            (Null, _) => false,
            (_, Null) => false,

            // Equality between equal types.
            (Boolean(l), Boolean(r)) => *l == *r,
            (UInt8(l), UInt8(r)) => *l == *r,
            (UInt16(l), UInt16(r)) => *l == *r,
            (UInt32(l), UInt32(r)) => *l == *r,
            (UInt64(l), UInt64(r)) => *l == *r,
            (Int8(l), Int8(r)) => *l == *r,
            (Int16(l), Int16(r)) => *l == *r,
            (Int32(l), Int32(r)) => *l == *r,
            (Int64(l), Int64(r)) => *l == *r,
            (Int128(l), Int128(r)) => *l == *r,
            (Float32(l), Float32(r)) => l.to_total_ord() == r.to_total_ord(),
            (Float64(l), Float64(r)) => l.to_total_ord() == r.to_total_ord(),
            (String(l), String(r)) => l == r,
            (Binary(l), Binary(r)) => l == r,
            #[cfg(feature = "dtype-time")]
            (Time(l), Time(r)) => *l == *r,
            #[cfg(all(feature = "dtype-datetime", feature = "dtype-date"))]
            (Date(l), Date(r)) => *l == *r,
            #[cfg(all(feature = "dtype-datetime", feature = "dtype-date"))]
            (Datetime(l, tul, tzl), Datetime(r, tur, tzr)) => {
                *l == *r && *tul == *tur && tzl == tzr
            },
            (List(l), List(r)) => l == r,
            #[cfg(feature = "dtype-categorical")]
            (Categorical(cat_l, map_l), Categorical(cat_r, map_r)) => {
                if !Arc::ptr_eq(map_l, map_r) {
                    // We can't support this because our Hash impl directly hashes the index. If you
                    // add support for this we must change the Hash impl.
                    unimplemented!(
                        "comparing categoricals with different Categories is not supported through AnyValue"
                    );
                }

                cat_l == cat_r
            },
            #[cfg(feature = "dtype-categorical")]
            (Enum(cat_l, map_l), Enum(cat_r, map_r)) => {
                if !Arc::ptr_eq(map_l, map_r) {
                    // We can't support this because our Hash impl directly hashes the index. If you
                    // add support for this we must change the Hash impl.
                    unimplemented!(
                        "comparing enums with different FrozenCategories is not supported through AnyValue"
                    );
                }

                cat_l == cat_r
            },
            #[cfg(feature = "dtype-duration")]
            (Duration(l, tu_l), Duration(r, tu_r)) => l == r && tu_l == tu_r,

            #[cfg(feature = "dtype-struct")]
            (StructOwned(l), StructOwned(r)) => struct_eq_missing(
                struct_owned_value_iter(l.as_ref()),
                struct_owned_value_iter(r.as_ref()),
                null_equal,
            ),
            #[cfg(feature = "dtype-struct")]
            (StructOwned(l), Struct(idx, arr, _)) => struct_eq_missing(
                struct_owned_value_iter(l.as_ref()),
                struct_value_iter(*idx, arr),
                null_equal,
            ),
            #[cfg(feature = "dtype-struct")]
            (Struct(idx, arr, _), StructOwned(r)) => struct_eq_missing(
                struct_value_iter(*idx, arr),
                struct_owned_value_iter(r.as_ref()),
                null_equal,
            ),
            #[cfg(feature = "dtype-struct")]
            (Struct(l_idx, l_arr, _), Struct(r_idx, r_arr, _)) => struct_eq_missing(
                struct_value_iter(*l_idx, l_arr),
                struct_value_iter(*r_idx, r_arr),
                null_equal,
            ),
            #[cfg(feature = "dtype-decimal")]
            (Decimal(l_v, l_s), Decimal(r_v, r_s)) => {
                // l_v / 10**l_s == r_v / 10**r_s
                if l_s == r_s && l_v == r_v || *l_v == 0 && *r_v == 0 {
                    true
                } else if l_s < r_s {
                    // l_v * 10**(r_s - l_s) == r_v
                    if let Some(lhs) = (|| {
                        let exp = i128::checked_pow(10, (r_s - l_s).try_into().ok()?)?;
                        l_v.checked_mul(exp)
                    })() {
                        lhs == *r_v
                    } else {
                        false
                    }
                } else {
                    // l_v == r_v * 10**(l_s - r_s)
                    if let Some(rhs) = (|| {
                        let exp = i128::checked_pow(10, (l_s - r_s).try_into().ok()?)?;
                        r_v.checked_mul(exp)
                    })() {
                        *l_v == rhs
                    } else {
                        false
                    }
                }
            },
            #[cfg(feature = "object")]
            (Object(l), Object(r)) => l == r,
            #[cfg(feature = "dtype-array")]
            (Array(l_values, l_size), Array(r_values, r_size)) => {
                if l_size != r_size {
                    return false;
                }

                debug_assert_eq!(l_values.len(), *l_size);
                debug_assert_eq!(r_values.len(), *r_size);

                let mut is_equal = true;
                for i in 0..*l_size {
                    let l = unsafe { l_values.get_unchecked(i) };
                    let r = unsafe { r_values.get_unchecked(i) };

                    is_equal &= l.eq_missing(&r, null_equal);
                }
                is_equal
            },

            (l, r) if l.to_i128().is_some() && r.to_i128().is_some() => l.to_i128() == r.to_i128(),
            (l, r) if l.to_f64().is_some() && r.to_f64().is_some() => {
                l.to_f64().unwrap().to_total_ord() == r.to_f64().unwrap().to_total_ord()
            },

            (_, _) => {
                unimplemented!(
                    "scalar eq_missing for mixed dtypes {self:?} and {other:?} is not supported"
                )
            },
        }
    }
}

impl PartialEq for AnyValue<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.eq_missing(other, true)
    }
}

impl PartialOrd for AnyValue<'_> {
    /// Only implemented for the same types and physical types!
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use AnyValue::*;
        match (self, &other) {
            // Map to borrowed.
            (StringOwned(l), r) => AnyValue::String(l.as_str()).partial_cmp(r),
            (BinaryOwned(l), r) => AnyValue::Binary(l.as_slice()).partial_cmp(r),
            #[cfg(feature = "object")]
            (ObjectOwned(l), r) => AnyValue::Object(&*l.0).partial_cmp(r),
            (l, StringOwned(r)) => l.partial_cmp(&AnyValue::String(r.as_str())),
            (l, BinaryOwned(r)) => l.partial_cmp(&AnyValue::Binary(r.as_slice())),
            #[cfg(feature = "object")]
            (l, ObjectOwned(r)) => l.partial_cmp(&AnyValue::Object(&*r.0)),
            #[cfg(feature = "dtype-datetime")]
            (DatetimeOwned(lv, ltu, ltz), r) => {
                Datetime(*lv, *ltu, ltz.as_ref().map(|v| v.as_ref())).partial_cmp(r)
            },
            #[cfg(feature = "dtype-datetime")]
            (l, DatetimeOwned(rv, rtu, rtz)) => {
                l.partial_cmp(&Datetime(*rv, *rtu, rtz.as_ref().map(|v| v.as_ref())))
            },
            #[cfg(feature = "dtype-categorical")]
            (CategoricalOwned(cat, map), r) => Categorical(*cat, map).partial_cmp(r),
            #[cfg(feature = "dtype-categorical")]
            (l, CategoricalOwned(cat, map)) => l.partial_cmp(&Categorical(*cat, map)),
            #[cfg(feature = "dtype-categorical")]
            (EnumOwned(cat, map), r) => Enum(*cat, map).partial_cmp(r),
            #[cfg(feature = "dtype-categorical")]
            (l, EnumOwned(cat, map)) => l.partial_cmp(&Enum(*cat, map)),

            // Comparison with null.
            (Null, Null) => Some(Ordering::Equal),
            (Null, _) => Some(Ordering::Less),
            (_, Null) => Some(Ordering::Greater),

            // Comparison between equal types.
            (Boolean(l), Boolean(r)) => l.partial_cmp(r),
            (UInt8(l), UInt8(r)) => l.partial_cmp(r),
            (UInt16(l), UInt16(r)) => l.partial_cmp(r),
            (UInt32(l), UInt32(r)) => l.partial_cmp(r),
            (UInt64(l), UInt64(r)) => l.partial_cmp(r),
            (Int8(l), Int8(r)) => l.partial_cmp(r),
            (Int16(l), Int16(r)) => l.partial_cmp(r),
            (Int32(l), Int32(r)) => l.partial_cmp(r),
            (Int64(l), Int64(r)) => l.partial_cmp(r),
            (Int128(l), Int128(r)) => l.partial_cmp(r),
            (Float32(l), Float32(r)) => Some(l.tot_cmp(r)),
            (Float64(l), Float64(r)) => Some(l.tot_cmp(r)),
            (String(l), String(r)) => l.partial_cmp(r),
            (Binary(l), Binary(r)) => l.partial_cmp(r),
            #[cfg(feature = "dtype-date")]
            (Date(l), Date(r)) => l.partial_cmp(r),
            #[cfg(feature = "dtype-datetime")]
            (Datetime(lt, lu, lz), Datetime(rt, ru, rz)) => {
                if lu != ru || lz != rz {
                    unimplemented!(
                        "comparing datetimes with different units or timezones is not supported"
                    );
                }

                lt.partial_cmp(rt)
            },
            #[cfg(feature = "dtype-duration")]
            (Duration(lt, lu), Duration(rt, ru)) => {
                if lu != ru {
                    unimplemented!("comparing durations with different units is not supported");
                }

                lt.partial_cmp(rt)
            },
            #[cfg(feature = "dtype-time")]
            (Time(l), Time(r)) => l.partial_cmp(r),
            #[cfg(feature = "dtype-categorical")]
            (Categorical(l_cat, l_map), Categorical(r_cat, r_map)) => unsafe {
                let l_str = l_map.cat_to_str_unchecked(*l_cat);
                let r_str = r_map.cat_to_str_unchecked(*r_cat);
                l_str.partial_cmp(r_str)
            },
            #[cfg(feature = "dtype-categorical")]
            (Enum(l_cat, l_map), Enum(r_cat, r_map)) => {
                if !Arc::ptr_eq(l_map, r_map) {
                    unimplemented!("can't order enums from different FrozenCategories")
                }
                l_cat.partial_cmp(r_cat)
            },
            (List(_), List(_)) => {
                unimplemented!("ordering for List dtype is not supported")
            },
            #[cfg(feature = "dtype-array")]
            (Array(..), Array(..)) => {
                unimplemented!("ordering for Array dtype is not supported")
            },
            #[cfg(feature = "object")]
            (Object(_), Object(_)) => {
                unimplemented!("ordering for Object dtype is not supported")
            },
            #[cfg(feature = "dtype-struct")]
            (StructOwned(_), StructOwned(_))
            | (StructOwned(_), Struct(..))
            | (Struct(..), StructOwned(_))
            | (Struct(..), Struct(..)) => {
                unimplemented!("ordering for Struct dtype is not supported")
            },
            #[cfg(feature = "dtype-decimal")]
            (Decimal(l_v, l_s), Decimal(r_v, r_s)) => {
                // l_v / 10**l_s <=> r_v / 10**r_s
                if l_s == r_s && l_v == r_v || *l_v == 0 && *r_v == 0 {
                    Some(Ordering::Equal)
                } else if l_s < r_s {
                    // l_v * 10**(r_s - l_s) <=> r_v
                    if let Some(lhs) = (|| {
                        let exp = i128::checked_pow(10, (r_s - l_s).try_into().ok()?)?;
                        l_v.checked_mul(exp)
                    })() {
                        lhs.partial_cmp(r_v)
                    } else {
                        Some(Ordering::Greater)
                    }
                } else {
                    // l_v <=> r_v * 10**(l_s - r_s)
                    if let Some(rhs) = (|| {
                        let exp = i128::checked_pow(10, (l_s - r_s).try_into().ok()?)?;
                        r_v.checked_mul(exp)
                    })() {
                        l_v.partial_cmp(&rhs)
                    } else {
                        Some(Ordering::Less)
                    }
                }
            },

            (_, _) => {
                unimplemented!(
                    "scalar ordering for mixed dtypes {self:?} and {other:?} is not supported"
                )
            },
        }
    }
}

impl TotalEq for AnyValue<'_> {
    #[inline]
    fn tot_eq(&self, other: &Self) -> bool {
        self.eq_missing(other, true)
    }
}

#[cfg(feature = "dtype-struct")]
fn struct_to_avs_static(idx: usize, arr: &StructArray, fields: &[Field]) -> Vec<AnyValue<'static>> {
    assert!(idx < arr.len());

    let arrs = arr.values();

    debug_assert_eq!(arrs.len(), fields.len());

    arrs.iter()
        .zip(fields)
        .map(|(arr, field)| {
            // SAFETY: We asserted above that the length of StructArray is larger than `idx`. Since
            // StructArray has the invariant that each array is the same length. This is okay to do
            // now.
            unsafe { arr_to_any_value(arr.as_ref(), idx, &field.dtype) }.into_static()
        })
        .collect()
}

pub trait GetAnyValue {
    /// # Safety
    ///
    /// Get an value without doing bound checks.
    unsafe fn get_unchecked(&self, index: usize) -> AnyValue<'_>;
}

impl GetAnyValue for ArrayRef {
    // Should only be called with physical types
    unsafe fn get_unchecked(&self, index: usize) -> AnyValue<'_> {
        match self.dtype() {
            ArrowDataType::Int8 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i8>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Int8(v),
                }
            },
            ArrowDataType::Int16 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i16>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Int16(v),
                }
            },
            ArrowDataType::Int32 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i32>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Int32(v),
                }
            },
            ArrowDataType::Int64 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i64>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Int64(v),
                }
            },
            ArrowDataType::Int128 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i128>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Int128(v),
                }
            },
            ArrowDataType::UInt8 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<u8>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::UInt8(v),
                }
            },
            ArrowDataType::UInt16 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<u16>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::UInt16(v),
                }
            },
            ArrowDataType::UInt32 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<u32>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::UInt32(v),
                }
            },
            ArrowDataType::UInt64 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<u64>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::UInt64(v),
                }
            },
            ArrowDataType::Float32 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<f32>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Float32(v),
                }
            },
            ArrowDataType::Float64 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<f64>>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Float64(v),
                }
            },
            ArrowDataType::Boolean => {
                let arr = self
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Boolean(v),
                }
            },
            ArrowDataType::LargeUtf8 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .unwrap_unchecked();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::String(v),
                }
            },
            _ => unimplemented!(),
        }
    }
}

impl<K: NumericNative> From<K> for AnyValue<'static> {
    fn from(value: K) -> Self {
        unsafe {
            match K::PRIMITIVE {
                PrimitiveType::Int8 => AnyValue::Int8(NumCast::from(value).unwrap_unchecked()),
                PrimitiveType::Int16 => AnyValue::Int16(NumCast::from(value).unwrap_unchecked()),
                PrimitiveType::Int32 => AnyValue::Int32(NumCast::from(value).unwrap_unchecked()),
                PrimitiveType::Int64 => AnyValue::Int64(NumCast::from(value).unwrap_unchecked()),
                PrimitiveType::Int128 => AnyValue::Int128(NumCast::from(value).unwrap_unchecked()),
                PrimitiveType::UInt8 => AnyValue::UInt8(NumCast::from(value).unwrap_unchecked()),
                PrimitiveType::UInt16 => AnyValue::UInt16(NumCast::from(value).unwrap_unchecked()),
                PrimitiveType::UInt32 => AnyValue::UInt32(NumCast::from(value).unwrap_unchecked()),
                PrimitiveType::UInt64 => AnyValue::UInt64(NumCast::from(value).unwrap_unchecked()),
                PrimitiveType::Float32 => {
                    AnyValue::Float32(NumCast::from(value).unwrap_unchecked())
                },
                PrimitiveType::Float64 => {
                    AnyValue::Float64(NumCast::from(value).unwrap_unchecked())
                },
                // not supported by polars
                _ => unreachable!(),
            }
        }
    }
}

impl<'a> From<&'a [u8]> for AnyValue<'a> {
    fn from(value: &'a [u8]) -> Self {
        AnyValue::Binary(value)
    }
}

impl<'a> From<&'a str> for AnyValue<'a> {
    fn from(value: &'a str) -> Self {
        AnyValue::String(value)
    }
}

impl From<bool> for AnyValue<'static> {
    fn from(value: bool) -> Self {
        AnyValue::Boolean(value)
    }
}

#[cfg(test)]
mod test {
    #[cfg(feature = "dtype-categorical")]
    use super::*;

    #[test]
    #[cfg(feature = "dtype-categorical")]
    fn test_arrow_dtypes_to_polars() {
        let dtypes = [
            (
                ArrowDataType::Duration(ArrowTimeUnit::Nanosecond),
                DataType::Duration(TimeUnit::Nanoseconds),
            ),
            (
                ArrowDataType::Duration(ArrowTimeUnit::Millisecond),
                DataType::Duration(TimeUnit::Milliseconds),
            ),
            (
                ArrowDataType::Date64,
                DataType::Datetime(TimeUnit::Milliseconds, None),
            ),
            (
                ArrowDataType::Timestamp(ArrowTimeUnit::Nanosecond, None),
                DataType::Datetime(TimeUnit::Nanoseconds, None),
            ),
            (
                ArrowDataType::Timestamp(ArrowTimeUnit::Microsecond, None),
                DataType::Datetime(TimeUnit::Microseconds, None),
            ),
            (
                ArrowDataType::Timestamp(ArrowTimeUnit::Millisecond, None),
                DataType::Datetime(TimeUnit::Milliseconds, None),
            ),
            (
                ArrowDataType::Timestamp(ArrowTimeUnit::Second, None),
                DataType::Datetime(TimeUnit::Milliseconds, None),
            ),
            (
                ArrowDataType::Timestamp(ArrowTimeUnit::Second, Some(PlSmallStr::EMPTY)),
                DataType::Datetime(TimeUnit::Milliseconds, None),
            ),
            (ArrowDataType::LargeUtf8, DataType::String),
            (ArrowDataType::Utf8, DataType::String),
            (ArrowDataType::LargeBinary, DataType::Binary),
            (ArrowDataType::Binary, DataType::Binary),
            (
                ArrowDataType::Time64(ArrowTimeUnit::Nanosecond),
                DataType::Time,
            ),
            (
                ArrowDataType::Time64(ArrowTimeUnit::Millisecond),
                DataType::Time,
            ),
            (
                ArrowDataType::Time64(ArrowTimeUnit::Microsecond),
                DataType::Time,
            ),
            (ArrowDataType::Time64(ArrowTimeUnit::Second), DataType::Time),
            (
                ArrowDataType::Time32(ArrowTimeUnit::Nanosecond),
                DataType::Time,
            ),
            (
                ArrowDataType::Time32(ArrowTimeUnit::Millisecond),
                DataType::Time,
            ),
            (
                ArrowDataType::Time32(ArrowTimeUnit::Microsecond),
                DataType::Time,
            ),
            (ArrowDataType::Time32(ArrowTimeUnit::Second), DataType::Time),
            (
                ArrowDataType::List(Box::new(ArrowField::new(
                    LIST_VALUES_NAME,
                    ArrowDataType::Float64,
                    true,
                ))),
                DataType::List(DataType::Float64.into()),
            ),
            (
                ArrowDataType::LargeList(Box::new(ArrowField::new(
                    LIST_VALUES_NAME,
                    ArrowDataType::Float64,
                    true,
                ))),
                DataType::List(DataType::Float64.into()),
            ),
        ];

        for (dt_a, dt_p) in dtypes {
            let dt = DataType::from_arrow_dtype(&dt_a);

            assert_eq!(dt_p, dt);
        }
    }
}
