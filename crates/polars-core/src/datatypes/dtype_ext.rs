//! The parts of a [`DataType`] that are about *values* rather than about the type.
//!
//! [`DataType`] lives in `polars-dtype`, below `polars-compute`, so that a kernel can be
//! dispatched on it. An [`AnyValue`] and a [`Scalar`] are values and live here with the rest of
//! them, so the handful of `DataType` methods that answer *in* values are an extension trait
//! rather than inherent methods.

use polars_utils::float16::pf16;

use crate::prelude::*;

/// The bounds of a [`DataType`], and whether a value is one of the ones it holds.
pub trait DataTypeValueExt {
    /// Try to get the maximum value for this datatype.
    fn max(&self) -> PolarsResult<Scalar>;

    /// Try to get the minimum value for this datatype.
    fn min(&self) -> PolarsResult<Scalar>;

    /// Whether `other` is a value this datatype holds.
    fn value_within_range(&self, other: AnyValue) -> bool;
}

impl DataTypeValueExt for DataType {
    fn max(&self) -> PolarsResult<Scalar> {
        use DataType::*;
        let v = match self {
            Int8 => Scalar::from(i8::MAX),
            Int16 => Scalar::from(i16::MAX),
            Int32 => Scalar::from(i32::MAX),
            Int64 => Scalar::from(i64::MAX),
            Int128 => Scalar::from(i128::MAX),
            UInt8 => Scalar::from(u8::MAX),
            UInt16 => Scalar::from(u16::MAX),
            UInt32 => Scalar::from(u32::MAX),
            UInt64 => Scalar::from(u64::MAX),
            UInt128 => Scalar::from(u128::MAX),
            Float16 => Scalar::from(pf16::INFINITY),
            Float32 => Scalar::from(f32::INFINITY),
            Float64 => Scalar::from(f64::INFINITY),
            #[cfg(feature = "dtype-time")]
            Time => Scalar::new(Time, AnyValue::Time(NS_IN_DAY - 1)),
            dt => polars_bail!(ComputeError: "cannot determine upper bound for dtype `{dt}`"),
        };
        Ok(v)
    }

    fn min(&self) -> PolarsResult<Scalar> {
        use DataType::*;
        let v = match self {
            Int8 => Scalar::from(i8::MIN),
            Int16 => Scalar::from(i16::MIN),
            Int32 => Scalar::from(i32::MIN),
            Int64 => Scalar::from(i64::MIN),
            Int128 => Scalar::from(i128::MIN),
            UInt8 => Scalar::from(u8::MIN),
            UInt16 => Scalar::from(u16::MIN),
            UInt32 => Scalar::from(u32::MIN),
            UInt64 => Scalar::from(u64::MIN),
            UInt128 => Scalar::from(u128::MIN),
            Float16 => Scalar::from(pf16::NEG_INFINITY),
            Float32 => Scalar::from(f32::NEG_INFINITY),
            Float64 => Scalar::from(f64::NEG_INFINITY),
            #[cfg(feature = "dtype-time")]
            Time => Scalar::new(Time, AnyValue::Time(0)),
            dt => polars_bail!(ComputeError: "cannot determine lower bound for dtype `{}`", dt),
        };
        Ok(v)
    }

    fn value_within_range(&self, other: AnyValue) -> bool {
        use DataType::*;
        match self {
            UInt8 => other.extract::<u8>().is_some(),
            #[cfg(feature = "dtype-u16")]
            UInt16 => other.extract::<u16>().is_some(),
            UInt32 => other.extract::<u32>().is_some(),
            UInt64 => other.extract::<u64>().is_some(),
            #[cfg(feature = "dtype-u128")]
            UInt128 => other.extract::<u128>().is_some(),
            #[cfg(feature = "dtype-i8")]
            Int8 => other.extract::<i8>().is_some(),
            #[cfg(feature = "dtype-i16")]
            Int16 => other.extract::<i16>().is_some(),
            Int32 => other.extract::<i32>().is_some(),
            Int64 => other.extract::<i64>().is_some(),
            #[cfg(feature = "dtype-i128")]
            Int128 => other.extract::<i128>().is_some(),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {

    /// The type of the integer a dynamic integer literal reads as is answered in `polars-dtype`,
    /// which has no [`AnyValue`] to build; the value itself is built here. The two have to agree,
    /// or a literal would be typed as one thing and materialized as another.
    #[test]
    fn a_dynamic_integer_is_typed_as_it_materializes() {
        for v in [
            0i128,
            1,
            i32::MAX as i128,
            i32::MAX as i128 + 1,
            i64::MAX as i128,
            i64::MAX as i128 + 1,
            u64::MAX as i128,
            i128::MIN,
        ] {
            assert_eq!(
                polars_dtype::dyn_int_dtype(v),
                crate::utils::materialize_dyn_int(v).dtype(),
                "{v} is typed as one thing and materialized as another",
            );
        }
    }
}
