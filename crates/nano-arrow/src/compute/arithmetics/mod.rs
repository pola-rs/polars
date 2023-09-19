//! Defines basic arithmetic kernels for [`PrimitiveArray`](crate::array::PrimitiveArray)s.
//!
//! The Arithmetics module is composed by basic arithmetics operations that can
//! be performed on [`PrimitiveArray`](crate::array::PrimitiveArray).
//!
//! Whenever possible, each operation declares variations
//! of the basic operation that offers different guarantees:
//! * plain: panics on overflowing and underflowing.
//! * checked: turns an overflowing to a null.
//! * saturating: turns the overflowing to the MAX or MIN value respectively.
//! * overflowing: returns an extra [`Bitmap`] denoting whether the operation overflowed.
//! * adaptive: for [`Decimal`](crate::datatypes::DataType::Decimal) only,
//!   adjusts the precision and scale to make the resulting value fit.
#[forbid(unsafe_code)]
pub mod basic;
#[cfg(feature = "compute_arithmetics_decimal")]
pub mod decimal;
pub mod time;

use crate::array::{Array, DictionaryArray, PrimitiveArray};
use crate::bitmap::Bitmap;
use crate::datatypes::{DataType, IntervalUnit, TimeUnit};
use crate::scalar::{PrimitiveScalar, Scalar};
use crate::types::NativeType;

fn binary_dyn<T: NativeType, F: Fn(&PrimitiveArray<T>, &PrimitiveArray<T>) -> PrimitiveArray<T>>(
    lhs: &dyn Array,
    rhs: &dyn Array,
    op: F,
) -> Box<dyn Array> {
    let lhs = lhs.as_any().downcast_ref().unwrap();
    let rhs = rhs.as_any().downcast_ref().unwrap();
    op(lhs, rhs).boxed()
}

// Macro to create a `match` statement with dynamic dispatch to functions based on
// the array's logical types
macro_rules! arith {
    ($lhs:expr, $rhs:expr, $op:tt $(, decimal = $op_decimal:tt )? $(, duration = $op_duration:tt )? $(, interval = $op_interval:tt )? $(, timestamp = $op_timestamp:tt )?) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        use DataType::*;
        match (lhs.data_type(), rhs.data_type()) {
            (Int8, Int8) => binary_dyn::<i8, _>(lhs, rhs, basic::$op),
            (Int16, Int16) => binary_dyn::<i16, _>(lhs, rhs, basic::$op),
            (Int32, Int32) => binary_dyn::<i32, _>(lhs, rhs, basic::$op),
            (Int64, Int64) | (Duration(_), Duration(_)) => {
                binary_dyn::<i64, _>(lhs, rhs, basic::$op)
            }
            (UInt8, UInt8) => binary_dyn::<u8, _>(lhs, rhs, basic::$op),
            (UInt16, UInt16) => binary_dyn::<u16, _>(lhs, rhs, basic::$op),
            (UInt32, UInt32) => binary_dyn::<u32, _>(lhs, rhs, basic::$op),
            (UInt64, UInt64) => binary_dyn::<u64, _>(lhs, rhs, basic::$op),
            (Float32, Float32) => binary_dyn::<f32, _>(lhs, rhs, basic::$op),
            (Float64, Float64) => binary_dyn::<f64, _>(lhs, rhs, basic::$op),
            $ (
            (Decimal(_, _), Decimal(_, _)) => {
                let lhs = lhs.as_any().downcast_ref().unwrap();
                let rhs = rhs.as_any().downcast_ref().unwrap();
                Box::new(decimal::$op_decimal(lhs, rhs)) as Box<dyn Array>
            }
            )?
            $ (
            (Time32(TimeUnit::Second), Duration(_))
            | (Time32(TimeUnit::Millisecond), Duration(_))
            | (Date32, Duration(_)) => {
                let lhs = lhs.as_any().downcast_ref().unwrap();
                let rhs = rhs.as_any().downcast_ref().unwrap();
                Box::new(time::$op_duration::<i32>(lhs, rhs)) as Box<dyn Array>
            }
            (Time64(TimeUnit::Microsecond), Duration(_))
            | (Time64(TimeUnit::Nanosecond), Duration(_))
            | (Date64, Duration(_))
            | (Timestamp(_, _), Duration(_)) => {
                let lhs = lhs.as_any().downcast_ref().unwrap();
                let rhs = rhs.as_any().downcast_ref().unwrap();
                Box::new(time::$op_duration::<i64>(lhs, rhs)) as Box<dyn Array>
            }
            )?
            $ (
            (Timestamp(_, _), Interval(IntervalUnit::MonthDayNano)) => {
                let lhs = lhs.as_any().downcast_ref().unwrap();
                let rhs = rhs.as_any().downcast_ref().unwrap();
                time::$op_interval(lhs, rhs).map(|x| Box::new(x) as Box<dyn Array>).unwrap()
            }
            )?
            $ (
            (Timestamp(_, None), Timestamp(_, None)) => {
                let lhs = lhs.as_any().downcast_ref().unwrap();
                let rhs = rhs.as_any().downcast_ref().unwrap();
                time::$op_timestamp(lhs, rhs).map(|x| Box::new(x) as Box<dyn Array>).unwrap()
            }
            )?
            _ => todo!(
                "Addition of {:?} with {:?} is not supported",
                lhs.data_type(),
                rhs.data_type()
            ),
        }
    }};
}

fn binary_scalar<T: NativeType, F: Fn(&PrimitiveArray<T>, &T) -> PrimitiveArray<T>>(
    lhs: &PrimitiveArray<T>,
    rhs: &PrimitiveScalar<T>,
    op: F,
) -> PrimitiveArray<T> {
    let rhs = if let Some(rhs) = *rhs.value() {
        rhs
    } else {
        return PrimitiveArray::<T>::new_null(lhs.data_type().clone(), lhs.len());
    };
    op(lhs, &rhs)
}

fn binary_scalar_dyn<T: NativeType, F: Fn(&PrimitiveArray<T>, &T) -> PrimitiveArray<T>>(
    lhs: &dyn Array,
    rhs: &dyn Scalar,
    op: F,
) -> Box<dyn Array> {
    let lhs = lhs.as_any().downcast_ref().unwrap();
    let rhs = rhs.as_any().downcast_ref().unwrap();
    binary_scalar(lhs, rhs, op).boxed()
}

// Macro to create a `match` statement with dynamic dispatch to functions based on
// the array's logical types
macro_rules! arith_scalar {
    ($lhs:expr, $rhs:expr, $op:tt $(, decimal = $op_decimal:tt )? $(, duration = $op_duration:tt )? $(, interval = $op_interval:tt )? $(, timestamp = $op_timestamp:tt )?) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        use DataType::*;
        match (lhs.data_type(), rhs.data_type()) {
            (Int8, Int8) => binary_scalar_dyn::<i8, _>(lhs, rhs, basic::$op),
            (Int16, Int16) => binary_scalar_dyn::<i16, _>(lhs, rhs, basic::$op),
            (Int32, Int32) => binary_scalar_dyn::<i32, _>(lhs, rhs, basic::$op),
            (Int64, Int64) | (Duration(_), Duration(_)) => {
                binary_scalar_dyn::<i64, _>(lhs, rhs, basic::$op)
            }
            (UInt8, UInt8) => binary_scalar_dyn::<u8, _>(lhs, rhs, basic::$op),
            (UInt16, UInt16) => binary_scalar_dyn::<u16, _>(lhs, rhs, basic::$op),
            (UInt32, UInt32) => binary_scalar_dyn::<u32, _>(lhs, rhs, basic::$op),
            (UInt64, UInt64) => binary_scalar_dyn::<u64, _>(lhs, rhs, basic::$op),
            (Float32, Float32) => binary_scalar_dyn::<f32, _>(lhs, rhs, basic::$op),
            (Float64, Float64) => binary_scalar_dyn::<f64, _>(lhs, rhs, basic::$op),
            $ (
            (Decimal(_, _), Decimal(_, _)) => {
                let lhs = lhs.as_any().downcast_ref().unwrap();
                let rhs = rhs.as_any().downcast_ref().unwrap();
                decimal::$op_decimal(lhs, rhs).boxed()
            }
            )?
            $ (
            (Time32(TimeUnit::Second), Duration(_))
            | (Time32(TimeUnit::Millisecond), Duration(_))
            | (Date32, Duration(_)) => {
                let lhs = lhs.as_any().downcast_ref().unwrap();
                let rhs = rhs.as_any().downcast_ref().unwrap();
                time::$op_duration::<i32>(lhs, rhs).boxed()
            }
            (Time64(TimeUnit::Microsecond), Duration(_))
            | (Time64(TimeUnit::Nanosecond), Duration(_))
            | (Date64, Duration(_))
            | (Timestamp(_, _), Duration(_)) => {
                let lhs = lhs.as_any().downcast_ref().unwrap();
                let rhs = rhs.as_any().downcast_ref().unwrap();
                time::$op_duration::<i64>(lhs, rhs).boxed()
            }
            )?
            $ (
            (Timestamp(_, _), Interval(IntervalUnit::MonthDayNano)) => {
                let lhs = lhs.as_any().downcast_ref().unwrap();
                let rhs = rhs.as_any().downcast_ref().unwrap();
                time::$op_interval(lhs, rhs).unwrap().boxed()
            }
            )?
            $ (
            (Timestamp(_, None), Timestamp(_, None)) => {
                let lhs = lhs.as_any().downcast_ref().unwrap();
                let rhs = rhs.as_any().downcast_ref().unwrap();
                time::$op_timestamp(lhs, rhs).unwrap().boxed()
            }
            )?
            _ => todo!(
                "Addition of {:?} with {:?} is not supported",
                lhs.data_type(),
                rhs.data_type()
            ),
        }
    }};
}

/// Adds two [`Array`]s.
/// # Panic
/// This function panics iff
/// * the operation is not supported for the logical types (use [`can_add`] to check)
/// * the arrays have a different length
/// * one of the arrays is a timestamp with timezone and the timezone is not valid.
pub fn add(lhs: &dyn Array, rhs: &dyn Array) -> Box<dyn Array> {
    arith!(
        lhs,
        rhs,
        add,
        duration = add_duration,
        interval = add_interval
    )
}

/// Adds an [`Array`] and a [`Scalar`].
/// # Panic
/// This function panics iff
/// * the operation is not supported for the logical types (use [`can_add`] to check)
/// * the arrays have a different length
/// * one of the arrays is a timestamp with timezone and the timezone is not valid.
pub fn add_scalar(lhs: &dyn Array, rhs: &dyn Scalar) -> Box<dyn Array> {
    arith_scalar!(
        lhs,
        rhs,
        add_scalar,
        duration = add_duration_scalar,
        interval = add_interval_scalar
    )
}

/// Returns whether two [`DataType`]s can be added by [`add`].
pub fn can_add(lhs: &DataType, rhs: &DataType) -> bool {
    use DataType::*;
    matches!(
        (lhs, rhs),
        (Int8, Int8)
            | (Int16, Int16)
            | (Int32, Int32)
            | (Int64, Int64)
            | (UInt8, UInt8)
            | (UInt16, UInt16)
            | (UInt32, UInt32)
            | (UInt64, UInt64)
            | (Float64, Float64)
            | (Float32, Float32)
            | (Duration(_), Duration(_))
            | (Decimal(_, _), Decimal(_, _))
            | (Date32, Duration(_))
            | (Date64, Duration(_))
            | (Time32(TimeUnit::Millisecond), Duration(_))
            | (Time32(TimeUnit::Second), Duration(_))
            | (Time64(TimeUnit::Microsecond), Duration(_))
            | (Time64(TimeUnit::Nanosecond), Duration(_))
            | (Timestamp(_, _), Duration(_))
            | (Timestamp(_, _), Interval(IntervalUnit::MonthDayNano))
    )
}

/// Subtracts two [`Array`]s.
/// # Panic
/// This function panics iff
/// * the operation is not supported for the logical types (use [`can_sub`] to check)
/// * the arrays have a different length
/// * one of the arrays is a timestamp with timezone and the timezone is not valid.
pub fn sub(lhs: &dyn Array, rhs: &dyn Array) -> Box<dyn Array> {
    arith!(
        lhs,
        rhs,
        sub,
        decimal = sub,
        duration = subtract_duration,
        timestamp = subtract_timestamps
    )
}

/// Adds an [`Array`] and a [`Scalar`].
/// # Panic
/// This function panics iff
/// * the operation is not supported for the logical types (use [`can_sub`] to check)
/// * the arrays have a different length
/// * one of the arrays is a timestamp with timezone and the timezone is not valid.
pub fn sub_scalar(lhs: &dyn Array, rhs: &dyn Scalar) -> Box<dyn Array> {
    arith_scalar!(
        lhs,
        rhs,
        sub_scalar,
        duration = sub_duration_scalar,
        timestamp = sub_timestamps_scalar
    )
}

/// Returns whether two [`DataType`]s can be subtracted by [`sub`].
pub fn can_sub(lhs: &DataType, rhs: &DataType) -> bool {
    use DataType::*;
    matches!(
        (lhs, rhs),
        (Int8, Int8)
            | (Int16, Int16)
            | (Int32, Int32)
            | (Int64, Int64)
            | (UInt8, UInt8)
            | (UInt16, UInt16)
            | (UInt32, UInt32)
            | (UInt64, UInt64)
            | (Float64, Float64)
            | (Float32, Float32)
            | (Duration(_), Duration(_))
            | (Decimal(_, _), Decimal(_, _))
            | (Date32, Duration(_))
            | (Date64, Duration(_))
            | (Time32(TimeUnit::Millisecond), Duration(_))
            | (Time32(TimeUnit::Second), Duration(_))
            | (Time64(TimeUnit::Microsecond), Duration(_))
            | (Time64(TimeUnit::Nanosecond), Duration(_))
            | (Timestamp(_, _), Duration(_))
            | (Timestamp(_, None), Timestamp(_, None))
    )
}

/// Multiply two [`Array`]s.
/// # Panic
/// This function panics iff
/// * the operation is not supported for the logical types (use [`can_mul`] to check)
/// * the arrays have a different length
pub fn mul(lhs: &dyn Array, rhs: &dyn Array) -> Box<dyn Array> {
    arith!(lhs, rhs, mul, decimal = mul)
}

/// Multiply an [`Array`] with a [`Scalar`].
/// # Panic
/// This function panics iff
/// * the operation is not supported for the logical types (use [`can_mul`] to check)
pub fn mul_scalar(lhs: &dyn Array, rhs: &dyn Scalar) -> Box<dyn Array> {
    arith_scalar!(lhs, rhs, mul_scalar, decimal = mul_scalar)
}

/// Returns whether two [`DataType`]s can be multiplied by [`mul`].
pub fn can_mul(lhs: &DataType, rhs: &DataType) -> bool {
    use DataType::*;
    matches!(
        (lhs, rhs),
        (Int8, Int8)
            | (Int16, Int16)
            | (Int32, Int32)
            | (Int64, Int64)
            | (UInt8, UInt8)
            | (UInt16, UInt16)
            | (UInt32, UInt32)
            | (UInt64, UInt64)
            | (Float64, Float64)
            | (Float32, Float32)
            | (Decimal(_, _), Decimal(_, _))
    )
}

/// Divide of two [`Array`]s.
/// # Panic
/// This function panics iff
/// * the operation is not supported for the logical types (use [`can_div`] to check)
/// * the arrays have a different length
pub fn div(lhs: &dyn Array, rhs: &dyn Array) -> Box<dyn Array> {
    arith!(lhs, rhs, div, decimal = div)
}

/// Divide an [`Array`] with a [`Scalar`].
/// # Panic
/// This function panics iff
/// * the operation is not supported for the logical types (use [`can_div`] to check)
pub fn div_scalar(lhs: &dyn Array, rhs: &dyn Scalar) -> Box<dyn Array> {
    arith_scalar!(lhs, rhs, div_scalar, decimal = div_scalar)
}

/// Returns whether two [`DataType`]s can be divided by [`div`].
pub fn can_div(lhs: &DataType, rhs: &DataType) -> bool {
    can_mul(lhs, rhs)
}

/// Remainder of two [`Array`]s.
/// # Panic
/// This function panics iff
/// * the operation is not supported for the logical types (use [`can_rem`] to check)
/// * the arrays have a different length
pub fn rem(lhs: &dyn Array, rhs: &dyn Array) -> Box<dyn Array> {
    arith!(lhs, rhs, rem)
}

/// Returns whether two [`DataType`]s "can be remainder" by [`rem`].
pub fn can_rem(lhs: &DataType, rhs: &DataType) -> bool {
    use DataType::*;
    matches!(
        (lhs, rhs),
        (Int8, Int8)
            | (Int16, Int16)
            | (Int32, Int32)
            | (Int64, Int64)
            | (UInt8, UInt8)
            | (UInt16, UInt16)
            | (UInt32, UInt32)
            | (UInt64, UInt64)
            | (Float64, Float64)
            | (Float32, Float32)
    )
}

macro_rules! with_match_negatable {(
    $key_type:expr, | $_:tt $T:ident | $($body:tt)*
) => ({
    macro_rules! __with_ty__ {( $_ $T:ident ) => ( $($body)* )}
    use crate::datatypes::PrimitiveType::*;
    use crate::types::{days_ms, months_days_ns, i256};
    match $key_type {
        Int8 => __with_ty__! { i8 },
        Int16 => __with_ty__! { i16 },
        Int32 => __with_ty__! { i32 },
        Int64 => __with_ty__! { i64 },
        Int128 => __with_ty__! { i128 },
        Int256 => __with_ty__! { i256 },
        DaysMs => __with_ty__! { days_ms },
        MonthDayNano => __with_ty__! { months_days_ns },
        UInt8 | UInt16 | UInt32 | UInt64 | Float16 => todo!(),
        Float32 => __with_ty__! { f32 },
        Float64 => __with_ty__! { f64 },
    }
})}

/// Negates an [`Array`].
/// # Panic
/// This function panics iff either
/// * the operation is not supported for the logical type (use [`can_neg`] to check)
/// * the operation overflows
pub fn neg(array: &dyn Array) -> Box<dyn Array> {
    use crate::datatypes::PhysicalType::*;
    match array.data_type().to_physical_type() {
        Primitive(primitive) => with_match_negatable!(primitive, |$T| {
            let array = array.as_any().downcast_ref().unwrap();

            let result = basic::negate::<$T>(array);
            Box::new(result) as Box<dyn Array>
        }),
        Dictionary(key) => match_integer_type!(key, |$T| {
            let array = array.as_any().downcast_ref::<DictionaryArray<$T>>().unwrap();

            let values = neg(array.values().as_ref());

            // safety - this operation only applies to values and thus preserves the dictionary's invariant
            unsafe{
                DictionaryArray::<$T>::try_new_unchecked(array.data_type().clone(), array.keys().clone(), values).unwrap().boxed()
            }
        }),
        _ => todo!(),
    }
}

/// Whether [`neg`] is supported for a given [`DataType`]
pub fn can_neg(data_type: &DataType) -> bool {
    if let DataType::Dictionary(_, values, _) = data_type.to_logical_type() {
        return can_neg(values.as_ref());
    }

    use crate::datatypes::PhysicalType::*;
    use crate::datatypes::PrimitiveType::*;
    matches!(
        data_type.to_physical_type(),
        Primitive(Int8)
            | Primitive(Int16)
            | Primitive(Int32)
            | Primitive(Int64)
            | Primitive(Float64)
            | Primitive(Float32)
            | Primitive(DaysMs)
            | Primitive(MonthDayNano)
    )
}

/// Defines basic addition operation for primitive arrays
pub trait ArrayAdd<Rhs>: Sized {
    /// Adds itself to `rhs`
    fn add(&self, rhs: &Rhs) -> Self;
}

/// Defines wrapping addition operation for primitive arrays
pub trait ArrayWrappingAdd<Rhs>: Sized {
    /// Adds itself to `rhs` using wrapping addition
    fn wrapping_add(&self, rhs: &Rhs) -> Self;
}

/// Defines checked addition operation for primitive arrays
pub trait ArrayCheckedAdd<Rhs>: Sized {
    /// Checked add
    fn checked_add(&self, rhs: &Rhs) -> Self;
}

/// Defines saturating addition operation for primitive arrays
pub trait ArraySaturatingAdd<Rhs>: Sized {
    /// Saturating add
    fn saturating_add(&self, rhs: &Rhs) -> Self;
}

/// Defines Overflowing addition operation for primitive arrays
pub trait ArrayOverflowingAdd<Rhs>: Sized {
    /// Overflowing add
    fn overflowing_add(&self, rhs: &Rhs) -> (Self, Bitmap);
}

/// Defines basic subtraction operation for primitive arrays
pub trait ArraySub<Rhs>: Sized {
    /// subtraction
    fn sub(&self, rhs: &Rhs) -> Self;
}

/// Defines wrapping subtraction operation for primitive arrays
pub trait ArrayWrappingSub<Rhs>: Sized {
    /// wrapping subtraction
    fn wrapping_sub(&self, rhs: &Rhs) -> Self;
}

/// Defines checked subtraction operation for primitive arrays
pub trait ArrayCheckedSub<Rhs>: Sized {
    /// checked subtraction
    fn checked_sub(&self, rhs: &Rhs) -> Self;
}

/// Defines saturating subtraction operation for primitive arrays
pub trait ArraySaturatingSub<Rhs>: Sized {
    /// saturarting subtraction
    fn saturating_sub(&self, rhs: &Rhs) -> Self;
}

/// Defines Overflowing subtraction operation for primitive arrays
pub trait ArrayOverflowingSub<Rhs>: Sized {
    /// overflowing subtraction
    fn overflowing_sub(&self, rhs: &Rhs) -> (Self, Bitmap);
}

/// Defines basic multiplication operation for primitive arrays
pub trait ArrayMul<Rhs>: Sized {
    /// multiplication
    fn mul(&self, rhs: &Rhs) -> Self;
}

/// Defines wrapping multiplication operation for primitive arrays
pub trait ArrayWrappingMul<Rhs>: Sized {
    /// wrapping multiplication
    fn wrapping_mul(&self, rhs: &Rhs) -> Self;
}

/// Defines checked multiplication operation for primitive arrays
pub trait ArrayCheckedMul<Rhs>: Sized {
    /// checked multiplication
    fn checked_mul(&self, rhs: &Rhs) -> Self;
}

/// Defines saturating multiplication operation for primitive arrays
pub trait ArraySaturatingMul<Rhs>: Sized {
    /// saturating multiplication
    fn saturating_mul(&self, rhs: &Rhs) -> Self;
}

/// Defines Overflowing multiplication operation for primitive arrays
pub trait ArrayOverflowingMul<Rhs>: Sized {
    /// overflowing multiplication
    fn overflowing_mul(&self, rhs: &Rhs) -> (Self, Bitmap);
}

/// Defines basic division operation for primitive arrays
pub trait ArrayDiv<Rhs>: Sized {
    /// division
    fn div(&self, rhs: &Rhs) -> Self;
}

/// Defines checked division operation for primitive arrays
pub trait ArrayCheckedDiv<Rhs>: Sized {
    /// checked division
    fn checked_div(&self, rhs: &Rhs) -> Self;
}

/// Defines basic reminder operation for primitive arrays
pub trait ArrayRem<Rhs>: Sized {
    /// remainder
    fn rem(&self, rhs: &Rhs) -> Self;
}

/// Defines checked reminder operation for primitive arrays
pub trait ArrayCheckedRem<Rhs>: Sized {
    /// checked remainder
    fn checked_rem(&self, rhs: &Rhs) -> Self;
}
