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

use crate::bitmap::Bitmap;

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
