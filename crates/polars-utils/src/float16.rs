#[cfg(feature = "python")]
use std::convert::Infallible;
use std::fmt::{Display, LowerExp};
use std::iter::Sum;
use std::ops::*;

use bytemuck::{Pod, Zeroable};
use half;
use num_derive::*;
use num_traits::real::Real;
use num_traits::{AsPrimitive, Bounded, FromBytes, One, Pow, ToBytes, Zero};
#[cfg(feature = "python")]
use numpy::Element;
#[cfg(feature = "python")]
use pyo3::types::PyFloat;
#[cfg(feature = "python")]
use pyo3::{FromPyObject, IntoPyObject, PyErr, Python};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::nulls::IsNull;

/// A portable float16 type.
///
/// This type is a newtype wrapper around `half::f16`.
/// We intend to replace it by Rust's builtin `f16` type once it is stabilized.
#[derive(
    Debug,
    Copy,
    Clone,
    Default,
    PartialEq,
    PartialOrd,
    Zeroable,
    Pod,
    Float,
    FromPrimitive,
    Num,
    NumCast,
    NumOps,
    One,
    ToPrimitive,
    Zero,
)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[allow(non_camel_case_types)]
#[repr(transparent)]
pub struct pf16(pub half::f16);

#[cfg(feature = "dsl-schema")]
impl schemars::JsonSchema for pf16 {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "f16".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "pf16"))
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        f32::json_schema(generator)
    }
}

impl pf16 {
    pub const NAN: Self = pf16(half::f16::NAN);
    pub const INFINITY: Self = pf16(half::f16::INFINITY);
    pub const NEG_INFINITY: Self = pf16(half::f16::NEG_INFINITY);

    #[inline]
    pub fn to_le_bytes(&self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    #[inline]
    pub fn is_nan(self) -> bool {
        self.0.is_nan()
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    #[inline]
    pub fn abs(self) -> Self {
        pf16(self.0.abs())
    }

    #[inline]
    pub fn trunc(self) -> Self {
        pf16(self.0.trunc())
    }

    #[inline]
    pub fn round(self) -> Self {
        pf16(self.0.round())
    }

    #[inline]
    pub fn floor(self) -> Self {
        pf16(self.0.floor())
    }

    #[inline]
    pub fn ceil(self) -> Self {
        pf16(self.0.ceil())
    }

    #[inline]
    pub fn log(self, base: Self) -> Self {
        pf16(self.0.log(base.0))
    }

    #[inline]
    pub fn to_bits(self) -> u16 {
        self.0.to_bits()
    }

    #[inline]
    pub fn from_bits(b: u16) -> Self {
        pf16(half::f16::from_bits(b))
    }

    #[inline]
    pub fn min(self, other: Self) -> Self {
        pf16(self.0.min(other.0))
    }

    #[inline]
    pub fn max(self, other: Self) -> Self {
        pf16(self.0.max(other.0))
    }
}

impl Display for pf16 {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0.to_f32(), f)
    }
}

impl Neg for pf16 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        pf16(-self.0)
    }
}

impl AddAssign for pf16 {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.0 = self.0 + other.0;
    }
}

impl SubAssign for pf16 {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.0 = self.0 - other.0;
    }
}

impl MulAssign for pf16 {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        self.0 = self.0 * other.0;
    }
}

impl DivAssign for pf16 {
    #[inline]
    fn div_assign(&mut self, other: Self) {
        self.0 = self.0 / other.0;
    }
}

impl RemAssign for pf16 {
    #[inline]
    fn rem_assign(&mut self, other: Self) {
        self.0 = self.0 % other.0;
    }
}

impl Sum for pf16 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut acc = Zero::zero();
        for v in iter {
            acc += v.0;
        }
        pf16(acc)
    }
}

impl Pow<pf16> for pf16 {
    type Output = Self;

    #[inline]
    fn pow(self, rhs: pf16) -> Self::Output {
        pf16(self.0.powf(rhs.0))
    }
}

impl From<bool> for pf16 {
    #[inline]
    fn from(value: bool) -> Self {
        if value { One::one() } else { Zero::zero() }
    }
}

impl From<pf16> for f32 {
    #[inline]
    fn from(value: pf16) -> Self {
        value.0.to_f32()
    }
}

impl From<pf16> for f64 {
    #[inline]
    fn from(value: pf16) -> Self {
        value.0.to_f64()
    }
}

impl From<f32> for pf16 {
    #[inline]
    fn from(value: f32) -> Self {
        pf16(half::f16::from_f32(value))
    }
}

impl From<f64> for pf16 {
    #[inline]
    fn from(value: f64) -> Self {
        pf16(half::f16::from_f64(value))
    }
}

impl Bounded for pf16 {
    fn min_value() -> Self {
        pf16(half::f16::MIN)
    }

    fn max_value() -> Self {
        pf16(half::f16::MAX)
    }
}

macro_rules! impl_as_primitive {
    ($ty:ty) => {
        impl AsPrimitive<$ty> for pf16 {
            #[inline]
            fn as_(self) -> $ty {
                self.0.as_()
            }
        }
    };
}

impl_as_primitive!(u8);
impl_as_primitive!(u16);
impl_as_primitive!(u32);
impl_as_primitive!(u64);
impl_as_primitive!(i8);
impl_as_primitive!(i16);
impl_as_primitive!(i32);
impl_as_primitive!(i64);
impl_as_primitive!(f32);
impl_as_primitive!(f64);

impl AsPrimitive<pf16> for pf16 {
    #[inline]
    fn as_(self) -> pf16 {
        self
    }
}

impl AsPrimitive<u128> for pf16 {
    #[inline]
    fn as_(self) -> u128 {
        self.0.to_f64() as u128
    }
}

impl AsPrimitive<i128> for pf16 {
    #[inline]
    fn as_(self) -> i128 {
        self.0.to_f64() as i128
    }
}

macro_rules! impl_as_primitive_pf16_from {
    ($ty:ty) => {
        impl AsPrimitive<pf16> for $ty {
            #[inline]
            fn as_(self) -> pf16 {
                pf16(<$ty>::as_(self))
            }
        }
    };
}
impl_as_primitive_pf16_from!(usize);
impl_as_primitive_pf16_from!(u8);
impl_as_primitive_pf16_from!(u16);
impl_as_primitive_pf16_from!(u32);
impl_as_primitive_pf16_from!(u64);
impl_as_primitive_pf16_from!(isize);
impl_as_primitive_pf16_from!(i8);
impl_as_primitive_pf16_from!(i16);
impl_as_primitive_pf16_from!(i32);
impl_as_primitive_pf16_from!(i64);
impl_as_primitive_pf16_from!(f32);
impl_as_primitive_pf16_from!(f64);

impl AsPrimitive<pf16> for i128 {
    #[inline]
    fn as_(self) -> pf16 {
        pf16(half::f16::from_f64(self as f64))
    }
}

impl AsPrimitive<pf16> for u128 {
    #[inline]
    fn as_(self) -> pf16 {
        pf16(half::f16::from_f64(self as f64))
    }
}

impl IsNull for pf16 {
    const HAS_NULLS: bool = false;
    type Inner = pf16;

    #[inline(always)]
    fn is_null(&self) -> bool {
        false
    }
    #[inline]
    fn unwrap_inner(self) -> Self::Inner {
        self
    }
}

impl ToBytes for pf16 {
    type Bytes = [u8; 2];

    #[inline]
    fn to_be_bytes(&self) -> Self::Bytes {
        self.0.to_be_bytes()
    }

    #[inline]
    fn to_le_bytes(&self) -> Self::Bytes {
        self.0.to_le_bytes()
    }

    #[inline]
    fn to_ne_bytes(&self) -> Self::Bytes {
        self.0.to_ne_bytes()
    }
}

impl FromBytes for pf16 {
    type Bytes = [u8; 2];

    #[inline]
    fn from_be_bytes(bytes: &Self::Bytes) -> Self {
        pf16(half::f16::from_be_bytes(*bytes))
    }

    #[inline]
    fn from_le_bytes(bytes: &Self::Bytes) -> Self {
        pf16(half::f16::from_le_bytes(*bytes))
    }

    #[inline]
    fn from_ne_bytes(bytes: &Self::Bytes) -> Self {
        pf16(half::f16::from_ne_bytes(*bytes))
    }
}

impl LowerExp for pf16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        LowerExp::fmt(&self.0.to_f32(), f)
    }
}

#[cfg(feature = "python")]
impl<'py> IntoPyObject<'py> for pf16 {
    type Target = PyFloat;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = Infallible;

    #[inline]
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        f32::into_pyobject(self.into(), py)
    }
}

#[cfg(feature = "python")]
impl<'a, 'py> FromPyObject<'a, 'py> for pf16 {
    type Error = PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let v: f32 = ob.extract()?;
        Ok(v.as_())
    }
}

#[cfg(feature = "python")]
unsafe impl Element for pf16 {
    const IS_COPY: bool = half::f16::IS_COPY;

    fn get_dtype(py: Python<'_>) -> pyo3::Bound<'_, numpy::PyArrayDescr> {
        half::f16::get_dtype(py)
    }
    fn clone_ref(&self, py: Python<'_>) -> Self {
        pf16(half::f16::clone_ref(&self.0, py))
    }
}
