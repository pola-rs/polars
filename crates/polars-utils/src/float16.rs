#[cfg(feature = "dsl-schema")]
use std::borrow::Cow;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::iter::Sum;
use std::ops::*;

use bytemuck::{Pod, Zeroable};
use half;
use num_derive::*;
use num_traits::real::Real;
use num_traits::{AsPrimitive, Bounded, One, Pow, Zero};
#[cfg(feature = "dsl-schema")]
use schemars::schema::Schema;
#[cfg(feature = "dsl-schema")]
use schemars::{JsonSchema, SchemaGenerator};
use serde::{Deserialize, Serialize};

use crate::nulls::IsNull;
use crate::total_ord::{ToTotalOrd, TotalEq, TotalHash, TotalOrd, TotalOrdWrap};

/// Type representation of the Float16 physical type
/// TODO: [amber] comment
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
impl JsonSchema for pf16 {
    fn schema_name() -> String {
        "pf16".into()
    }

    fn schema_id() -> Cow<'static, str> {
        concat!(module_path!(), "::pf16").into()
    }

    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        use schemars::schema::*;

        SchemaObject {
            instance_type: Some(InstanceType::Number.into()),
            format: Some("half".to_owned()),
            ..Default::default()
        }
        .into()
    }
}

/// Converts an f32 into a canonical form, where -0 == 0 and all NaNs map to
/// the same value.
#[inline]
pub fn canonical_f16(x: pf16) -> pf16 {
    let convert_zero = x.0.abs(); // zero out the sign bit if the f16 is zero.
    pf16(if convert_zero.is_nan() {
        half::f16::from_bits(0x7c00) // Canonical quiet NaN.
    } else {
        convert_zero
    })
}

impl pf16 {
    pub const NAN: Self = pf16(half::f16::NAN);
    pub const INFINITY: Self = pf16(half::f16::INFINITY);
    pub const NEG_INFINITY: Self = pf16(half::f16::NEG_INFINITY);

    #[inline]
    pub fn to_ne_bytes(&self) -> [u8; 2] {
        self.0.to_ne_bytes()
    }

    #[inline]
    pub fn from_ne_bytes(bytes: [u8; 2]) -> Self {
        pf16(half::f16::from_ne_bytes(bytes))
    }

    #[inline]
    pub fn to_le_bytes(&self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    #[inline]
    pub fn from_le_bytes(bytes: [u8; 2]) -> Self {
        pf16(half::f16::from_le_bytes(bytes))
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
        self.0.to_f32().fmt(f)
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

// TODO: [amber] Maybe implement more combinations?
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

impl TotalHash for pf16 {
    #[inline]
    fn tot_hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        canonical_f16(*self).0.to_bits().hash(state)
    }
}

impl ToTotalOrd for pf16 {
    type TotalOrdItem = TotalOrdWrap<pf16>;
    type SourceItem = pf16;

    #[inline]
    fn to_total_ord(&self) -> Self::TotalOrdItem {
        TotalOrdWrap(*self)
    }

    #[inline]
    fn peel_total_ord(ord_item: Self::TotalOrdItem) -> Self::SourceItem {
        ord_item.0
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

impl TotalEq for pf16 {
    #[inline]
    fn tot_eq(&self, other: &Self) -> bool {
        if self.0.is_nan() {
            other.0.is_nan()
        } else {
            self == other
        }
    }
}

impl TotalOrd for pf16 {
    #[inline]
    fn tot_cmp(&self, _other: &Self) -> std::cmp::Ordering {
        unimplemented!()
    }
}
