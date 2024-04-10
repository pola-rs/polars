use std::ops::Neg;
use std::panic::RefUnwindSafe;

use bytemuck::{Pod, Zeroable};
use polars_utils::min_max::MinMax;
use polars_utils::nulls::IsNull;
use polars_utils::total_ord::{TotalEq, TotalOrd};

use super::PrimitiveType;

/// Sealed trait implemented by all physical types that can be allocated,
/// serialized and deserialized by this crate.
/// All O(N) allocations in this crate are done for this trait alone.
pub trait NativeType:
    super::private::Sealed
    + Pod
    + Send
    + Sync
    + Sized
    + RefUnwindSafe
    + std::fmt::Debug
    + std::fmt::Display
    + PartialEq
    + Default
    + Copy
    + TotalOrd
    + IsNull
    + MinMax
{
    /// The corresponding variant of [`PrimitiveType`].
    const PRIMITIVE: PrimitiveType;

    /// Type denoting its representation as bytes.
    /// This is `[u8; N]` where `N = size_of::<T>`.
    type Bytes: AsRef<[u8]>
        + AsMut<[u8]>
        + std::ops::Index<usize, Output = u8>
        + std::ops::IndexMut<usize, Output = u8>
        + for<'a> TryFrom<&'a [u8]>
        + std::fmt::Debug
        + Default
        + IntoIterator<Item = u8>;

    /// To bytes in little endian
    fn to_le_bytes(&self) -> Self::Bytes;

    /// To bytes in big endian
    fn to_be_bytes(&self) -> Self::Bytes;

    /// From bytes in little endian
    fn from_le_bytes(bytes: Self::Bytes) -> Self;

    /// From bytes in big endian
    fn from_be_bytes(bytes: Self::Bytes) -> Self;
}

macro_rules! native_type {
    ($type:ty, $primitive_type:expr) => {
        impl NativeType for $type {
            const PRIMITIVE: PrimitiveType = $primitive_type;

            type Bytes = [u8; std::mem::size_of::<Self>()];
            #[inline]
            fn to_le_bytes(&self) -> Self::Bytes {
                Self::to_le_bytes(*self)
            }

            #[inline]
            fn to_be_bytes(&self) -> Self::Bytes {
                Self::to_be_bytes(*self)
            }

            #[inline]
            fn from_le_bytes(bytes: Self::Bytes) -> Self {
                Self::from_le_bytes(bytes)
            }

            #[inline]
            fn from_be_bytes(bytes: Self::Bytes) -> Self {
                Self::from_be_bytes(bytes)
            }
        }
    };
}

native_type!(u8, PrimitiveType::UInt8);
native_type!(u16, PrimitiveType::UInt16);
native_type!(u32, PrimitiveType::UInt32);
native_type!(u64, PrimitiveType::UInt64);
native_type!(i8, PrimitiveType::Int8);
native_type!(i16, PrimitiveType::Int16);
native_type!(i32, PrimitiveType::Int32);
native_type!(i64, PrimitiveType::Int64);
native_type!(f32, PrimitiveType::Float32);
native_type!(f64, PrimitiveType::Float64);
native_type!(i128, PrimitiveType::Int128);
native_type!(u128, PrimitiveType::UInt128);

/// The in-memory representation of the DayMillisecond variant of arrow's "Interval" logical type.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable, Pod)]
#[allow(non_camel_case_types)]
#[repr(C)]
pub struct days_ms(pub i32, pub i32);

impl days_ms {
    /// A new [`days_ms`].
    #[inline]
    pub fn new(days: i32, milliseconds: i32) -> Self {
        Self(days, milliseconds)
    }

    /// The number of days
    #[inline]
    pub fn days(&self) -> i32 {
        self.0
    }

    /// The number of milliseconds
    #[inline]
    pub fn milliseconds(&self) -> i32 {
        self.1
    }
}

impl TotalEq for days_ms {
    #[inline]
    fn tot_eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl TotalOrd for days_ms {
    #[inline]
    fn tot_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.days()
            .cmp(&other.days())
            .then(self.milliseconds().cmp(&other.milliseconds()))
    }
}

impl MinMax for days_ms {
    fn nan_min_lt(&self, other: &Self) -> bool {
        self < other
    }

    fn nan_max_lt(&self, other: &Self) -> bool {
        self < other
    }
}

impl NativeType for days_ms {
    const PRIMITIVE: PrimitiveType = PrimitiveType::DaysMs;
    type Bytes = [u8; 8];
    #[inline]
    fn to_le_bytes(&self) -> Self::Bytes {
        let days = self.0.to_le_bytes();
        let ms = self.1.to_le_bytes();
        let mut result = [0; 8];
        result[0] = days[0];
        result[1] = days[1];
        result[2] = days[2];
        result[3] = days[3];
        result[4] = ms[0];
        result[5] = ms[1];
        result[6] = ms[2];
        result[7] = ms[3];
        result
    }

    #[inline]
    fn to_be_bytes(&self) -> Self::Bytes {
        let days = self.0.to_be_bytes();
        let ms = self.1.to_be_bytes();
        let mut result = [0; 8];
        result[0] = days[0];
        result[1] = days[1];
        result[2] = days[2];
        result[3] = days[3];
        result[4] = ms[0];
        result[5] = ms[1];
        result[6] = ms[2];
        result[7] = ms[3];
        result
    }

    #[inline]
    fn from_le_bytes(bytes: Self::Bytes) -> Self {
        let mut days = [0; 4];
        days[0] = bytes[0];
        days[1] = bytes[1];
        days[2] = bytes[2];
        days[3] = bytes[3];
        let mut ms = [0; 4];
        ms[0] = bytes[4];
        ms[1] = bytes[5];
        ms[2] = bytes[6];
        ms[3] = bytes[7];
        Self(i32::from_le_bytes(days), i32::from_le_bytes(ms))
    }

    #[inline]
    fn from_be_bytes(bytes: Self::Bytes) -> Self {
        let mut days = [0; 4];
        days[0] = bytes[0];
        days[1] = bytes[1];
        days[2] = bytes[2];
        days[3] = bytes[3];
        let mut ms = [0; 4];
        ms[0] = bytes[4];
        ms[1] = bytes[5];
        ms[2] = bytes[6];
        ms[3] = bytes[7];
        Self(i32::from_be_bytes(days), i32::from_be_bytes(ms))
    }
}

/// The in-memory representation of the MonthDayNano variant of the "Interval" logical type.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable, Pod)]
#[allow(non_camel_case_types)]
#[repr(C)]
pub struct months_days_ns(pub i32, pub i32, pub i64);

impl IsNull for months_days_ns {
    const HAS_NULLS: bool = false;
    type Inner = months_days_ns;

    fn is_null(&self) -> bool {
        false
    }

    fn unwrap_inner(self) -> Self::Inner {
        self
    }
}

impl months_days_ns {
    /// A new [`months_days_ns`].
    #[inline]
    pub fn new(months: i32, days: i32, nanoseconds: i64) -> Self {
        Self(months, days, nanoseconds)
    }

    /// The number of months
    #[inline]
    pub fn months(&self) -> i32 {
        self.0
    }

    /// The number of days
    #[inline]
    pub fn days(&self) -> i32 {
        self.1
    }

    /// The number of nanoseconds
    #[inline]
    pub fn ns(&self) -> i64 {
        self.2
    }
}

impl TotalEq for months_days_ns {
    #[inline]
    fn tot_eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl TotalOrd for months_days_ns {
    #[inline]
    fn tot_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.months()
            .cmp(&other.months())
            .then(self.days().cmp(&other.days()))
            .then(self.ns().cmp(&other.ns()))
    }
}

impl MinMax for months_days_ns {
    fn nan_min_lt(&self, other: &Self) -> bool {
        self < other
    }

    fn nan_max_lt(&self, other: &Self) -> bool {
        self < other
    }
}

impl NativeType for months_days_ns {
    const PRIMITIVE: PrimitiveType = PrimitiveType::MonthDayNano;
    type Bytes = [u8; 16];
    #[inline]
    fn to_le_bytes(&self) -> Self::Bytes {
        let months = self.months().to_le_bytes();
        let days = self.days().to_le_bytes();
        let ns = self.ns().to_le_bytes();
        let mut result = [0; 16];
        result[0] = months[0];
        result[1] = months[1];
        result[2] = months[2];
        result[3] = months[3];
        result[4] = days[0];
        result[5] = days[1];
        result[6] = days[2];
        result[7] = days[3];
        (0..8).for_each(|i| {
            result[8 + i] = ns[i];
        });
        result
    }

    #[inline]
    fn to_be_bytes(&self) -> Self::Bytes {
        let months = self.months().to_be_bytes();
        let days = self.days().to_be_bytes();
        let ns = self.ns().to_be_bytes();
        let mut result = [0; 16];
        result[0] = months[0];
        result[1] = months[1];
        result[2] = months[2];
        result[3] = months[3];
        result[4] = days[0];
        result[5] = days[1];
        result[6] = days[2];
        result[7] = days[3];
        (0..8).for_each(|i| {
            result[8 + i] = ns[i];
        });
        result
    }

    #[inline]
    fn from_le_bytes(bytes: Self::Bytes) -> Self {
        let mut months = [0; 4];
        months[0] = bytes[0];
        months[1] = bytes[1];
        months[2] = bytes[2];
        months[3] = bytes[3];
        let mut days = [0; 4];
        days[0] = bytes[4];
        days[1] = bytes[5];
        days[2] = bytes[6];
        days[3] = bytes[7];
        let mut ns = [0; 8];
        (0..8).for_each(|i| {
            ns[i] = bytes[8 + i];
        });
        Self(
            i32::from_le_bytes(months),
            i32::from_le_bytes(days),
            i64::from_le_bytes(ns),
        )
    }

    #[inline]
    fn from_be_bytes(bytes: Self::Bytes) -> Self {
        let mut months = [0; 4];
        months[0] = bytes[0];
        months[1] = bytes[1];
        months[2] = bytes[2];
        months[3] = bytes[3];
        let mut days = [0; 4];
        days[0] = bytes[4];
        days[1] = bytes[5];
        days[2] = bytes[6];
        days[3] = bytes[7];
        let mut ns = [0; 8];
        (0..8).for_each(|i| {
            ns[i] = bytes[8 + i];
        });
        Self(
            i32::from_be_bytes(months),
            i32::from_be_bytes(days),
            i64::from_be_bytes(ns),
        )
    }
}

impl IsNull for days_ms {
    const HAS_NULLS: bool = false;
    type Inner = days_ms;
    fn is_null(&self) -> bool {
        false
    }
    fn unwrap_inner(self) -> Self::Inner {
        self
    }
}

impl std::fmt::Display for days_ms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}d {}ms", self.days(), self.milliseconds())
    }
}

impl std::fmt::Display for months_days_ns {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}m {}d {}ns", self.months(), self.days(), self.ns())
    }
}

impl Neg for days_ms {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::new(-self.days(), -self.milliseconds())
    }
}

impl Neg for months_days_ns {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::new(-self.months(), -self.days(), -self.ns())
    }
}

/// Type representation of the Float16 physical type
#[derive(Copy, Clone, Default, Zeroable, Pod)]
#[allow(non_camel_case_types)]
#[repr(C)]
pub struct f16(pub u16);

impl PartialEq for f16 {
    #[inline]
    fn eq(&self, other: &f16) -> bool {
        if self.is_nan() || other.is_nan() {
            false
        } else {
            (self.0 == other.0) || ((self.0 | other.0) & 0x7FFFu16 == 0)
        }
    }
}

impl IsNull for f16 {
    const HAS_NULLS: bool = false;
    type Inner = f16;

    #[inline(always)]
    fn is_null(&self) -> bool {
        false
    }
    fn unwrap_inner(self) -> Self::Inner {
        self
    }
}

// see https://github.com/starkat99/half-rs/blob/main/src/binary16.rs
impl f16 {
    /// The difference between 1.0 and the next largest representable number.
    pub const EPSILON: f16 = f16(0x1400u16);

    #[inline]
    #[must_use]
    pub(crate) const fn is_nan(self) -> bool {
        self.0 & 0x7FFFu16 > 0x7C00u16
    }

    /// Casts from u16.
    #[inline]
    pub const fn from_bits(bits: u16) -> f16 {
        f16(bits)
    }

    /// Casts to u16.
    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Casts this `f16` to `f32`
    pub fn to_f32(self) -> f32 {
        let i = self.0;
        // Check for signed zero
        if i & 0x7FFFu16 == 0 {
            return f32::from_bits((i as u32) << 16);
        }

        let half_sign = (i & 0x8000u16) as u32;
        let half_exp = (i & 0x7C00u16) as u32;
        let half_man = (i & 0x03FFu16) as u32;

        // Check for an infinity or NaN when all exponent bits set
        if half_exp == 0x7C00u32 {
            // Check for signed infinity if mantissa is zero
            if half_man == 0 {
                let number = (half_sign << 16) | 0x7F80_0000u32;
                return f32::from_bits(number);
            } else {
                // NaN, keep current mantissa but also set most significiant mantissa bit
                let number = (half_sign << 16) | 0x7FC0_0000u32 | (half_man << 13);
                return f32::from_bits(number);
            }
        }

        // Calculate single-precision components with adjusted exponent
        let sign = half_sign << 16;
        // Unbias exponent
        let unbiased_exp = ((half_exp as i32) >> 10) - 15;

        // Check for subnormals, which will be normalized by adjusting exponent
        if half_exp == 0 {
            // Calculate how much to adjust the exponent by
            let e = (half_man as u16).leading_zeros() - 6;

            // Rebias and adjust exponent
            let exp = (127 - 15 - e) << 23;
            let man = (half_man << (14 + e)) & 0x7F_FF_FFu32;
            return f32::from_bits(sign | exp | man);
        }

        // Rebias exponent for a normalized normal
        let exp = ((unbiased_exp + 127) as u32) << 23;
        let man = (half_man & 0x03FFu32) << 13;
        f32::from_bits(sign | exp | man)
    }

    /// Casts an `f32` into `f16`
    pub fn from_f32(value: f32) -> Self {
        let x: u32 = value.to_bits();

        // Extract IEEE754 components
        let sign = x & 0x8000_0000u32;
        let exp = x & 0x7F80_0000u32;
        let man = x & 0x007F_FFFFu32;

        // Check for all exponent bits being set, which is Infinity or NaN
        if exp == 0x7F80_0000u32 {
            // Set mantissa MSB for NaN (and also keep shifted mantissa bits)
            let nan_bit = if man == 0 { 0 } else { 0x0200u32 };
            return f16(((sign >> 16) | 0x7C00u32 | nan_bit | (man >> 13)) as u16);
        }

        // The number is normalized, start assembling half precision version
        let half_sign = sign >> 16;
        // Unbias the exponent, then bias for half precision
        let unbiased_exp = ((exp >> 23) as i32) - 127;
        let half_exp = unbiased_exp + 15;

        // Check for exponent overflow, return +infinity
        if half_exp >= 0x1F {
            return f16((half_sign | 0x7C00u32) as u16);
        }

        // Check for underflow
        if half_exp <= 0 {
            // Check mantissa for what we can do
            if 14 - half_exp > 24 {
                // No rounding possibility, so this is a full underflow, return signed zero
                return f16(half_sign as u16);
            }
            // Don't forget about hidden leading mantissa bit when assembling mantissa
            let man = man | 0x0080_0000u32;
            let mut half_man = man >> (14 - half_exp);
            // Check for rounding (see comment above functions)
            let round_bit = 1 << (13 - half_exp);
            if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
                half_man += 1;
            }
            // No exponent for subnormals
            return f16((half_sign | half_man) as u16);
        }

        // Rebias the exponent
        let half_exp = (half_exp as u32) << 10;
        let half_man = man >> 13;
        // Check for rounding (see comment above functions)
        let round_bit = 0x0000_1000u32;
        if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
            // Round it
            f16(((half_sign | half_exp | half_man) + 1) as u16)
        } else {
            f16((half_sign | half_exp | half_man) as u16)
        }
    }
}

impl std::fmt::Debug for f16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.to_f32())
    }
}

impl std::fmt::Display for f16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl TotalEq for f16 {
    #[inline]
    fn tot_eq(&self, other: &Self) -> bool {
        if self.is_nan() {
            other.is_nan()
        } else {
            self == other
        }
    }
}

impl TotalOrd for f16 {
    #[inline]
    fn tot_cmp(&self, _other: &Self) -> std::cmp::Ordering {
        unimplemented!()
    }
}

impl MinMax for f16 {
    fn nan_min_lt(&self, _other: &Self) -> bool {
        unimplemented!()
    }

    fn nan_max_lt(&self, _other: &Self) -> bool {
        unimplemented!()
    }
}

impl NativeType for f16 {
    const PRIMITIVE: PrimitiveType = PrimitiveType::Float16;
    type Bytes = [u8; 2];
    #[inline]
    fn to_le_bytes(&self) -> Self::Bytes {
        self.0.to_le_bytes()
    }

    #[inline]
    fn to_be_bytes(&self) -> Self::Bytes {
        self.0.to_be_bytes()
    }

    #[inline]
    fn from_be_bytes(bytes: Self::Bytes) -> Self {
        Self(u16::from_be_bytes(bytes))
    }

    #[inline]
    fn from_le_bytes(bytes: Self::Bytes) -> Self {
        Self(u16::from_le_bytes(bytes))
    }
}

/// Physical representation of a decimal
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq, PartialOrd, Ord)]
#[allow(non_camel_case_types)]
#[repr(C)]
pub struct i256(pub ethnum::I256);

impl i256 {
    /// Returns a new [`i256`] from two `i128`.
    pub fn from_words(hi: i128, lo: i128) -> Self {
        Self(ethnum::I256::from_words(hi, lo))
    }
}

impl IsNull for i256 {
    const HAS_NULLS: bool = false;
    type Inner = i256;
    #[inline(always)]
    fn is_null(&self) -> bool {
        false
    }
    fn unwrap_inner(self) -> Self::Inner {
        self
    }
}

impl Neg for i256 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        let (a, b) = self.0.into_words();
        Self(ethnum::I256::from_words(-a, b))
    }
}

impl std::fmt::Debug for i256 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl std::fmt::Display for i256 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

unsafe impl Pod for i256 {}
unsafe impl Zeroable for i256 {}

impl TotalEq for i256 {
    #[inline]
    fn tot_eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl TotalOrd for i256 {
    #[inline]
    fn tot_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cmp(other)
    }
}

impl MinMax for i256 {
    fn nan_min_lt(&self, other: &Self) -> bool {
        self < other
    }

    fn nan_max_lt(&self, other: &Self) -> bool {
        self < other
    }
}

impl NativeType for i256 {
    const PRIMITIVE: PrimitiveType = PrimitiveType::Int256;

    type Bytes = [u8; 32];

    #[inline]
    fn to_le_bytes(&self) -> Self::Bytes {
        let mut bytes = [0u8; 32];
        let (a, b) = self.0.into_words();
        let a = a.to_le_bytes();
        (0..16).for_each(|i| {
            bytes[i] = a[i];
        });

        let b = b.to_le_bytes();
        (0..16).for_each(|i| {
            bytes[i + 16] = b[i];
        });

        bytes
    }

    #[inline]
    fn to_be_bytes(&self) -> Self::Bytes {
        let mut bytes = [0u8; 32];
        let (a, b) = self.0.into_words();

        let a = a.to_be_bytes();
        (0..16).for_each(|i| {
            bytes[i] = a[i];
        });

        let b = b.to_be_bytes();
        (0..16).for_each(|i| {
            bytes[i + 16] = b[i];
        });

        bytes
    }

    #[inline]
    fn from_be_bytes(bytes: Self::Bytes) -> Self {
        let (a, b) = bytes.split_at(16);
        let a: [u8; 16] = a.try_into().unwrap();
        let b: [u8; 16] = b.try_into().unwrap();
        let a = i128::from_be_bytes(a);
        let b = i128::from_be_bytes(b);
        Self(ethnum::I256::from_words(a, b))
    }

    #[inline]
    fn from_le_bytes(bytes: Self::Bytes) -> Self {
        let (b, a) = bytes.split_at(16);
        let a: [u8; 16] = a.try_into().unwrap();
        let b: [u8; 16] = b.try_into().unwrap();
        let a = i128::from_le_bytes(a);
        let b = i128::from_le_bytes(b);
        Self(ethnum::I256::from_words(a, b))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_f16_to_f32() {
        let f = f16::from_f32(7.0);
        assert_eq!(f.to_f32(), 7.0f32);

        // 7.1 is NOT exactly representable in 16-bit, it's rounded
        let f = f16::from_f32(7.1);
        let diff = (f.to_f32() - 7.1f32).abs();
        // diff must be <= 4 * EPSILON, as 7 has two more significant bits than 1
        assert!(diff <= 4.0 * f16::EPSILON.to_f32());

        assert_eq!(f16(0x0000_0001).to_f32(), 2.0f32.powi(-24));
        assert_eq!(f16(0x0000_0005).to_f32(), 5.0 * 2.0f32.powi(-24));

        assert_eq!(f16(0x0000_0001), f16::from_f32(2.0f32.powi(-24)));
        assert_eq!(f16(0x0000_0005), f16::from_f32(5.0 * 2.0f32.powi(-24)));

        assert_eq!(format!("{}", f16::from_f32(7.0)), "7".to_string());
        assert_eq!(format!("{:?}", f16::from_f32(7.0)), "7.0".to_string());
    }
}
