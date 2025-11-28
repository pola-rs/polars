use std::hash::Hash;
use std::ops::Neg;
use std::panic::RefUnwindSafe;

use bytemuck::{Pod, Zeroable};
use half;
use polars_utils::float16::pf16;
use polars_utils::min_max::MinMax;
use polars_utils::nulls::IsNull;
use polars_utils::total_ord::{TotalEq, TotalOrd};

use super::PrimitiveType;
use super::aligned_bytes::*;

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

    /// Type denoting its representation as aligned bytes.
    ///
    /// This is `[u8; N]` where `N = size_of::<Self>` and has alignment `align_of::<Self>`.
    type AlignedBytes: AlignedBytes<Unaligned = Self::Bytes> + From<Self> + Into<Self>;

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
    ($type:ty, $aligned:ty, $primitive_type:expr) => {
        impl NativeType for $type {
            const PRIMITIVE: PrimitiveType = $primitive_type;

            type Bytes = [u8; std::mem::size_of::<Self>()];
            type AlignedBytes = $aligned;

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

native_type!(u8, Bytes1Alignment1, PrimitiveType::UInt8);
native_type!(u16, Bytes2Alignment2, PrimitiveType::UInt16);
native_type!(u32, Bytes4Alignment4, PrimitiveType::UInt32);
native_type!(u64, Bytes8Alignment8, PrimitiveType::UInt64);
native_type!(u128, Bytes16Alignment16, PrimitiveType::UInt128);
native_type!(i8, Bytes1Alignment1, PrimitiveType::Int8);
native_type!(i16, Bytes2Alignment2, PrimitiveType::Int16);
native_type!(i32, Bytes4Alignment4, PrimitiveType::Int32);
native_type!(i64, Bytes8Alignment8, PrimitiveType::Int64);
native_type!(i128, Bytes16Alignment16, PrimitiveType::Int128);
native_type!(f32, Bytes4Alignment4, PrimitiveType::Float32);
native_type!(f64, Bytes8Alignment8, PrimitiveType::Float64);

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
    type AlignedBytes = Bytes8Alignment4;

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
    type AlignedBytes = Bytes16Alignment8;

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

impl TryFrom<i256> for i128 {
    type Error = core::num::TryFromIntError;

    fn try_from(value: i256) -> Result<Self, Self::Error> {
        value.0.try_into()
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
    type AlignedBytes = Bytes32Alignment16;

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

impl NativeType for pf16 {
    const PRIMITIVE: PrimitiveType = PrimitiveType::Float16;
    type Bytes = [u8; 2];
    type AlignedBytes = Bytes2Alignment2;

    #[inline]
    fn to_le_bytes(&self) -> Self::Bytes {
        self.0.to_le_bytes()
    }

    #[inline]
    fn to_be_bytes(&self) -> Self::Bytes {
        self.0.to_be_bytes()
    }

    #[inline]
    fn from_le_bytes(bytes: Self::Bytes) -> Self {
        pf16(half::f16::from_le_bytes(bytes))
    }

    #[inline]
    fn from_be_bytes(bytes: Self::Bytes) -> Self {
        pf16(half::f16::from_be_bytes(bytes))
    }
}
