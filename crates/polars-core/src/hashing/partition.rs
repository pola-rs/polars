use polars_utils::hash_to_partition;

use super::*;

// Used to to get a u64 from the hashing keys
// We need to modify the hashing algorithm to use the hash for this and only compute the hash once.
pub trait AsU64 {
    #[allow(clippy::wrong_self_convention)]
    fn as_u64(self) -> u64;
}

// Identity gives a poorly distributed hash. So we spread values around by a simple wrapping
// multiplication by a 'random' odd number.
const RANDOM_ODD: u64 = 0x55fbfd6bfc5458e9;

impl AsU64 for u8 {
    #[inline]
    fn as_u64(self) -> u64 {
        (self as u64).wrapping_mul(RANDOM_ODD)
    }
}

impl AsU64 for u16 {
    #[inline]
    fn as_u64(self) -> u64 {
        (self as u64).wrapping_mul(RANDOM_ODD)
    }
}

impl AsU64 for u32 {
    #[inline]
    fn as_u64(self) -> u64 {
        (self as u64).wrapping_mul(RANDOM_ODD)
    }
}

impl AsU64 for u64 {
    #[inline]
    fn as_u64(self) -> u64 {
        self.wrapping_mul(RANDOM_ODD)
    }
}

impl AsU64 for i8 {
    #[inline]
    fn as_u64(self) -> u64 {
        (self as u64).wrapping_mul(RANDOM_ODD)
    }
}

impl AsU64 for i16 {
    #[inline]
    fn as_u64(self) -> u64 {
        (self as u64).wrapping_mul(RANDOM_ODD)
    }
}

impl AsU64 for i32 {
    #[inline]
    fn as_u64(self) -> u64 {
        let asu32: u32 = unsafe { std::mem::transmute(self) };
        (asu32 as u64).wrapping_mul(RANDOM_ODD)
    }
}

impl AsU64 for i64 {
    #[inline]
    fn as_u64(self) -> u64 {
        (unsafe { std::mem::transmute::<_, u64>(self) }).wrapping_mul(RANDOM_ODD)
    }
}

impl<T: AsU64 + Copy> AsU64 for Option<&T> {
    #[inline]
    fn as_u64(self) -> u64 {
        match self {
            Some(v) => v.as_u64().wrapping_mul(RANDOM_ODD),
            None => RANDOM_ODD,
        }
    }
}

impl<T: AsU64 + Copy> AsU64 for &T {
    fn as_u64(self) -> u64 {
        (*self).as_u64().wrapping_mul(RANDOM_ODD)
    }
}

impl AsU64 for Option<u32> {
    #[inline]
    fn as_u64(self) -> u64 {
        match self {
            Some(v) => (v as u64).wrapping_mul(RANDOM_ODD),
            None => RANDOM_ODD,
        }
    }
}

#[cfg(feature = "performant")]
impl AsU64 for Option<u8> {
    #[inline]
    fn as_u64(self) -> u64 {
        match self {
            Some(v) => (v as u64).wrapping_mul(RANDOM_ODD),
            None => RANDOM_ODD,
        }
    }
}

#[cfg(feature = "performant")]
impl AsU64 for Option<u16> {
    #[inline]
    fn as_u64(self) -> u64 {
        match self {
            Some(v) => (v as u64).wrapping_mul(RANDOM_ODD),
            None => RANDOM_ODD,
        }
    }
}

impl AsU64 for Option<u64> {
    #[inline]
    fn as_u64(self) -> u64 {
        match self {
            Some(v) => v.wrapping_mul(RANDOM_ODD),
            None => RANDOM_ODD,
        }
    }
}

impl<'a> AsU64 for BytesHash<'a> {
    fn as_u64(self) -> u64 {
        self.hash
    }
}

#[inline]
pub fn this_partition(h: u64, thread_no: u64, n_partitions: u64) -> bool {
    hash_to_partition(h, n_partitions as usize) as u64 == thread_no
}
