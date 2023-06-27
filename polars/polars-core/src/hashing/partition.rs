use super::*;

// Used to to get a u64 from the hashing keys
// We need to modify the hashing algorithm to use the hash for this and only compute the hash once.
pub trait AsU64 {
    #[allow(clippy::wrong_self_convention)]
    fn as_u64(self) -> u64;
}

impl AsU64 for u8 {
    #[inline]
    fn as_u64(self) -> u64 {
        self as u64
    }
}

impl AsU64 for u16 {
    #[inline]
    fn as_u64(self) -> u64 {
        self as u64
    }
}

impl AsU64 for u32 {
    #[inline]
    fn as_u64(self) -> u64 {
        self as u64
    }
}

impl AsU64 for u64 {
    #[inline]
    fn as_u64(self) -> u64 {
        self
    }
}

impl AsU64 for i8 {
    #[inline]
    fn as_u64(self) -> u64 {
        self as u64
    }
}

impl AsU64 for i16 {
    #[inline]
    fn as_u64(self) -> u64 {
        self as u64
    }
}

impl AsU64 for i32 {
    #[inline]
    fn as_u64(self) -> u64 {
        let asu32: u32 = unsafe { std::mem::transmute(self) };
        asu32 as u64
    }
}

impl AsU64 for i64 {
    #[inline]
    fn as_u64(self) -> u64 {
        unsafe { std::mem::transmute(self) }
    }
}

impl<T: AsU64 + Copy> AsU64 for Option<&T> {
    #[inline]
    fn as_u64(self) -> u64 {
        match self {
            Some(v) => v.as_u64(),
            // just a number safe from overflow
            None => u64::MAX >> 2,
        }
    }
}

impl<T: AsU64 + Copy> AsU64 for &T {
    fn as_u64(self) -> u64 {
        (*self).as_u64()
    }
}

impl AsU64 for Option<u32> {
    #[inline]
    fn as_u64(self) -> u64 {
        match self {
            Some(v) => v as u64,
            // just a number safe from overflow
            None => u64::MAX >> 2,
        }
    }
}

#[cfg(feature = "performant")]
impl AsU64 for Option<u8> {
    #[inline]
    fn as_u64(self) -> u64 {
        match self {
            Some(v) => v as u64,
            // just a number safe from overflow
            None => u64::MAX >> 2,
        }
    }
}

#[cfg(feature = "performant")]
impl AsU64 for Option<u16> {
    #[inline]
    fn as_u64(self) -> u64 {
        match self {
            Some(v) => v as u64,
            // just a number safe from overflow
            None => u64::MAX >> 2,
        }
    }
}

impl AsU64 for Option<u64> {
    #[inline]
    fn as_u64(self) -> u64 {
        self.unwrap_or(u64::MAX >> 2)
    }
}

impl<'a> AsU64 for BytesHash<'a> {
    fn as_u64(self) -> u64 {
        self.hash
    }
}

#[inline]
/// For partitions that are a power of 2 we can use a bitshift instead of a modulo.
pub fn this_partition(h: u64, thread_no: u64, n_partitions: u64) -> bool {
    debug_assert!(n_partitions.is_power_of_two());
    // n % 2^i = n & (2^i - 1)
    (h & n_partitions.wrapping_sub(1)) == thread_no
}
