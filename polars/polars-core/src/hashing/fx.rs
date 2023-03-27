use super::*;
// This should only be used for small integers as high bits will collide

macro_rules! fx_hash_8_bit {
    ($val: expr, $k: expr ) => {{
        let val = std::mem::transmute::<_, u8>($val);
        (val as u64).wrapping_mul($k)
    }};
}
macro_rules! fx_hash_16_bit {
    ($val: expr, $k: expr ) => {{
        let val = std::mem::transmute::<_, u16>($val);
        (val as u64).wrapping_mul($k)
    }};
}
macro_rules! fx_hash_32_bit {
    ($val: expr, $k: expr ) => {{
        let val = std::mem::transmute::<_, u32>($val);
        (val as u64).wrapping_mul($k)
    }};
}
macro_rules! fx_hash_64_bit {
    ($val: expr, $k: expr ) => {{
        ($val as u64).wrapping_mul($k)
    }};
}
pub(super) const FXHASH_K: u64 = 0x517cc1b727220a95;

/// Ensure that the same hash is used as with `VecHash`.
pub trait FxHash {
    fn get_k(random_state: RandomState) -> u64 {
        random_state.hash_one(FXHASH_K)
    }
    fn _fx_hash(self, k: u64) -> u64;
}
impl FxHash for i8 {
    #[inline]
    fn _fx_hash(self, k: u64) -> u64 {
        unsafe { fx_hash_8_bit!(self, k) }
    }
}
impl FxHash for u8 {
    #[inline]
    fn _fx_hash(self, k: u64) -> u64 {
        #[allow(clippy::useless_transmute)]
        unsafe {
            fx_hash_8_bit!(self, k)
        }
    }
}
impl FxHash for i16 {
    #[inline]
    fn _fx_hash(self, k: u64) -> u64 {
        unsafe { fx_hash_16_bit!(self, k) }
    }
}
impl FxHash for u16 {
    #[inline]
    fn _fx_hash(self, k: u64) -> u64 {
        #[allow(clippy::useless_transmute)]
        unsafe {
            fx_hash_16_bit!(self, k)
        }
    }
}

impl FxHash for i32 {
    #[inline]
    fn _fx_hash(self, k: u64) -> u64 {
        unsafe { fx_hash_32_bit!(self, k) }
    }
}
impl FxHash for u32 {
    #[inline]
    fn _fx_hash(self, k: u64) -> u64 {
        #[allow(clippy::useless_transmute)]
        unsafe {
            fx_hash_32_bit!(self, k)
        }
    }
}

impl FxHash for i64 {
    #[inline]
    fn _fx_hash(self, k: u64) -> u64 {
        fx_hash_64_bit!(self, k)
    }
}
impl FxHash for u64 {
    #[inline]
    fn _fx_hash(self, k: u64) -> u64 {
        fx_hash_64_bit!(self, k)
    }
}
