#[cfg(target_endian = "little")]
#[inline]
pub fn is_native_little_endian() -> bool {
    true
}

#[cfg(target_endian = "big")]
#[inline]
pub fn is_native_little_endian() -> bool {
    false
}
