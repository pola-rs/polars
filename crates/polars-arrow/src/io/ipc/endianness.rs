#[inline]
pub const fn is_native_little_endian() -> bool {
    cfg!(target_endian = "little")
}
