use super::Index;

/// Sealed trait describing the subset (`i32` and `i64`) of [`Index`] that can be used
/// as offsets of variable-length Arrow arrays.
pub trait Offset: super::private::Sealed + Index {
    /// Whether it is `i32` (false) or `i64` (true).
    const IS_LARGE: bool;
}

impl Offset for i32 {
    const IS_LARGE: bool = false;
}

impl Offset for i64 {
    const IS_LARGE: bool = true;
}
