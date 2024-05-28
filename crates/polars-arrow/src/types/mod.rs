//! Sealed traits and implementations to handle all _physical types_ used in this crate.
//!
//! Most physical types used in this crate are native Rust types, such as `i32`.
//! The trait [`NativeType`] describes the interfaces required by this crate to be conformant
//! with Arrow.
//!
//! Every implementation of [`NativeType`] has an associated variant in [`PrimitiveType`],
//! available via [`NativeType::PRIMITIVE`].
//! Combined, these allow structs generic over [`NativeType`] to be trait objects downcastable
//! to concrete implementations based on the matched [`NativeType::PRIMITIVE`] variant.
//!
//! Another important trait in this module is [`Offset`], the subset of [`NativeType`] that can
//! be used in Arrow offsets (`i32` and `i64`).
//!
//! Another important trait in this module is [`BitChunk`], describing types that can be used to
//! represent chunks of bits (e.g. 8 bits via `u8`, 16 via `u16`), and [`BitChunkIter`],
//! that can be used to iterate over bitmaps in [`BitChunk`]s according to
//! Arrow's definition of bitmaps.
//!
//! Finally, this module contains traits used to compile code based on [`NativeType`] optimized
//! for SIMD, at [`mod@simd`].

mod bit_chunk;
pub use bit_chunk::{BitChunk, BitChunkIter, BitChunkOnes};
mod index;
pub mod simd;
pub use index::*;
mod native;
pub use native::*;
mod offset;
pub use offset::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// The set of all implementations of the sealed trait [`NativeType`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PrimitiveType {
    /// A signed 8-bit integer.
    Int8,
    /// A signed 16-bit integer.
    Int16,
    /// A signed 32-bit integer.
    Int32,
    /// A signed 64-bit integer.
    Int64,
    /// A signed 128-bit integer.
    Int128,
    /// A signed 256-bit integer.
    Int256,
    /// An unsigned 8-bit integer.
    UInt8,
    /// An unsigned 16-bit integer.
    UInt16,
    /// An unsigned 32-bit integer.
    UInt32,
    /// An unsigned 64-bit integer.
    UInt64,
    /// An unsigned 128-bit integer.
    UInt128,
    /// A 16-bit floating point number.
    Float16,
    /// A 32-bit floating point number.
    Float32,
    /// A 64-bit floating point number.
    Float64,
    /// Two i32 representing days and ms
    DaysMs,
    /// months_days_ns(i32, i32, i64)
    MonthDayNano,
}

mod private {
    use crate::array::View;

    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for i128 {}
    impl Sealed for u128 {}
    impl Sealed for super::i256 {}
    impl Sealed for super::f16 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for super::days_ms {}
    impl Sealed for super::months_days_ns {}
    impl Sealed for View {}
}
