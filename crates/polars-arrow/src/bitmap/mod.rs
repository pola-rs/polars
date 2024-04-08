//! contains [`Bitmap`] and [`MutableBitmap`], containers of `bool`.
mod immutable;
pub use immutable::*;

pub mod iterator;
pub use iterator::IntoIter;

mod mutable;
pub use mutable::MutableBitmap;

mod bitmap_ops;
pub use bitmap_ops::*;

pub mod aligned;

mod assign_ops;
pub use assign_ops::*;

pub mod utils;

pub mod bitmask;
