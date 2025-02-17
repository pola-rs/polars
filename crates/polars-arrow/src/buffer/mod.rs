//! Contains [`Buffer`], an immutable container for all Arrow physical types (e.g. i32, f64).

mod immutable;
mod iterator;

pub use immutable::Buffer;
pub(super) use iterator::IntoIter;
