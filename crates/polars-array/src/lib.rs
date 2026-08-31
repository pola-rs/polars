//! The Polars vector format.
//!
//! The arrays in this crate are the intended replacement for the array types in `polars-arrow`.
//! They differ from their Arrow counterparts in two important ways:
//!
//! * They are cheaply cloneable: all buffers are backed by [`Buffer`](polars_buffer::Buffer),
//!   so cloning and slicing are `O(1)`.
//! * They carry their logical length in a separate `length` field instead of deriving it from the
//!   backing buffers. This makes a *scalar* (broadcast) array — one logical value repeated
//!   `length` times — representable in `O(1)` memory. See [`broadcast`] for the exact rules.
//!
//! Unlike the `polars-arrow` arrays, these arrays carry no logical type; they are purely a physical
//! representation. Logical typing lives at a higher level.

pub mod bitmap;
pub mod boolean;
pub mod broadcast;
pub mod primitive;

pub use bitmap::{PlBitmap, PlBitmapIter, PlBitmapRef};
pub use boolean::PlBooleanArray;
pub use primitive::PlPrimitiveArray;
