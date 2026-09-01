//! The Polars vector format.
//!
//! The arrays in this crate are the intended replacement for the array types in `polars-arrow`.
//! They differ from their Arrow counterparts in two important ways:
//!
//! * They are cheaply cloneable: all buffers are backed by [`Buffer`](polars_buffer::Buffer),
//!   so cloning and slicing are `O(1)`.
//! * They carry their logical length in a separate `length` field instead of deriving it from the
//!   backing buffers. This makes a *scalar* (scalar) array — one logical value repeated
//!   `length` times — representable in `O(1)` memory. See [`scalar`] for the exact rules.
//!
//! Unlike the `polars-arrow` arrays, these arrays carry no logical type; they are purely a physical
//! representation. Logical typing lives at a higher level.
//!
//! Every array implements the trait object [`PlArray`] and can be downcast to a concrete struct
//! based on the [`PlArrayType`] available from [`PlArray::array_type`]. Code that is generic over
//! the array rather than over its element type is written against [`StaticArray`], the typed
//! counterpart of that trait, and an array that is not laid out already is built by a
//! [`StaticArrayBuilder`] — see [`builder`]. An array of a known type is also collected from an
//! iterator of its elements — see [`collect`].

pub mod array;
pub mod array_type;
pub mod binview;
pub mod bitmap;
pub mod boolean;
pub mod broadcast;
pub mod builder;
pub mod collect;
pub mod concatenate;
pub mod fixed_size_list;
pub mod flat;
pub mod list;
mod macros;
pub mod null;
pub mod primitive;
pub mod static_array;
pub mod struct_;

pub use array::PlArray;
pub use array_type::{PlArrayType, PrimitiveType};
pub use binview::{PlBinaryViewArray, PlBinaryViewArrayBuilder};
pub use bitmap::{PlBitmap, PlBitmapIter, PlBitmapRef};
pub use boolean::{PlBooleanArray, PlBooleanArrayBuilder};
pub use builder::{PlArrayBuilder, StaticArrayBuilder};
pub use collect::{ArrayCollectIterExt, ArrayFromIter};
pub use fixed_size_list::{PlFixedSizeListArray, PlFixedSizeListArrayBuilder};
pub use flat::Flat;
pub use list::{PlListArray, PlListArrayBuilder};
pub use null::{PlNullArray, PlNullArrayBuilder};
pub use primitive::{PlPrimitiveArray, PlPrimitiveArrayBuilder};
pub use static_array::StaticArray;
pub use struct_::{PlStructArray, PlStructArrayBuilder};
