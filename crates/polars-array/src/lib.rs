//! The Polars vector format.

pub mod array;
pub mod array_type;
pub mod arrow;
pub mod binary;
pub mod binview;
pub mod bitmap;
pub mod boolean;
pub mod broadcast;
pub mod builder;
pub mod collect;
pub mod concatenate;
pub mod fixed_size_binary;
pub mod fixed_size_list;
pub mod flat;
#[cfg(test)]
mod iterator_tests;
pub mod list;
mod macros;
pub mod null;
pub mod primitive;
pub mod static_array;
pub mod struct_;
pub mod utf8view;

pub use array::PlArray;
pub use array_type::{PlArrayType, PrimitiveType};
pub use binary::{PlBinaryArray, PlBinaryArrayBuilder};
pub use binview::{PlBinaryViewArray, PlBinaryViewArrayBuilder};
pub use bitmap::{PlBitmap, PlBitmapIter, PlBitmapRef};
pub use boolean::{PlBooleanArray, PlBooleanArrayBuilder};
pub use builder::{PlArrayBuilder, StaticArrayBuilder};
pub use collect::{ArrayCollectIterExt, ArrayFromIter, ZeroableArrayFromIter};
pub use fixed_size_binary::{PlFixedSizeBinaryArray, PlFixedSizeBinaryArrayBuilder};
pub use fixed_size_list::{PlFixedSizeListArray, PlFixedSizeListArrayBuilder};
pub use flat::Flat;
pub use list::{PlListArray, PlListArrayBuilder};
pub use null::{PlNullArray, PlNullArrayBuilder};
pub use primitive::{PlPrimitiveArray, PlPrimitiveArrayBuilder};
pub use static_array::StaticArray;
pub use struct_::{PlStructArray, PlStructArrayBuilder};
pub use utf8view::{PlUtf8ViewArray, PlUtf8ViewArrayBuilder};
