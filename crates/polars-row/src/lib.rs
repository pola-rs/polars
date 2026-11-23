//! Row format as defined in `arrow-rs`.
//! This currently partially implements that format only for needed types.
//! For completeness sake the format as defined by `arrow-rs` is as followed:
//! Converts [`ArrayRef`] columns into a [row-oriented](self) format.
//!
//! ## Overview
//!
//! The row format is a variable length byte sequence created by
//! concatenating the encoded form of each column. The encoding for
//! each column depends on its datatype (and sort options).
//!
//! The encoding is carefully designed in such a way that escaping is
//! unnecessary: it is never ambiguous as to whether a byte is part of
//! a sentinel (e.g. null) or a value.
//!
//! ## Unsigned Integer Encoding
//!
//! A null integer is encoded as a `0_u8`, followed by a zero-ed number of bytes corresponding
//! to the integer's length.
//!
//! A valid integer is encoded as `1_u8`, followed by the big-endian representation of the
//! integer.
//!
//! ```text
//!               ┌──┬──┬──┬──┐      ┌──┬──┬──┬──┬──┐
//!    3          │03│00│00│00│      │01│00│00│00│03│
//!               └──┴──┴──┴──┘      └──┴──┴──┴──┴──┘
//!               ┌──┬──┬──┬──┐      ┌──┬──┬──┬──┬──┐
//!   258         │02│01│00│00│      │01│00│00│01│02│
//!               └──┴──┴──┴──┘      └──┴──┴──┴──┴──┘
//!               ┌──┬──┬──┬──┐      ┌──┬──┬──┬──┬──┐
//!  23423        │7F│5B│00│00│      │01│00│00│5B│7F│
//!               └──┴──┴──┴──┘      └──┴──┴──┴──┴──┘
//!               ┌──┬──┬──┬──┐      ┌──┬──┬──┬──┬──┐
//!  NULL         │??│??│??│??│      │00│00│00│00│00│
//!               └──┴──┴──┴──┘      └──┴──┴──┴──┴──┘
//!
//!              32-bit (4 bytes)        Row Format
//!  Value        Little Endian
//! ```
//!
//! ## Signed Integer Encoding
//!
//! Signed integers have their most significant sign bit flipped, and are then encoded in the
//! same manner as an unsigned integer.
//!
//! ```text
//!        ┌──┬──┬──┬──┐       ┌──┬──┬──┬──┐       ┌──┬──┬──┬──┬──┐
//!     5  │05│00│00│00│       │05│00│00│80│       │01│80│00│00│05│
//!        └──┴──┴──┴──┘       └──┴──┴──┴──┘       └──┴──┴──┴──┴──┘
//!        ┌──┬──┬──┬──┐       ┌──┬──┬──┬──┐       ┌──┬──┬──┬──┬──┐
//!    -5  │FB│FF│FF│FF│       │FB│FF│FF│7F│       │01│7F│FF│FF│FB│
//!        └──┴──┴──┴──┘       └──┴──┴──┴──┘       └──┴──┴──┴──┴──┘
//!
//!  Value  32-bit (4 bytes)    High bit flipped      Row Format
//!          Little Endian
//! ```
//!
//! ## Float Encoding
//!
//! Floats are converted from IEEE 754 representation to a signed integer representation
//! by flipping all bar the sign bit if they are negative after normalizing nans
//! and signed zeros to a canonical representation.
//!
//! They are then encoded in the same manner as a signed integer.
//!
//! ## Fixed Length Bytes Encoding
//!
//! Fixed length bytes are encoded in the same fashion as primitive types above.
//!
//! For a fixed length array of length `n`:
//!
//! A null is encoded as `0_u8` null sentinel followed by `n` `0_u8` bytes
//!
//! A valid value is encoded as `1_u8` followed by the value bytes
//!
//! ## Variable Length Bytes (including Strings) Encoding
//!
//! A null is encoded as a `0_u8`.
//!
//! An empty byte array is encoded as `1_u8`.
//!
//! A non-null, non-empty byte array is encoded as `2_u8` followed by the byte array
//! encoded using a block based scheme described below.
//!
//! The byte array is broken up into 32-byte blocks, each block is written in turn
//! to the output, followed by `0xFF_u8`. The final block is padded to 32-bytes
//! with `0_u8` and written to the output, followed by the un-padded length in bytes
//! of this final block as a `u8`.
//!
//! Note the following example encodings use a block size of 4 bytes,
//! as opposed to 32 bytes for brevity:
//!
//! ```text
//!                       ┌───┬───┬───┬───┬───┬───┐
//!  "MEEP"               │02 │'M'│'E'│'E'│'P'│04 │
//!                       └───┴───┴───┴───┴───┴───┘
//!
//!                       ┌───┐
//!  ""                   │01 |
//!                       └───┘
//!
//!  NULL                 ┌───┐
//!                       │00 │
//!                       └───┘
//!
//! "Defenestration"      ┌───┬───┬───┬───┬───┬───┐
//!                       │02 │'D'│'e'│'f'│'e'│FF │
//!                       └───┼───┼───┼───┼───┼───┤
//!                           │'n'│'e'│'s'│'t'│FF │
//!                           ├───┼───┼───┼───┼───┤
//!                           │'r'│'a'│'t'│'r'│FF │
//!                           ├───┼───┼───┼───┼───┤
//!                           │'a'│'t'│'i'│'o'│FF │
//!                           ├───┼───┼───┼───┼───┤
//!                           │'n'│00 │00 │00 │01 │
//!                           └───┴───┴───┴───┴───┘
//! ```
//!
//! This approach is loosely inspired by [COBS] encoding, and chosen over more traditional
//! [byte stuffing] as it is more amenable to vectorisation, in particular AVX-256.
//!
//! For the unordered row encoding we use a simpler scheme, we prepend the length
//! encoded as 4 bytes followed by the raw data, with nulls being marked with a
//! length of u32::MAX.
//!
//! ## Dictionary Encoding
//!
//! The row encoding needs to support converting dictionary encoded arrays with unsorted, and
//! potentially distinct dictionaries. One simple mechanism to avoid this would be to reverse
//! the dictionary encoding, and encode the array values directly, however, this would lose
//! the benefits of dictionary encoding to reduce memory and CPU consumption.
//!
//! As such the row encoding creates an order-preserving mapping
//! for each dictionary encoded column, which allows new dictionary
//! values to be added whilst preserving the sort order.
//!
//! A null dictionary value is encoded as `0_u8`.
//!
//! A non-null dictionary value is encoded as `1_u8` followed by a null-terminated byte array
//! key determined by the order-preserving dictionary encoding
//!
//! ```text
//! ┌──────────┐                 ┌─────┐
//! │  "Bar"   │ ───────────────▶│ 01  │
//! └──────────┘                 └─────┘
//! ┌──────────┐                 ┌─────┬─────┐
//! │"Fabulous"│ ───────────────▶│ 01  │ 02  │
//! └──────────┘                 └─────┴─────┘
//! ┌──────────┐                 ┌─────┐
//! │  "Soup"  │ ───────────────▶│ 05  │
//! └──────────┘                 └─────┘
//! ┌──────────┐                 ┌─────┐
//! │   "ZZ"   │ ───────────────▶│ 07  │
//! └──────────┘                 └─────┘
//!
//! Example Order Preserving Mapping
//! ```
//! Using the map above, the corresponding row format will be
//!
//! ```text
//!                           ┌─────┬─────┬─────┬─────┐
//!    "Fabulous"             │ 01  │ 03  │ 05  │ 00  │
//!                           └─────┴─────┴─────┴─────┘
//!
//!                           ┌─────┬─────┬─────┐
//!    "ZZ"                   │ 01  │ 07  │ 00  │
//!                           └─────┴─────┴─────┘
//!
//!                           ┌─────┐
//!     NULL                  │ 00  │
//!                           └─────┘
//!
//!      Input                  Row Format
//! ```
//!
//! ## Struct Encoding
//!
//! A null is encoded as a `0_u8`.
//!
//! A valid value is encoded as `1_u8` followed by the row encoding of each child.
//!
//! This encoding effectively flattens the schema in a depth-first fashion.
//!
//! For example
//!
//! ```text
//! ┌───────┬────────────────────────┬───────┐
//! │ Int32 │ Struct[Int32, Float32] │ Int32 │
//! └───────┴────────────────────────┴───────┘
//! ```
//!
//! Is encoded as
//!
//! ```text
//! ┌───────┬───────────────┬───────┬─────────┬───────┐
//! │ Int32 │ Null Sentinel │ Int32 │ Float32 │ Int32 │
//! └───────┴───────────────┴───────┴─────────┴───────┘
//! ```
//!
//! ## List Encoding
//!
//! Lists are encoded by first encoding all child elements to the row format.
//!
//! A "canonical byte array" is then constructed by concatenating the row
//! encodings of all their elements into a single binary array, followed
//! by the lengths of each encoded row, and the number of elements, encoded
//! as big endian `u32`.
//!
//! This canonical byte array is then encoded using the variable length byte
//! encoding described above.
//!
//! _The lengths are not strictly necessary but greatly simplify decode, they
//! may be removed in a future iteration_.
//!
//! For example given:
//!
//! ```text
//! [1_u8, 2_u8, 3_u8]
//! [1_u8, null]
//! []
//! null
//! ```
//!
//! The elements would be converted to:
//!
//! ```text
//!     ┌──┬──┐     ┌──┬──┐     ┌──┬──┐     ┌──┬──┐        ┌──┬──┐
//!  1  │01│01│  2  │01│02│  3  │01│03│  1  │01│01│  null  │00│00│
//!     └──┴──┘     └──┴──┘     └──┴──┘     └──┴──┘        └──┴──┘
//!```
//!
//! Which would be grouped into the following canonical byte arrays:
//!
//! ```text
//!                         ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
//!  [1_u8, 2_u8, 3_u8]     │01│01│01│02│01│03│00│00│00│02│00│00│00│02│00│00│00│02│00│00│00│03│
//!                         └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
//!                          └──── rows ────┘   └───────── row lengths ─────────┘  └─ count ─┘
//!
//!                         ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
//!  [1_u8, null]           │01│01│00│00│00│00│00│02│00│00│00│02│00│00│00│02│
//!                         └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
//!```
//!
//! With `[]` represented by an empty byte array, and `null` a null byte array.
//!
//! These byte arrays will then be encoded using the variable length byte encoding
//! described above.
//!
//! # Ordering
//!
//! ## Float Ordering
//!
//! Floats are totally ordered just like in the rest of Polars,
//! -inf < neg < -0.0 = 0.0 < pos < inf < nan, with all nans being equal.
//!
//! ## Null Ordering
//!
//! The encoding described above will order nulls first, this can be inverted by representing
//! nulls as `0xFF_u8` instead of `0_u8`
//!
//! ## Reverse Column Ordering
//!
//! The order of a given column can be reversed by negating the encoded bytes of non-null values
//!
//! # Columns Without Nulls
//!
//! A column that does not encode nulls, see [`RowEncodingOptions::SKIP_NULL_ENCODING`], leaves
//! out the null sentinel byte of fixed-width values, structs and fixed-size lists, and the
//! validity bit of decimals. Nested values, booleans, variable length bytes and lists are encoded
//! as usual.
//!
//! With `propagate_nulls` in [`convert_columns`] a row is null if any of its values is null.
//!
//! # Output Types
//!
//! [`RowEncodingLayout`] describes how columns are encoded and is needed to decode them again.
//! [`RowEncodingOutput`] picks the type of the rows, given by [`RowEncodingLayout::dtype`]:
//! `AlwaysBinaryOffset` and `AlwaysBinary` return the format above as `LargeBinary` or
//! `BinaryView`. `Optimal` returns bit-packed rows when every column is a null, boolean, integer,
//! float, decimal, enum or unordered categorical column and the packed rows fit in 128 bits. A
//! single binary or string column that is unordered or ascending with nulls first is returned
//! unchanged, nulls included, with strings as `BinaryView`. Otherwise it returns the format above
//! as `LargeBinary`. The choice only depends on the layout, never on the data.
//!
//! ## Bit-Packed Rows
//!
//! Each column takes a null bit, if it encodes nulls, followed by its value bits: 0 for
//! nulls, 1 for booleans, the bit width for integers and floats, and `ceil(p * log2(10)) + 1`
//! for decimals of precision `p` (at most 128). The columns are placed in order from the most
//! significant bit down to bit 0 of one number of `B <= 128` bits. For `B <= 1` the rows are
//! booleans, otherwise the smallest unsigned integer that fits. Unused high bits are zero.
//!
//! The null bit is 1 for valid values and 0 for nulls, or the other way around with nulls
//! last. The value bits of a null are zero. The value bits of a valid value are the same
//! bits as the big-endian value bytes in the format above, inverted when descending. So
//! packed rows compare like the format above. Unordered columns are packed like ascending
//! columns with nulls first.
//!
//! [COBS]: https://en.wikipedia.org/wiki/Consistent_Overhead_Byte_Stuffing
//! [byte stuffing]: https://en.wikipedia.org/wiki/High-Level_Data_Link_Control#Asynchronous_framing

extern crate core;

pub mod decode;
pub mod encode;
pub(crate) mod fixed;
mod layout;
mod row;
mod utils;
pub(crate) mod variable;
mod widths;

use polars_arrow::array::*;
pub type ArrayRef = Box<dyn Array>;

pub use encode::convert_columns;
pub use layout::{RowEncodingField, RowEncodingLayout, RowEncodingOutput};
pub use row::{RowEncodingCategoricalContext, RowEncodingContext, RowEncodingOptions};
