//! Conversion between the arrays of this crate and the Arrow arrays of `polars-arrow`.
//!
//! The two lay their elements out the same way and are built on the same
//! [`Buffer`](polars_buffer::Buffer) and [`Bitmap`](arrow::bitmap::Bitmap), so a conversion is a
//! matter of handing the backing buffers over rather than of copying the elements. What does not
//! carry over is the logical type: an Arrow array names one and the arrays of this crate do not,
//! so importing drops it — see [`import`] — and exporting has to put one back, which is why every
//! Arrow array has an export function of its own that inlines the data type it exports as — see
//! [`export`].
//!
//! [`bridge`] is the typed crossing built on those two: it names, for each array of this crate,
//! the Arrow array that holds the same elements, so a kernel written for one can be called on the
//! other.

pub mod bridge;
pub mod export;
pub mod import;
