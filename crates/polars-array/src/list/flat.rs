//! What a [`PlListArray`] gains from being known to be [`Flat`].

use arrow::bitmap::Bitmap;
use polars_buffer::Buffer;

use super::PlListArray;
use crate::array::PlArray;
use crate::flat::Flat;

/// The methods a [`PlListArray`] gains from holding one range and one validity bit per element.
impl Flat<PlListArray> {
    /// The backing offsets buffer, holding exactly [`len`](PlListArray::len) `+ 1` offsets.
    #[inline(always)]
    pub const fn offsets(&self) -> &Buffer<u64> {
        &self.as_array().offsets
    }

    /// Consumes this array into its internal components, whose ranges and bits are one per element.
    #[inline]
    pub fn into_inner(self) -> (Box<dyn PlArray>, Buffer<u64>, Option<Bitmap>) {
        let PlListArray {
            values,
            offsets,
            length: _,
            validity,
        } = self.into_array();

        (values, offsets, validity)
    }
}

crate::impl_flat_methods!(PlListArray);
