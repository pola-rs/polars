//! What a [`PlFixedSizeListArray`] gains from being known to be [`Flat`].

use arrow::bitmap::Bitmap;

use super::PlFixedSizeListArray;
use crate::array::PlArray;
use crate::flat::Flat;

/// The methods a [`PlFixedSizeListArray`] gains from holding one slot and one bit per element.
impl Flat<PlFixedSizeListArray> {
    /// The values array, holding `len * width` values.
    #[inline]
    pub fn values(&self) -> &dyn PlArray {
        &*self.as_array().values
    }

    /// Consumes this array into its internal components, whose values and bits are one per element.
    #[inline]
    pub fn into_inner(self) -> (Box<dyn PlArray>, usize, Option<Bitmap>) {
        let PlFixedSizeListArray {
            values,
            width,
            length: _,
            validity,
        } = self.into_array();

        (values, width, validity)
    }
}

crate::impl_flat_methods!(PlFixedSizeListArray);
