//! What a [`PlFixedSizeListArray`] gains from being known to be [`Flat`].

use arrow::bitmap::Bitmap;

use super::PlFixedSizeListArray;
use crate::array::PlArray;
use crate::flat::Flat;

/// The methods a [`PlFixedSizeListArray`] gains from holding the values of every element and one
/// validity bit per element.
impl Flat<PlFixedSizeListArray> {
    /// The values array, holding exactly [`len`](PlFixedSizeListArray::len) `*`
    /// [`width`](PlFixedSizeListArray::width) values.
    #[inline]
    pub fn values(&self) -> &dyn PlArray {
        &*self.0.values
    }

    /// The validity mask, if any element may be null, as an ordinary [`Bitmap`] of exactly
    /// [`len`](PlFixedSizeListArray::len) bits.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.0.validity.as_ref()
    }

    /// Consumes this array into its internal components, whose values and bits are one per element.
    #[inline]
    pub fn into_inner(self) -> (Box<dyn PlArray>, usize, Option<Bitmap>) {
        let PlFixedSizeListArray {
            values,
            width,
            length: _,
            validity,
        } = self.0;

        (values, width, validity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PlPrimitiveArray;

    #[test]
    fn buffers_are_handed_out_as_they_are() {
        let arr = PlFixedSizeListArray::new_scalar(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
            3,
        );
        let flat = arr.to_flat();

        assert_eq!(flat.values().len(), 6);
        assert!(flat.validity().is_none());
        assert_eq!(flat, arr);
    }

    #[test]
    fn a_scalar_mask_is_written_out() {
        let arr = PlFixedSizeListArray::new_full_null(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
            3,
        );
        let flat = arr.to_flat();

        assert_eq!(flat.validity().unwrap().len(), 3);
        assert_eq!(flat.null_count(), 3);
    }

    #[test]
    fn into_inner_gives_up_the_length() {
        let arr = PlFixedSizeListArray::from_values(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4])),
            2,
        );
        let (values, width, validity) = arr.to_flat().into_inner();

        assert_eq!(values.len(), 4);
        assert_eq!(width, 2);
        assert!(validity.is_none());
    }
}
