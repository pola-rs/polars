use crate::bitmap::{PlBitmapIter, PlBitmapRef, ValidityFold, ValidityIter};

/// Iterator over the optional elements of a [`PlBooleanArray`](super::PlBooleanArray).
#[derive(Clone)]
pub struct PlBooleanIter<'a> {
    values: PlBitmapIter<'a>,
    validity: ValidityIter<'a>,
}

impl<'a> PlBooleanIter<'a> {
    #[inline]
    pub(super) fn new(
        values: PlBitmapRef<'a>,
        validity: Option<PlBitmapRef<'a>>,
        length: usize,
    ) -> Self {
        assert_eq!(values.len(), length);
        assert!(validity.is_none_or(|validity| validity.len() == length));

        Self {
            values: values.iter(),
            validity: ValidityIter::new(validity),
        }
    }

    /// The values and the mask that says which of them are elements, to walk in one loop.
    #[inline]
    fn split(self) -> (PlBitmapIter<'a>, ValidityFold<'a>) {
        (self.values, self.validity.into_mask())
    }
}

crate::impl_optional_iter!(PlBooleanIter<'a>, bool);
