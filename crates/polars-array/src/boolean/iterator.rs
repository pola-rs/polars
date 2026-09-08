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

#[cfg(test)]
mod tests {

    use crate::PlBooleanArray;
    use crate::iterator_tests::assert_iterates;

    #[test]
    fn flat() {
        let array = PlBooleanArray::from_vec(vec![true, false, true]);

        assert_iterates(array.values_iter(), &[true, false, true]);
        assert_iterates(array.iter(), &[Some(true), Some(false), Some(true)]);
    }

    #[test]
    fn scalar() {
        let array = PlBooleanArray::new_scalar(true, 4);

        assert_iterates(array.values_iter(), &[true; 4]);
        assert_iterates(array.iter(), &[Some(true); 4]);
    }

    #[test]
    fn a_broadcast_array_is_not_materialized() {
        // Walking a billion elements would not finish; the scalar path must hit.
        let array = PlBooleanArray::new_scalar(true, 1_000_000_000);

        assert_eq!(array.values_iter().count(), 1_000_000_000);
        assert_eq!(array.iter().nth(999_999_999), Some(Some(true)));
        assert_eq!(array.iter().nth_back(999_999_999), Some(Some(true)));
        assert_eq!(array.iter().len(), 1_000_000_000);
    }
}
