//! The iterators of a [`PlFixedSizeListArray`](super::PlFixedSizeListArray).
//!
//! A fixed size list array cuts its values every width, which is the one thing its walk does not
//! share with every other nested array — see [`crate::nested`].

use crate::nested::{NestedIter, NestedValuesIter, Stride};

/// Iterator over the elements of a [`PlFixedSizeListArray`](super::PlFixedSizeListArray),
/// ignoring validity.
pub type PlFixedSizeListValuesIter<'a> = NestedValuesIter<'a, Stride>;

/// Iterator over the optional elements of a
/// [`PlFixedSizeListArray`](super::PlFixedSizeListArray).
pub type PlFixedSizeListIter<'a> = NestedIter<'a, Stride>;

#[cfg(test)]
mod tests {

    use crate::iterator_tests::assert_iterates;
    use crate::{PlArray, PlFixedSizeListArray, PlPrimitiveArray};

    /// The list `values` are, as an element of a fixed size list array is.
    fn element(values: &[i32]) -> Box<dyn PlArray> {
        Box::new(PlPrimitiveArray::from_vec(values.to_vec()))
    }

    /// A flat array of the lists `[1, 2]`, `[3, 4]` and `[5, 6]`.
    fn flat_array() -> PlFixedSizeListArray {
        PlFixedSizeListArray::new(element(&[1, 2, 3, 4, 5, 6]), 2, 3, None)
    }

    fn elements() -> [Box<dyn PlArray>; 3] {
        [element(&[1, 2]), element(&[3, 4]), element(&[5, 6])]
    }

    #[test]
    fn flat() {
        let array = flat_array();

        assert_iterates(array.values_iter(), &elements());
        assert_iterates(array.iter(), &elements().map(Some));
    }

    #[test]
    fn scalar() {
        let array = PlFixedSizeListArray::new_scalar(element(&[1, 2]), 4);
        let expected = [(); 4].map(|()| element(&[1, 2]));

        assert_iterates(array.values_iter(), &expected);
        assert_iterates(array.iter(), &expected.map(Some));
    }

    /// The elements of a sliced array start partway into the values, cut out by the width.
    #[test]
    fn sliced() {
        let array = flat_array().sliced(1, 2);

        assert_iterates(array.values_iter(), &elements()[1..]);
        assert_iterates(
            array.iter(),
            &elements()[1..]
                .iter()
                .cloned()
                .map(Some)
                .collect::<Vec<_>>(),
        );
    }
}
