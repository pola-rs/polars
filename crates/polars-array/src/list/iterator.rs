//! The iterators of a [`PlListArray`](super::PlListArray).
//!
//! A list array cuts its values at the offsets it holds, which is the one thing its walk does not
//! share with every other nested array — see [`crate::nested`].

use crate::nested::{NestedIter, NestedValuesIter, Offsets};

/// Iterator over the elements of a [`PlListArray`](super::PlListArray), ignoring validity.
pub type PlListValuesIter<'a> = NestedValuesIter<'a, Offsets<'a>>;

/// Iterator over the optional elements of a [`PlListArray`](super::PlListArray).
pub type PlListIter<'a> = NestedIter<'a, Offsets<'a>>;

#[cfg(test)]
mod tests {
    use polars_buffer::Buffer;

    use crate::iterator_tests::assert_iterates;
    use crate::{PlArray, PlListArray, PlPrimitiveArray};

    /// The list `values` are, as an element of a list array is.
    fn element(values: &[i32]) -> Box<dyn PlArray> {
        Box::new(PlPrimitiveArray::from_vec(values.to_vec()))
    }

    /// A flat array of the lists `[1, 2]`, `[]` and `[3]`.
    fn flat_array() -> PlListArray {
        PlListArray::new(
            element(&[1, 2, 3]),
            Buffer::from_owner([0, 2, 2, 3]),
            3,
            None,
        )
    }

    fn elements() -> [Box<dyn PlArray>; 3] {
        [element(&[1, 2]), element(&[]), element(&[3])]
    }

    #[test]
    fn flat() {
        let array = flat_array();

        assert_iterates(array.values_iter(), &elements());
        assert_iterates(array.iter(), &elements().map(Some));
    }

    #[test]
    fn scalar() {
        let array = PlListArray::new_scalar(element(&[1, 2]), 4);
        let expected = [(); 4].map(|()| element(&[1, 2]));

        assert_iterates(array.values_iter(), &expected);
        assert_iterates(array.iter(), &expected.map(Some));
    }

    /// The elements of a sliced array start partway into the offsets, which hold the last end.
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
