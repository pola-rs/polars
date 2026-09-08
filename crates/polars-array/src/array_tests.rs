//! The fixtures the tests that run over every array type of this crate share.

use polars_buffer::Buffer;

use crate::bitmap::PlBitmap;
use crate::concatenate::concatenate;
use crate::{
    PlArray, PlBinaryArray, PlBinaryViewArray, PlBooleanArray, PlFixedSizeBinaryArray,
    PlFixedSizeListArray, PlListArray, PlNullArray, PlPrimitiveArray, PlStructArray,
};

/// One array of every array type, all of three elements, with a null in the middle.
///
/// A test that says nothing about an array's own values reads its elements out of this, which is
/// what keeps a new array type from needing a copy of the test written for it.
pub(crate) fn arrays() -> Vec<Box<dyn PlArray>> {
    let validity = || Some(PlBitmap::from_iter([true, false, true]));
    vec![
        Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3]).with_validity(validity())),
        Box::new(PlBooleanArray::from_vec(vec![true, false, true]).with_validity(validity())),
        Box::new(
            PlBinaryArray::from_values_iter([b"foo".as_slice(), b"", b"bar"])
                .with_validity(validity()),
        ),
        Box::new(
            PlBinaryViewArray::from_values_iter([
                b"foo".as_slice(),
                b"bar",
                b"a value that is too long to inline",
            ])
            .with_validity(validity()),
        ),
        Box::new(
            PlFixedSizeBinaryArray::from_vec(vec![1u8, 2, 3, 4, 5, 6], 2).with_validity(validity()),
        ),
        Box::new(PlStructArray::new(
            vec![Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3]))],
            3,
            validity(),
        )),
        Box::new(
            PlListArray::from_offsets(
                Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])),
                Buffer::from(vec![0u64, 1, 2, 3]),
            )
            .with_validity(validity()),
        ),
        Box::new(
            PlFixedSizeListArray::from_values(
                Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4, 5, 6])),
                2,
            )
            .with_validity(validity()),
        ),
        Box::new(PlNullArray::new(3)),
    ]
}

/// The elements of `array` at `indices`, laid down as one array of its type.
///
/// An index of `None` stands for a null element, which is what an appended null comes back as.
pub(crate) fn picked(array: &dyn PlArray, indices: &[Option<usize>]) -> Box<dyn PlArray> {
    let picks: Vec<Box<dyn PlArray>> = indices
        .iter()
        .map(|index| match index {
            Some(index) => array.new_from_index(*index, 1),
            None => array.new_full_null_like_self(1),
        })
        .collect();

    let picks: Vec<&dyn PlArray> = picks.iter().map(|pick| &**pick).collect();
    concatenate(&picks).expect("one array of every element picked")
}

/// Asserts that `built` holds the elements of `array` at `indices`, in that order.
#[track_caller]
pub(crate) fn assert_picked(built: &dyn PlArray, array: &dyn PlArray, indices: &[Option<usize>]) {
    let expected = picked(array, indices);
    assert_eq!(built.len(), indices.len(), "length of {built:?}");
    assert_eq!(built, &*expected, "elements of {built:?}");
}
