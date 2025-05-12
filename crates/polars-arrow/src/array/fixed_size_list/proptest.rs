use proptest::prelude::{Just, Strategy};
use proptest::prop_oneof;
use proptest::sample::SizeRange;

use super::FixedSizeListArray;
use crate::bitmap::proptest::bitmap;
use crate::datatypes::{ArrowDataType, Field};

pub fn fixed_size_list_array_with_dtype(
    size_range: impl Into<SizeRange>,
    field: Box<Field>,
    width: usize,
) -> impl Strategy<Value = FixedSizeListArray> {
    let size_range = size_range.into();
    let (min, max) = size_range.start_end_incl();
    (min..=max).prop_flat_map(move |length| {
        let field = field.clone();
        (
            crate::array::proptest::array_with_dtype(field.dtype.clone(), length * width),
            prop_oneof![Just(None), bitmap(length).prop_map(Some)],
        )
            .prop_map(move |(child_array, validity)| {
                FixedSizeListArray::new(
                    ArrowDataType::FixedSizeList(field.clone(), width),
                    length,
                    child_array,
                    validity,
                )
            })
    })
}
