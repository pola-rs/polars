use proptest::prelude::{Strategy, any, any_with};
use proptest::sample::SizeRange;

use super::ListArray;
use crate::bitmap::Bitmap;
use crate::datatypes::{ArrowDataType, Field};
use crate::offset::OffsetsBuffer;

pub fn list_array_with_dtype(
    size_range: impl Into<SizeRange>,
    field: Box<Field>,
) -> impl Strategy<Value = ListArray<i64>> {
    let size_range = size_range.into();
    (
        any::<bool>(),
        any_with::<Vec<(bool, u32)>>(size_range.lift()),
    )
        .prop_flat_map(move |(do_validity, values)| {
            let field = field.clone();
            let validity = do_validity.then(|| Bitmap::from_iter(values.iter().map(|(v, _)| *v)));

            let mut offsets = Vec::with_capacity(values.len() + 1);
            offsets.push(0i64);
            for (_, value) in &values {
                let value = *value % 7;
                offsets.push(*offsets.last().unwrap() + value as i64);
            }

            let child_length = *offsets.last().unwrap();
            let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets.into()) };

            crate::array::proptest::array_with_dtype(field.dtype.clone(), child_length as usize)
                .prop_map(move |child_array| {
                    ListArray::<i64>::new(
                        ArrowDataType::LargeList(field.clone()),
                        offsets.clone(),
                        child_array,
                        validity.clone(),
                    )
                })
        })
}
