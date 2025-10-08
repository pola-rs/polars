use proptest::prelude::{Strategy, any};
use proptest::sample::SizeRange;

use super::BinaryArray;
use crate::bitmap::Bitmap;
use crate::datatypes::ArrowDataType;
use crate::offset::OffsetsBuffer;

pub fn binary_array(size_range: impl Into<SizeRange>) -> impl Strategy<Value = BinaryArray<i64>> {
    let size_range = size_range.into();
    (
        any::<bool>(),
        proptest::prelude::any_with::<Vec<(bool, Vec<u8>)>>(size_range.lift()),
    )
        .prop_map(|(do_validity, values)| {
            let validity = do_validity.then(|| Bitmap::from_iter(values.iter().map(|(v, _)| *v)));

            let mut offsets = Vec::with_capacity(values.len() + 1);
            offsets.push(0i64);
            for (_, value) in &values {
                let value = value.len() % 7;
                offsets.push(*offsets.last().unwrap() + value as i64);
            }

            let mut buffer = Vec::with_capacity((*offsets.last().unwrap()) as usize);
            for (_, value) in &values {
                buffer.extend_from_slice(value);
            }

            let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets.into()) };
            BinaryArray::new(ArrowDataType::LargeBinary, offsets, buffer.into(), validity)
        })
}
