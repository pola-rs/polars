use proptest::prelude::{Arbitrary, Strategy, any_with};
use proptest::sample::SizeRange;

use super::PrimitiveArray;
use crate::bitmap::Bitmap;
use crate::types::NativeType;

pub fn primitive_array<T: NativeType + Arbitrary>(
    size_range: impl Into<SizeRange>,
) -> impl Strategy<Value = PrimitiveArray<T>> {
    let size_range = size_range.into();
    proptest::prop_oneof![
        any_with::<Vec<T>>(size_range.clone().lift()).prop_map(|v| PrimitiveArray::new(
            T::PRIMITIVE.into(),
            v.into(),
            None
        )),
        any_with::<Vec<(bool, T)>>(size_range.lift()).prop_map(|v| {
            let (validity, values): (Vec<bool>, Vec<T>) = v.into_iter().collect();
            let validity = Bitmap::from_iter(validity);
            PrimitiveArray::new(T::PRIMITIVE.into(), values.into(), Some(validity))
        }),
    ]
}
