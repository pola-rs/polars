use proptest::prelude::{Strategy, any_with};
use proptest::sample::SizeRange;

use super::{ArrowDataType, BooleanArray};
use crate::bitmap::Bitmap;

pub fn boolean_array(size_range: impl Into<SizeRange>) -> impl Strategy<Value = BooleanArray> {
    let size_range = size_range.into();
    proptest::prop_oneof![
        any_with::<Vec<bool>>(size_range.clone().lift()).prop_map(|v| BooleanArray::new(
            ArrowDataType::Boolean,
            v.into(),
            None
        )),
        any_with::<Vec<(bool, bool)>>(size_range.lift()).prop_map(|v| {
            let values = Bitmap::from_iter(v.iter().map(|(v, _)| *v));
            let validity = Bitmap::from_iter(v.iter().map(|(_, v)| *v));
            BooleanArray::new(ArrowDataType::Boolean, values, Some(validity))
        }),
    ]
}
