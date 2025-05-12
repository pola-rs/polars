use proptest::prelude::{Strategy, any_with};
use proptest::sample::SizeRange;

use super::{BinaryViewArray, MutableBinaryViewArray, Utf8ViewArray};
use crate::bitmap::Bitmap;

pub fn utf8view_array(size_range: impl Into<SizeRange>) -> impl Strategy<Value = Utf8ViewArray> {
    let size_range = size_range.into();
    proptest::prop_oneof![
        any_with::<Vec<String>>(size_range.clone().lift()).prop_map(|v| {
            let mut builder = MutableBinaryViewArray::<str>::with_capacity(v.len());
            builder.extend_values(v.into_iter());
            builder.freeze()
        }),
        any_with::<Vec<(bool, String)>>(size_range.lift()).prop_map(|v| {
            let mut builder = MutableBinaryViewArray::<str>::with_capacity(v.len());
            builder.extend_values(v.iter().map(|(_, s)| s));
            builder
                .freeze()
                .with_validity(Some(Bitmap::from_iter(v.iter().map(|(v, _)| *v))))
        }),
    ]
}

pub fn binview_array(size_range: impl Into<SizeRange>) -> impl Strategy<Value = BinaryViewArray> {
    let size_range = size_range.into();
    proptest::prop_oneof![
        any_with::<Vec<Vec<u8>>>(size_range.clone().lift()).prop_map(|v| {
            let mut builder = MutableBinaryViewArray::<[u8]>::with_capacity(v.len());
            builder.extend_values(v.into_iter());
            builder.freeze()
        }),
        any_with::<Vec<(bool, Vec<u8>)>>(size_range.lift()).prop_map(|v| {
            let mut builder = MutableBinaryViewArray::<[u8]>::with_capacity(v.len());
            builder.extend_values(v.iter().map(|(_, s)| s));
            builder
                .freeze()
                .with_validity(Some(Bitmap::from_iter(v.iter().map(|(v, _)| *v))))
        }),
    ]
}
