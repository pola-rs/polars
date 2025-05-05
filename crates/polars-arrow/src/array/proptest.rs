use proptest::prelude::Strategy;
use proptest::prop_oneof;
use proptest::sample::SizeRange;

use super::Array;
use crate::array::binview::proptest::utf8view_array;
use crate::array::primitive::proptest::primitive_array;
use crate::array::{PrimitiveArray, Utf8ViewArray};

pub fn array(size_range: impl Into<SizeRange>) -> impl Strategy<Value = Box<dyn Array>> {
    let size_range = size_range.into();
    prop_oneof![
        utf8view_array(size_range.clone()).prop_map(Utf8ViewArray::boxed),
        primitive_array::<i8>(size_range.clone()).prop_map(PrimitiveArray::boxed),
        primitive_array::<i16>(size_range.clone()).prop_map(PrimitiveArray::boxed),
        primitive_array::<i32>(size_range.clone()).prop_map(PrimitiveArray::boxed),
        primitive_array::<i64>(size_range.clone()).prop_map(PrimitiveArray::boxed),
        primitive_array::<u8>(size_range.clone()).prop_map(PrimitiveArray::boxed),
        primitive_array::<u16>(size_range.clone()).prop_map(PrimitiveArray::boxed),
        primitive_array::<u32>(size_range.clone()).prop_map(PrimitiveArray::boxed),
        primitive_array::<u64>(size_range.clone()).prop_map(PrimitiveArray::boxed),
        primitive_array::<f32>(size_range.clone()).prop_map(PrimitiveArray::boxed),
        primitive_array::<f64>(size_range.clone()).prop_map(PrimitiveArray::boxed),
    ]
}
