use proptest::prelude::{Just, Strategy};
use proptest::prop_oneof;
use proptest::sample::SizeRange;

use super::StructArray;
use crate::bitmap::proptest::bitmap;
use crate::datatypes::{ArrowDataType, Field};

pub fn struct_array_with_fields(
    size_range: impl Into<SizeRange>,
    fields: Vec<Field>,
) -> impl Strategy<Value = StructArray> {
    let size_range = size_range.into();
    let (min, max) = size_range.start_end_incl();
    (min..=max).prop_flat_map(move |length| {
        let fields = fields.clone();

        let strategies: Vec<_> = fields
            .iter()
            .map(|field| crate::array::proptest::array_with_dtype(field.dtype.clone(), length))
            .collect();

        (
            strategies,
            prop_oneof![Just(None), bitmap(length).prop_map(Some)],
        )
            .prop_map(move |(child_arrays, validity)| {
                StructArray::new(
                    ArrowDataType::Struct(fields.clone()),
                    length,
                    child_arrays,
                    validity,
                )
            })
    })
}
