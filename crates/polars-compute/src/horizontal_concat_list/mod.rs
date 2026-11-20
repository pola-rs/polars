#![allow(unsafe_op_in_unsafe_fn)]
use polars_arrow::array::builder::{ShareStrategy, StaticArrayBuilder, make_builder};
use polars_arrow::array::{
    Array, BinaryArray, BinaryArrayBuilder, BinaryViewArray, BinaryViewArrayBuilder, BooleanArray,
    BooleanArrayBuilder, FixedSizeBinaryArray, FixedSizeBinaryArrayBuilder, FixedSizeListArray,
    FixedSizeListArrayBuilder, ListArray, ListArrayBuilder, NullArray, PrimitiveArray,
    PrimitiveArrayBuilder, StructArray, StructArrayBuilder, Utf8ViewArray, Utf8ViewArrayBuilder,
};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::{ArrowDataType, PhysicalType};
use polars_arrow::offset::{Offsets, OffsetsBuffer};
use polars_arrow::with_match_primitive_type_full;
use polars_error::PolarsResult;

/// Low-level operation used by `concat_list`. For every output row `r`, the resulting list is
/// `arrays[0][r] ++ arrays[1][r] ++ .. ++ arrays[n-1][r]`.
///
/// Rows that are NULL in `outer_validity` are output as empty lists.
///
/// # Safety
/// * `arrays` is non-empty
/// * Every array `arrays[i]` has a length of either
///   * `output_height`
///   * `1` (this would be broadcasted)
/// * All arrays in `arrays` have the same type
/// * `outer_validity`, if given, has a length of `output_height`
///
/// # Errors
/// Errors if the total number of output values overflows `i64`.
pub unsafe fn horizontal_concat_list_unchecked(
    arrays: &[ListArray<i64>],
    output_height: usize,
    outer_validity: Option<Bitmap>,
) -> PolarsResult<ListArray<i64>> {
    use PhysicalType::*;

    assert!(!arrays.is_empty());

    debug_assert!(arrays.iter().all(|x| x.dtype() == arrays[0].dtype()));
    debug_assert!(
        arrays
            .iter()
            .all(|x| x.len() == output_height || x.len() == 1)
    );
    debug_assert!(
        outer_validity
            .as_ref()
            .is_none_or(|x| x.len() == output_height)
    );

    // Sum the row lengths column-wise, this auto-vectorizes.
    let mut lengths = vec![0usize; output_height];

    for arr in arrays {
        let offsets = arr.offsets().as_slice();

        if arr.len() == 1 {
            let len = (offsets[1] - offsets[0]) as usize;
            lengths.iter_mut().for_each(|x| *x += len);
        } else {
            lengths
                .iter_mut()
                .zip(offsets.windows(2))
                .for_each(|(x, w)| *x += (w[1] - w[0]) as usize);
        }
    }

    if let Some(validity) = outer_validity.as_ref() {
        lengths
            .iter_mut()
            .zip(validity.iter())
            .for_each(|(x, valid)| *x *= valid as usize);
    }

    let out_offsets: OffsetsBuffer<i64> = Offsets::try_from_lengths(lengths.into_iter())?.into();

    let dtype = arrays[0].values().dtype();

    let values: Box<dyn Array> = match dtype.to_physical_type() {
        Null => Box::new(NullArray::new(dtype.clone(), *out_offsets.last() as usize)),
        Boolean => horizontal_concat_list_unchecked_impl_generic(
            BooleanArrayBuilder::new(dtype.clone()),
            arrays,
            &downcast_values::<BooleanArray>(arrays),
            &out_offsets,
        ),
        Primitive(primitive) => with_match_primitive_type_full!(primitive, |$T| {
            horizontal_concat_list_unchecked_impl_generic(
                PrimitiveArrayBuilder::<$T>::new(dtype.clone()),
                arrays,
                &downcast_values::<PrimitiveArray<$T>>(arrays),
                &out_offsets,
            )
        }),
        LargeBinary => horizontal_concat_list_unchecked_impl_generic(
            BinaryArrayBuilder::<i64>::new(dtype.clone()),
            arrays,
            &downcast_values::<BinaryArray<i64>>(arrays),
            &out_offsets,
        ),
        FixedSizeBinary => horizontal_concat_list_unchecked_impl_generic(
            FixedSizeBinaryArrayBuilder::new(dtype.clone()),
            arrays,
            &downcast_values::<FixedSizeBinaryArray>(arrays),
            &out_offsets,
        ),
        Struct => {
            let ArrowDataType::Struct(fields) = dtype else {
                unreachable!()
            };

            horizontal_concat_list_unchecked_impl_generic(
                StructArrayBuilder::new(
                    dtype.clone(),
                    fields.iter().map(|f| make_builder(f.dtype())).collect(),
                ),
                arrays,
                &downcast_values::<StructArray>(arrays),
                &out_offsets,
            )
        },
        LargeList => {
            let ArrowDataType::LargeList(inner) = dtype else {
                unreachable!()
            };

            horizontal_concat_list_unchecked_impl_generic(
                ListArrayBuilder::<i64, _>::new(dtype.clone(), make_builder(inner.dtype())),
                arrays,
                &downcast_values::<ListArray<i64>>(arrays),
                &out_offsets,
            )
        },
        FixedSizeList => {
            let ArrowDataType::FixedSizeList(inner, _) = dtype else {
                unreachable!()
            };

            horizontal_concat_list_unchecked_impl_generic(
                FixedSizeListArrayBuilder::new(dtype.clone(), make_builder(inner.dtype())),
                arrays,
                &downcast_values::<FixedSizeListArray>(arrays),
                &out_offsets,
            )
        },
        BinaryView => horizontal_concat_list_unchecked_impl_generic(
            BinaryViewArrayBuilder::new(dtype.clone()),
            arrays,
            &downcast_values::<BinaryViewArray>(arrays),
            &out_offsets,
        ),
        Utf8View => horizontal_concat_list_unchecked_impl_generic(
            Utf8ViewArrayBuilder::new(dtype.clone()),
            arrays,
            &downcast_values::<Utf8ViewArray>(arrays),
            &out_offsets,
        ),
        t => unimplemented!("horizontal_concat_list not supported for data type {:?}", t),
    };

    Ok(ListArray::new(
        arrays[0].dtype().clone(),
        out_offsets,
        values,
        outer_validity,
    ))
}

fn downcast_values<T: Array + 'static>(arrays: &[ListArray<i64>]) -> Vec<&T> {
    arrays
        .iter()
        .map(|x| x.values().as_any().downcast_ref::<T>().unwrap())
        .collect()
}

/// Generic over the builder so that the per-row extend is statically dispatched.
unsafe fn horizontal_concat_list_unchecked_impl_generic<B>(
    mut builder: B,
    arrays: &[ListArray<i64>],
    values: &[&B::Array],
    out_offsets: &OffsetsBuffer<i64>,
) -> Box<dyn Array>
where
    B: StaticArrayBuilder,
{
    assert_eq!(values.len(), arrays.len());

    let output_height = out_offsets.len_proxy();
    let out_offsets = out_offsets.as_slice();
    let out_len = *out_offsets.last().unwrap() as usize;

    let offsets = arrays
        .iter()
        .map(|x| x.offsets().as_slice())
        .collect::<Vec<_>>();
    // We multiply the row index by this to support broadcasting.
    let strides = arrays
        .iter()
        .map(|x| (x.len() != 1) as usize)
        .collect::<Vec<_>>();

    builder.reserve(out_len);

    for row_idx in 0..output_height {
        // Skips empty and NULL rows.
        if out_offsets.get_unchecked(row_idx) == out_offsets.get_unchecked(1 + row_idx) {
            continue;
        }

        for ((values, offsets), stride) in values.iter().zip(&offsets).zip(&strides) {
            let i = row_idx * stride;
            let start = *offsets.get_unchecked(i) as usize;
            let end = *offsets.get_unchecked(1 + i) as usize;

            if start != end {
                builder.subslice_extend(values, start, end - start, ShareStrategy::Always);
            }
        }
    }

    debug_assert_eq!(builder.len(), out_len);

    Box::new(builder.freeze())
}
