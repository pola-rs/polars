use polars_arrow::compute::utils::combine_validities_and;
use polars_compute::horizontal_concat_list::horizontal_concat_list_unchecked;
use polars_core::prelude::{Column, DataType, IntoColumn, ListChunked};
use polars_error::{PolarsContext, PolarsResult};
use polars_utils::broadcast::broadcast_len;

/// Note: The caller must ensure all columns in `args` have the type `dtype`.
///
/// # Panics
/// Panics if
/// * `args` is empty
/// * `dtype` is not a `DataType::List`
pub fn concat_list(args: &[Column], dtype: &DataType) -> PolarsResult<Column> {
    let DataType::List(inner_dtype) = dtype else {
        panic!("{}", dtype);
    };

    let inner_dtype = inner_dtype.as_ref();

    let output_height = broadcast_len(args.iter()).context("concat_list")?;
    // If there is a `List` column with a single NULL, the output will be entirely NULL.
    let mut return_all_null = false;
    // Indicates whether all `arrays` have unit length
    let mut all_unit_len = true;
    let mut validities = Vec::with_capacity(args.len());

    let arrays = args
        .iter()
        .map(|c| {
            let len = c.len();

            all_unit_len &= len == 1;

            // Don't expand scalars to height, this is handled by the `horizontal_concat_list` kernel.
            let s = c.as_materialized_series_maintain_scalar();
            debug_assert_eq!(s.dtype(), dtype);

            let arr = s.list().unwrap().rechunk();
            let validity = arr.rechunk_validity();

            return_all_null |= len == 1 && validity.as_ref().is_some_and(|x| !x.get_bit(0));

            // Ignore unit-length validities. If they are non-valid then `return_all_null` will
            // cause an early return.
            if let Some(v) = validity.filter(|_| len > 1) {
                validities.push(v)
            }

            arr.downcast_as_array().clone()
        })
        .collect::<Vec<_>>();

    if return_all_null || output_height == 0 {
        return Ok(ListChunked::full_null_with_dtype(
            args[0].name().clone(),
            output_height,
            inner_dtype,
        )
        .into_column());
    }

    // Combine validities
    let outer_validity = validities.into_iter().fold(None, |a, b| {
        debug_assert_eq!(b.len(), output_height);
        combine_validities_and(a.as_ref(), Some(&b))
    });

    // At this point the output height and all arrays should have non-zero length
    let out = if all_unit_len {
        // Fast-path for all scalars
        let arr = unsafe { horizontal_concat_list_unchecked(&arrays, 1, outer_validity) }?;

        let mut out = ListChunked::with_chunk(args[0].name().clone(), arr);
        unsafe { out.to_logical(inner_dtype.clone()) };

        return Ok(out.into_column().new_from_index(0, output_height));
    } else {
        let arr =
            unsafe { horizontal_concat_list_unchecked(&arrays, output_height, outer_validity) }?;

        let mut out = ListChunked::with_chunk(args[0].name().clone(), arr);
        unsafe { out.to_logical(inner_dtype.clone()) };

        out.into_column()
    };

    Ok(out)
}
