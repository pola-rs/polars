use polars_core::chunked_array::cast::CastOptions;
use polars_utils::total_ord::TotalEq;

use super::*;
use crate::series::index_of;

fn check_if_cast_lossless(dtype1: &DataType, dtype2: &DataType, result: bool) -> PolarsResult<()> {
    polars_ensure!(
        result,
        InvalidOperation: "cannot cast lossless between {} and {}",
        dtype1, dtype2,
    );
    Ok(())
}

pub fn list_index_of_in(ca: &ListChunked, needles: &Series) -> PolarsResult<Series> {
    let mut builder = PrimitiveChunkedBuilder::<IdxType>::new(ca.name().clone(), ca.len());
    let inner_dtype = ca.dtype().inner_dtype().unwrap();
    let needle_dtype = needles.dtype();
    // We need to do casting ourselves, unless we grow a new CastingRules
    // variant.
    check_if_cast_lossless(
        &needle_dtype,
        inner_dtype,
        (inner_dtype.leaf_dtype().is_float() == needle_dtype.is_float()) || needle_dtype.is_null(),
    )?;
    if needles.len() == 1 {
        let needle = needles.get(0).unwrap();
        let cast_needle = needle.cast(inner_dtype);
        check_if_cast_lossless(&needle_dtype, inner_dtype, cast_needle.tot_eq(&needle))?;
        let mut needle_dtype = cast_needle.dtype().clone();
        if needle_dtype.is_null() {
            needle_dtype = inner_dtype.clone();
        }
        let needle = Scalar::new(needle_dtype, cast_needle.into_static());
        ca.amortized_iter().for_each(|opt_series| {
            if let Some(subseries) = opt_series {
                builder.append_option(
                    // TODO clone() sucks, maybe need to change the API for
                    // index_of so it takes AnyValue<'_> instead of a Scalar
                    // which implies AnyValue<'static>?
                    index_of(subseries.as_ref(), needle.clone())
                        .unwrap()
                        .map(|v| v.try_into().unwrap()),
                );
            } else {
                builder.append_null();
            }
        });
    } else {
        let needles = needles.cast_with_options(ca.inner_dtype(), CastOptions::Strict)?;
        ca.amortized_iter()
            // TODO iter() assumes a single chunk. could continue to use this
            // and just rechunk(), or have needles also be a ChunkedArray, in
            // which case we'd need to have to use one of the
            // dispatch-on-dtype-and-cast-to-relevant-chunkedarray-type macros
            // to duplicate the implementation code per dtype.
            .zip(needles.iter())
            .for_each(|(opt_series, needle)| match (opt_series, needle) {
                (None, _) => builder.append_null(),
                (Some(subseries), needle) => {
                    let needle = Scalar::new(needles.dtype().clone(), needle.into_static());
                    builder.append_option(
                        index_of(subseries.as_ref(), needle)
                            .unwrap()
                            .map(|v| v.try_into().unwrap()),
                    );
                },
            });
    }
    Ok(builder.finish().into())
}
