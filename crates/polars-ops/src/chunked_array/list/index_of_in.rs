use super::*;
use crate::series::index_of;

pub fn list_index_of_in(ca: &ListChunked, needles: &Series) -> PolarsResult<Series> {
    let mut builder = PrimitiveChunkedBuilder::<IdxType>::new(ca.name().clone(), ca.len());
    let inner_dtype = ca.dtype().inner_dtype().unwrap();
    if needles.len() == 1 {
        let needle = needles.get(0).unwrap();
        let needle_dtype = inner_dtype.clone();
        let needle = Scalar::new(needle_dtype, needle.into_static());
        ca.amortized_iter().for_each(|opt_series| {
            if let Some(subseries) = opt_series {
                builder.append_option(
                    // The clone() could perhaps be removed by refactoring
                    // index_of() to take a (scalar) Column, and then we could
                    // just pass through the Column as is. Still a bunch of
                    // duplicate work even in that case though, so possibly the
                    // real solution would be duplicating the code in
                    // index_of(), or refactoring so its guts are more
                    // available.
                    index_of(subseries.as_ref(), needle.clone())
                        .unwrap()
                        .map(|v| v.try_into().unwrap()),
                );
            } else {
                builder.append_null();
            }
        });
    } else {
        polars_ensure!(
            ca.len() == needles.len(),
            ComputeError: "shapes don't match: expected {} elements in 'index_of_in' comparison, got {}",
            ca.len(),
            needles.len()
        );
        let needles = needles.rechunk();
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
