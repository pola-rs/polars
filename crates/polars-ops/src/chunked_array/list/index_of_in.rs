use super::*;
use crate::series::index_of;

pub fn list_index_of_in(ca: &ListChunked, needles: &Series) -> PolarsResult<Series> {
    let mut builder = PrimitiveChunkedBuilder::<IdxType>::new(ca.name().clone(), ca.len());
    if needles.len() == 1 {
        // For some reason we need to do casting ourselves.
        let needle = needles.get(0).unwrap();
        let cast_needle = needle.cast(ca.dtype().inner_dtype().unwrap());
        if cast_needle != needle {
            todo!("nicer error handling");
        }
        let needle = Scalar::new(
            cast_needle.dtype().clone(),
            cast_needle.into_static(),
        );
        ca.amortized_iter().for_each(|opt_series| {
            if let Some(subseries) = opt_series {
                // TODO justify why unwrap()s are ok
                builder.append_option(
                    // TODO clone() sucks, maybe need to change the API for index_of?
                    index_of(subseries.as_ref(), needle.clone())
                        .unwrap()
                        .map(|v| v.try_into().unwrap()),
                );
            } else {
                builder.append_null();
            }
        });
    } else {
        ca.amortized_iter()
            .zip(needles.iter())
            .for_each(|(opt_series, needle)| {
                match (opt_series, needle) {
                    (None, _) => builder.append_null(),
                    (Some(subseries), needle) => {
                        let needle = Scalar::new(needles.dtype().clone(), needle.into_static());
                        // TODO justify why unwrap()s are ok
                        builder.append_option(
                            index_of(subseries.as_ref(), needle)
                                .unwrap()
                                .map(|v| v.try_into().unwrap()),
                        );
                    },
                }
            });
    }
    Ok(builder.finish().into())
}
