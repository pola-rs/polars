pub(crate) mod aggregates;
mod generic;
mod ooc;
mod ooc_state;
mod primitive;
mod string;
mod utils;

pub(crate) use generic::GenericGroupby2;
use polars_core::prelude::*;
#[cfg(feature = "dtype-categorical")]
use polars_core::using_string_cache;
pub(crate) use primitive::*;
pub(crate) use string::*;

pub(super) fn physical_agg_to_logical(cols: &mut [Series], output_schema: &Schema) {
    for (s, (name, dtype)) in cols.iter_mut().zip(output_schema.iter()) {
        if s.name() != name {
            s.rename(name);
        }
        match dtype {
            #[cfg(feature = "dtype-categorical")]
            dt @ (DataType::Categorical(rev_map, ordering) | DataType::Enum(rev_map, ordering)) => {
                if let Some(rev_map) = rev_map {
                    let cats = s.u32().unwrap().clone();
                    // SAFETY:
                    // the rev-map comes from these categoricals
                    unsafe {
                        *s = CategoricalChunked::from_cats_and_rev_map_unchecked(
                            cats,
                            rev_map.clone(),
                            matches!(dt, DataType::Enum(_, _)),
                            *ordering,
                        )
                        .into_series()
                    }
                } else {
                    let cats = s.u32().unwrap().clone();
                    if using_string_cache() {
                        // SAFETY, we go from logical to primitive back to logical so the categoricals should still match the global map.
                        *s = unsafe {
                            CategoricalChunked::from_global_indices_unchecked(cats, *ordering)
                                .into_series()
                        };
                    } else {
                        // we set the global string cache once we start a streaming pipeline
                        unreachable!()
                    }
                }
            },
            _ => {
                let dtype_left = s.dtype();
                if dtype_left != dtype
                    && !matches!(dtype, DataType::Boolean)
                    && !(dtype.is_float() && dtype_left.is_float())
                {
                    *s = s.cast(dtype).unwrap()
                }
            },
        }
    }
}
