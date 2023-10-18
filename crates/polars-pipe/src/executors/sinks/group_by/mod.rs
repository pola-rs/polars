pub(crate) mod aggregates;
mod generic;
mod ooc;
mod ooc_state;
mod primitive;
mod string;
mod utils;

use arrow::array::ListArray;
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
        physical_series_to_logical(s, dtype);
    }
}

fn physical_series_to_logical(s: &mut Series, dtype: &DataType) {
    match dtype {
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(rev_map) => {
            if let Some(rev_map) = rev_map {
                let cats = s.u32().unwrap().clone();
                // safety:
                // the rev-map comes from these categoricals
                unsafe {
                    *s = CategoricalChunked::from_cats_and_rev_map_unchecked(cats, rev_map.clone())
                        .into_series()
                }
            } else {
                let cats = s.u32().unwrap().clone();
                if using_string_cache() {
                    // Safety, we go from logical to primitive back to logical so the categoricals should still match the global map.
                    *s = unsafe {
                        CategoricalChunked::from_global_indices_unchecked(cats).into_series()
                    }
                } else {
                    // we set the global string cache once we start a streaming pipeline
                    unreachable!()
                }
            }
        },
        #[cfg(feature = "dtype-categorical")]
        DataType::List(inner_dtype) => {
            // This match arm is for handling Categoricals that are within Lists.
            let list = s.list().unwrap();
            let mut inner_series = list.get_inner();
            physical_series_to_logical(&mut inner_series, inner_dtype);
            let inner_values = inner_series.array_ref(0).clone();

            let list = list.rechunk();
            let arr = list.downcast_iter().next().unwrap();

            let data_type = ListArray::<i64>::default_datatype(inner_values.data_type().clone());
            let new_arr = ListArray::<i64>::new(
                data_type,
                arr.offsets().clone(),
                inner_values,
                arr.validity().cloned(),
            );

            // Safety: we just casted so the dtype matches.
            unsafe {
                *s = Series::from_chunks_and_dtype_unchecked(
                    list.name(),
                    vec![Box::new(new_arr)],
                    &DataType::List(inner_dtype.clone()),
                );
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
