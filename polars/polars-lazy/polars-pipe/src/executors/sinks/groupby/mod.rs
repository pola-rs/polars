pub(crate) mod aggregates;
mod generic;
mod primitive;
mod string;
mod utils;

pub(crate) use generic::*;
use polars_core::prelude::*;
pub(crate) use primitive::*;
pub(crate) use string::*;

pub(super) fn physical_agg_to_logical(cols: &mut [Series], output_schema: &Schema) {
    for (s, (name, dtype)) in cols.iter_mut().zip(output_schema.iter()) {
        if s.name() != name {
            s.rename(name);
        }
        match dtype {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(Some(rev_map)) => {
                let cats = s.u32().unwrap().clone();
                // safety:
                // the rev-map comes from these categoricals
                unsafe {
                    *s = CategoricalChunked::from_cats_and_rev_map_unchecked(cats, rev_map.clone())
                        .into_series()
                }
            }
            _ => {
                let dtype_left = s.dtype();
                if dtype_left != dtype
                    && !matches!(dtype, DataType::Boolean)
                    && !(dtype.is_float() && dtype_left.is_float())
                {
                    *s = s.cast(dtype).unwrap()
                }
            }
        }
    }
}
