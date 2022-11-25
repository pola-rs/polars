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
        let dtype_left = s.dtype();
        if dtype_left != dtype
            && !matches!(dtype, DataType::Boolean)
            && !(dtype.is_float() && dtype_left.is_float())
        {
            *s = s.cast(dtype).unwrap()
        }
    }
}
