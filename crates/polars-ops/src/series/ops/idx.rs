use num_traits::{FromPrimitive, Zero};
use polars_core::prelude::*;
use polars_utils::index::ToIdx;

fn prepare_gather_index_impl<T>(ca: &ChunkedArray<T>, length: usize) -> IdxCa
where T: PolarsNumericType,
T::Native: ToIdx
{
    T::Native::from_usize()

    ca.apply_generic(|v| {
        v.and_then(|v|{
            if v < T::Native::zero() {

            }

            v.to_idx_size()
        })
    })
}

pub fn convert_to_index(s: &Series, length: usize)
