use std::hash::Hash;

use polars_core::prelude::*;
use polars_core::with_match_physical_integer_polars_type;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash};

#[cfg(feature = "approx_unique")]
use crate::series::ops::approx_algo::HyperLogLog;

fn approx_n_unique_ca<'a, T>(ca: &'a ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsDataType,
    T::Physical<'a>: TotalHash + TotalEq + Copy + ToTotalOrd,
    <Option<T::Physical<'a>> as ToTotalOrd>::TotalOrdItem: Hash + Eq,
{
    let mut hllp = HyperLogLog::new();
    ca.iter().for_each(|item| hllp.add(&item.to_total_ord()));
    let c = hllp.count() as IdxSize;

    Ok(Series::new(ca.name(), &[c]))
}

fn dispatcher(s: &Series) -> PolarsResult<Series> {
    let s = s.to_physical_repr();
    use DataType::*;
    match s.dtype() {
        Boolean => s.bool().and_then(approx_n_unique_ca),
        Binary => s.binary().and_then(approx_n_unique_ca),
        String => {
            let ca = s.str().unwrap().as_binary();
            approx_n_unique_ca(&ca)
        },
        Float32 => approx_n_unique_ca(AsRef::<ChunkedArray<Float32Type>>::as_ref(
            s.as_ref().as_ref(),
        )),
        Float64 => approx_n_unique_ca(AsRef::<ChunkedArray<Float64Type>>::as_ref(
            s.as_ref().as_ref(),
        )),
        dt if dt.is_numeric() => {
            with_match_physical_integer_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                approx_n_unique_ca(ca)
            })
        },
        dt => polars_bail!(opq = approx_n_unique, dt),
    }
}

/// Approx count unique values.
///
/// This is done using the HyperLogLog++ algorithm for cardinality estimation.
///
/// # Example
///
/// ```ignore
///
/// # #[macro_use] extern crate polars_core;
/// # fn main() {
///
///  use polars_core::prelude::*;
///
///  let s = Series::new("s", [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]);
///
///   let approx_count = approx_n_unique(&s).unwrap();
///   println!("{}", approx_count);
/// # }
/// ```
/// Outputs:
/// ```text
/// approx_count = shape: (1,)
/// Series: 's' [u32]
/// [
///     3
/// ]
/// ```
pub fn approx_n_unique(s: &Series) -> PolarsResult<Series> {
    dispatcher(s)
}
