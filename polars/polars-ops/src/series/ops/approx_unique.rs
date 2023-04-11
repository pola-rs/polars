use std::hash::Hash;

use polars_core::prelude::*;
use polars_core::with_match_physical_integer_polars_type;

#[cfg(feature = "approx_unique")]
use crate::series::HyperLogLog;

fn approx_unique_ca<'a, T>(ca: &'a ChunkedArray<T>, precision: u8) -> PolarsResult<Series>
where
    T: PolarsDataType,
    &'a ChunkedArray<T>: IntoIterator,
    <<&'a ChunkedArray<T> as IntoIterator>::IntoIter as IntoIterator>::Item: Hash + Eq,
{
    let mut hllp = HyperLogLog::new();
    ca.into_iter().for_each(|item| hllp.add(&item));
    let c = hllp.count() as IdxSize;

    Ok(Series::new(ca.name(), &[c]))
}

fn dispatcher(s: &Series, precision: u8) -> PolarsResult<Series> {
    let s = s.to_physical_repr();
    use DataType::*;
    match s.dtype() {
        Boolean => s.bool().and_then(|ca| approx_unique_ca(ca, precision)),
        Binary => s.binary().and_then(|ca| approx_unique_ca(ca, precision)),
        Utf8 => {
            let s = s.cast(&Binary).unwrap();
            let ca = s.binary().unwrap();
            approx_unique_ca(ca, precision)
        }
        Float32 => approx_unique_ca(&s.bit_repr_small(), precision),
        Float64 => approx_unique_ca(&s.bit_repr_large(), precision),
        dt if dt.is_numeric() => {
            with_match_physical_integer_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                approx_unique_ca(ca, precision)
            })
        }
        dt => polars_bail!(opq = approx_unique, dt),
    }
}

pub fn approx_unique(s: &Series, precision: u8) -> PolarsResult<Series> {
    dispatcher(s, precision)
}
