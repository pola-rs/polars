use std::hash::Hash;

use arrow::array::PrimitiveArray;
use num::Float;
use polars::prelude::*;
use pyo3_polars::export::polars_core::utils::arrow::types::NativeType;
use pyo3_polars::export::polars_core::with_match_physical_integer_type;

#[allow(clippy::all)]
pub(super) fn naive_hamming_dist(a: &str, b: &str) -> u32 {
    let x = a.as_bytes();
    let y = b.as_bytes();
    x.iter()
        .zip(y)
        .fold(0, |a, (b, c)| a + (*b ^ *c).count_ones() as u32)
}

fn jacc_helper<T: NativeType + Hash + Eq>(a: &PrimitiveArray<T>, b: &PrimitiveArray<T>) -> f64 {
    // convert to hashsets over Option<T>
    let s1 = a.into_iter().collect::<PlHashSet<_>>();
    let s2 = b.into_iter().collect::<PlHashSet<_>>();

    // count the number of intersections
    let s3_len = s1.intersection(&s2).count();
    // return similarity
    s3_len as f64 / (s1.len() + s2.len() - s3_len) as f64
}

#[allow(unexpected_cfgs)]
pub(super) fn naive_jaccard_sim(a: &ListChunked, b: &ListChunked) -> PolarsResult<Float64Chunked> {
    polars_ensure!(
        a.inner_dtype() == b.inner_dtype(),
        ComputeError: "inner data types don't match"
    );
    polars_ensure!(
        a.inner_dtype().is_integer(),
        ComputeError: "inner data types must be integer"
    );
    Ok(with_match_physical_integer_type!(a.inner_dtype(), |$T| {
    polars::prelude::arity::binary_elementwise(a, b, |a, b| {
        match (a, b) {
            (Some(a), Some(b)) => {
                let a = a.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                let b = b.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
                Some(jacc_helper(a, b))
            },
            _ => None
        }
    })
    }))
}

fn haversine_elementwise<T: Float>(start_lat: T, start_long: T, end_lat: T, end_long: T) -> T {
    let r_in_km = T::from(6371.0).unwrap();
    let two = T::from(2.0).unwrap();
    let one = T::one();

    let d_lat = (end_lat - start_lat).to_radians();
    let d_lon = (end_long - start_long).to_radians();
    let lat1 = (start_lat).to_radians();
    let lat2 = (end_lat).to_radians();

    let a = ((d_lat / two).sin()) * ((d_lat / two).sin())
        + ((d_lon / two).sin()) * ((d_lon / two).sin()) * (lat1.cos()) * (lat2.cos());
    let c = two * ((a.sqrt()).atan2((one - a).sqrt()));
    r_in_km * c
}

pub(super) fn naive_haversine<T>(
    start_lat: &ChunkedArray<T>,
    start_long: &ChunkedArray<T>,
    end_lat: &ChunkedArray<T>,
    end_long: &ChunkedArray<T>,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Float,
{
    let out: ChunkedArray<T> = start_lat
        .iter()
        .zip(start_long.iter())
        .zip(end_lat.iter())
        .zip(end_long.iter())
        .map(|(((start_lat, start_long), end_lat), end_long)| {
            let start_lat = start_lat?;
            let start_long = start_long?;
            let end_lat = end_lat?;
            let end_long = end_long?;
            Some(haversine_elementwise(
                start_lat, start_long, end_lat, end_long,
            ))
        })
        .collect();

    Ok(out.with_name(start_lat.name().clone()))
}
