//! Utilities for benchmarking

use rand::distributions::{Alphanumeric, Distribution, Standard};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{array::*, offset::Offset, types::NativeType};

/// Returns fixed seedable RNG
pub fn seedable_rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

/// Creates an random (but fixed-seeded) array of a given size and null density
pub fn create_primitive_array<T>(size: usize, null_density: f32) -> PrimitiveArray<T>
where
    T: NativeType,
    Standard: Distribution<T>,
{
    let mut rng = seedable_rng();

    (0..size)
        .map(|_| {
            if rng.gen::<f32>() < null_density {
                None
            } else {
                Some(rng.gen())
            }
        })
        .collect::<PrimitiveArray<T>>()
}

/// Creates a new [`PrimitiveArray`] from random values with a pre-set seed.
pub fn create_primitive_array_with_seed<T>(
    size: usize,
    null_density: f32,
    seed: u64,
) -> PrimitiveArray<T>
where
    T: NativeType,
    Standard: Distribution<T>,
{
    let mut rng = StdRng::seed_from_u64(seed);

    (0..size)
        .map(|_| {
            if rng.gen::<f32>() < null_density {
                None
            } else {
                Some(rng.gen())
            }
        })
        .collect::<PrimitiveArray<T>>()
}

/// Creates an random (but fixed-seeded) array of a given size and null density
pub fn create_boolean_array(size: usize, null_density: f32, true_density: f32) -> BooleanArray
where
    Standard: Distribution<bool>,
{
    let mut rng = seedable_rng();
    (0..size)
        .map(|_| {
            if rng.gen::<f32>() < null_density {
                None
            } else {
                let value = rng.gen::<f32>() < true_density;
                Some(value)
            }
        })
        .collect()
}

/// Creates an random (but fixed-seeded) [`Utf8Array`] of a given length, number of characters and null density.
pub fn create_string_array<O: Offset>(
    length: usize,
    size: usize,
    null_density: f32,
    seed: u64,
) -> Utf8Array<O> {
    let mut rng = StdRng::seed_from_u64(seed);

    (0..length)
        .map(|_| {
            if rng.gen::<f32>() < null_density {
                None
            } else {
                let value = (&mut rng)
                    .sample_iter(&Alphanumeric)
                    .take(size)
                    .map(char::from)
                    .collect::<String>();
                Some(value)
            }
        })
        .collect()
}
