use std::ops::{Add, Div, Mul, Sub};

#[cfg(feature = "interpolate")]
pub mod interpolate;
#[cfg(feature = "interpolate_by")]
pub mod interpolate_by;

fn linear_itp<T>(low: T, step: T, slope: T) -> T
where
    T: Sub<Output = T> + Mul<Output = T> + Add<Output = T> + Div<Output = T>,
{
    low + step * slope
}
