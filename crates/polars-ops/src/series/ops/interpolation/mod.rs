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

fn nearest_itp<T>(low: T, step: T, diff: T, steps_n: T) -> T
where
    T: Sub<Output = T> + Mul<Output = T> + Add<Output = T> + Div<Output = T> + PartialOrd + Copy,
{
    // 5 - 1 = 5 -> low
    // 5 - 2 = 3 -> low
    // 5 - 3 = 2 -> high
    if (steps_n - step) > step {
        low
    } else {
        low + diff
    }
}
