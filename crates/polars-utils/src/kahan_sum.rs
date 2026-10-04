use std::ops::{Add, AddAssign};

use num_traits::Num;

use crate::float::IsFloat;

#[derive(Debug, Clone)]
pub struct KahanSum<T> {
    sum: T,
    err: T,
}

impl<T: IsFloat + Num + Copy> KahanSum<T> {
    #[inline]
    pub fn new(v: T) -> Self {
        KahanSum {
            sum: v,
            err: T::zero(),
        }
    }

    #[inline(always)]
    pub fn sum(&self) -> T {
        self.sum
    }
}

impl<T: Num> Default for KahanSum<T> {
    #[inline]
    fn default() -> Self {
        KahanSum {
            sum: T::zero(),
            err: T::zero(),
        }
    }
}

impl<T: IsFloat + Num + AddAssign + Copy> AddAssign<T> for KahanSum<T> {
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        let y = rhs - self.err;
        let new_sum = self.sum + y;
        let new_err = (new_sum - self.sum) - y;
        self.sum = new_sum;
        if new_err.is_finite() {
            // Ensure err stays finite so we don't introduce NaNs through Inf - Inf.
            self.err = new_err;
        }
    }
}

impl<T: IsFloat + Num + AddAssign + Copy> Add<T> for KahanSum<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        let mut rv = self;
        rv += rhs;
        rv
    }
}
