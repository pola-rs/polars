use std::ops::{Add, AddAssign};

use num_traits::Float;

#[derive(Debug, Clone)]
pub struct KahanSum<T: Float> {
    sum: T,
    err: T,
}

impl<T: Float> KahanSum<T> {
    pub fn new(v: T) -> Self {
        KahanSum {
            sum: v,
            err: T::zero(),
        }
    }

    pub fn sum(&self) -> T {
        self.sum
    }
}

impl<T: Float> Default for KahanSum<T> {
    fn default() -> Self {
        KahanSum {
            sum: T::zero(),
            err: T::zero(),
        }
    }
}

impl<T: Float + AddAssign> AddAssign<T> for KahanSum<T> {
    fn add_assign(&mut self, rhs: T) {
        if rhs.is_finite() {
            let y = rhs - self.err;
            let new_sum = self.sum + y;
            self.err = (new_sum - self.sum) - y;
            self.sum = new_sum;
        } else {
            self.sum += rhs
        }
    }
}

impl<T: Float + AddAssign> Add<T> for KahanSum<T> {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        let mut rv = self;
        rv += rhs;
        rv
    }
}
