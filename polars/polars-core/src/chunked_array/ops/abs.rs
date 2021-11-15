use crate::prelude::*;
use num::Signed;

impl<T: PolarsNumericType> ChunkedArray<T>
where
    T::Native: Signed,
{
    /// Convert all values to their absolute/positive value.
    pub fn abs(&self) -> Self {
        self.apply(|v| v.abs())
    }
}
