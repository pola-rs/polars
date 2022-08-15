use num::Signed;

use crate::prelude::*;

impl<T: PolarsNumericType> ChunkedArray<T>
where
    T::Native: Signed,
{
    /// Convert all values to their absolute/positive value.
    #[must_use]
    pub fn abs(&self) -> Self {
        self.apply(|v| v.abs())
    }
}
