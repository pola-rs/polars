use either::Either;

use crate::prelude::*;

impl<T: PolarsNumericType> ChunkedArray<T> {
    /// Convert to a [`Vec`] of [`Option<T::Native>`].
    pub fn to_vec(&self) -> Vec<Option<T::Native>> {
        let mut buf = Vec::with_capacity(self.len());
        for arr in self.downcast_iter() {
            buf.extend(arr.into_iter().map(|v| v.copied()))
        }
        buf
    }

    /// Convert to a [`Vec`] but don't return [`Option<T::Native>`] if there are no null values
    pub fn to_vec_null_aware(&self) -> Either<Vec<T::Native>, Vec<Option<T::Native>>> {
        if self.null_count() == 0 {
            let mut buf = Vec::with_capacity(self.len());

            for arr in self.downcast_iter() {
                buf.extend_from_slice(arr.values())
            }
            Either::Left(buf)
        } else {
            Either::Right(self.to_vec())
        }
    }
}
