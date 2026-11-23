use either::Either;
use polars_compute::filter::filter_values;

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

    /// Convert to a [`Vec`] of the non-null values.
    pub fn to_vec_non_null(&self) -> Vec<T::Native> {
        let filtered = |arr: &PrimitiveArray<T::Native>| match arr.validity() {
            Some(validity) if validity.unset_bits() > 0 => {
                Some(filter_values(arr.values(), validity))
            },
            _ => None,
        };
        if self.chunks().len() == 1 {
            let arr = self.downcast_iter().next().unwrap();
            return filtered(arr).unwrap_or_else(|| arr.values().as_slice().to_vec());
        }
        let mut vals = Vec::with_capacity(self.len() - self.null_count());
        for arr in self.downcast_iter() {
            match filtered(arr) {
                Some(nonnull) => vals.extend_from_slice(&nonnull),
                None => vals.extend_from_slice(arr.values()),
            }
        }
        vals
    }
}
