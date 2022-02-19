use crate::prelude::*;
use crate::utils::{get_supertype, CustomIterTools};
use hashbrown::hash_set::HashSet;
use std::hash::Hash;

unsafe fn is_in_helper<T, P>(ca: &ChunkedArray<T>, other: &Series) -> Result<BooleanChunked>
where
    T: PolarsNumericType,
    P: Eq + Hash + Copy,
{
    let mut set = HashSet::with_capacity(other.len());

    let other = ca.unpack_series_matching_type(other)?;
    other.downcast_iter().for_each(|iter| {
        iter.into_iter().for_each(|opt_val| {
            // Safety
            // bit sizes are/ should be equal
            let ptr = &opt_val.copied() as *const Option<T::Native> as *const Option<P>;
            let opt_val = *ptr;
            set.insert(opt_val);
        })
    });

    let name = ca.name();
    let mut ca: BooleanChunked = ca
        .into_iter()
        .map(|opt_val| {
            // Safety
            // bit sizes are/ should be equal
            let ptr = &opt_val as *const Option<T::Native> as *const Option<P>;
            let opt_val = *ptr;
            set.contains(&opt_val)
        })
        .collect_trusted();
    ca.rename(name);
    Ok(ca)
}

impl<T> IsIn for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn is_in(&self, other: &Series) -> Result<BooleanChunked> {
        // We check implicitly cast to supertype here
        match other.dtype() {
            DataType::List(dt) => {
                let st = get_supertype(self.dtype(), dt)?;
                if &st != self.dtype() {
                    let left = self.cast(&st)?;
                    let right = other.cast(&DataType::List(Box::new(st)))?;
                    return left.is_in(&right);
                }

                let mut ca: BooleanChunked = if self.len() == 1 && other.len() != 1 {
                    let value = self.get(0);

                    other
                        .list()?
                        .amortized_iter()
                        .map(|opt_s| {
                            opt_s.map(|s| {
                                let ca = s.as_ref().unpack::<T>().unwrap();
                                ca.into_iter().any(|a| a == value)
                            }) == Some(true)
                        })
                        .collect_trusted()
                } else {
                    self.into_iter()
                        .zip(other.list()?.amortized_iter())
                        .map(|(value, series)| match (value, series) {
                            (val, Some(series)) => {
                                let ca = series.as_ref().unpack::<T>().unwrap();
                                ca.into_iter().any(|a| a == val)
                            }
                            _ => false,
                        })
                        .collect_trusted()
                };
                ca.rename(self.name());
                Ok(ca)
            }
            _ => {
                // first make sure that the types are equal
                let st = get_supertype(self.dtype(), other.dtype())?;
                if self.dtype() != other.dtype() {
                    let left = self.cast(&st)?;
                    let right = other.cast(&st)?;
                    return left.is_in(&right);
                }
                // now that the types are equal, we coerce every 32 bit array to u32
                // and every 64 bit array to u64 (including floats)
                // this allows hashing them and greatly reduces the number of code paths.
                match self.dtype() {
                    DataType::UInt64 | DataType::Int64 | DataType::Float64 => unsafe {
                        is_in_helper::<T, u64>(self, other)
                    },
                    DataType::UInt32 | DataType::Int32 | DataType::Float32 => unsafe {
                        is_in_helper::<T, u32>(self, other)
                    },
                    DataType::UInt8 | DataType::Int8 => unsafe {
                        is_in_helper::<T, u8>(self, other)
                    },
                    DataType::UInt16 | DataType::Int16 => unsafe {
                        is_in_helper::<T, u16>(self, other)
                    },
                    _ => Err(PolarsError::ComputeError(
                        format!(
                            "Data type {:?} not supported in is_in operation",
                            self.dtype()
                        )
                        .into(),
                    )),
                }
            }
        }
        .map(|mut ca| {
            ca.rename(self.name());
            ca
        })
    }
}
impl IsIn for Utf8Chunked {
    fn is_in(&self, other: &Series) -> Result<BooleanChunked> {
        match other.dtype() {
            DataType::List(dt) if self.dtype() == &**dt => {
                let mut ca: BooleanChunked = if self.len() == 1 && other.len() != 1 {
                    let value = self.get(0);
                    other
                        .list()?
                        .amortized_iter()
                        .map(|opt_s| {
                            opt_s.map(|s| {
                                let ca = s.as_ref().unpack::<Utf8Type>().unwrap();
                                ca.into_iter().any(|a| a == value)
                            }) == Some(true)
                        })
                        .collect_trusted()
                } else {
                    self.into_iter()
                        .zip(other.list()?.amortized_iter())
                        .map(|(value, series)| match (value, series) {
                            (val, Some(series)) => {
                                let ca = series.as_ref().unpack::<Utf8Type>().unwrap();
                                ca.into_iter().any(|a| a == val)
                            }
                            _ => false,
                        })
                        .collect_trusted()
                };
                ca.rename(self.name());
                Ok(ca)
            }
            DataType::Utf8 => {
                let mut set = HashSet::with_capacity(other.len());

                let other = other.utf8()?;
                other.downcast_iter().for_each(|iter| {
                    iter.into_iter().for_each(|opt_val| {
                        set.insert(opt_val);
                    })
                });
                let mut ca: BooleanChunked = self
                    .into_iter()
                    .map(|opt_val| set.contains(&opt_val))
                    .collect_trusted();
                ca.rename(self.name());
                Ok(ca)
            }
            _ => Err(PolarsError::SchemaMisMatch(
                format!(
                    "cannot do is_in operation with left a dtype: {:?} and right a dtype {:?}",
                    self.dtype(),
                    other.dtype()
                )
                .into(),
            )),
        }
        .map(|mut ca| {
            ca.rename(self.name());
            ca
        })
    }
}

impl IsIn for BooleanChunked {
    fn is_in(&self, other: &Series) -> Result<BooleanChunked> {
        match other.dtype() {
            DataType::List(dt) if self.dtype() == &**dt => {
                let mut ca: BooleanChunked = if self.len() == 1 && other.len() != 1 {
                    let value = self.get(0);
                    // safety: we know the iterators len
                    unsafe {
                        other
                            .list()?
                            .amortized_iter()
                            .map(|opt_s| {
                                opt_s.map(|s| {
                                    let ca = s.as_ref().unpack::<BooleanType>().unwrap();
                                    ca.into_iter().any(|a| a == value)
                                }) == Some(true)
                            })
                            .trust_my_length(other.len())
                            .collect_trusted()
                    }
                } else {
                    self.into_iter()
                        .zip(other.list()?.amortized_iter())
                        .map(|(value, series)| match (value, series) {
                            (val, Some(series)) => {
                                let ca = series.as_ref().unpack::<BooleanType>().unwrap();
                                ca.into_iter().any(|a| a == val)
                            }
                            _ => false,
                        })
                        .collect_trusted()
                };
                ca.rename(self.name());
                Ok(ca)
            }
            _ => Err(PolarsError::SchemaMisMatch(
                format!(
                    "cannot do is_in operation with left a dtype: {:?} and right a dtype {:?}",
                    self.dtype(),
                    other.dtype()
                )
                .into(),
            )),
        }
        .map(|mut ca| {
            ca.rename(self.name());
            ca
        })
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_is_in() -> Result<()> {
        let a = Int32Chunked::new("a", &[1, 2, 3, 4]);
        let b = Int64Chunked::new("b", &[4, 5, 1]);

        let out = a.is_in(&b.into_series())?;
        assert_eq!(
            Vec::from(&out),
            [Some(true), Some(false), Some(false), Some(true)]
        );

        let a = Utf8Chunked::new("a", &["a", "b", "c", "d"]);
        let b = Utf8Chunked::new("b", &["d", "e", "c"]);

        let out = a.is_in(&b.into_series())?;
        assert_eq!(
            Vec::from(&out),
            [Some(false), Some(false), Some(true), Some(true)]
        );
        Ok(())
    }
}
