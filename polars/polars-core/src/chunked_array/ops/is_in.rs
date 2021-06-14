use crate::prelude::*;
use crate::utils::get_supertype;
use hashbrown::hash_set::HashSet;
use num::NumCast;
use std::hash::Hash;

unsafe fn is_in_helper<T, P>(ca: &ChunkedArray<T>, other: &Series) -> Result<BooleanChunked>
where
    T: PolarsNumericType,
    T::Native: NumCast,
    P: Eq + Hash + Copy,
{
    let mut set = HashSet::with_capacity(other.len());

    let other = ca.unpack_series_matching_type(other)?;
    other.downcast_iter().for_each(|iter| {
        iter.into_iter().for_each(|opt_val| {
            // Safety
            // bit sizes are/ should be equal
            let ptr = &opt_val as *const Option<T::Native> as *const Option<P>;
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
        .collect();
    ca.rename(name);
    Ok(ca)
}

impl<T> IsIn for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast + Copy,
{
    fn is_in(&self, other: &Series) -> Result<BooleanChunked> {
        // We check implicitly cast to supertype here
        match other.dtype() {
            DataType::List(dt) => {
                let st = get_supertype(self.dtype(), &dt.into())?;
                if &st != self.dtype() {
                    let left = self.cast_with_dtype(&st)?;
                    let right = other.cast_with_dtype(&DataType::List(st.to_arrow()))?;
                    return left.is_in(&right);
                }

                let ca: BooleanChunked = self
                    .into_iter()
                    .zip(other.list()?.into_iter())
                    .map(|(value, series)| match (value, series) {
                        (val, Some(series)) => {
                            let ca = series.unpack::<T>().unwrap();
                            ca.into_iter().any(|a| a == val)
                        }
                        _ => false,
                    })
                    .collect();
                Ok(ca)
            }
            _ => {
                // first make sure that the types are equal
                let st = get_supertype(self.dtype(), other.dtype())?;
                if self.dtype() != other.dtype() {
                    let left = self.cast_with_dtype(&st)?;
                    let right = other.cast_with_dtype(&st)?;
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
                    _ => Err(PolarsError::Other(
                        format!(
                            "Data type {:?} not supported in is_in operation",
                            self.dtype()
                        )
                        .into(),
                    )),
                }
            }
        }
    }
}
impl IsIn for Utf8Chunked {
    fn is_in(&self, other: &Series) -> Result<BooleanChunked> {
        match other.dtype() {
            DataType::List(dt) if self.dtype() == dt => {
                let ca: BooleanChunked = self
                    .into_iter()
                    .zip(other.list()?.into_iter())
                    .map(|(value, series)| match (value, series) {
                        (val, Some(series)) => {
                            let ca = series.unpack::<Utf8Type>().unwrap();
                            ca.into_iter().any(|a| a == val)
                        }
                        _ => false,
                    })
                    .collect();
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
                    .collect();
                ca.rename(self.name());
                Ok(ca)
            }
            _ => Err(PolarsError::DataTypeMisMatch(
                format!(
                    "cannot do is_in operation with left a dtype: {:?} and right a dtype {:?}",
                    self.dtype(),
                    other.dtype()
                )
                .into(),
            )),
        }
    }
}

impl IsIn for BooleanChunked {
    fn is_in(&self, other: &Series) -> Result<BooleanChunked> {
        match other.dtype() {
            DataType::List(dt) if self.dtype() == dt => {
                let ca: BooleanChunked = self
                    .into_iter()
                    .zip(other.list()?.into_iter())
                    .map(|(value, series)| match (value, series) {
                        (val, Some(series)) => {
                            let ca = series.unpack::<BooleanType>().unwrap();
                            ca.into_iter().any(|a| a == val)
                        }
                        _ => false,
                    })
                    .collect();
                Ok(ca)
            }
            _ => Err(PolarsError::DataTypeMisMatch(
                format!(
                    "cannot do is_in operation with left a dtype: {:?} and right a dtype {:?}",
                    self.dtype(),
                    other.dtype()
                )
                .into(),
            )),
        }
    }
}

impl IsIn for CategoricalChunked {
    fn is_in(&self, other: &Series) -> Result<BooleanChunked> {
        self.cast::<UInt32Type>().unwrap().is_in(other)
    }
}

impl IsIn for ListChunked {}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_is_in() -> Result<()> {
        let a = Int32Chunked::new_from_slice("a", &[1, 2, 3, 4]);
        let b = Int64Chunked::new_from_slice("b", &[4, 5, 1]);

        let out = a.is_in(&b.into_series())?;
        assert_eq!(
            Vec::from(&out),
            [Some(true), Some(false), Some(false), Some(true)]
        );

        let a = Utf8Chunked::new_from_slice("a", &["a", "b", "c", "d"]);
        let b = Utf8Chunked::new_from_slice("b", &["d", "e", "c"]);

        let out = a.is_in(&b.into_series())?;
        assert_eq!(
            Vec::from(&out),
            [Some(false), Some(false), Some(true), Some(true)]
        );
        Ok(())
    }
}
