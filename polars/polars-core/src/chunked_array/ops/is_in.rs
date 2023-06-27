use std::hash::Hash;

use crate::prelude::*;
use crate::utils::{try_get_supertype, CustomIterTools};

unsafe fn is_in_helper<T, P>(ca: &ChunkedArray<T>, other: &Series) -> PolarsResult<BooleanChunked>
where
    T: PolarsNumericType,
    P: Eq + Hash + Copy,
{
    let mut set = PlHashSet::with_capacity(other.len());

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
    fn is_in(&self, other: &Series) -> PolarsResult<BooleanChunked> {
        // We check implicitly cast to supertype here
        match other.dtype() {
            DataType::List(dt) => {
                let st = try_get_supertype(self.dtype(), dt)?;
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
                if self.dtype() != other.dtype() {
                    let st = try_get_supertype(self.dtype(), other.dtype())?;
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
                    dt => polars_bail!(opq = is_in, dt),
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
    fn is_in(&self, other: &Series) -> PolarsResult<BooleanChunked> {
        match other.dtype() {
            #[cfg(feature = "dtype-categorical")]
            DataType::List(dt) if matches!(&**dt, DataType::Categorical(_)) => {
                if let DataType::Categorical(Some(rev_map)) = &**dt {
                    let opt_val = self.get(0);

                    let other = other.list()?;
                    match opt_val {
                        None => {
                            let mut ca: BooleanChunked = other
                                .amortized_iter()
                                .map(|opt_s| {
                                    opt_s.map(|s| s.as_ref().null_count() > 0) == Some(true)
                                })
                                .collect_trusted();
                            ca.rename(self.name());
                            Ok(ca)
                        }
                        Some(value) => {
                            match rev_map.find(value) {
                                // all false
                                None => Ok(BooleanChunked::full(self.name(), false, other.len())),
                                Some(idx) => {
                                    let mut ca: BooleanChunked = other
                                        .amortized_iter()
                                        .map(|opt_s| {
                                            opt_s.map(|s| {
                                                let s = s.as_ref().to_physical_repr();
                                                let ca = s.as_ref().u32().unwrap();
                                                if ca.null_count() == 0 {
                                                    ca.into_no_null_iter().any(|a| a == idx)
                                                } else {
                                                    ca.into_iter().any(|a| a == Some(idx))
                                                }
                                            }) == Some(true)
                                        })
                                        .collect_trusted();
                                    ca.rename(self.name());
                                    Ok(ca)
                                }
                            }
                        }
                    }
                } else {
                    unreachable!()
                }
            }
            DataType::List(dt) if DataType::Utf8 == **dt => self.as_binary().is_in(
                &other
                    .cast(&DataType::List(Box::new(DataType::Binary)))
                    .unwrap(),
            ),
            DataType::Utf8 => self
                .as_binary()
                .is_in(&other.cast(&DataType::Binary).unwrap()),
            _ => polars_bail!(opq = is_in, self.dtype(), other.dtype()),
        }
    }
}

impl IsIn for BinaryChunked {
    fn is_in(&self, other: &Series) -> PolarsResult<BooleanChunked> {
        match other.dtype() {
            DataType::List(dt) if DataType::Binary == **dt => {
                let mut ca: BooleanChunked = if self.len() == 1 && other.len() != 1 {
                    let value = self.get(0);
                    other
                        .list()?
                        .amortized_iter()
                        .map(|opt_b| {
                            opt_b.map(|s| {
                                let ca = s.as_ref().unpack::<BinaryType>().unwrap();
                                ca.into_iter().any(|a| a == value)
                            }) == Some(true)
                        })
                        .collect_trusted()
                } else {
                    self.into_iter()
                        .zip(other.list()?.amortized_iter())
                        .map(|(value, series)| match (value, series) {
                            (val, Some(series)) => {
                                let ca = series.as_ref().unpack::<BinaryType>().unwrap();
                                ca.into_iter().any(|a| a == val)
                            }
                            _ => false,
                        })
                        .collect_trusted()
                };
                ca.rename(self.name());
                Ok(ca)
            }
            DataType::Binary => {
                let mut set = PlHashSet::with_capacity(other.len());

                let other = other.binary()?;
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
            _ => polars_bail!(opq = is_in, self.dtype(), other.dtype()),
        }
        .map(|mut ca| {
            ca.rename(self.name());
            ca
        })
    }
}

impl IsIn for BooleanChunked {
    fn is_in(&self, other: &Series) -> PolarsResult<BooleanChunked> {
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
            DataType::Boolean => {
                let other = other.bool().unwrap();
                let has_true = other.any();
                let has_false = !other.all();
                Ok(self.apply(|v| if v { has_true } else { has_false }))
            }
            _ => polars_bail!(opq = is_in, self.dtype(), other.dtype()),
        }
        .map(|mut ca| {
            ca.rename(self.name());
            ca
        })
    }
}

#[cfg(feature = "dtype-struct")]
impl IsIn for StructChunked {
    fn is_in(&self, other: &Series) -> PolarsResult<BooleanChunked> {
        match other.dtype() {
            DataType::List(_) => {
                let mut ca: BooleanChunked = if self.len() == 1 && other.len() != 1 {
                    let mut value = vec![];
                    let left = self.clone().into_series();
                    let av = left.get(0).unwrap();
                    if let AnyValue::Struct(_, _, _) = av {
                        av._materialize_struct_av(&mut value);
                    }
                    other
                        .list()?
                        .amortized_iter()
                        .map(|opt_s| {
                            opt_s.map(|s| {
                                let ca = s.as_ref().struct_().unwrap();
                                ca.into_iter().any(|a| a == value)
                            }) == Some(true)
                        })
                        .collect()
                } else {
                    self.into_iter()
                        .zip(other.list()?.amortized_iter())
                        .map(|(value, series)| match (value, series) {
                            (val, Some(series)) => {
                                let ca = series.as_ref().struct_().unwrap();
                                ca.into_iter().any(|a| a == val)
                            }
                            _ => false,
                        })
                        .collect()
                };
                ca.rename(self.name());
                Ok(ca)
            }
            _ => {
                let other = other.cast(&other.dtype().to_physical()).unwrap();
                let other = other.struct_()?;

                polars_ensure!(
                    self.fields().len() == other.fields().len(),
                    ComputeError: "`is_in`: mismatch in the number of struct fields: {} and {}",
                    self.fields().len(), other.fields().len()
                );

                // first make sure that the types are equal
                let self_dtypes: Vec<_> = self.fields().iter().map(|f| f.dtype()).collect();
                let other_dtypes: Vec<_> = other.fields().iter().map(|f| f.dtype()).collect();
                if self_dtypes != other_dtypes {
                    let self_names = self.fields().iter().map(|f| f.name());
                    let other_names = other.fields().iter().map(|f| f.name());
                    let supertypes = self_dtypes
                        .iter()
                        .zip(other_dtypes.iter())
                        .map(|(dt1, dt2)| try_get_supertype(dt1, dt2))
                        .collect::<Result<Vec<_>, _>>()?;
                    let self_supertype_fields = self_names
                        .zip(supertypes.iter())
                        .map(|(name, st)| Field::new(name, st.clone()))
                        .collect();
                    let self_super = self.cast(&DataType::Struct(self_supertype_fields))?;
                    let other_supertype_fields = other_names
                        .zip(supertypes.iter())
                        .map(|(name, st)| Field::new(name, st.clone()))
                        .collect();
                    let other_super = other.cast(&DataType::Struct(other_supertype_fields))?;
                    return self_super.is_in(&other_super);
                }

                let mut anyvalues = Vec::with_capacity(other.len() * other.fields().len());
                // Safety:
                // the iterator is unsafe as the lifetime is tied to the iterator
                // so we copy to an owned buffer first
                other.into_iter().for_each(|val| {
                    anyvalues.extend_from_slice(val);
                });

                // then we fill the set
                let mut set = PlHashSet::with_capacity(other.len());
                for key in anyvalues.chunks_exact(other.fields().len()) {
                    set.insert(key);
                }
                // physical self
                let self_ca = self.cast(&self.dtype().to_physical()).unwrap();
                let self_ca = self_ca.struct_().unwrap();

                // and then we check for membership
                let mut ca: BooleanChunked = self_ca
                    .into_iter()
                    .map(|vals| set.contains(&vals))
                    .collect();
                ca.rename(self.name());
                Ok(ca)
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_is_in() -> PolarsResult<()> {
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
