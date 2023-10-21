use std::hash::Hash;

use polars_core::prelude::*;
use polars_core::utils::{try_get_supertype, CustomIterTools};
use polars_core::with_match_physical_integer_polars_type;

fn is_in_helper<'a, T>(ca: &'a ChunkedArray<T>, other: &Series) -> PolarsResult<BooleanChunked>
where
    T: PolarsDataType,
    T::Physical<'a>: Hash + Eq + Copy,
{
    let mut set = PlHashSet::with_capacity(other.len());

    let other = ca.unpack_series_matching_type(other)?;
    other.downcast_iter().for_each(|iter| {
        iter.iter().for_each(|opt_val| {
            if let Some(v) = opt_val {
                set.insert(v);
            }
        })
    });
    Ok(ca.apply_values_generic(|val| set.contains(&val)))
}

fn is_in_numeric<T>(ca_in: &ChunkedArray<T>, other: &Series) -> PolarsResult<BooleanChunked>
where
    T: PolarsIntegerType,
    T::Native: Hash + Eq,
{
    // We check implicitly cast to supertype here
    match other.dtype() {
        DataType::List(dt) => {
            let st = try_get_supertype(ca_in.dtype(), dt)?;
            if &st != ca_in.dtype() || **dt != st {
                let left = ca_in.cast(&st)?;
                let right = other.cast(&DataType::List(Box::new(st)))?;
                return is_in(&left, &right);
            }

            let mut ca: BooleanChunked = if ca_in.len() == 1 && other.len() != 1 {
                let value = ca_in.get(0);

                other.list()?.apply_amortized_generic(|opt_s| {
                    Some(opt_s.map(|s| {
                        let ca = s.as_ref().unpack::<T>().unwrap();
                        ca.into_iter().any(|a| a == value)
                    }) == Some(true))
                })
            } else {
                polars_ensure!(ca_in.len() == other.len(), ComputeError: "shapes don't match: expected {} elements in 'is_in' comparison, got {}", ca_in.len(), other.len());
                // SAFETY: unstable series never lives longer than the iterator.
                unsafe { ca_in.into_iter()
                    .zip(other.list()?.amortized_iter())
                    .map(|(value, series)| match (value, series) {
                        (val, Some(series)) => {
                            let ca = series.as_ref().unpack::<T>().unwrap();
                            ca.into_iter().any(|a| a == val)
                        }
                        _ => false,
                    })
                    .collect_trusted()}
            };
            ca.rename(ca_in.name());
            Ok(ca)
        }
        _ => {
            // first make sure that the types are equal
            if ca_in.dtype() != other.dtype() {
                let st = try_get_supertype(ca_in.dtype(), other.dtype())?;
                let left = ca_in.cast(&st)?;
                let right = other.cast(&st)?;
                return is_in(&left, &right);
            }
            is_in_helper(ca_in, other)
        }
    }
        .map(|mut ca| {
            ca.rename(ca_in.name());
            ca
        })
}

fn is_in_utf8(ca_in: &Utf8Chunked, other: &Series) -> PolarsResult<BooleanChunked> {
    match other.dtype() {
        #[cfg(feature = "dtype-categorical")]
        DataType::List(dt) if matches!(&**dt, DataType::Categorical(_)) => {
            if let DataType::Categorical(Some(rev_map)) = &**dt {
                let opt_val = ca_in.get(0);

                let other = other.list()?;
                match opt_val {
                    None => Ok(other
                        .apply_amortized_generic(|opt_s| {
                            opt_s.map(|s| Some(s.as_ref().null_count() > 0) == Some(true))
                        })
                        .with_name(ca_in.name())),
                    Some(value) => {
                        match rev_map.find(value) {
                            // all false
                            None => Ok(BooleanChunked::full(ca_in.name(), false, other.len())),
                            Some(idx) => Ok(other
                                .apply_amortized_generic(|opt_s| {
                                    Some(
                                        opt_s.map(|s| {
                                            let s = s.as_ref().to_physical_repr();
                                            let ca = s.as_ref().u32().unwrap();
                                            if ca.null_count() == 0 {
                                                ca.into_no_null_iter().any(|a| a == idx)
                                            } else {
                                                ca.into_iter().any(|a| a == Some(idx))
                                            }
                                        }) == Some(true),
                                    )
                                })
                                .with_name(ca_in.name())),
                        }
                    },
                }
            } else {
                unreachable!()
            }
        },
        DataType::List(dt) if DataType::Utf8 == **dt => is_in_binary(
            &ca_in.as_binary(),
            &other
                .cast(&DataType::List(Box::new(DataType::Binary)))
                .unwrap(),
        ),
        DataType::Utf8 => is_in_binary(&ca_in.as_binary(), &other.cast(&DataType::Binary).unwrap()),
        _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
    }
}

fn is_in_binary(ca_in: &BinaryChunked, other: &Series) -> PolarsResult<BooleanChunked> {
    match other.dtype() {
            DataType::List(dt) if DataType::Binary == **dt => {
                let mut ca: BooleanChunked = if ca_in.len() == 1 && other.len() != 1 {
                    let value = ca_in.get(0);

                    other.list()?.apply_amortized_generic(|opt_b| {
                        Some(opt_b.map(|s| {
                            let ca = s.as_ref().unpack::<BinaryType>().unwrap();
                            ca.into_iter().any(|a| a == value)
                        }) == Some(true))
                    })
                } else {
                    polars_ensure!(ca_in.len() == other.len(), ComputeError: "shapes don't match: expected {} elements in 'is_in' comparison, got {}", ca_in.len(), other.len());
                    // SAFETY: unstable series never lives longer than the iterator.
                    unsafe { ca_in.into_iter()
                        .zip(other.list()?.amortized_iter())
                        .map(|(value, series)| match (value, series) {
                            (val, Some(series)) => {
                                let ca = series.as_ref().unpack::<BinaryType>().unwrap();
                                ca.into_iter().any(|a| a == val)
                            }
                            _ => false,
                        })
                        .collect_trusted()}
                };
                ca.rename(ca_in.name());
                Ok(ca)
            }
            DataType::Binary => {
                is_in_helper(ca_in, other)
            }
            _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
        }
            .map(|mut ca| {
                ca.rename(ca_in.name());
                ca
            })
}

fn is_in_boolean(ca_in: &BooleanChunked, other: &Series) -> PolarsResult<BooleanChunked> {
    match other.dtype() {
            DataType::List(dt) if ca_in.dtype() == &**dt => {
                let mut ca: BooleanChunked = if ca_in.len() == 1 && other.len() != 1 {
                    let value = ca_in.get(0);
                    // SAFETY: we know the iterators len
                    // SAFETY: unstable series never lives longer than the iterator.
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
                    polars_ensure!(ca_in.len() == other.len(), ComputeError: "shapes don't match: expected {} elements in 'is_in' comparison, got {}", ca_in.len(), other.len());
                    // SAFETY: unstable series never lives longer than the iterator.
                    unsafe { ca_in.into_iter()
                        .zip(other.list()?.amortized_iter())
                        .map(|(value, series)| match (value, series) {
                            (val, Some(series)) => {
                                let ca = series.as_ref().unpack::<BooleanType>().unwrap();
                                ca.into_iter().any(|a| a == val)
                            }
                            _ => false,
                        })
                        .collect_trusted()
                }};
                ca.rename(ca_in.name());
                Ok(ca)
            }
            DataType::Boolean => {
                let other = other.bool().unwrap();
                let has_true = other.any();
                let nc = other.null_count();

                let has_false = if nc == 0 {
                    !other.all()
                } else {
                    (other.sum().unwrap() as usize + nc) != other.len()
                };
                Ok(ca_in.apply_values(|v| if v { has_true } else { has_false }))
            }
            _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
        }
            .map(|mut ca| {
                ca.rename(ca_in.name());
                ca
            })
}

#[cfg(feature = "dtype-struct")]
fn is_in_struct(ca_in: &StructChunked, other: &Series) -> PolarsResult<BooleanChunked> {
    match other.dtype() {
        DataType::List(_) => {
            let mut ca: BooleanChunked = if ca_in.len() == 1 && other.len() != 1 {
                let mut value = vec![];
                let left = ca_in.clone().into_series();
                let av = left.get(0).unwrap();
                if let AnyValue::Struct(_, _, _) = av {
                    av._materialize_struct_av(&mut value);
                }
                // SAFETY: unstable series never lives longer than the iterator.
                other.list()?.apply_amortized_generic(|opt_s| {
                    Some(
                        opt_s.map(|s| {
                            let ca = s.as_ref().struct_().unwrap();
                            ca.into_iter().any(|a| a == value)
                        }) == Some(true),
                    )
                })
            } else {
                polars_ensure!(ca_in.len() == other.len(), ComputeError: "shapes don't match: expected {} elements in 'is_in' comparison, got {}", ca_in.len(), other.len());
                unsafe {
                    ca_in
                        .into_iter()
                        .zip(other.list()?.amortized_iter())
                        .map(|(value, series)| match (value, series) {
                            (val, Some(series)) => {
                                let ca = series.as_ref().struct_().unwrap();
                                ca.into_iter().any(|a| a == val)
                            },
                            _ => false,
                        })
                        .collect()
                }
            };
            ca.rename(ca_in.name());
            Ok(ca)
        },
        _ => {
            let other = other.cast(&other.dtype().to_physical()).unwrap();
            let other = other.struct_()?;

            polars_ensure!(
                ca_in.fields().len() == other.fields().len(),
                ComputeError: "`is_in`: mismatch in the number of struct fields: {} and {}",
                ca_in.fields().len(), other.fields().len()
            );

            // first make sure that the types are equal
            let ca_in_dtypes: Vec<_> = ca_in.fields().iter().map(|f| f.dtype()).collect();
            let other_dtypes: Vec<_> = other.fields().iter().map(|f| f.dtype()).collect();
            if ca_in_dtypes != other_dtypes {
                let ca_in_names = ca_in.fields().iter().map(|f| f.name());
                let other_names = other.fields().iter().map(|f| f.name());
                let supertypes = ca_in_dtypes
                    .iter()
                    .zip(other_dtypes.iter())
                    .map(|(dt1, dt2)| try_get_supertype(dt1, dt2))
                    .collect::<Result<Vec<_>, _>>()?;
                let ca_in_supertype_fields = ca_in_names
                    .zip(supertypes.iter())
                    .map(|(name, st)| Field::new(name, st.clone()))
                    .collect();
                let ca_in_super = ca_in.cast(&DataType::Struct(ca_in_supertype_fields))?;
                let other_supertype_fields = other_names
                    .zip(supertypes.iter())
                    .map(|(name, st)| Field::new(name, st.clone()))
                    .collect();
                let other_super = other.cast(&DataType::Struct(other_supertype_fields))?;
                return is_in(&ca_in_super, &other_super);
            }

            let mut anyvalues = Vec::with_capacity(other.len() * other.fields().len());
            // SAFETY:
            // the iterator is unsafe as the lifetime is tied to the iterator
            // so we copy to an owned buffer first
            other.into_iter().for_each(|vals| {
                anyvalues.extend_from_slice(vals);
            });

            // then we fill the set
            let mut set = PlHashSet::with_capacity(other.len());
            for key in anyvalues.chunks_exact(other.fields().len()) {
                set.insert(key);
            }
            // physical ca_in
            let ca_in_ca = ca_in.cast(&ca_in.dtype().to_physical()).unwrap();
            let ca_in_ca = ca_in_ca.struct_().unwrap();

            // and then we check for membership
            let mut ca: BooleanChunked = ca_in_ca
                .into_iter()
                .map(|vals| {
                    // If all rows are null we see the struct row as missing.
                    if !vals.iter().all(|val| matches!(val, AnyValue::Null)) {
                        Some(set.contains(&vals))
                    } else {
                        None
                    }
                })
                .collect();
            ca.rename(ca_in.name());
            Ok(ca)
        },
    }
}

pub fn is_in(s: &Series, other: &Series) -> PolarsResult<BooleanChunked> {
    match s.dtype() {
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(_) => {
            use crate::frame::join::_check_categorical_src;
            _check_categorical_src(s.dtype(), other.dtype())?;
            let ca = s.categorical().unwrap();
            let ca = ca.physical();
            is_in_numeric(ca, &other.to_physical_repr())
        },
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(_) => {
            let ca = s.struct_().unwrap();
            is_in_struct(ca, other)
        },
        DataType::Utf8 => {
            let ca = s.utf8().unwrap();
            is_in_utf8(ca, other)
        },
        DataType::Binary => {
            let ca = s.binary().unwrap();
            is_in_binary(ca, other)
        },
        DataType::Boolean => {
            let ca = s.bool().unwrap();
            is_in_boolean(ca, other)
        },
        DataType::Float32 => {
            let other = match other.dtype() {
                DataType::List(_) => {
                    let other = other.cast(&DataType::List(Box::new(DataType::Float32)))?;
                    let other = other.list().unwrap();
                    other.reinterpret_unsigned()
                },
                _ => {
                    let other = other.cast(&DataType::Float32)?;
                    let other = other.f32().unwrap();
                    other.reinterpret_unsigned()
                },
            };
            let ca = s.f32().unwrap();
            let s = ca.reinterpret_unsigned();
            is_in(&s, &other)
        },
        DataType::Float64 => {
            let other = match other.dtype() {
                DataType::List(_) => {
                    let other = other.cast(&DataType::List(Box::new(DataType::Float64)))?;
                    let other = other.list().unwrap();
                    other.reinterpret_unsigned()
                },
                _ => {
                    let other = other.cast(&DataType::Float64)?;
                    let other = other.f64().unwrap();
                    other.reinterpret_unsigned()
                },
            };
            let ca = s.f64().unwrap();
            let s = ca.reinterpret_unsigned();
            is_in(&s, &other)
        },
        dt if dt.to_physical().is_integer() => {
            let s = s.to_physical_repr();
            with_match_physical_integer_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                is_in_numeric(ca, other)
            })
        },
        DataType::Null => {
            let series_bool = s.cast(&DataType::Boolean)?;
            let ca = series_bool.bool().unwrap();
            Ok(ca.clone())
        },
        dt => polars_bail!(opq = is_in, dt),
    }
}
