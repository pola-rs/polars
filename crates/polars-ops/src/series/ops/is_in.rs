use std::hash::Hash;

use polars_core::prelude::arity::unary_elementwise_values;
use polars_core::prelude::*;
use polars_core::utils::{try_get_supertype, CustomIterTools};
use polars_core::with_match_physical_numeric_polars_type;
#[cfg(feature = "dtype-categorical")]
use polars_utils::itertools::Itertools;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash};

fn is_in_helper_ca<'a, T>(
    ca: &'a ChunkedArray<T>,
    other: &'a ChunkedArray<T>,
) -> PolarsResult<BooleanChunked>
where
    T: PolarsDataType,
    T::Physical<'a>: TotalHash + TotalEq + ToTotalOrd + Copy,
    <T::Physical<'a> as ToTotalOrd>::TotalOrdItem: Hash + Eq + Copy,
{
    let mut set = PlHashSet::with_capacity(other.len());
    other.downcast_iter().for_each(|iter| {
        iter.iter().for_each(|opt_val| {
            if let Some(v) = opt_val {
                set.insert(v.to_total_ord());
            }
        })
    });
    Ok(unary_elementwise_values(ca, |val| set.contains(&val.to_total_ord())).with_name(ca.name()))
}

fn is_in_helper<'a, T>(ca: &'a ChunkedArray<T>, other: &Series) -> PolarsResult<BooleanChunked>
where
    T: PolarsDataType,
    T::Physical<'a>: TotalHash + TotalEq + Copy + ToTotalOrd,
    <T::Physical<'a> as ToTotalOrd>::TotalOrdItem: Hash + Eq + Copy,
{
    let other = ca.unpack_series_matching_type(other)?;
    is_in_helper_ca(ca, other)
}

fn is_in_numeric_list<T>(ca_in: &ChunkedArray<T>, other: &Series) -> PolarsResult<BooleanChunked>
where
    T: PolarsNumericType,
    T::Native: TotalHash + TotalEq,
{
    let mut ca: BooleanChunked = if ca_in.len() == 1 && other.len() != 1 {
        let value = ca_in.get(0);

        other.list()?.apply_amortized_generic(|opt_s| {
            Some(
                opt_s.map(|s| {
                    let ca = s.as_ref().unpack::<T>().unwrap();
                    ca.iter().any(|a| a == value)
                }) == Some(true),
            )
        })
    } else {
        polars_ensure!(ca_in.len() == other.len(), ComputeError: "shapes don't match: expected {} elements in 'is_in' comparison, got {}", ca_in.len(), other.len());
        {
            ca_in
                .iter()
                .zip(other.list()?.amortized_iter())
                .map(|(value, series)| match (value, series) {
                    (val, Some(series)) => {
                        let ca = series.as_ref().unpack::<T>().unwrap();
                        ca.iter().any(|a| a == val)
                    },
                    _ => false,
                })
                .collect_trusted()
        }
    };
    ca.rename(ca_in.name());
    Ok(ca)
}

#[cfg(feature = "dtype-array")]
fn is_in_numeric_array<T>(ca_in: &ChunkedArray<T>, other: &Series) -> PolarsResult<BooleanChunked>
where
    T: PolarsNumericType,
    T::Native: TotalHash + TotalEq,
{
    let mut ca: BooleanChunked = if ca_in.len() == 1 && other.len() != 1 {
        let value = ca_in.get(0);

        other.array()?.apply_amortized_generic(|opt_s| {
            Some(
                opt_s.map(|s| {
                    let ca = s.as_ref().unpack::<T>().unwrap();
                    ca.iter().any(|a| a == value)
                }) == Some(true),
            )
        })
    } else {
        polars_ensure!(ca_in.len() == other.len(), ComputeError: "shapes don't match: expected {} elements in 'is_in' comparison, got {}", ca_in.len(), other.len());
        ca_in
            .iter()
            .zip(other.array()?.amortized_iter())
            .map(|(value, series)| match (value, series) {
                (val, Some(series)) => {
                    let ca = series.as_ref().unpack::<T>().unwrap();
                    ca.iter().any(|a| a == val)
                },
                _ => false,
            })
            .collect_trusted()
    };
    ca.rename(ca_in.name());
    Ok(ca)
}

fn is_in_numeric<T>(ca_in: &ChunkedArray<T>, other: &Series) -> PolarsResult<BooleanChunked>
where
    T: PolarsNumericType,
    T::Native: TotalHash + TotalEq + ToTotalOrd,
    <T::Native as ToTotalOrd>::TotalOrdItem: Hash + Eq + Copy,
{
    // We check implicitly cast to supertype here
    match other.dtype() {
        DataType::List(dt) => {
            let st = try_get_supertype(ca_in.dtype(), dt)?;
            if &st != ca_in.dtype() || **dt != st {
                let left = ca_in.cast(&st)?;
                let right = other.cast(&DataType::List(Box::new(st)))?;
                return is_in(&left, &right);
            };
            is_in_numeric_list(ca_in, other)
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(dt, width) => {
            let st = try_get_supertype(ca_in.dtype(), dt)?;
            if &st != ca_in.dtype() || **dt != st {
                let left = ca_in.cast(&st)?;
                let right = other.cast(&DataType::Array(Box::new(st), *width))?;
                return is_in(&left, &right);
            };
            is_in_numeric_array(ca_in, other)
        },
        _ => {
            // first make sure that the types are equal
            if ca_in.dtype() != other.dtype() {
                let st = try_get_supertype(ca_in.dtype(), other.dtype())?;
                let left = ca_in.cast(&st)?;
                let right = other.cast(&st)?;
                return is_in(&left, &right);
            }
            is_in_helper(ca_in, other)
        },
    }
}

#[cfg(feature = "dtype-categorical")]
fn is_in_string_list_categorical(
    ca_in: &StringChunked,
    other: &Series,
    rev_map: &Arc<RevMapping>,
) -> PolarsResult<BooleanChunked> {
    let mut ca = if ca_in.len() == 1 && other.len() != 1 {
        let opt_val = ca_in.get(0);
        match opt_val.map(|val| rev_map.find(val)) {
            None => other.list()?.apply_amortized_generic(|opt_s| {
                {
                    opt_s.map(|s| s.as_ref().null_count() > 0)
                }
            }),
            Some(None) => other
                .list()?
                .apply_amortized_generic(|opt_s| opt_s.map(|_| false)),
            Some(Some(idx)) => other.list()?.apply_amortized_generic(|opt_s| {
                opt_s.map(|s| {
                    let s = s.as_ref().to_physical_repr();
                    let ca = s.as_ref().u32().unwrap();
                    if ca.null_count() == 0 {
                        ca.into_no_null_iter().any(|a| a == idx)
                    } else {
                        ca.iter().any(|a| a == Some(idx))
                    }
                })
            }),
        }
    } else {
        polars_ensure!(ca_in.len() == other.len(), ComputeError: "shapes don't match: expected {} elements in 'is_in' comparison, got {}", ca_in.len(), other.len());
        {
            ca_in
                .iter()
                .zip(other.list()?.amortized_iter())
                .map(|(opt_val, series)| match (opt_val, series) {
                    (opt_val, Some(series)) => match opt_val.map(|val| rev_map.find(val)) {
                        None => Some(series.as_ref().null_count() > 0),
                        Some(None) => Some(false),
                        Some(Some(idx)) => {
                            let ca = series.as_ref().categorical().unwrap();
                            Some(ca.physical().iter().any(|el| el == Some(idx)))
                        },
                    },
                    _ => None,
                })
                .collect()
        }
    };
    ca.rename(ca_in.name());
    Ok(ca)
}

fn is_in_string(ca_in: &StringChunked, other: &Series) -> PolarsResult<BooleanChunked> {
    match other.dtype() {
        #[cfg(feature = "dtype-categorical")]
        DataType::List(dt)
            if matches!(&**dt, DataType::Categorical(_, _) | DataType::Enum(_, _)) =>
        {
            match &**dt {
                DataType::Enum(Some(rev_map), _) | DataType::Categorical(Some(rev_map), _) => {
                    is_in_string_list_categorical(ca_in, other, rev_map)
                },
                _ => unreachable!(),
            }
        },
        DataType::List(dt) if DataType::String == **dt => is_in_binary(
            &ca_in.as_binary(),
            &other
                .cast(&DataType::List(Box::new(DataType::Binary)))
                .unwrap(),
        ),
        #[cfg(feature = "dtype-array")]
        DataType::Array(dt, width) if DataType::String == **dt => is_in_binary(
            &ca_in.as_binary(),
            &other
                .cast(&DataType::Array(Box::new(DataType::Binary), *width))
                .unwrap(),
        ),
        DataType::String => {
            is_in_binary(&ca_in.as_binary(), &other.cast(&DataType::Binary).unwrap())
        },
        #[cfg(feature = "dtype-categorical")]
        DataType::Enum(_, _) | DataType::Categorical(_, _) => {
            is_in_string_categorical(ca_in, other.categorical().unwrap())
        },
        _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
    }
}

fn is_in_binary_list(ca_in: &BinaryChunked, other: &Series) -> PolarsResult<BooleanChunked> {
    let mut ca: BooleanChunked = if ca_in.len() == 1 && other.len() != 1 {
        let value = ca_in.get(0);

        other.list()?.apply_amortized_generic(|opt_b| {
            Some(
                opt_b.map(|s| {
                    let ca = s.as_ref().unpack::<BinaryType>().unwrap();
                    ca.iter().any(|a| a == value)
                }) == Some(true),
            )
        })
    } else {
        polars_ensure!(ca_in.len() == other.len(), ComputeError: "shapes don't match: expected {} elements in 'is_in' comparison, got {}", ca_in.len(), other.len());
        {
            ca_in
                .iter()
                .zip(other.list()?.amortized_iter())
                .map(|(value, series)| match (value, series) {
                    (val, Some(series)) => {
                        let ca = series.as_ref().unpack::<BinaryType>().unwrap();
                        ca.iter().any(|a| a == val)
                    },
                    _ => false,
                })
                .collect_trusted()
        }
    };
    ca.rename(ca_in.name());
    Ok(ca)
}

#[cfg(feature = "dtype-array")]
fn is_in_binary_array(ca_in: &BinaryChunked, other: &Series) -> PolarsResult<BooleanChunked> {
    let mut ca: BooleanChunked = if ca_in.len() == 1 && other.len() != 1 {
        let value = ca_in.get(0);

        other.array()?.apply_amortized_generic(|opt_b| {
            Some(
                opt_b.map(|s| {
                    let ca = s.as_ref().unpack::<BinaryType>().unwrap();
                    ca.iter().any(|a| a == value)
                }) == Some(true),
            )
        })
    } else {
        polars_ensure!(ca_in.len() == other.len(), ComputeError: "shapes don't match: expected {} elements in 'is_in' comparison, got {}", ca_in.len(), other.len());
        ca_in
            .iter()
            .zip(other.array()?.amortized_iter())
            .map(|(value, series)| match (value, series) {
                (val, Some(series)) => {
                    let ca = series.as_ref().unpack::<BinaryType>().unwrap();
                    ca.iter().any(|a| a == val)
                },
                _ => false,
            })
            .collect_trusted()
    };
    ca.rename(ca_in.name());
    Ok(ca)
}

fn is_in_binary(ca_in: &BinaryChunked, other: &Series) -> PolarsResult<BooleanChunked> {
    match other.dtype() {
        DataType::List(dt) if DataType::Binary == **dt => is_in_binary_list(ca_in, other),
        #[cfg(feature = "dtype-array")]
        DataType::Array(dt, _) if DataType::Binary == **dt => is_in_binary_array(ca_in, other),
        DataType::Binary => is_in_helper(ca_in, other),
        _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
    }
}

fn is_in_boolean_list(ca_in: &BooleanChunked, other: &Series) -> PolarsResult<BooleanChunked> {
    let mut ca: BooleanChunked = if ca_in.len() == 1 && other.len() != 1 {
        let value = ca_in.get(0);
        // SAFETY: we know the iterators len
        unsafe {
            other
                .list()?
                .amortized_iter()
                .map(|opt_s| {
                    opt_s.map(|s| {
                        let ca = s.as_ref().unpack::<BooleanType>().unwrap();
                        ca.iter().any(|a| a == value)
                    }) == Some(true)
                })
                .trust_my_length(other.len())
                .collect_trusted()
        }
    } else {
        polars_ensure!(ca_in.len() == other.len(), ComputeError: "shapes don't match: expected {} elements in 'is_in' comparison, got {}", ca_in.len(), other.len());
        {
            ca_in
                .iter()
                .zip(other.list()?.amortized_iter())
                .map(|(value, series)| match (value, series) {
                    (val, Some(series)) => {
                        let ca = series.as_ref().unpack::<BooleanType>().unwrap();
                        ca.iter().any(|a| a == val)
                    },
                    _ => false,
                })
                .collect_trusted()
        }
    };
    ca.rename(ca_in.name());
    Ok(ca)
}

#[cfg(feature = "dtype-array")]
fn is_in_boolean_array(ca_in: &BooleanChunked, other: &Series) -> PolarsResult<BooleanChunked> {
    let mut ca: BooleanChunked = if ca_in.len() == 1 && other.len() != 1 {
        let value = ca_in.get(0);
        // SAFETY: we know the iterators len
        unsafe {
            other
                .array()?
                .amortized_iter()
                .map(|opt_s| {
                    opt_s.map(|s| {
                        let ca = s.as_ref().unpack::<BooleanType>().unwrap();
                        ca.iter().any(|a| a == value)
                    }) == Some(true)
                })
                .trust_my_length(other.len())
                .collect_trusted()
        }
    } else {
        polars_ensure!(ca_in.len() == other.len(), ComputeError: "shapes don't match: expected {} elements in 'is_in' comparison, got {}", ca_in.len(), other.len());
        ca_in
            .iter()
            .zip(other.array()?.amortized_iter())
            .map(|(value, series)| match (value, series) {
                (val, Some(series)) => {
                    let ca = series.as_ref().unpack::<BooleanType>().unwrap();
                    ca.iter().any(|a| a == val)
                },
                _ => false,
            })
            .collect_trusted()
    };
    ca.rename(ca_in.name());
    Ok(ca)
}

fn is_in_boolean(ca_in: &BooleanChunked, other: &Series) -> PolarsResult<BooleanChunked> {
    match other.dtype() {
        DataType::List(dt) if ca_in.dtype() == &**dt => is_in_boolean_list(ca_in, other),
        #[cfg(feature = "dtype-array")]
        DataType::Array(dt, _) if ca_in.dtype() == &**dt => is_in_boolean_array(ca_in, other),
        DataType::Boolean => {
            let other = other.bool().unwrap();
            let has_true = other.any();
            let nc = other.null_count();

            let has_false = if nc == 0 {
                !other.all()
            } else {
                (other.sum().unwrap() as usize + nc) != other.len()
            };
            Ok(ca_in
                .apply_values(|v| if v { has_true } else { has_false })
                .with_name(ca_in.name()))
        },
        _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
    }
}

#[cfg(feature = "dtype-struct")]
fn is_in_struct_list(ca_in: &StructChunked, other: &Series) -> PolarsResult<BooleanChunked> {
    let mut ca: BooleanChunked = if ca_in.len() == 1 && other.len() != 1 {
        let left = ca_in.get_row_encoded(Default::default())?;
        let value = left.get(0).unwrap();
        other.list()?.apply_amortized_generic(|opt_s| {
            Some(
                opt_s.map(|s| {
                    let ca = s.as_ref().struct_().unwrap();
                    let arr = ca.get_row_encoded_array(Default::default()).unwrap();
                    arr.values_iter().any(|a| a == value)
                }) == Some(true),
            )
        })
    } else {
        polars_ensure!(ca_in.len() == other.len(), ComputeError: "shapes don't match: expected {} elements in 'is_in' comparison, got {}", ca_in.len(), other.len());

        // TODO! improve this.
        let ca = if ca_in.null_count() > 0 {
            let ca_in = ca_in.rechunk();
            let mut ca = ca_in.get_row_encoded(Default::default())?;
            ca.merge_validities(ca_in.chunks());
            ca
        } else {
            ca_in.get_row_encoded(Default::default())?
        };
        {
            ca.iter()
                .zip(other.list()?.amortized_iter())
                .map(|(value, series)| match (value, series) {
                    (val, Some(series)) => {
                        let val = val.expect("no_nulls");
                        let ca = series.as_ref().struct_().unwrap();
                        let arr = ca.get_row_encoded_array(Default::default()).unwrap();
                        arr.values_iter().any(|a| a == val)
                    },
                    _ => false,
                })
                .collect()
        }
    };
    ca.rename(ca_in.name());
    Ok(ca)
}

#[cfg(all(feature = "dtype-struct", feature = "dtype-array"))]
fn is_in_struct_array(ca_in: &StructChunked, other: &Series) -> PolarsResult<BooleanChunked> {
    let mut ca: BooleanChunked = if ca_in.len() == 1 && other.len() != 1 {
        let left = ca_in.get_row_encoded(Default::default())?;
        let value = left.get(0).unwrap();
        other.array()?.apply_amortized_generic(|opt_s| {
            Some(
                opt_s.map(|s| {
                    let ca = s.as_ref().struct_().unwrap();
                    let arr = ca.get_row_encoded_array(Default::default()).unwrap();
                    arr.values_iter().any(|a| a == value)
                }) == Some(true),
            )
        })
    } else {
        polars_ensure!(ca_in.len() == other.len(), ComputeError: "shapes don't match: expected {} elements in 'is_in' comparison, got {}", ca_in.len(), other.len());

        // TODO! improve this.
        let ca = if ca_in.null_count() > 0 {
            let ca_in = ca_in.rechunk();
            let mut ca = ca_in.get_row_encoded(Default::default())?;
            ca.merge_validities(ca_in.chunks());
            ca
        } else {
            ca_in.get_row_encoded(Default::default())?
        };
        {
            ca.iter()
                .zip(other.array()?.amortized_iter())
                .map(|(value, series)| match (value, series) {
                    (val, Some(series)) => {
                        let val = val.expect("no nulls");
                        let ca = series.as_ref().struct_().unwrap();
                        let arr = ca.get_row_encoded_array(Default::default()).unwrap();
                        arr.values_iter().any(|a| a == val)
                    },
                    _ => false,
                })
                .collect()
        }
    };
    ca.rename(ca_in.name());
    Ok(ca)
}

#[cfg(feature = "dtype-struct")]
fn is_in_struct(ca_in: &StructChunked, other: &Series) -> PolarsResult<BooleanChunked> {
    match other.dtype() {
        DataType::List(_) => is_in_struct_list(ca_in, other),
        #[cfg(feature = "dtype-array")]
        DataType::Array(_, _) => is_in_struct_array(ca_in, other),
        _ => {
            let ca_in = ca_in.cast(&ca_in.dtype().to_physical()).unwrap();
            let ca_in = ca_in.struct_()?;
            let other = other.cast(&other.dtype().to_physical()).unwrap();
            let other = other.struct_()?;

            polars_ensure!(
                ca_in.struct_fields().len() == other.struct_fields().len(),
                ComputeError: "`is_in`: mismatch in the number of struct fields: {} and {}",
                ca_in.struct_fields().len(), other.struct_fields().len()
            );

            // first make sure that the types are equal
            let ca_in_dtypes: Vec<_> = ca_in
                .struct_fields()
                .iter()
                .map(|f| f.data_type())
                .collect();
            let other_dtypes: Vec<_> = other
                .struct_fields()
                .iter()
                .map(|f| f.data_type())
                .collect();
            if ca_in_dtypes != other_dtypes {
                let ca_in_names = ca_in.struct_fields().iter().map(|f| f.name());
                let other_names = other.struct_fields().iter().map(|f| f.name());
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

            if ca_in.null_count() > 0 {
                let ca_in = ca_in.rechunk();
                let mut ca_in_o = ca_in.get_row_encoded(Default::default())?;
                ca_in_o.merge_validities(ca_in.chunks());
                let ca_other = other.get_row_encoded(Default::default())?;
                is_in_helper_ca(&ca_in_o, &ca_other)
            } else {
                let ca_in = ca_in.get_row_encoded(Default::default())?;
                let ca_other = other.get_row_encoded(Default::default())?;
                is_in_helper_ca(&ca_in, &ca_other)
            }
        },
    }
}

#[cfg(feature = "dtype-categorical")]
fn is_in_string_categorical(
    ca_in: &StringChunked,
    other: &CategoricalChunked,
) -> PolarsResult<BooleanChunked> {
    // In case of fast unique, we can directly use the categories. Otherwise we need to
    // first get the unique physicals
    let categories = StringChunked::with_chunk("", other.get_rev_map().get_categories().clone());
    let other = if other._can_fast_unique() {
        categories
    } else {
        let s = other.physical().unique()?.cast(&IDX_DTYPE)?;
        // SAFETY: Invariant of categorical means indices are in bound
        unsafe { categories.take_unchecked(s.idx()?) }
    };
    is_in_helper_ca(&ca_in.as_binary(), &other.as_binary())
}

#[cfg(feature = "dtype-categorical")]
fn is_in_cat(ca_in: &CategoricalChunked, other: &Series) -> PolarsResult<BooleanChunked> {
    match other.dtype() {
        DataType::Categorical(_, _) | DataType::Enum(_, _) => {
            let (ca_in, other_in) =
                make_categoricals_compatible(ca_in, other.categorical().unwrap())?;
            is_in_helper_ca(ca_in.physical(), other_in.physical())
        },
        DataType::String => {
            let ca_other = other.str().unwrap();
            let rev_map = ca_in.get_rev_map();
            let categories = rev_map.get_categories();
            let others: PlHashSet<&str> = ca_other.downcast_iter().flatten().flatten().collect();
            let mut set = PlHashSet::with_capacity(std::cmp::min(categories.len(), ca_other.len()));

            // Either store the global or local indices of the overlapping strings
            match &**rev_map {
                RevMapping::Global(hash_map, categories, _) => {
                    for (global_idx, local_idx) in hash_map.iter() {
                        // SAFETY: index is in bounds
                        if others
                            .contains(unsafe { categories.value_unchecked(*local_idx as usize) })
                        {
                            #[allow(clippy::unnecessary_cast)]
                            set.insert((*global_idx as u32).to_total_ord());
                        }
                    }
                },
                RevMapping::Local(categories, _) => {
                    categories
                        .values_iter()
                        .enumerate_idx()
                        .for_each(|(idx, v)| {
                            if others.contains(v) {
                                #[allow(clippy::unnecessary_cast)]
                                set.insert((idx as u32).to_total_ord());
                            }
                        });
                },
            }

            Ok(
                unary_elementwise_values(ca_in.physical(), |val| set.contains(&val.to_total_ord()))
                    .with_name(ca_in.name()),
            )
        },

        DataType::List(dt)
            if matches!(&**dt, DataType::Categorical(_, _) | DataType::Enum(_, _)) =>
        {
            is_in_cat_list(ca_in, other)
        },

        _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
    }
}

#[cfg(feature = "dtype-categorical")]
fn is_in_cat_list(ca_in: &CategoricalChunked, other: &Series) -> PolarsResult<BooleanChunked> {
    let list_chunked = other.list()?;

    let mut ca: BooleanChunked = if ca_in.len() == 1 && other.len() != 1 {
        let (DataType::Categorical(Some(rev_map), _) | DataType::Enum(Some(rev_map), _)) =
            list_chunked.inner_dtype()
        else {
            unreachable!();
        };

        let idx = ca_in.physical().get(0);
        let new_phys = idx
            .map(|idx| ca_in.get_rev_map().get(idx))
            .map(|s| rev_map.find(s));

        match new_phys {
            None => list_chunked
                .apply_amortized_generic(|opt_s| opt_s.map(|s| s.as_ref().null_count() > 0)),
            Some(None) => list_chunked.apply_amortized_generic(|opt_s| opt_s.map(|_| false)),
            Some(Some(idx)) => list_chunked.apply_amortized_generic(|opt_s| {
                opt_s.map(|s| {
                    let ca = s.as_ref().categorical().unwrap();
                    ca.physical().iter().any(|a| a == Some(idx))
                })
            }),
        }
    } else {
        polars_ensure!(ca_in.len() == other.len(), ComputeError: "shapes don't match: expected {} elements in 'is_in' comparison, got {}", ca_in.len(), other.len());
        let list_chunked_inner = list_chunked.get_inner();
        let inner_cat = list_chunked_inner.categorical()?;
        // Make physicals compatible of ca_in with those of the list
        let (_, ca_in) = make_categoricals_compatible(inner_cat, ca_in)?;

        {
            ca_in
                .physical()
                .iter()
                .zip(list_chunked.amortized_iter())
                .map(|(value, series)| match (value, series) {
                    (val, Some(series)) => {
                        let ca = series.as_ref().categorical().unwrap();
                        Some(ca.physical().iter().any(|a| a == val))
                    },
                    _ => None,
                })
                .collect_trusted()
        }
    };
    ca.rename(ca_in.name());
    Ok(ca)
}

pub fn is_in(s: &Series, other: &Series) -> PolarsResult<BooleanChunked> {
    match s.dtype() {
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(_, _) | DataType::Enum(_, _) => {
            let ca = s.categorical().unwrap();
            is_in_cat(ca, other)
        },
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(_) => {
            let ca = s.struct_().unwrap();
            is_in_struct(ca, other)
        },
        DataType::String => {
            let ca = s.str().unwrap();
            is_in_string(ca, other)
        },
        DataType::Binary => {
            let ca = s.binary().unwrap();
            is_in_binary(ca, other)
        },
        DataType::Boolean => {
            let ca = s.bool().unwrap();
            is_in_boolean(ca, other)
        },
        DataType::Null => {
            let series_bool = s.cast(&DataType::Boolean)?;
            let ca = series_bool.bool().unwrap();
            Ok(ca.clone())
        },
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(_, _) => {
            let s = s.decimal()?;
            let other = other.decimal()?;
            let scale = s.scale().max(other.scale());
            let s = s.to_scale(scale)?;
            let other = other.to_scale(scale)?.into_owned().into_series();

            is_in_numeric(s.physical(), &other)
        },
        dt if dt.to_physical().is_numeric() => {
            let s = s.to_physical_repr();
            with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                is_in_numeric(ca, other)
            })
        },
        dt => polars_bail!(opq = is_in, dt),
    }
}
