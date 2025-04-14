use std::hash::Hash;

use arrow::array::BooleanArray;
use arrow::bitmap::BitmapBuilder;
use polars_core::prelude::arity::{unary_elementwise, unary_elementwise_values};
use polars_core::prelude::*;
use polars_core::utils::{CustomIterTools, try_get_supertype};
use polars_core::with_match_physical_numeric_polars_type;
#[cfg(feature = "dtype-categorical")]
use polars_utils::itertools::Itertools;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash};

use self::row_encode::_get_rows_encoded_ca_unordered;

fn is_in_helper_ca<'a, T>(
    ca: &'a ChunkedArray<T>,
    other: &'a ChunkedArray<T>,
    nulls_equal: bool,
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

    if nulls_equal {
        if other.has_nulls() {
            // If the rhs has nulls, then nulls in the left set evaluates to true.
            Ok(unary_elementwise(ca, |val| {
                val.is_none_or(|v| set.contains(&v.to_total_ord()))
            }))
        } else {
            // The rhs has no nulls; nulls in the left evaluates to false.
            Ok(unary_elementwise(ca, |val| {
                val.is_some_and(|v| set.contains(&v.to_total_ord()))
            }))
        }
    } else {
        Ok(
            unary_elementwise_values(ca, |v| set.contains(&v.to_total_ord()))
                .with_name(ca.name().clone()),
        )
    }
}

fn is_in_helper_list_ca<'a, T>(
    ca_in: &'a ChunkedArray<T>,
    other: &'a ListChunked,
    nulls_equal: bool,
) -> PolarsResult<BooleanChunked>
where
    T: PolarsDataType<IsLogical = FalseT>,
    for<'b> T::Physical<'b>: TotalHash + TotalEq + ToTotalOrd + Copy,
    for<'b> <T::Physical<'b> as ToTotalOrd>::TotalOrdItem: Hash + Eq + Copy,
{
    let offsets = other.offsets()?;
    let inner = other.get_inner();
    let inner: &ChunkedArray<T> = inner.as_ref().as_ref();
    let validity = other.rechunk_validity();

    let mut ca: BooleanChunked = if ca_in.len() == 1 && other.len() != 1 {
        let value = ca_in.get(0);

        match value {
            None if !nulls_equal => BooleanChunked::full_null(PlSmallStr::EMPTY, other.len()),
            value => {
                let mut builder = BitmapBuilder::with_capacity(other.len());

                for (start, length) in offsets.offset_and_length_iter() {
                    let mut is_in = false;
                    for i in 0..length {
                        is_in |= value.to_total_ord() == inner.get(start + i).to_total_ord();
                    }
                    builder.push(is_in);
                }

                let values = builder.freeze();

                let result = BooleanArray::new(ArrowDataType::Boolean, values, validity);
                BooleanChunked::from_chunk_iter(PlSmallStr::EMPTY, [result])
            },
        }
    } else {
        assert_eq!(ca_in.len(), offsets.len_proxy());
        {
            if nulls_equal {
                let mut builder = BitmapBuilder::with_capacity(ca_in.len());

                for (value, (start, length)) in ca_in.iter().zip(offsets.offset_and_length_iter()) {
                    let mut is_in = false;
                    for i in 0..length {
                        is_in |= value.to_total_ord() == inner.get(start + i).to_total_ord();
                    }
                    builder.push(is_in);
                }

                let values = builder.freeze();

                let result = BooleanArray::new(ArrowDataType::Boolean, values, validity);
                BooleanChunked::from_chunk_iter(PlSmallStr::EMPTY, [result])
            } else {
                let mut builder = BitmapBuilder::with_capacity(ca_in.len());

                for (value, (start, length)) in ca_in.iter().zip(offsets.offset_and_length_iter()) {
                    let mut is_in = false;
                    if value.is_some() {
                        for i in 0..length {
                            is_in |= value.to_total_ord() == inner.get(start + i).to_total_ord();
                        }
                    }
                    builder.push(is_in);
                }

                let values = builder.freeze();

                let validity = match (validity, ca_in.rechunk_validity()) {
                    (None, None) => None,
                    (Some(v), None) | (None, Some(v)) => Some(v),
                    (Some(l), Some(r)) => Some(arrow::bitmap::and(&l, &r)),
                };

                let result = BooleanArray::new(ArrowDataType::Boolean, values, validity);
                BooleanChunked::from_chunk_iter(PlSmallStr::EMPTY, [result])
            }
        }
    };
    ca.rename(ca_in.name().clone());
    Ok(ca)
}

#[cfg(feature = "dtype-array")]
fn is_in_helper_array_ca<'a, T>(
    ca_in: &'a ChunkedArray<T>,
    other: &'a ArrayChunked,
    nulls_equal: bool,
) -> PolarsResult<BooleanChunked>
where
    T: PolarsDataType<IsLogical = FalseT>,
    for<'b> T::Physical<'b>: TotalHash + TotalEq + ToTotalOrd + Copy,
    for<'b> <T::Physical<'b> as ToTotalOrd>::TotalOrdItem: Hash + Eq + Copy,
{
    let width = other.width();
    let inner = other.get_inner();
    let inner: &ChunkedArray<T> = inner.as_ref().as_ref();
    let validity = other.rechunk_validity();

    let mut ca: BooleanChunked = if ca_in.len() == 1 && other.len() != 1 {
        let value = ca_in.get(0);

        match value {
            None if !nulls_equal => BooleanChunked::full_null(PlSmallStr::EMPTY, other.len()),
            value => {
                let mut builder = BitmapBuilder::with_capacity(other.len());

                for i in 0..other.len() {
                    let mut is_in = false;
                    for j in 0..width {
                        is_in |= value.to_total_ord() == inner.get(i * width + j).to_total_ord();
                    }
                    builder.push(is_in);
                }

                let values = builder.freeze();

                let result = BooleanArray::new(ArrowDataType::Boolean, values, validity);
                BooleanChunked::from_chunk_iter(PlSmallStr::EMPTY, [result])
            },
        }
    } else {
        assert_eq!(ca_in.len(), other.len());
        {
            if nulls_equal {
                let mut builder = BitmapBuilder::with_capacity(ca_in.len());

                for (i, value) in ca_in.iter().enumerate() {
                    let mut is_in = false;
                    for j in 0..width {
                        is_in |= value.to_total_ord() == inner.get(i * width + j).to_total_ord();
                    }
                    builder.push(is_in);
                }

                let values = builder.freeze();

                let result = BooleanArray::new(ArrowDataType::Boolean, values, validity);
                BooleanChunked::from_chunk_iter(PlSmallStr::EMPTY, [result])
            } else {
                let mut builder = BitmapBuilder::with_capacity(ca_in.len());

                for (i, value) in ca_in.iter().enumerate() {
                    let mut is_in = false;
                    if value.is_some() {
                        for j in 0..width {
                            is_in |=
                                value.to_total_ord() == inner.get(i * width + j).to_total_ord();
                        }
                    }
                    builder.push(is_in);
                }

                let values = builder.freeze();

                let validity = match (validity, ca_in.rechunk_validity()) {
                    (None, None) => None,
                    (Some(v), None) | (None, Some(v)) => Some(v),
                    (Some(l), Some(r)) => Some(arrow::bitmap::and(&l, &r)),
                };

                let result = BooleanArray::new(ArrowDataType::Boolean, values, validity);
                BooleanChunked::from_chunk_iter(PlSmallStr::EMPTY, [result])
            }
        }
    };
    ca.rename(ca_in.name().clone());
    Ok(ca)
}

fn is_in_numeric<T>(
    ca_in: &ChunkedArray<T>,
    other: &Series,
    nulls_equal: bool,
) -> PolarsResult<BooleanChunked>
where
    T: PolarsNumericType,
    T::Native: TotalHash + TotalEq + ToTotalOrd,
    <T::Native as ToTotalOrd>::TotalOrdItem: Hash + Eq + Copy,
{
    match other.dtype() {
        DataType::List(..) => {
            let other = other.list()?;
            if other.len() == 1 {
                let other = other.explode()?;
                let other = other.as_ref().as_ref();
                is_in_helper_ca(ca_in, other, nulls_equal)
            } else {
                is_in_helper_list_ca(ca_in, other, nulls_equal)
            }
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(..) => {
            let other = other.array()?;
            if other.len() == 1 {
                let other = other.explode()?;
                let other = other.as_ref().as_ref();
                is_in_helper_ca(ca_in, other, nulls_equal)
            } else {
                is_in_helper_array_ca(ca_in, other, nulls_equal)
            }
        },
        _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
    }
}

#[cfg(feature = "dtype-categorical")]
fn is_in_string_list_categorical(
    ca_in: &StringChunked,
    other: &ListChunked,
    rev_map: &Arc<RevMapping>,
) -> PolarsResult<BooleanChunked> {
    let mut ca = if ca_in.len() == 1 && other.len() != 1 {
        let opt_val = ca_in.get(0);
        match opt_val.map(|val| rev_map.find(val)) {
            None => {
                other.apply_amortized_generic(|opt_s| opt_s.map(|s| s.as_ref().null_count() > 0))
            },
            Some(None) => other.apply_amortized_generic(|opt_s| opt_s.map(|_| false)),
            Some(Some(idx)) => other.apply_amortized_generic(|opt_s| {
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
                .zip(other.amortized_iter())
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
    ca.rename(ca_in.name().clone());
    Ok(ca)
}

fn is_in_string(
    ca_in: &StringChunked,
    other: &Series,
    nulls_equal: bool,
) -> PolarsResult<BooleanChunked> {
    fn is_in_string_broadcast(
        ca_in: &StringChunked,
        other: &Series,
        nulls_equal: bool,
    ) -> PolarsResult<BooleanChunked> {
        match other.dtype() {
            DataType::String => {
                is_in_helper_ca(&ca_in.as_binary(), &other.str()?.as_binary(), nulls_equal)
            },
            #[cfg(feature = "dtype-categorical")]
            DataType::Enum(_, _) | DataType::Categorical(_, _) => {
                is_in_string_categorical(ca_in, other.categorical().unwrap(), nulls_equal)
            },
            _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
        }
    }

    match other.dtype() {
        #[cfg(feature = "dtype-categorical")]
        DataType::List(dt)
            if matches!(&**dt, DataType::Categorical(_, _) | DataType::Enum(_, _)) =>
        {
            let other = other.list()?;
            if other.len() == 1 {
                let other = other.explode()?;
                is_in_string_broadcast(ca_in, &other, nulls_equal)
            } else {
                match &**dt {
                    DataType::Enum(Some(rev_map), _) | DataType::Categorical(Some(rev_map), _) => {
                        is_in_string_list_categorical(ca_in, other, rev_map)
                    },
                    _ => unreachable!(),
                }
            }
        },
        DataType::List(dt) if DataType::String == **dt => {
            let other = other.list()?;
            if other.len() == 1 {
                let other = other.explode()?;
                is_in_string_broadcast(ca_in, &other, nulls_equal)
            } else {
                is_in_binary(
                    &ca_in.as_binary(),
                    &other
                        .cast(&DataType::List(Box::new(DataType::Binary)))
                        .unwrap(),
                    nulls_equal,
                )
            }
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(dt, width) if DataType::String == **dt => {
            let other = other.array()?;
            if other.len() == 1 {
                let other = other.explode()?;
                is_in_string_broadcast(ca_in, &other, nulls_equal)
            } else {
                is_in_binary(
                    &ca_in.as_binary(),
                    &other
                        .cast(&DataType::Array(Box::new(DataType::Binary), *width))
                        .unwrap(),
                    nulls_equal,
                )
            }
        },
        _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
    }
}

fn is_in_binary(
    ca_in: &BinaryChunked,
    other: &Series,
    nulls_equal: bool,
) -> PolarsResult<BooleanChunked> {
    match other.dtype() {
        DataType::List(dt) if DataType::Binary == **dt => {
            let other = other.list()?;
            if other.len() == 1 {
                let other = other.explode()?;
                let other = other.binary()?;
                is_in_helper_ca(ca_in, other, nulls_equal)
            } else {
                is_in_helper_list_ca(ca_in, other, nulls_equal)
            }
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(dt, _) if DataType::Binary == **dt => {
            let other = other.array()?;
            if other.len() == 1 {
                let other = other.explode()?;
                let other = other.binary()?;
                is_in_helper_ca(ca_in, other, nulls_equal)
            } else {
                is_in_helper_array_ca(ca_in, other, nulls_equal)
            }
        },
        _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
    }
}

fn is_in_boolean(
    ca_in: &BooleanChunked,
    other: &Series,
    nulls_equal: bool,
) -> PolarsResult<BooleanChunked> {
    fn is_in_boolean_broadcast(
        ca_in: &BooleanChunked,
        other: &BooleanChunked,
        nulls_equal: bool,
    ) -> PolarsResult<BooleanChunked> {
        let has_true = other.any();
        let nc = other.null_count();

        let has_false = if nc == 0 {
            !other.all()
        } else {
            (other.sum().unwrap() as usize + nc) != other.len()
        };
        let value_map = |v| if v { has_true } else { has_false };
        if nulls_equal {
            if other.has_nulls() {
                // If the rhs has nulls, then nulls in the left set evaluates to true.
                Ok(ca_in.apply(|opt_v| Some(opt_v.is_none_or(value_map))))
            } else {
                // The rhs has no nulls; nulls in the left evaluates to false.
                Ok(ca_in.apply(|opt_v| Some(opt_v.is_some_and(value_map))))
            }
        } else {
            Ok(ca_in
                .apply_values(value_map)
                .with_name(ca_in.name().clone()))
        }
    }

    match other.dtype() {
        DataType::List(dt) if ca_in.dtype() == &**dt => {
            let other = other.list()?;
            if other.len() == 1 {
                let other = other.explode()?;
                let other = other.bool()?;
                is_in_boolean_broadcast(ca_in, other, nulls_equal)
            } else {
                is_in_helper_list_ca(ca_in, other, nulls_equal)
            }
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(dt, _) if ca_in.dtype() == &**dt => {
            let other = other.array()?;
            if other.len() == 1 {
                let other = other.explode()?;
                let other = other.bool()?;
                is_in_boolean_broadcast(ca_in, other, nulls_equal)
            } else {
                is_in_helper_array_ca(ca_in, other, nulls_equal)
            }
        },
        _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
    }
}

#[cfg(feature = "dtype-categorical")]
fn is_in_string_categorical(
    ca_in: &StringChunked,
    other: &CategoricalChunked,
    nulls_equal: bool,
) -> PolarsResult<BooleanChunked> {
    // In case of fast unique, we can directly use the categories. Otherwise we need to
    // first get the unique physicals
    let categories = StringChunked::with_chunk(
        PlSmallStr::EMPTY,
        other.get_rev_map().get_categories().clone(),
    );
    let other = if other._can_fast_unique() {
        categories
    } else {
        let s = other.physical().unique()?.cast(&IDX_DTYPE)?;
        // SAFETY: Invariant of categorical means indices are in bound
        unsafe { categories.take_unchecked(s.idx()?) }
    };
    is_in_helper_ca(&ca_in.as_binary(), &other.as_binary(), nulls_equal)
}

#[cfg(feature = "dtype-categorical")]
fn is_in_cat(
    ca_in: &CategoricalChunked,
    other: &Series,
    nulls_equal: bool,
) -> PolarsResult<BooleanChunked> {
    fn is_in_cat_broadcast(
        ca_in: &CategoricalChunked,
        other: &Series,
        nulls_equal: bool,
    ) -> PolarsResult<BooleanChunked> {
        match other.dtype() {
            DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                let (ca_in, other_in) =
                    make_rhs_categoricals_compatible(ca_in, other.categorical().unwrap())?;
                is_in_helper_ca(ca_in.physical(), other_in.physical(), nulls_equal)
            },
            DataType::String => {
                let ca_other = other.str().unwrap();
                let rev_map = ca_in.get_rev_map();
                let categories = rev_map.get_categories();
                let others: PlHashSet<&str> =
                    ca_other.downcast_iter().flatten().flatten().collect();
                let mut set =
                    PlHashSet::with_capacity(std::cmp::min(categories.len(), ca_other.len()));

                // Either store the global or local indices of the overlapping strings
                match &**rev_map {
                    RevMapping::Global(hash_map, categories, _) => {
                        for (global_idx, local_idx) in hash_map.iter() {
                            // SAFETY: index is in bounds
                            if others.contains(unsafe {
                                categories.value_unchecked(*local_idx as usize)
                            }) {
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

                let ca = ca_in.physical();
                if nulls_equal {
                    if other.has_nulls() {
                        // If the rhs has nulls, then nulls in the left set evaluates to true.
                        Ok(unary_elementwise(ca, |val| {
                            val.is_none_or(|v| set.contains(&v.to_total_ord()))
                        }))
                    } else {
                        // The rhs has no nulls; nulls in the left evaluates to false.
                        Ok(unary_elementwise(ca, |val| {
                            val.is_some_and(|v| set.contains(&v.to_total_ord()))
                        }))
                    }
                } else {
                    Ok(
                        unary_elementwise_values(ca, |val| set.contains(&val.to_total_ord()))
                            .with_name(ca.name().clone()),
                    )
                }
            },
            _ => unreachable!(),
        }
    }

    match other.dtype() {
        DataType::List(dt)
            if matches!(
                &**dt,
                DataType::Categorical(_, _) | DataType::Enum(_, _) | DataType::String
            ) =>
        {
            let other = other.list()?;
            if other.len() == 1 {
                let other = other.explode()?;
                is_in_cat_broadcast(ca_in, &other, nulls_equal)
            } else {
                is_in_cat_list(ca_in, other)
            }
        },
        _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
    }
}

#[cfg(feature = "dtype-categorical")]
fn is_in_cat_list(ca_in: &CategoricalChunked, other: &ListChunked) -> PolarsResult<BooleanChunked> {
    let inner_dtype = other.inner_dtype();
    if inner_dtype.is_string() {
        polars_bail!(nyi = "`is_in` with elementwise categorical and string");
    }

    let mut ca: BooleanChunked = if ca_in.len() == 1 && other.len() != 1 {
        let (DataType::Categorical(Some(rev_map), _) | DataType::Enum(Some(rev_map), _)) =
            inner_dtype
        else {
            unreachable!();
        };

        let idx = ca_in.physical().get(0);
        let new_phys = idx
            .map(|idx| ca_in.get_rev_map().get(idx))
            .map(|s| rev_map.find(s));

        match new_phys {
            None => {
                other.apply_amortized_generic(|opt_s| opt_s.map(|s| s.as_ref().null_count() > 0))
            },
            Some(None) => other.apply_amortized_generic(|opt_s| opt_s.map(|_| false)),
            Some(Some(idx)) => other.apply_amortized_generic(|opt_s| {
                opt_s.map(|s| {
                    let ca = s.as_ref().categorical().unwrap();
                    ca.physical().iter().any(|a| a == Some(idx))
                })
            }),
        }
    } else {
        polars_ensure!(ca_in.len() == other.len(), ComputeError: "shapes don't match: expected {} elements in 'is_in' comparison, got {}", ca_in.len(), other.len());
        let list_chunked_inner = other.explode()?;
        let inner_cat = list_chunked_inner.categorical()?;
        // Make physicals compatible of ca_in with those of the list
        let (_, ca_in) = make_rhs_categoricals_compatible(inner_cat, ca_in)?;

        {
            ca_in
                .physical()
                .iter()
                .zip(other.amortized_iter())
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
    ca.rename(ca_in.name().clone());
    Ok(ca)
}

fn is_in_null(s: &Series, other: &Series, nulls_equal: bool) -> PolarsResult<BooleanChunked> {
    if nulls_equal {
        let ca_in = s.null()?;
        Ok(match other.dtype() {
            DataType::List(_) => {
                let other = other.list()?;
                if other.len() == 1 {
                    let other = other.explode()?;
                    BooleanChunked::from_iter_values(
                        ca_in.name().clone(),
                        std::iter::repeat_n(other.has_nulls(), ca_in.len()),
                    )
                } else {
                    other.apply_amortized_generic(|opt_s| {
                        Some(opt_s.map(|s| s.as_ref().has_nulls()) == Some(true))
                    })
                }
            },
            #[cfg(feature = "dtype-array")]
            DataType::Array(_, _) => {
                let other = other.array()?;
                if other.len() == 1 {
                    let other = other.explode()?;
                    BooleanChunked::from_iter_values(
                        ca_in.name().clone(),
                        std::iter::repeat_n(other.has_nulls(), ca_in.len()),
                    )
                } else {
                    other.apply_amortized_generic(|opt_s| {
                        Some(opt_s.map(|s| s.as_ref().has_nulls()) == Some(true))
                    })
                }
            },
            _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
        })
    } else {
        let out = s.cast(&DataType::Boolean)?;
        let ca_bool = out.bool()?.clone();
        Ok(ca_bool)
    }
}

#[cfg(feature = "dtype-decimal")]
fn is_in_decimal(
    ca_in: &DecimalChunked,
    other: &Series,
    nulls_equal: bool,
) -> PolarsResult<BooleanChunked> {
    let Some(DataType::Decimal(_, other_scale)) = other.dtype().inner_dtype() else {
        polars_bail!(opq = is_in, ca_in.dtype(), other.dtype());
    };
    let other_scale = other_scale.unwrap();
    let scale = ca_in.scale().max(other_scale);
    let ca_in = ca_in.to_scale(scale)?;

    match other.dtype() {
        DataType::List(_) => {
            let other = other.list()?;
            let other = other.apply_to_inner(&|s| {
                let s = s.decimal()?;
                let s = s.to_scale(scale)?;
                let s = s.physical();
                Ok(s.to_owned().into_series())
            })?;
            let other = other.into_series();
            is_in_numeric(ca_in.physical(), &other, nulls_equal)
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(_, _) => {
            let other = other.array()?;
            let other = other.apply_to_inner(&|s| {
                let s = s.decimal()?;
                let s = s.to_scale(scale)?;
                let s = s.physical();
                Ok(s.to_owned().into_series())
            })?;
            let other = other.into_series();
            is_in_numeric(ca_in.physical(), &other, nulls_equal)
        },
        _ => unreachable!(),
    }
}

fn is_in_row_encoded(
    s: &Series,
    other: &Series,
    nulls_equal: bool,
) -> PolarsResult<BooleanChunked> {
    let ca_in = _get_rows_encoded_ca_unordered(s.name().clone(), &[s.clone().into_column()])?;
    let mut mask = match other.dtype() {
        DataType::List(_) => {
            let other = other.list()?;
            let other = other.apply_to_inner(&|s| {
                Ok(
                    _get_rows_encoded_ca_unordered(s.name().clone(), &[s.into_column()])?
                        .into_series(),
                )
            })?;
            if other.len() == 1 {
                let other = other.explode()?;
                let other = other.binary_offset()?;
                is_in_helper_ca(&ca_in, other, nulls_equal)
            } else {
                is_in_helper_list_ca(&ca_in, &other, nulls_equal)
            }
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(_, _) => {
            let other = other.array()?;
            let other = other.apply_to_inner(&|s| {
                Ok(
                    _get_rows_encoded_ca_unordered(s.name().clone(), &[s.into_column()])?
                        .into_series(),
                )
            })?;
            if other.len() == 1 {
                let other = other.explode()?;
                let other = other.binary_offset()?;
                is_in_helper_ca(&ca_in, other, nulls_equal)
            } else {
                is_in_helper_array_ca(&ca_in, &other, nulls_equal)
            }
        },
        _ => unreachable!(),
    }?;

    let mut validity = other.rechunk_validity();
    if !nulls_equal {
        validity = match (validity, s.rechunk_validity()) {
            (None, None) => None,
            (Some(v), None) | (None, Some(v)) => Some(v),
            (Some(l), Some(r)) => Some(arrow::bitmap::and(&l, &r)),
        };
    }

    assert_eq!(mask.null_count(), 0);
    mask.with_validities(&[validity]);

    Ok(mask)
}

pub fn is_in(s: &Series, other: &Series, nulls_equal: bool) -> PolarsResult<BooleanChunked> {
    polars_ensure!(
        s.len() == other.len() || s.len() == 1 || other.len() == 1,
        length_mismatch = "is_in",
        s.len(),
        other.len()
    );

    #[allow(unused_mut)]
    let mut other_is_valid_type = matches!(other.dtype(), DataType::List(_));
    #[cfg(feature = "dtype-array")]
    {
        other_is_valid_type |= matches!(other.dtype(), DataType::Array(..))
    }
    polars_ensure!(other_is_valid_type, opq = is_in, s.dtype(), other.dtype());

    match s.dtype() {
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(_, _) | DataType::Enum(_, _) => {
            let ca = s.categorical().unwrap();
            is_in_cat(ca, other, nulls_equal)
        },
        DataType::String => {
            let ca = s.str().unwrap();
            is_in_string(ca, other, nulls_equal)
        },
        DataType::Binary => {
            let ca = s.binary().unwrap();
            is_in_binary(ca, other, nulls_equal)
        },
        DataType::Boolean => {
            let ca = s.bool().unwrap();
            is_in_boolean(ca, other, nulls_equal)
        },
        DataType::Null => is_in_null(s, other, nulls_equal),
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(_, _) => {
            let ca_in = s.decimal()?;
            is_in_decimal(ca_in, other, nulls_equal)
        },
        dt if dt.is_nested() => is_in_row_encoded(s, other, nulls_equal),
        dt if dt.to_physical().is_primitive_numeric() => {
            let s = s.to_physical_repr();
            let other = other.to_physical_repr();
            let other = other.as_ref();
            with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                is_in_numeric(ca, other, nulls_equal)
            })
        },
        dt => polars_bail!(opq = is_in, dt),
    }
}
