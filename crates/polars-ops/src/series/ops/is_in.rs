use std::hash::Hash;

use arrow::array::BooleanArray;
use arrow::bitmap::BitmapBuilder;
use polars_core::prelude::arity::{unary_elementwise, unary_elementwise_values};
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
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
                if other.has_nulls() {
                    return Ok(BooleanChunked::full_null(ca_in.name().clone(), ca_in.len()));
                }

                let other = other.explode(true)?;
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
                if other.has_nulls() {
                    return Ok(BooleanChunked::full_null(ca_in.name().clone(), ca_in.len()));
                }

                let other = other.explode(true)?;
                let other = other.as_ref().as_ref();
                is_in_helper_ca(ca_in, other, nulls_equal)
            } else {
                is_in_helper_array_ca(ca_in, other, nulls_equal)
            }
        },
        _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
    }
}

fn is_in_string(
    ca_in: &StringChunked,
    other: &Series,
    nulls_equal: bool,
) -> PolarsResult<BooleanChunked> {
    let other = match other.dtype() {
        DataType::List(dt) if dt.is_string() || dt.is_enum() || dt.is_categorical() => {
            let other = other.list()?;
            other
                .apply_to_inner(&|mut s| {
                    if dt.is_enum() || dt.is_categorical() {
                        s = s.cast(&DataType::String)?;
                    }
                    let s = s.str()?;
                    Ok(s.as_binary().into_series())
                })?
                .into_series()
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(dt, _) if dt.is_string() || dt.is_enum() || dt.is_categorical() => {
            let other = other.array()?;
            other
                .apply_to_inner(&|mut s| {
                    if dt.is_enum() || dt.is_categorical() {
                        s = s.cast(&DataType::String)?;
                    }
                    Ok(s.str()?.as_binary().into_series())
                })?
                .into_series()
        },
        _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
    };
    is_in_binary(&ca_in.as_binary(), &other, nulls_equal)
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
                if other.has_nulls() {
                    return Ok(BooleanChunked::full_null(ca_in.name().clone(), ca_in.len()));
                }

                let other = other.explode(true)?;
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
                if other.has_nulls() {
                    return Ok(BooleanChunked::full_null(ca_in.name().clone(), ca_in.len()));
                }

                let other = other.explode(true)?;
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
                if other.has_nulls() {
                    return Ok(BooleanChunked::full_null(ca_in.name().clone(), ca_in.len()));
                }

                let other = other.explode(true)?;
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
                if other.has_nulls() {
                    return Ok(BooleanChunked::full_null(ca_in.name().clone(), ca_in.len()));
                }

                let other = other.explode(true)?;
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
fn is_in_cat_and_enum(
    ca_in: &CategoricalChunked,
    other: &Series,
    nulls_equal: bool,
) -> PolarsResult<BooleanChunked> {
    use std::borrow::Cow;

    use arrow::array::{Array, FixedSizeListArray, IntoBoxedArray, ListArray};

    let mut needs_remap = false;
    let to_categories = match (ca_in.dtype(), other.dtype().inner_dtype().unwrap()) {
        (DataType::Enum(revmap, ordering), DataType::String) => {
            let categories = revmap.as_deref().unwrap().get_categories();
            (&|s: Series| {
                let ca = s.str()?;
                let ca = CategoricalChunked::from_string_to_enum(ca, categories, *ordering)?;
                let ca = ca.into_physical();
                Ok(ca.into_series())
            }) as _
        },
        (DataType::Categorical(revmap, ordering), DataType::String) => {
            (&|s: Series| {
                let categories = revmap.as_deref().unwrap().get_categories();
                let ca = s.str()?;
                let ca =
                    if ca_in.get_rev_map().is_local() {
                        assert!(categories.len() < u32::MAX as usize);
                        let cats = PlIndexSet::from_iter(categories.values_iter());
                        UInt32Chunked::from_iter(ca.iter().map(|v| {
                            v.map(|v| cats.get_index_of(v).map_or(u32::MAX, |n| n as u32))
                        }))
                    } else {
                        let cat = ca.cast(&DataType::Categorical(None, *ordering))?;
                        cat.categorical()?.physical().clone()
                    };
                Ok(ca.into_series())
            }) as _
        },
        (DataType::Categorical(revmap, _), DataType::Categorical(other_revmap, _)) => {
            let (Some(revmap), Some(other_revmap)) = (revmap, other_revmap) else {
                polars_bail!(ComputeError: "expected revmap to be set at this point");
            };
            needs_remap = !revmap.same_src(other_revmap);
            (&|s: Series| {
                let ca = s.categorical()?;
                let ca = ca.physical().clone();
                Ok(ca.into_series())
            }) as _
        },
        (DataType::Enum(revmap, _), DataType::Enum(other_revmap, _)) => {
            let (Some(revmap), Some(other_revmap)) = (revmap, other_revmap) else {
                polars_bail!(ComputeError: "expected revmap to be set at this point");
            };
            polars_ensure!(
                revmap.same_src(other_revmap),
                opq = is_in,
                ca_in.dtype(),
                other.dtype()
            );
            (&|s: Series| {
                let ca = s.categorical()?;
                let ca = ca.physical().clone();
                Ok(ca.into_series())
            }) as _
        },
        _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
    };

    let mut ca_in = Cow::Borrowed(ca_in);
    let other = match other.dtype() {
        DataType::List(_) => {
            let mut other = Cow::Borrowed(other.list()?);
            if needs_remap {
                let other_rechunked = other.rechunk();
                let other_arr = other_rechunked.downcast_as_array();

                let other_inner = other.get_inner();
                let other_offsets = other_arr.offsets().clone();
                let other_inner = other_inner.categorical()?;
                let (ca_in_remapped, other_inner) =
                    make_rhs_categoricals_compatible(&ca_in, other_inner)?;

                let other_inner_phys = other_inner.physical().rechunk();
                let other_inner_phys = other_inner_phys.downcast_as_array();

                let other_phys = ListArray::try_new(
                    other_arr.dtype().clone(),
                    other_offsets,
                    other_inner_phys.clone().into_boxed(),
                    other_arr.validity().cloned(),
                )?;

                other = Cow::Owned(unsafe {
                    ListChunked::from_chunks_and_dtype(
                        other.name().clone(),
                        vec![other_phys.into_boxed()],
                        DataType::List(Box::new(other_inner.dtype().clone())),
                    )
                });
                ca_in = Cow::Owned(ca_in_remapped);
            }
            let other = other.apply_to_inner(to_categories)?;
            other.into_series()
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(_, _) => {
            let mut other = Cow::Borrowed(other.array()?);
            if needs_remap {
                let other_rechunked = other.rechunk();
                let other_arr = other_rechunked.downcast_as_array();

                let other_inner = other.get_inner();
                let other_inner = other_inner.categorical()?;
                let (ca_in_remapped, other_inner) =
                    make_rhs_categoricals_compatible(&ca_in, other_inner)?;

                let other_inner_phys = other_inner.physical().rechunk();
                let other_inner_phys = other_inner_phys.downcast_as_array();

                let other_phys = FixedSizeListArray::try_new(
                    other_arr.dtype().clone(),
                    other.len(),
                    other_inner_phys.clone().into_boxed(),
                    other_arr.validity().cloned(),
                )?;

                other = Cow::Owned(unsafe {
                    ArrayChunked::from_chunks_and_dtype(
                        other.name().clone(),
                        vec![other_phys.into_boxed()],
                        DataType::Array(Box::new(other_inner.dtype().clone()), other.width()),
                    )
                });
                ca_in = Cow::Owned(ca_in_remapped);
            }
            let other = other.apply_to_inner(to_categories)?;
            other.into_series()
        },
        _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
    };

    is_in_numeric(ca_in.physical(), &other, nulls_equal)
}

fn is_in_null(s: &Series, other: &Series, nulls_equal: bool) -> PolarsResult<BooleanChunked> {
    if nulls_equal {
        let ca_in = s.null()?;
        Ok(match other.dtype() {
            DataType::List(_) => {
                let other = other.list()?;
                if other.len() == 1 {
                    if other.has_nulls() {
                        return Ok(BooleanChunked::full_null(ca_in.name().clone(), ca_in.len()));
                    }

                    let other = other.explode(true)?;
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
                    if other.has_nulls() {
                        return Ok(BooleanChunked::full_null(ca_in.name().clone(), ca_in.len()));
                    }

                    let other = other.explode(true)?;
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
                if other.has_nulls() {
                    return Ok(BooleanChunked::full_null(ca_in.name().clone(), ca_in.len()));
                }

                let other = other.explode(true)?;
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
                if other.has_nulls() {
                    return Ok(BooleanChunked::full_null(ca_in.name().clone(), ca_in.len()));
                }

                let other = other.explode(true)?;
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
            is_in_cat_and_enum(ca, other, nulls_equal)
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
