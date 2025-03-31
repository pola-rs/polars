use std::borrow::Cow;
use std::hash::Hash;

use arrow::array::BooleanArray;
use arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_core::prelude::arity::{unary_elementwise, unary_elementwise_values};
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::itertools::Itertools;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash};

use self::row_encode::_get_rows_encoded_ca_unordered;
use crate::prelude::ListNameSpaceImpl;

enum ListLengths {
    Array(IdxSize),
    List(IdxCa),
}

fn is_in_helper_ca_list<T>(
    ca_in: &ChunkedArray<T>,
    other_length: usize,
    other_exploded: ChunkedArray<T>,
    other_offsets: ListLengths,
    nulls_equal: bool,
) -> PolarsResult<BooleanChunked>
where
    T: PolarsDataType,
    for<'a> T::Physical<'a>: TotalHash + TotalEq + ToTotalOrd + Copy,
    for<'a> <T::Physical<'a> as ToTotalOrd>::TotalOrdItem: Hash + Eq + Copy,
{
    let mask = match (ca_in.len(), other_length) {
        (l, r) if l == r => {
            let mut other_exploded_iter = other_exploded.iter();
            let mut bm = BitmapBuilder::with_capacity(l);

            match other_offsets {
                ListLengths::Array(width) => {
                    for value in ca_in.iter() {
                        let mut is_in = false;
                        for is_in_value in other_exploded_iter.by_ref().take(width as usize) {
                            is_in |= value.tot_eq(&is_in_value);
                        }
                        // SAFETY: We allocated ca_in.len() items.
                        unsafe { bm.push_unchecked(is_in) };
                    }
                },
                ListLengths::List(lengths) => {
                    assert_eq!(ca_in.len(), lengths.len());
                    for (value, width) in ca_in.iter().zip(lengths.iter()) {
                        let mut is_in = false;
                        for is_in_value in other_exploded_iter
                            .by_ref()
                            .take(width.unwrap_or(0) as usize)
                        {
                            is_in |= value.tot_eq(&is_in_value);
                        }
                        // SAFETY: We allocated ca_in.len() items.
                        unsafe { bm.push_unchecked(is_in) };
                    }
                },
            }

            bm.freeze()
        },
        (1, r) => {
            let value = ca_in.get(0);

            let mut other_exploded_iter = other_exploded.iter();
            let mut bm = BitmapBuilder::with_capacity(r);

            match other_offsets {
                ListLengths::Array(width) => {
                    for _ in 0..r {
                        let mut is_in = false;
                        for is_in_value in other_exploded_iter.by_ref().take(width as usize) {
                            is_in |= value.tot_eq(&is_in_value);
                        }
                        // SAFETY: We allocated ca_in.len() items.
                        unsafe { bm.push_unchecked(is_in) };
                    }
                },
                ListLengths::List(lengths) => {
                    assert_eq!(other_length, lengths.len());
                    for width in lengths.iter() {
                        let mut is_in = false;
                        for is_in_value in other_exploded_iter
                            .by_ref()
                            .take(width.unwrap_or(0) as usize)
                        {
                            is_in |= value.tot_eq(&is_in_value);
                        }
                        // SAFETY: We allocated lengths.len() items.
                        unsafe { bm.push_unchecked(is_in) };
                    }
                },
            }

            bm.freeze()
        },
        (_, 1) => {
            let mut set = PlHashSet::with_capacity(other_exploded.len());
            other_exploded.iter().for_each(|opt_val| {
                if let Some(v) = opt_val {
                    set.insert(v.to_total_ord());
                }
            });

            return if nulls_equal {
                if other_exploded.has_nulls() {
                    // If the rhs has nulls, then nulls in the left set evaluates to true.
                    Ok(unary_elementwise(ca_in, |val| {
                        val.is_none_or(|v| set.contains(&v.to_total_ord()))
                    }))
                } else {
                    // The rhs has no nulls; nulls in the left evaluates to false.
                    Ok(unary_elementwise(ca_in, |val| {
                        val.is_some_and(|v| set.contains(&v.to_total_ord()))
                    }))
                }
            } else {
                Ok(
                    unary_elementwise_values(ca_in, |v| set.contains(&v.to_total_ord()))
                        .with_name(ca_in.name().clone()),
                )
            };
        },
        (l, r) => {
            polars_bail!(ShapeMismatch: "expected {l} elements in 'is_in' comparison, got {r}")
        },
    };

    let mut validity = None;
    if !nulls_equal {
        if ca_in.len() == 1 && other_length != 1 {
            validity = ca_in.has_nulls().then(|| Bitmap::new_zeroed(other_length));
        } else {
            validity = ca_in.rechunk_validity();
        }
    }
    let mask = BooleanArray::new(ArrowDataType::Boolean, mask, validity);
    Ok(BooleanChunked::from_chunk_iter(
        ca_in.name().clone(),
        [mask],
    ))
}

fn explode_other(ca_in_dtype: &DataType, other: &Series) -> PolarsResult<(Series, ListLengths)> {
    Ok(match other.dtype() {
        DataType::List(_) => {
            let other = other.list()?;
            let exploded = other.explode()?;
            (exploded, ListLengths::List(other.lst_lengths()))
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(_, width) => {
            let other = other.array()?;
            let exploded = other.explode()?;
            (
                exploded,
                ListLengths::Array(IdxSize::try_from(*width).unwrap()),
            )
        },
        _ => polars_bail!(op = "is_in", ca_in_dtype, other.dtype()),
    })
}

fn is_in_helper<T>(
    ca_in: &ChunkedArray<T>,
    other: &Series,
    nulls_equal: bool,
) -> PolarsResult<BooleanChunked>
where
    T: PolarsDataType<IsLogical = FalseT>,
    for<'a> T::Physical<'a>: TotalHash + TotalEq + Copy + ToTotalOrd,
    for<'a> <T::Physical<'a> as ToTotalOrd>::TotalOrdItem: Hash + Eq + Copy,
{
    let (exploded, lengths) = explode_other(ca_in.dtype(), other)?;
    let exploded = exploded.take_inner();
    let mut mask = is_in_helper_ca_list(ca_in, other.len(), exploded, lengths, nulls_equal)?;

    if other.has_nulls() {
        mask = mask.zip_with(
            &other.is_not_null(),
            &BooleanChunked::from_iter_options(PlSmallStr::EMPTY, [None].into_iter()),
        )?;
    }

    Ok(mask)
}

fn is_in_row_encoded(
    s: &Series,
    other: &Series,
    nulls_equal: bool,
) -> PolarsResult<BooleanChunked> {
    let ca_in = _get_rows_encoded_ca_unordered(s.name().clone(), &[s.clone().into_column()])?;

    // We check implicitly cast to supertype here
    let mut mask = match other.dtype() {
        DataType::List(_) => {
            let other = other.list()?;
            let exploded = other.explode()?;
            let exploded =
                _get_rows_encoded_ca_unordered(PlSmallStr::EMPTY, &[exploded.into_column()])?;
            is_in_helper_ca_list(
                &ca_in,
                other.len(),
                exploded,
                ListLengths::List(other.lst_lengths()),
                nulls_equal,
            )
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(_, width) => {
            let other = other.array()?;
            let exploded = other.explode()?;
            let exploded =
                _get_rows_encoded_ca_unordered(PlSmallStr::EMPTY, &[exploded.into_column()])?;
            is_in_helper_ca_list(
                &ca_in,
                other.len(),
                exploded,
                ListLengths::Array(IdxSize::try_from(*width).unwrap()),
                nulls_equal,
            )
        },
        _ => polars_bail!(op = "is_in", s.dtype(), other.dtype()),
    }?;

    if !nulls_equal {
        if s.has_nulls() {
            mask = mask.zip_with(
                &s.is_not_null(),
                &BooleanChunked::from_iter_options(PlSmallStr::EMPTY, [None].into_iter()),
            )?;
        }
    }
    if other.has_nulls() {
        mask = mask.zip_with(
            &other.is_not_null(),
            &BooleanChunked::from_iter_options(PlSmallStr::EMPTY, [None].into_iter()),
        )?;
    }

    Ok(mask)
}

// FIXME! I think this incorrect.
fn is_in_null(s: &Series, other: &Series, nulls_equal: bool) -> PolarsResult<BooleanChunked> {
    let output_length = match (s.len(), other.len()) {
        (l, r) if l == r => l,
        (1, r) => r,
        (l, 1) => l,
        (l, r) => {
            polars_bail!(ShapeMismatch: "expected {l} elements in 'is_in' comparison, got {r}")
        },
    };

    let (_, lengths) = explode_other(&DataType::Null, other)?;
    if nulls_equal {
        match lengths {
            ListLengths::Array(value) => Ok(BooleanChunked::from_iter_values(
                s.name().clone(),
                std::iter::repeat_n(value != 0, output_length),
            )),
            ListLengths::List(offsets) if offsets.len() == 1 => {
                Ok(BooleanChunked::from_iter_values(
                    s.name().clone(),
                    std::iter::repeat_n(offsets.get(0).unwrap() != 0, output_length),
                ))
            },
            ListLengths::List(offsets) => Ok(BooleanChunked::from_iter_values(
                s.name().clone(),
                offsets.into_iter().map(|v| v.unwrap() != 0),
            )),
        }
    } else {
        let out = s.cast(&DataType::Boolean)?;
        let ca_bool = out.bool()?.clone();
        Ok(ca_bool)
    }
}

pub fn is_in(s: &Series, other: &Series, nulls_equal: bool) -> PolarsResult<BooleanChunked> {
    let Some(other_inner_dtype) = other.dtype().inner_dtype() else {
        polars_bail!(op = "is_in", s.dtype(), other.dtype());
    };

    // @HACK. I hate this as much as the next guy, but we don't really have a choice because local
    // categoricals are a bit broken.
    if other_inner_dtype.is_string() {
        if let DataType::Categorical(Some(rm), _) = s.dtype() {
            if let RevMapping::Local(categories, _) = rm.as_ref() {
                let ca = s.categorical().unwrap();
                let (exploded, lengths) = explode_other(s.dtype(), other)?;
                let exploded = exploded.str().unwrap();

                assert!(categories.len() < u32::MAX as usize);

                let categories_map = PlHashMap::from_iter(
                    categories
                        .values_iter()
                        .enumerate_u32()
                        .map(|(k, v)| (v, k)),
                );
                let exploded = unary_elementwise(exploded, |v| {
                    v.map(|v| categories_map.get(v).copied().unwrap_or(u32::MAX))
                });

                let mut mask = is_in_helper_ca_list(
                    ca.physical(),
                    other.len(),
                    exploded,
                    lengths,
                    nulls_equal,
                )?;
                if other.has_nulls() {
                    mask = mask.zip_with(
                        &other.is_not_null(),
                        &BooleanChunked::from_iter_options(PlSmallStr::EMPTY, [None].into_iter()),
                    )?;
                }
                return Ok(mask);
            }
        }
    }
    if s.dtype().is_string() {
        if let DataType::Categorical(Some(rm), _) = other_inner_dtype {
            if let RevMapping::Local(categories, _) = rm.as_ref() {
                let ca = s.str().unwrap();
                let (exploded, lengths) = explode_other(s.dtype(), other)?;
                let exploded = exploded.categorical().unwrap();

                assert!(categories.len() < u32::MAX as usize);

                let categories_map = PlHashMap::from_iter(
                    categories
                        .values_iter()
                        .enumerate_u32()
                        .map(|(k, v)| (v, k)),
                );
                let ca = unary_elementwise(ca, |v| {
                    v.map(|v| categories_map.get(v).copied().unwrap_or(u32::MAX))
                });

                let mut mask = is_in_helper_ca_list(
                    &ca,
                    other.len(),
                    exploded.physical().clone(),
                    lengths,
                    nulls_equal,
                )?;
                if other.has_nulls() {
                    mask = mask.zip_with(
                        &other.is_not_null(),
                        &BooleanChunked::from_iter_options(PlSmallStr::EMPTY, [None].into_iter()),
                    )?;
                }
                return Ok(mask);
            }
        }
    }

    // For eager execution, we need to cast here.
    let mut other = Cow::Borrowed(other);
    if other_inner_dtype != s.dtype() {
        other = Cow::Owned(match other.dtype() {
            DataType::List(_) => other
                .as_ref()
                .cast(&DataType::List(Box::new(s.dtype().clone())))?,
            DataType::Array(_, size) => other
                .as_ref()
                .cast(&DataType::Array(Box::new(s.dtype().clone()), *size))?,
            _ => unreachable!(),
        });
    }
    let other = other.as_ref();

    match s.dtype() {
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(_, _) | DataType::Enum(_, _) => {
            let ca = s.categorical().unwrap();
            let (exploded, lengths) = explode_other(s.dtype(), other)?;
            let exploded = exploded.categorical().unwrap();
            let (ca, exploded) = make_categoricals_compatible(ca, exploded)?;

            let mut mask = is_in_helper_ca_list(
                ca.physical(),
                other.len(),
                exploded.into_physical(),
                lengths,
                nulls_equal,
            )?;
            if other.has_nulls() {
                mask = mask.zip_with(
                    &other.is_not_null(),
                    &BooleanChunked::from_iter_options(PlSmallStr::EMPTY, [None].into_iter()),
                )?;
            }
            Ok(mask)
        },

        #[cfg(feature = "dtype-struct")]
        DataType::Struct(_) => is_in_row_encoded(s, other, nulls_equal),
        #[cfg(feature = "dtype-array")]
        DataType::Array(_, _) => is_in_row_encoded(s, other, nulls_equal),
        DataType::List(_) => is_in_row_encoded(s, other, nulls_equal),

        DataType::String => {
            let ca = s.str().unwrap();
            is_in_helper(&ca, other, nulls_equal)
        },
        DataType::Binary => {
            let ca = s.binary().unwrap();
            is_in_helper(ca, other, nulls_equal)
        },
        DataType::Boolean => {
            let ca = s.bool().unwrap();
            is_in_helper(ca, other, nulls_equal)
        },
        DataType::Null => is_in_null(s, other, nulls_equal),
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(_, _) => {
            // Oh, decimal. Why to do you always have be so quirky and different.
            let (exploded, lengths) = explode_other(s.dtype(), other)?;
            let s = s.decimal()?;
            let exploded = exploded.decimal()?;

            let scale = s.scale().max(exploded.scale());
            let s = s.to_scale(scale)?;
            let exploded = exploded.to_scale(scale)?;

            let mut mask = is_in_helper_ca_list(
                s.physical(),
                other.len(),
                exploded.physical().clone(),
                lengths,
                nulls_equal,
            )?;
            if other.has_nulls() {
                mask = mask.zip_with(
                    &other.is_not_null(),
                    &BooleanChunked::from_iter_options(PlSmallStr::EMPTY, [None].into_iter()),
                )?;
            }
            Ok(mask)
        },
        dt if dt.to_physical().is_primitive_numeric() => {
            let s = s.to_physical_repr();
            with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                is_in_helper(ca, other.to_physical_repr().as_ref(), nulls_equal)
            })
        },
        dt => polars_bail!(opq = is_in, dt),
    }
}
