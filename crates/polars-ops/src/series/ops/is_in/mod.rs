use std::hash::Hash;

use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::{Bitmap, BitmapBuilder};
use polars_core::prelude::*;
use polars_core::{with_match_categorical_physical_type, with_match_physical_numeric_polars_type};
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash};

use self::binary::{BinaryLookup, RowEncodedLookup};
use self::primitive::{PrimitiveLookup, PrimitiveProbe};
use self::row_encode::_get_rows_encoded_ca_unordered;

mod binary;
mod primitive;

/// Haystacks with at most this many values are probed with a linear scan.
const SMALL_MAX: usize = 8;

/// Pushes one bit per value; `values.len()` bits in total.
#[inline(always)]
fn probe_words<V: Copy>(values: &[V], out: &mut BitmapBuilder, f: impl Fn(V) -> bool) {
    out.reserve(values.len());
    let mut chunks = values.chunks_exact(64);
    for chunk in &mut chunks {
        let mut word = 0u64;
        for (i, &v) in chunk.iter().enumerate() {
            word |= (f(v) as u64) << i;
        }
        // SAFETY: reserved above.
        unsafe { out.push_word_with_len_unchecked(word, 64) };
    }
    let rem = chunks.remainder();
    if !rem.is_empty() {
        let mut word = 0u64;
        for (i, &v) in rem.iter().enumerate() {
            word |= (f(v) as u64) << i;
        }
        // SAFETY: reserved above.
        unsafe { out.push_word_with_len_unchecked(word, rem.len()) };
    }
}

/// Combines the membership of the values with the validity of the needles.
fn finish_chunk(
    values: Bitmap,
    validity: Option<&Bitmap>,
    nulls_equal: bool,
    has_null: bool,
) -> BooleanArray {
    match validity {
        None => BooleanArray::new(ArrowDataType::Boolean, values, None),
        Some(validity) if nulls_equal => {
            let values = if has_null {
                polars_arrow::bitmap::or_not(&values, validity)
            } else {
                polars_arrow::bitmap::and(&values, validity)
            };
            BooleanArray::new(ArrowDataType::Boolean, values, None)
        },
        Some(validity) => BooleanArray::new(ArrowDataType::Boolean, values, Some(validity.clone())),
    }
}

enum Lookup {
    /// The haystack itself is null.
    OuterNull,
    /// Only the null count of the haystack matters.
    NullNeedle,
    Primitive(Box<dyn PrimitiveProbe>),
    Binary(BinaryLookup),
    RowEncoded(RowEncodedLookup),
    Boolean {
        has_true: bool,
        has_false: bool,
    },
    #[cfg(feature = "dtype-decimal")]
    Decimal {
        prec: usize,
        scale: usize,
        lookup: PrimitiveLookup<Int128Type>,
    },
}

/// A single haystack prepared for probing with needles of a fixed dtype.
pub struct IsInHaystack {
    lookup: Lookup,
    has_null: bool,
    needle_dtype: DataType,
}

#[cfg(feature = "dtype-decimal")]
const DECIMAL_SENTINEL_NEEDLE: i128 = i128::MAX;
#[cfg(feature = "dtype-decimal")]
const DECIMAL_SENTINEL_HAYSTACK: i128 = i128::MAX - 1;

impl IsInHaystack {
    /// `haystack` must be a `List` or `Array` series of length 1.
    pub fn new(haystack: &Series, needle_dtype: &DataType) -> PolarsResult<Self> {
        let is_container = matches!(haystack.dtype(), DataType::List(_));
        #[cfg(feature = "dtype-array")]
        let is_container = is_container || matches!(haystack.dtype(), DataType::Array(..));
        polars_ensure!(is_container, opq = is_in, needle_dtype, haystack.dtype());
        assert_eq!(haystack.len(), 1);

        if haystack.has_nulls() {
            return Ok(Self {
                lookup: Lookup::OuterNull,
                has_null: false,
                needle_dtype: needle_dtype.clone(),
            });
        }

        let flat = haystack.explode(ExplodeOptions {
            empty_as_null: false,
            keep_nulls: true,
        })?;
        let has_null = flat.has_nulls();
        let mismatch = || polars_err!(opq = is_in, needle_dtype, haystack.dtype());

        let lookup = match needle_dtype {
            #[cfg(feature = "dtype-categorical")]
            dt @ (DataType::Categorical(_, _) | DataType::Enum(_, _)) => {
                with_match_categorical_physical_type!(dt.cat_physical().unwrap(), |$C| {
                    let phys = categorical_haystack::<$C>(dt, haystack.dtype(), &flat)?;
                    Lookup::Primitive(Box::new(PrimitiveLookup::<<$C as PolarsCategoricalType>::PolarsPhysical>::new(&phys)))
                })
            },
            DataType::String => {
                let flat = match flat.dtype() {
                    DataType::String => flat,
                    #[cfg(feature = "dtype-categorical")]
                    DataType::Enum(_, _) | DataType::Categorical(_, _) => {
                        flat.cast(&DataType::String)?
                    },
                    _ => return Err(mismatch()),
                };
                let ca = flat.str().unwrap().as_binary();
                Lookup::Binary(BinaryLookup::new(
                    ca.downcast_iter()
                        .flat_map(|arr| arr.non_null_values_iter()),
                ))
            },
            DataType::Binary => {
                let ca = flat.binary().map_err(|_| mismatch())?;
                Lookup::Binary(BinaryLookup::new(
                    ca.downcast_iter()
                        .flat_map(|arr| arr.non_null_values_iter()),
                ))
            },
            DataType::Boolean => {
                let ca = flat.bool().map_err(|_| mismatch())?;
                let num_true = ca.sum().unwrap_or(0) as usize;
                Lookup::Boolean {
                    has_true: num_true > 0,
                    has_false: ca.len() - ca.null_count() > num_true,
                }
            },
            DataType::Null => Lookup::NullNeedle,
            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(needle_prec, needle_scale) => {
                let ca = flat.decimal().map_err(|_| mismatch())?;
                let prec = (*needle_prec).max(ca.precision());
                let scale = (*needle_scale).max(ca.scale());
                let phys = ca.into_phys_with_prec_scale_or_sentinel(
                    prec,
                    scale,
                    DECIMAL_SENTINEL_HAYSTACK,
                );
                Lookup::Decimal {
                    prec,
                    scale,
                    lookup: PrimitiveLookup::new(&phys),
                }
            },
            dt if dt.is_nested() => {
                let encoded =
                    _get_rows_encoded_ca_unordered(PlSmallStr::EMPTY, &[flat.into_column()])?;
                let rows = encoded.rechunk().downcast_as_array().clone();
                Lookup::RowEncoded(RowEncodedLookup::new(rows))
            },
            dt if dt.to_physical().is_primitive_numeric() => {
                let flat = flat.to_physical_repr();
                polars_ensure!(
                    flat.dtype() == &dt.to_physical(),
                    opq = is_in,
                    dt,
                    haystack.dtype()
                );
                with_match_physical_numeric_polars_type!(flat.dtype(), |$T| {
                    let ca: &ChunkedArray<$T> = flat.as_ref().as_ref().as_ref();
                    Lookup::Primitive(Box::new(PrimitiveLookup::<$T>::new(ca)))
                })
            },
            dt => polars_bail!(opq = is_in, dt),
        };

        Ok(Self {
            lookup,
            has_null,
            needle_dtype: needle_dtype.clone(),
        })
    }

    pub fn probe(&self, needle: &Series, nulls_equal: bool) -> PolarsResult<BooleanChunked> {
        polars_ensure!(
            needle.dtype() == &self.needle_dtype,
            SchemaMismatch: "is_in: expected needle of dtype {}, got {}",
            self.needle_dtype, needle.dtype()
        );
        let name = needle.name().clone();
        let has_null = self.has_null;

        let out = match &self.lookup {
            Lookup::OuterNull => BooleanChunked::full_null(name, needle.len()),
            Lookup::NullNeedle => {
                if nulls_equal {
                    BooleanChunked::full(name, has_null, needle.len())
                } else {
                    BooleanChunked::full_null(name, needle.len())
                }
            },
            Lookup::Primitive(lookup) => {
                lookup.probe_series(&needle.to_physical_repr(), nulls_equal, has_null)
            },
            Lookup::Binary(lookup) => match needle.dtype() {
                DataType::String => {
                    lookup.probe(&needle.str().unwrap().as_binary(), nulls_equal, has_null)
                },
                _ => lookup.probe(needle.binary().unwrap(), nulls_equal, has_null),
            },
            Lookup::RowEncoded(lookup) => {
                let encoded =
                    _get_rows_encoded_ca_unordered(name, &[needle.clone().into_column()])?;
                let mut out = lookup.probe(&encoded, nulls_equal, has_null);
                if !nulls_equal {
                    out.with_validities(&[needle.rechunk_validity()]);
                }
                out
            },
            Lookup::Boolean {
                has_true,
                has_false,
            } => {
                let ca = needle.bool().unwrap();
                let chunks = ca.downcast_iter().map(|arr| {
                    let values = match (has_true, has_false) {
                        (true, true) => Bitmap::new_with_value(true, arr.len()),
                        (true, false) => arr.values().clone(),
                        (false, true) => !arr.values(),
                        (false, false) => Bitmap::new_with_value(false, arr.len()),
                    };
                    finish_chunk(values, arr.validity(), nulls_equal, has_null)
                });
                BooleanChunked::from_chunk_iter(name, chunks)
            },
            #[cfg(feature = "dtype-decimal")]
            Lookup::Decimal {
                prec,
                scale,
                lookup,
            } => {
                let phys = needle
                    .decimal()
                    .unwrap()
                    .into_phys_with_prec_scale_or_sentinel(*prec, *scale, DECIMAL_SENTINEL_NEEDLE);
                lookup.probe(&phys, nulls_equal, has_null)
            },
        };
        Ok(out)
    }
}

/// Maps a String, Categorical or Enum haystack to the physical categories of the needle.
#[cfg(feature = "dtype-categorical")]
fn categorical_haystack<T: PolarsCategoricalType>(
    needle_dtype: &DataType,
    haystack_dtype: &DataType,
    flat: &Series,
) -> PolarsResult<ChunkedArray<T::PolarsPhysical>> {
    let out = match (needle_dtype, flat.dtype()) {
        (DataType::Enum(_, mapping) | DataType::Categorical(_, mapping), DataType::String) => {
            let ca = flat.str().unwrap();
            // Strings without a category can never match; only nulls stay null.
            ca.iter()
                .filter_map(|opt_s| match opt_s {
                    None => Some(None),
                    Some(s) => mapping.get_cat(s).map(|c| Some(T::Native::from_cat(c))),
                })
                .collect_ca(PlSmallStr::EMPTY)
        },
        (DataType::Categorical(lcats, _), DataType::Categorical(rcats, _)) => {
            ensure_same_categories(lcats, rcats)?;
            flat.cat::<T>().unwrap().physical().clone()
        },
        (DataType::Enum(lfcats, _), DataType::Enum(rfcats, _)) => {
            ensure_same_frozen_categories(lfcats, rfcats)?;
            flat.cat::<T>().unwrap().physical().clone()
        },
        _ => polars_bail!(opq = is_in, needle_dtype, haystack_dtype),
    };
    Ok(out)
}

fn is_in_helper_list_ca<'a, T>(
    ca_in: &'a ChunkedArray<T>,
    other: &'a ListChunked,
    nulls_equal: bool,
) -> PolarsResult<BooleanChunked>
where
    T: PolarsPhysicalType,
    for<'b> T::Physical<'b>: TotalHash + TotalEq + ToTotalOrd + Copy,
    for<'b> <T::Physical<'b> as ToTotalOrd>::TotalOrdItem: Hash + Eq + Copy,
{
    debug_assert_ne!(other.len(), 1);
    let offsets = other.offsets()?;
    let inner = other.get_inner();
    let inner: &ChunkedArray<T> = inner.as_ref().as_ref();
    let validity = other.rechunk_validity();

    let mut ca: BooleanChunked = if ca_in.len() == 1 {
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
                    (Some(l), Some(r)) => Some(polars_arrow::bitmap::and(&l, &r)),
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
    T: PolarsPhysicalType,
    for<'b> T::Physical<'b>: TotalHash + TotalEq + ToTotalOrd + Copy,
    for<'b> <T::Physical<'b> as ToTotalOrd>::TotalOrdItem: Hash + Eq + Copy,
{
    debug_assert_ne!(other.len(), 1);
    let width = other.width();
    let inner = other.get_inner();
    let inner: &ChunkedArray<T> = inner.as_ref().as_ref();
    let validity = other.rechunk_validity();

    let mut ca: BooleanChunked = if ca_in.len() == 1 {
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
                    (Some(l), Some(r)) => Some(polars_arrow::bitmap::and(&l, &r)),
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
        DataType::List(..) => is_in_helper_list_ca(ca_in, other.list()?, nulls_equal),
        #[cfg(feature = "dtype-array")]
        DataType::Array(..) => is_in_helper_array_ca(ca_in, other.array()?, nulls_equal),
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
            is_in_helper_list_ca(ca_in, other.list()?, nulls_equal)
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(dt, _) if DataType::Binary == **dt => {
            is_in_helper_array_ca(ca_in, other.array()?, nulls_equal)
        },
        _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
    }
}

fn is_in_boolean(
    ca_in: &BooleanChunked,
    other: &Series,
    nulls_equal: bool,
) -> PolarsResult<BooleanChunked> {
    match other.dtype() {
        DataType::List(dt) if ca_in.dtype() == &**dt => {
            is_in_helper_list_ca(ca_in, other.list()?, nulls_equal)
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(dt, _) if ca_in.dtype() == &**dt => {
            is_in_helper_array_ca(ca_in, other.array()?, nulls_equal)
        },
        _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
    }
}

#[cfg(feature = "dtype-categorical")]
fn is_in_cat_and_enum<T: PolarsCategoricalType>(
    ca_in: &CategoricalChunked<T>,
    other: &Series,
    nulls_equal: bool,
) -> PolarsResult<BooleanChunked>
where
    T::Native: ToTotalOrd<TotalOrdItem = T::Native>,
{
    let to_categories = match (ca_in.dtype(), other.dtype().inner_dtype().unwrap()) {
        (DataType::Enum(_, mapping) | DataType::Categorical(_, mapping), DataType::String) => {
            (&|s: Series| {
                let ca = s.str()?;
                let ca: ChunkedArray<T::PolarsPhysical> = ca
                    .iter()
                    .map(|opt_s| opt_s.and_then(|s| mapping.get_cat(s).map(T::Native::from_cat)))
                    .collect_ca(PlSmallStr::EMPTY);
                Ok(ca.into_series())
            }) as _
        },
        (DataType::Categorical(lcats, _), DataType::Categorical(rcats, _)) => {
            ensure_same_categories(lcats, rcats)?;
            (&|s: Series| Ok(s.cat::<T>()?.physical().clone().into_series())) as _
        },
        (DataType::Enum(lfcats, _), DataType::Enum(rfcats, _)) => {
            ensure_same_frozen_categories(lfcats, rfcats)?;
            (&|s: Series| Ok(s.cat::<T>()?.physical().clone().into_series())) as _
        },
        _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
    };

    let other = match other.dtype() {
        DataType::List(_) => other.list()?.apply_to_inner(to_categories)?.into_series(),
        #[cfg(feature = "dtype-array")]
        DataType::Array(_, _) => other.array()?.apply_to_inner(to_categories)?.into_series(),
        _ => polars_bail!(opq = is_in, ca_in.dtype(), other.dtype()),
    };

    is_in_numeric(ca_in.physical(), &other, nulls_equal)
}

fn is_in_null(s: &Series, other: &Series, nulls_equal: bool) -> PolarsResult<BooleanChunked> {
    if nulls_equal {
        let ca_in = s.null()?;
        Ok(match other.dtype() {
            DataType::List(_) => other.list()?.apply_amortized_generic(|opt_s| {
                Some(opt_s.map(|s| s.as_ref().has_nulls()) == Some(true))
            }),
            #[cfg(feature = "dtype-array")]
            DataType::Array(_, _) => other.array()?.apply_amortized_generic(|opt_s| {
                Some(opt_s.map(|s| s.as_ref().has_nulls()) == Some(true))
            }),
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
    let Some(DataType::Decimal(other_precision, other_scale)) = other.dtype().inner_dtype() else {
        polars_bail!(opq = is_in, ca_in.dtype(), other.dtype());
    };
    let prec = ca_in.precision().max(*other_precision);
    let scale = ca_in.scale().max(*other_scale);

    // We convert both sides to a common scale, mapping any out-of-range values to unique integers,
    // allowing us to then use is_in on the integer representation.
    let ca_in_phys =
        ca_in.into_phys_with_prec_scale_or_sentinel(prec, scale, DECIMAL_SENTINEL_NEEDLE);

    match other.dtype() {
        DataType::List(_) => {
            let other = other.list()?;
            let other = other.apply_to_inner(&|s| {
                let s = s.decimal()?;
                let s =
                    s.into_phys_with_prec_scale_or_sentinel(prec, scale, DECIMAL_SENTINEL_HAYSTACK);
                Ok(s.to_owned().into_series())
            })?;
            let other = other.into_series();
            is_in_numeric(&ca_in_phys, &other, nulls_equal)
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(_, _) => {
            let other = other.array()?;
            let other = other.apply_to_inner(&|s| {
                let s = s.decimal()?;
                let s =
                    s.into_phys_with_prec_scale_or_sentinel(prec, scale, DECIMAL_SENTINEL_HAYSTACK);
                Ok(s.to_owned().into_series())
            })?;
            let other = other.into_series();
            is_in_numeric(&ca_in_phys, &other, nulls_equal)
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
            is_in_helper_list_ca(&ca_in, &other, nulls_equal)
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
            is_in_helper_array_ca(&ca_in, &other, nulls_equal)
        },
        _ => unreachable!(),
    }?;

    let mut validity = other.rechunk_validity();
    if !nulls_equal {
        validity = match (validity, s.rechunk_validity()) {
            (None, None) => None,
            (Some(v), None) | (None, Some(v)) => Some(v),
            (Some(l), Some(r)) => Some(polars_arrow::bitmap::and(&l, &r)),
        };
    }

    assert_eq!(mask.null_count(), 0);
    mask.with_validities(&[validity]);

    Ok(mask)
}

pub fn is_in(
    needle: &Series,
    haystack: &Series,
    nulls_equal: bool,
) -> PolarsResult<BooleanChunked> {
    polars_ensure!(
        needle.len() == haystack.len() || needle.len() == 1 || haystack.len() == 1,
        length_mismatch = "is_in",
        needle.len(),
        haystack.len()
    );

    #[allow(unused_mut)]
    let mut other_is_valid_type = matches!(haystack.dtype(), DataType::List(_));
    #[cfg(feature = "dtype-array")]
    {
        other_is_valid_type |= matches!(haystack.dtype(), DataType::Array(..))
    }
    polars_ensure!(
        other_is_valid_type,
        opq = is_in,
        needle.dtype(),
        haystack.dtype()
    );

    if haystack.len() == 1 {
        return IsInHaystack::new(haystack, needle.dtype())?.probe(needle, nulls_equal);
    }

    match needle.dtype() {
        #[cfg(feature = "dtype-categorical")]
        dt @ DataType::Categorical(_, _) | dt @ DataType::Enum(_, _) => {
            with_match_categorical_physical_type!(dt.cat_physical().unwrap(), |$C| {
                is_in_cat_and_enum(needle.cat::<$C>().unwrap(), haystack, nulls_equal)
            })
        },
        DataType::String => {
            let ca = needle.str().unwrap();
            is_in_string(ca, haystack, nulls_equal)
        },
        DataType::Binary => {
            let ca = needle.binary().unwrap();
            is_in_binary(ca, haystack, nulls_equal)
        },
        DataType::Boolean => {
            let ca = needle.bool().unwrap();
            is_in_boolean(ca, haystack, nulls_equal)
        },
        DataType::Null => is_in_null(needle, haystack, nulls_equal),
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(_, _) => {
            let ca_in = needle.decimal()?;
            is_in_decimal(ca_in, haystack, nulls_equal)
        },
        dt if dt.is_nested() => is_in_row_encoded(needle, haystack, nulls_equal),
        dt if dt.to_physical().is_primitive_numeric() => {
            let s = needle.to_physical_repr();
            let other = haystack.to_physical_repr();
            let other = other.as_ref();
            with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                is_in_numeric(ca, other, nulls_equal)
            })
        },
        dt => polars_bail!(opq = is_in, dt),
    }
}
