//! Implementations of the ChunkCast Trait.

use std::borrow::Cow;

use polars_compute::cast::CastOptionsImpl;
#[cfg(feature = "serde-lazy")]
use serde::{Deserialize, Serialize};

use super::flags::StatisticsFlags;
#[cfg(feature = "dtype-datetime")]
use crate::prelude::DataType::Datetime;
use crate::chunked_array::arrow_bridge::as_flat;
use crate::prelude::*;
use crate::utils::{handle_array_casting_failures, handle_casting_failures};

#[derive(Copy, Clone, Debug, Default, PartialEq, Hash, Eq)]
#[cfg_attr(feature = "serde-lazy", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[repr(u8)]
pub enum CastOptions {
    /// Raises on overflow
    #[default]
    Strict,
    /// Overflow is replaced with null
    NonStrict,
    /// Allows wrapping overflow
    Overflowing,
}

impl CastOptions {
    pub fn is_strict(&self) -> bool {
        matches!(self, CastOptions::Strict)
    }
}

impl From<CastOptions> for CastOptionsImpl {
    fn from(value: CastOptions) -> Self {
        let wrapped = match value {
            CastOptions::Strict | CastOptions::NonStrict => false,
            CastOptions::Overflowing => true,
        };
        CastOptionsImpl {
            wrapped,
            partial: false,
        }
    }
}

/// Casts the chunks of a [`ChunkedArray`] to `dtype`, through the Arrow cast kernel.
///
/// Each chunk crosses to Arrow and back, which is `O(1)` in each direction — see
/// [`with_arrow_chunk`](crate::chunked_array::arrow_bridge::with_arrow_chunk) — so what the cast
/// costs is the cast itself.
pub(crate) fn cast_chunks(
    chunks: &[PlArrayRef],
    dtype: &DataType,
    options: CastOptions,
) -> PolarsResult<Vec<PlArrayRef>> {
    let arrow_chunks: Vec<ArrayRef> = chunks
        .iter()
        .map(|chunk| polars_array::arrow::export::to_arrow(&**chunk))
        .collect();
    let cast = cast_arrow_chunks(&arrow_chunks, dtype, options)?;
    Ok(crate::chunked_array::from::import_arrow_chunks(cast))
}

/// Casts Arrow chunks to `dtype`, which is what the boundaries where data arrives as Arrow use.
pub(crate) fn cast_arrow_chunks(
    chunks: &[ArrayRef],
    dtype: &DataType,
    options: CastOptions,
) -> PolarsResult<Vec<ArrayRef>> {
    let check_nulls = matches!(options, CastOptions::Strict);
    let options = options.into();

    let arrow_dtype = dtype.try_to_arrow(CompatLevel::newest())?;
    chunks
        .iter()
        .map(|arr| {
            let out = polars_compute::cast::cast(arr.as_ref(), &arrow_dtype, options);
            if check_nulls {
                out.and_then(|new| {
                    if arr.null_count() != new.null_count() {
                        handle_array_casting_failures(&**arr, &*new)?;
                    }
                    Ok(new)
                })
            } else {
                out
            }
        })
        .collect::<PolarsResult<Vec<_>>>()
}

fn cast_impl_inner(
    name: PlSmallStr,
    chunks: &[PlArrayRef],
    dtype: &DataType,
    options: CastOptions,
) -> PolarsResult<Series> {
    let chunks = match dtype {
        // @NOTE: We cast to the decimal itself rather than to its physical type, as casting to
        // the physical type would lower the scale. The chunks carry no logical type, so what is
        // left of the decimal after the cast is the `i128` it is stored as.
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(_, _) => cast_chunks(chunks, dtype, options)?,
        _ => cast_chunks(chunks, &dtype.to_physical(), options)?,
    };

    // SAFETY: the chunks were just cast to the physical type of `dtype`.
    let out = unsafe {
        Series::from_chunks_and_dtype_unchecked(name, chunks, &dtype.to_physical())
    };
    use DataType::*;
    let out = match dtype {
        Date => out.into_date(),
        Datetime(tu, tz) => match tz {
            #[cfg(feature = "timezones")]
            Some(tz) => {
                TimeZone::validate_time_zone(tz)?;
                out.into_datetime(*tu, Some(tz.clone()))
            },
            _ => out.into_datetime(*tu, None),
        },
        Duration(tu) => out.into_duration(*tu),
        #[cfg(feature = "dtype-time")]
        Time => out.into_time(),
        #[cfg(feature = "dtype-decimal")]
        Decimal(precision, scale) => out.into_decimal(*precision, *scale)?,
        #[cfg(feature = "dtype-extension")]
        Extension(typ, _) => out.into_extension(typ.clone()),
        _ => out,
    };

    Ok(out)
}

fn cast_impl(
    name: PlSmallStr,
    chunks: &[PlArrayRef],
    dtype: &DataType,
    options: CastOptions,
) -> PolarsResult<Series> {
    cast_impl_inner(name, chunks, dtype, options)
}

#[cfg(feature = "dtype-struct")]
fn cast_single_to_struct(
    name: PlSmallStr,
    chunks: &[PlArrayRef],
    fields: &[Field],
    options: CastOptions,
) -> PolarsResult<Series> {
    polars_ensure!(fields.len() == 1, InvalidOperation: "must specify one field in the struct");
    let mut new_fields = Vec::with_capacity(fields.len());
    // cast to first field dtype
    let mut fields = fields.iter();
    let fld = fields.next().unwrap();
    let s = cast_impl_inner(fld.name.clone(), chunks, &fld.dtype, options)?;
    let length = s.len();
    new_fields.push(s);

    for fld in fields {
        new_fields.push(Series::full_null(fld.name.clone(), length, &fld.dtype));
    }

    StructChunked::from_series(name, length, new_fields.iter()).map(|ca| ca.into_series())
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn cast_impl(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        if self.dtype() == dtype {
            // SAFETY: chunks are correct dtype
            let mut out = unsafe {
                Series::from_chunks_and_dtype_unchecked(
                    self.name().clone(),
                    self.chunks.clone(),
                    dtype,
                )
            };
            out.set_sorted_flag(self.is_sorted_flag());
            return Ok(out);
        }
        match dtype {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(..) | DataType::Enum(..) => {
                polars_bail!(
                    ComputeError:
                    "casting from {} to {dtype} is not supported.\n\
                    Instead of `.cast({dtype:?}`, use `.cat.to({dtype:?})`.",
                    T::get_static_dtype()
                );
            },

            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => {
                cast_single_to_struct(self.name().clone(), &self.chunks, fields, options)
            },
            _ => cast_impl_inner(self.name().clone(), &self.chunks, dtype, options).map(|mut s| {
                // maintain sorted if data types
                // - remain signed
                // - unsigned -> signed
                // this may still fail with overflow?
                let to_signed = dtype.is_signed_integer();
                let unsigned2unsigned =
                    self.dtype().is_unsigned_integer() && dtype.is_unsigned_integer();
                let allowed = to_signed || unsigned2unsigned;

                if (allowed)
                    && (s.null_count() == self.null_count())
                    // physical to logicals
                    || (self.dtype().to_physical() == dtype.to_physical())
                {
                    let is_sorted = self.is_sorted_flag();
                    s.set_sorted_flag(is_sorted)
                }
                s
            }),
        }
    }
}

impl<T> ChunkCast for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        self.cast_impl(dtype, options)
    }

    unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Series> {
        match dtype {
            // LEGACY
            // TODO @ cat-rework: remove after exposing to/from physical functions.
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(cats, _mapping) => {
                polars_ensure!(self.dtype() == &cats.physical().dtype(), ComputeError: "cannot cast numeric types to 'Categorical'");
                with_match_categorical_physical_type!(cats.physical(), |$C| {
                    // SAFETY: we are guarded by the type system.
                    type PhysCa = ChunkedArray<<$C as PolarsCategoricalType>::PolarsPhysical>;
                    let ca = unsafe { &*(self as *const ChunkedArray<T> as *const PhysCa) };
                    Ok(CategoricalChunked::<$C>::from_cats_and_dtype_unchecked(ca.clone(), dtype.clone())
                        .into_series())
                })
            },

            // LEGACY
            // TODO @ cat-rework: remove after exposing to/from physical functions.
            #[cfg(feature = "dtype-categorical")]
            DataType::Enum(fcats, _mapping) => {
                polars_ensure!(self.dtype() == &fcats.physical().dtype(), ComputeError: "cannot cast numeric types to 'Enum'");
                with_match_categorical_physical_type!(fcats.physical(), |$C| {
                    // SAFETY: we are guarded by the type system.
                    type PhysCa = ChunkedArray<<$C as PolarsCategoricalType>::PolarsPhysical>;
                    let ca = unsafe { &*(self as *const ChunkedArray<T> as *const PhysCa) };
                    Ok(CategoricalChunked::<$C>::from_cats_and_dtype_unchecked(ca.clone(), dtype.clone()).into_series())
                })
            },

            _ => self.cast_impl(dtype, CastOptions::Overflowing),
        }
    }
}

impl ChunkCast for StringChunked {
    fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        match dtype {
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(cats, _mapping) => {
                with_match_categorical_physical_type!(cats.physical(), |$C| {
                    Ok(CategoricalChunked::<$C>::from_str_iter(self.name().clone(), dtype.clone(), self.iter())?
                        .into_series())
                })
            },
            #[cfg(feature = "dtype-categorical")]
            DataType::Enum(fcats, _mapping) => {
                let ret = with_match_categorical_physical_type!(fcats.physical(), |$C| {
                    CategoricalChunked::<$C>::from_str_iter(self.name().clone(), dtype.clone(), self.iter())?
                        .into_series()
                });

                if options.is_strict() && self.null_count() != ret.null_count() {
                    handle_casting_failures(&self.clone().into_series(), &ret)?;
                }

                Ok(ret)
            },
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => {
                cast_single_to_struct(self.name().clone(), &self.chunks, fields, options)
            },
            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(precision, scale) => {
                let chunks = self.downcast_iter().map(|arr| {
                    let arr = <PlUtf8ViewArray as ToArrow>::to_arrow(&as_flat(arr)).to_binview();
                    let arr = polars_compute::cast::binview_to_decimal(&arr, *precision, *scale);
                    polars_array::arrow::import::primitive_from_arrow(&arr)
                });
                let ca = Int128Chunked::from_chunk_iter(self.name().clone(), chunks);
                Ok(ca.into_decimal_unchecked(*precision, *scale).into_series())
            },
            #[cfg(feature = "dtype-date")]
            DataType::Date => {
                let result = cast_chunks(&self.chunks, dtype, options)?;
                // SAFETY: the chunks were just cast to the physical type of a date.
                Ok(unsafe {
                    Series::from_chunks_and_dtype_unchecked(self.name().clone(), result, dtype)
                })
            },
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(time_unit, time_zone) => match time_zone {
                #[cfg(feature = "timezones")]
                Some(time_zone) => {
                    TimeZone::validate_time_zone(time_zone)?;
                    let dtype = Datetime(time_unit.to_owned(), Some(time_zone.clone()));
                    let result = cast_chunks(&self.chunks, &dtype, options)?;
                    // SAFETY: the chunks were just cast to the physical type of a datetime.
                    Ok(unsafe {
                        Series::from_chunks_and_dtype_unchecked(self.name().clone(), result, &dtype)
                    })
                },
                _ => {
                    let dtype = Datetime(time_unit.to_owned(), None);
                    let result = cast_chunks(&self.chunks, &dtype, options)?;
                    // SAFETY: as above.
                    Ok(unsafe {
                        Series::from_chunks_and_dtype_unchecked(self.name().clone(), result, &dtype)
                    })
                },
            },
            _ => cast_impl(self.name().clone(), &self.chunks, dtype, options),
        }
    }

    unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Series> {
        self.cast_with_options(dtype, CastOptions::Overflowing)
    }
}

impl BinaryChunked {
    /// # Safety
    /// String is not validated
    pub unsafe fn to_string_unchecked(&self) -> StringChunked {
        // SAFETY: the caller promises the bytes are valid UTF-8, which is the invariant a
        // `StringChunked` chunk carries.
        let chunks = self
            .downcast_iter()
            .map(|arr| unsafe { PlUtf8ViewArray::from_binview_unchecked(arr.clone()) }.into_boxed())
            .collect();
        let field = Arc::new(Field::new(self.name().clone(), DataType::String));

        let mut ca = StringChunked::new_with_compute_len(field, chunks);

        use StatisticsFlags as F;
        ca.retain_flags_from(self, F::IS_SORTED_ANY | F::CAN_FAST_EXPLODE_LIST);
        ca
    }
}

impl StringChunked {
    pub fn as_binary(&self) -> BinaryChunked {
        let chunks = self
            .downcast_iter()
            .map(|arr| arr.as_binview().to_boxed())
            .collect();
        let field = Arc::new(Field::new(self.name().clone(), DataType::Binary));

        let mut ca = BinaryChunked::new_with_compute_len(field, chunks);

        use StatisticsFlags as F;
        ca.retain_flags_from(self, F::IS_SORTED_ANY | F::CAN_FAST_EXPLODE_LIST);
        ca
    }
}

impl ChunkCast for BinaryChunked {
    fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        match dtype {
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => {
                cast_single_to_struct(self.name().clone(), &self.chunks, fields, options)
            },
            _ => cast_impl(self.name().clone(), &self.chunks, dtype, options),
        }
    }

    unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Series> {
        match dtype {
            DataType::String => unsafe { Ok(self.to_string_unchecked().into_series()) },
            _ => self.cast_with_options(dtype, CastOptions::Overflowing),
        }
    }
}

impl ChunkCast for BinaryOffsetChunked {
    fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        match dtype {
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => {
                cast_single_to_struct(self.name().clone(), &self.chunks, fields, options)
            },
            _ => cast_impl(self.name().clone(), &self.chunks, dtype, options),
        }
    }

    unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Series> {
        self.cast_with_options(dtype, CastOptions::Overflowing)
    }
}

impl ChunkCast for BooleanChunked {
    fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        match dtype {
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => {
                cast_single_to_struct(self.name().clone(), &self.chunks, fields, options)
            },
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                polars_bail!(InvalidOperation: "cannot cast Boolean to Categorical");
            },
            _ => cast_impl(self.name().clone(), &self.chunks, dtype, options),
        }
    }

    unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Series> {
        self.cast_with_options(dtype, CastOptions::Overflowing)
    }
}

/// We cannot cast anything to or from List/LargeList
/// So this implementation casts the inner type
impl ChunkCast for ListChunked {
    fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        let ca = self
            .trim_lists_to_normalized_offsets()
            .map_or(Cow::Borrowed(self), Cow::Owned);
        let ca = ca.propagate_nulls().map_or(ca, Cow::Owned);

        use DataType::*;
        match dtype {
            List(child_type) => {
                match (ca.inner_dtype(), &**child_type) {
                    (old, new) if old == new => Ok(ca.into_owned().into_series()),
                    // TODO @ cat-rework: can we implement this now?
                    #[cfg(feature = "dtype-categorical")]
                    (dt, Categorical(_, _) | Enum(_, _))
                        if !matches!(dt, Categorical(_, _) | Enum(_, _) | String | Null) =>
                    {
                        polars_bail!(InvalidOperation: "cannot cast List inner type: '{:?}' to Categorical", dt)
                    },
                    _ => {
                        // ensure the inner logical type bubbles up
                        let (arr, child_type) = cast_list(ca.as_ref(), child_type, options)?;
                        // SAFETY: we just cast so the dtype matches.
                        // we must take this path to correct for physical types.
                        unsafe {
                            Ok(Series::from_chunks_and_dtype_unchecked(
                                ca.name().clone(),
                                vec![arr],
                                &List(Box::new(child_type)),
                            ))
                        }
                    },
                }
            },
            #[cfg(feature = "dtype-array")]
            Array(child_type, width) => {
                let physical_type = dtype.to_physical();

                // cast to the physical type to avoid logical chunks.
                let chunks = cast_chunks(ca.chunks(), &physical_type, options)?;
                // SAFETY: we just cast so the dtype matches.
                // we must take this path to correct for physical types.
                unsafe {
                    Ok(Series::from_chunks_and_dtype_unchecked(
                        ca.name().clone(),
                        chunks,
                        &Array(child_type.clone(), *width),
                    ))
                }
            },
            #[cfg(feature = "dtype-u8")]
            Binary => {
                polars_ensure!(
                    matches!(self.inner_dtype(), UInt8),
                    InvalidOperation: "cannot cast List type (inner: '{:?}', to: '{:?}')",
                    self.inner_dtype(),
                    dtype,
                );
                let chunks = cast_chunks(self.chunks(), &DataType::Binary, options)?;

                // SAFETY: we just cast so the dtype matches.
                unsafe {
                    Ok(Series::from_chunks_and_dtype_unchecked(
                        self.name().clone(),
                        chunks,
                        &DataType::Binary,
                    ))
                }
            },
            _ => {
                polars_bail!(
                    InvalidOperation: "cannot cast List type (inner: '{:?}', to: '{:?}')",
                    ca.inner_dtype(),
                    dtype,
                )
            },
        }
    }

    unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Series> {
        use DataType::*;
        match dtype {
            List(child_type) => cast_list_unchecked(self, child_type),
            _ => self.cast_with_options(dtype, CastOptions::Overflowing),
        }
    }
}

/// We cannot cast anything to or from List/LargeList
/// So this implementation casts the inner type
#[cfg(feature = "dtype-array")]
impl ChunkCast for ArrayChunked {
    fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        let ca = self
            .trim_lists_to_normalized_offsets()
            .map_or(Cow::Borrowed(self), Cow::Owned);
        let ca = ca.propagate_nulls().map_or(ca, Cow::Owned);

        use DataType::*;
        match dtype {
            Array(child_type, width) => {
                polars_ensure!(
                    *width == ca.width(),
                    InvalidOperation: "cannot cast Array to a different width"
                );

                match (ca.inner_dtype(), &**child_type) {
                    (old, new) if old == new => Ok(ca.into_owned().into_series()),
                    // TODO @ cat-rework: can we implement this now?
                    #[cfg(feature = "dtype-categorical")]
                    (dt, Categorical(_, _) | Enum(_, _)) if !matches!(dt, String) => {
                        polars_bail!(InvalidOperation: "cannot cast Array inner type: '{:?}' to dtype: {:?}", dt, child_type)
                    },
                    _ => {
                        // ensure the inner logical type bubbles up
                        let (arr, child_type) =
                            cast_fixed_size_list(ca.as_ref(), child_type, options)?;
                        // SAFETY: we just cast so the dtype matches.
                        // we must take this path to correct for physical types.
                        unsafe {
                            Ok(Series::from_chunks_and_dtype_unchecked(
                                ca.name().clone(),
                                vec![arr],
                                &Array(Box::new(child_type), *width),
                            ))
                        }
                    },
                }
            },
            List(child_type) => {
                let physical_type = dtype.to_physical();
                // cast to the physical type to avoid logical chunks.
                let chunks = cast_chunks(ca.chunks(), &physical_type, options)?;
                // SAFETY: we just cast so the dtype matches.
                // we must take this path to correct for physical types.
                unsafe {
                    Ok(Series::from_chunks_and_dtype_unchecked(
                        ca.name().clone(),
                        chunks,
                        &List(child_type.clone()),
                    ))
                }
            },
            _ => {
                polars_bail!(
                    InvalidOperation: "cannot cast Array type (inner: '{:?}', to: '{:?}')",
                    ca.inner_dtype(),
                    dtype,
                )
            },
        }
    }

    unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Series> {
        self.cast_with_options(dtype, CastOptions::Overflowing)
    }
}

// Returns inner data type. This is needed because a cast can instantiate the dtype inner
// values for instance with categoricals
fn cast_list(
    ca: &ListChunked,
    child_type: &DataType,
    options: CastOptions,
) -> PolarsResult<(PlArrayRef, DataType)> {
    // We still rechunk because we must bubble up a single data-type
    // TODO!: consider a version that works on chunks and merges the data-types and arrays.
    let ca = ca.rechunk();
    let arr = ca.downcast_as_array().to_flat();
    // SAFETY: inner dtype is passed correctly
    let s = unsafe {
        Series::from_chunks_and_dtype_unchecked(
            PlSmallStr::EMPTY,
            vec![arr.values().to_boxed()],
            ca.inner_dtype(),
        )
    };
    let new_inner = s.cast_with_options(child_type, options)?;

    let inner_dtype = new_inner.dtype().clone();
    debug_assert_eq!(&inner_dtype, child_type);

    let new_values = new_inner.rechunk().array_ref(0).clone();

    // The offsets and the mask are handed over as they are: only the values were cast.
    let (_, offsets, length, validity) = arr.into_array().into_inner();
    let new_arr = PlListArray::new(new_values, offsets, length, validity);
    Ok((Box::new(new_arr), inner_dtype))
}

unsafe fn cast_list_unchecked(ca: &ListChunked, child_type: &DataType) -> PolarsResult<Series> {
    // TODO! add chunked, but this must correct for list offsets.
    let ca = ca.rechunk();
    let arr = ca.downcast_as_array().to_flat();
    // SAFETY: inner dtype is passed correctly
    let s = unsafe {
        Series::from_chunks_and_dtype_unchecked(
            PlSmallStr::EMPTY,
            vec![arr.values().to_boxed()],
            ca.inner_dtype(),
        )
    };
    let new_inner = s.cast_unchecked(child_type)?;
    let new_values = new_inner.rechunk().array_ref(0).clone();

    let (_, offsets, length, validity) = arr.into_array().into_inner();
    let new_arr = PlListArray::new(new_values, offsets, length, validity);
    Ok(ListChunked::from_chunks_and_dtype_unchecked(
        ca.name().clone(),
        vec![Box::new(new_arr)],
        DataType::List(Box::new(child_type.clone())),
    )
    .into_series())
}

// Returns inner data type. This is needed because a cast can instantiate the dtype inner
// values for instance with categoricals
#[cfg(feature = "dtype-array")]
fn cast_fixed_size_list(
    ca: &ArrayChunked,
    child_type: &DataType,
    options: CastOptions,
) -> PolarsResult<(PlArrayRef, DataType)> {
    let ca = ca.rechunk();
    let arr = ca.downcast_as_array().to_flat();
    // SAFETY: inner dtype is passed correctly
    let s = unsafe {
        Series::from_chunks_and_dtype_unchecked(
            PlSmallStr::EMPTY,
            vec![arr.values().to_boxed()],
            ca.inner_dtype(),
        )
    };
    let new_inner = s.cast_with_options(child_type, options)?;

    let inner_dtype = new_inner.dtype().clone();
    debug_assert_eq!(&inner_dtype, child_type);

    let new_values = new_inner.rechunk().array_ref(0).clone();

    // The width and the mask are handed over as they are: only the values were cast.
    let (_, width, length, validity) = arr.into_array().into_inner();
    let new_arr = PlFixedSizeListArray::new(new_values, width, length, validity);
    Ok((Box::new(new_arr), inner_dtype))
}

#[cfg(test)]
mod test {
    use crate::chunked_array::cast::CastOptions;
    use crate::prelude::*;

    #[test]
    fn test_cast_list() -> PolarsResult<()> {
        let mut builder = ListPrimitiveChunkedBuilder::<Int32Type>::new(
            PlSmallStr::from_static("a"),
            10,
            10,
            DataType::Int32,
        );
        builder.append_opt_slice(Some(&[1i32, 2, 3]));
        builder.append_opt_slice(Some(&[1i32, 2, 3]));
        let ca = builder.finish();

        let new = ca.cast_with_options(
            &DataType::List(DataType::Float64.into()),
            CastOptions::Strict,
        )?;

        assert_eq!(new.dtype(), &DataType::List(DataType::Float64.into()));
        Ok(())
    }

    #[test]
    #[cfg(feature = "dtype-categorical")]
    fn test_cast_noop() {
        // check if we can cast categorical twice without panic
        let ca = StringChunked::new(PlSmallStr::from_static("foo"), &["bar", "ham"]);
        let cats = Categories::global();
        let out = ca
            .cast_with_options(
                &DataType::from_categories(cats.clone()),
                CastOptions::Strict,
            )
            .unwrap();
        let out = out.cast(&DataType::from_categories(cats)).unwrap();
        assert!(matches!(out.dtype(), &DataType::Categorical(_, _)))
    }
}
