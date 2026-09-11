//! Special fixed-size-list utility methods

mod iterator;

use std::borrow::Cow;

use arrow::bitmap::BitmapBuilder;
use either::Either;
use polars_array::builder::{PlArrayBuilder, builder_like};
use polars_array::concatenate::concatenate;

use super::align_inner_chunks;
use crate::chunked_array::new_empty_chunk;
use crate::prelude::*;

/// The values `arr` is taken over: the values of every element, laid end to end.
pub(crate) fn array_values(arr: &PlFixedSizeListArray) -> PlArrayRef {
    if let Some(values) = arr.flat_values() {
        return values.to_boxed();
    }

    // The mask is dropped first: it is not part of what is handed over, and a repeated one would
    // otherwise be written out along with the values for no reader at all.
    arr.clone()
        .with_validity(None)
        .to_flat()
        .values()
        .to_boxed()
}

/// Returns `arr` with its values replaced, keeping its width and validity mask.
///
/// The replacement stands for the values it replaces one for one, so it is in the
/// representation they were in: one element's values where `arr` repeats a single element, and
/// one element's values per element otherwise.
pub(crate) fn array_with_values(
    arr: &PlFixedSizeListArray,
    values: PlArrayRef,
) -> PlFixedSizeListArray {
    assert_eq!(arr.values().len(), values.len());
    let (width, length) = (arr.width(), arr.len());
    let values_are_scalar = arr.values_are_scalar();
    let validity = arr.validity().map(PlBitmap::from);

    // SAFETY: only the values are replaced, by an array of the same length, so they still hold
    // the width of every element in the representation they were taken out in.
    let out = unsafe {
        if values_are_scalar {
            PlFixedSizeListArray::new_broadcast_unchecked(values, width, length, None)
        } else {
            PlFixedSizeListArray::new_unchecked(values, width, length, None)
        }
    };
    out.with_validity(validity)
}

/// Lays `elements` out as the chunk of an [`ArrayChunked`] of `width` and `inner_dtype`.
pub(crate) fn collect_array_chunk(
    elements: Vec<Option<PlArrayRef>>,
    width: usize,
    inner_dtype: &DataType,
) -> PlFixedSizeListArray {
    let length = elements.len();
    let mut validity = BitmapBuilder::with_capacity(length);
    let mut has_nulls = false;
    for element in &elements {
        if let Some(values) = element {
            assert_eq!(
                values.len(),
                width,
                "a fixed size list element of the wrong width"
            );
        }
        has_nulls |= element.is_none();
        validity.push(element.is_some());
    }

    // The values of a null element are the `width` nulls that stand in for the element it does not
    // hold; the array is built once and shared by every null element.
    let null_element = has_nulls.then(|| {
        let mut builder = builder_like(&*new_empty_chunk(inner_dtype));
        builder.extend_nulls(width);
        builder.freeze_reset()
    });

    let values = elements
        .iter()
        .map(|element| match element {
            Some(values) => &**values,
            None => &**null_element.as_ref().unwrap(),
        })
        .collect::<Vec<_>>();
    let values = if values.is_empty() {
        new_empty_chunk(inner_dtype)
    } else {
        concatenate(&values).expect("the elements of a fixed size list are all of the same type")
    };

    // SAFETY: every element covers `width` values, which were laid end to end.
    unsafe {
        PlFixedSizeListArray::new_unchecked(
            values,
            width,
            length,
            (has_nulls.then(|| validity.freeze())).map(PlBitmap::from_bitmap),
        )
    }
}

impl ArrayChunked {
    /// Get the inner data type of the fixed size list.
    pub fn inner_dtype(&self) -> &DataType {
        match self.dtype() {
            DataType::Array(dt, _size) => dt.as_ref(),
            _ => unreachable!(),
        }
    }

    /// Relabel the inner dtype, checking its physical representation.
    ///
    /// # Safety
    /// The values must be valid for `dtype`, see [`Self::to_logical`].
    ///
    /// # Panics
    /// Panics if the physical representation of `dtype` differs the physical
    /// representation of the existing inner `dtype`.
    pub unsafe fn set_inner_dtype(&mut self, dtype: DataType) {
        // A chunk carries no inner type, so a `ChunkedArray` built from one alone names `Null`
        // as its inner type until it is set here.
        assert!(
            self.inner_dtype().is_null() || dtype.to_physical() == self.inner_dtype().to_physical()
        );
        unsafe { self.to_logical(dtype) }
    }

    pub fn width(&self) -> usize {
        match self.dtype() {
            DataType::Array(_dt, size) => *size,
            _ => unreachable!(),
        }
    }

    /// Relabel the inner dtype without changing values.
    ///
    /// # Safety
    /// Same requirements as [`ListChunked::to_logical`].
    pub unsafe fn to_logical(&mut self, inner_dtype: DataType) {
        // A chunk carries no inner type, so a `ChunkedArray` built from one alone names `Null`
        // as its inner type until it is set here.
        debug_assert!(
            self.inner_dtype().is_null()
                || inner_dtype.to_physical() == self.inner_dtype().to_physical()
        );
        let width = self.width();
        let fld = Arc::make_mut(&mut self.field);
        fld.set_dtype(DataType::Array(Box::new(inner_dtype), width))
    }

    /// Convert the datatype of the array into the physical datatype.
    pub fn to_physical_repr(&self) -> Cow<'_, ArrayChunked> {
        // Whether the values change is a question about the inner type alone, so it is asked of
        // the type rather than of the values: `get_inner` writes them out one list per element,
        // which for an inner type that is already physical is a copy of the whole column for an
        // answer of "nothing to do" — and a chunk that repeats one list pays it in full.
        if !self.inner_dtype().is_logical() {
            return Cow::Borrowed(self);
        }

        let Cow::Owned(physical_repr) = self.get_inner().to_physical_repr() else {
            return Cow::Borrowed(self);
        };

        let chunk_len_validity_iter =
            if physical_repr.chunks().len() == 1 && self.chunks().len() > 1 {
                // Physical repr got rechunked, rechunk our validity as well.
                Either::Left(std::iter::once((
                    self.len(),
                    // Rechunking writes the mask out one bit per element.
                    self.rechunk_validity(),
                )))
            } else {
                // No rechunking, expect the same number of chunks.
                assert_eq!(self.chunks().len(), physical_repr.chunks().len());
                Either::Right(
                    self.chunks()
                        .iter()
                        .map(|c| (c.len(), c.validity().map(PlBitmap::from))),
                )
            };

        let width = self.width();
        let chunks: Vec<_> = chunk_len_validity_iter
            .zip(physical_repr.into_chunks())
            .map(|((len, validity), values)| {
                // SAFETY: the values are the physical repr of the ones taken out, so they still
                // hold the width of every element, laid end to end.
                unsafe { PlFixedSizeListArray::new_unchecked(values, width, len, None) }
                    .with_validity(validity)
                    .into_boxed()
            })
            .collect();

        let name = self.name().clone();
        let dtype = DataType::Array(Box::new(self.inner_dtype().to_physical()), width);
        Cow::Owned(unsafe { ArrayChunked::from_chunks_and_dtype_unchecked(name, chunks, dtype) })
    }

    /// Convert a non-logical [`ArrayChunked`] back into a logical [`ArrayChunked`] without casting.
    ///
    /// # Safety
    ///
    /// This can lead to invalid memory access in downstream code.
    pub unsafe fn from_physical_unchecked(&self, to_inner_dtype: DataType) -> PolarsResult<Self> {
        debug_assert!(!self.inner_dtype().is_logical());

        // The values are re-tagged one for one, so they are taken as they are laid out: a chunk
        // that repeats a single array holds that one array's values rather than a copy of them
        // per element, and `array_with_values` puts the re-tagged ones back the same way.
        let chunks = self
            .downcast_iter()
            .map(|arr| arr.values().to_boxed())
            .collect();

        let inner = unsafe {
            Series::from_chunks_and_dtype_unchecked(PlSmallStr::EMPTY, chunks, self.inner_dtype())
        };
        let inner = unsafe { inner.from_physical_unchecked(&to_inner_dtype) }?;

        let chunks: Vec<_> = self
            .downcast_iter()
            .zip(inner.into_chunks())
            .map(|(chunk, values)| array_with_values(chunk, values).into_boxed())
            .collect();

        let name = self.name().clone();
        let dtype = DataType::Array(Box::new(to_inner_dtype), self.width());
        Ok(unsafe { Self::from_chunks_and_dtype_unchecked(name, chunks, dtype) })
    }

    /// Get the inner values as `Series`
    pub fn get_inner(&self) -> Series {
        let chunks: Vec<_> = self.downcast_iter().map(array_values).collect();

        // SAFETY: Data type of arrays matches because they are chunks from the same array.
        unsafe {
            Series::from_chunks_and_dtype_unchecked(self.name().clone(), chunks, self.inner_dtype())
        }
    }

    /// The total number of inner values across all chunks, i.e. `len() * width()`
    /// discounting sliced-away chunks.
    pub fn inner_length(&self) -> usize {
        self.downcast_iter().map(|c| c.len() * c.width()).sum()
    }

    /// Rebuild the arrays around new inner values, reusing the widths and outer validity.
    ///
    /// `values` must have `inner_length()` elements; its chunks need not line up with
    /// this array's, but nothing is copied when they do.
    pub fn with_inner_values(&self, values: &Series) -> ArrayChunked {
        if cfg!(debug_assertions) {
            assert_eq!(values.len(), self.inner_length());
        }

        // Align the chunks of the array's inner values and the values series.
        let values = align_inner_chunks(
            self.downcast_iter().map(|arr| arr.len() * arr.width()),
            values,
        );
        let values_dtype = values.dtype().clone();
        let width = self.width();

        let chunks = self
            .downcast_iter()
            .zip(values.into_chunks())
            .map(|(ca_arr, v_arr)| array_with_values(ca_arr, v_arr).into_boxed())
            .collect::<Vec<_>>();

        // SAFETY: the chunks' inner dtype is derived from `values`' own chunks.
        unsafe {
            ArrayChunked::from_chunks_and_dtype_unchecked(
                self.name().clone(),
                chunks,
                DataType::Array(Box::new(values_dtype), width),
            )
        }
    }

    /// Ignore the list indices and apply `func` to the inner type as [`Series`].
    ///
    /// `func` is handed the values of one element for a chunk that repeats a single array,
    /// since every element reads the same ones, and its answer is repeated in turn.
    pub fn apply_to_inner(
        &self,
        func: &dyn Fn(Series) -> PolarsResult<Series>,
    ) -> PolarsResult<ArrayChunked> {
        // Rechunk or the generated Series will have wrong length.
        let ca = self.rechunk();
        let arr = ca.downcast_as_array();

        // SAFETY:
        // Inner dtype is passed correctly
        let elements = unsafe {
            Series::from_chunks_and_dtype_unchecked(
                self.name().clone(),
                vec![arr.values().to_boxed()],
                ca.inner_dtype(),
            )
        };

        let expected_len = elements.len();
        let out: Series = func(elements)?;
        polars_ensure!(
            out.len() == expected_len,
            ComputeError: "the function should apply element-wise, it removed elements instead"
        );
        let out = out.rechunk();
        let values = out.chunks()[0].clone();

        let arr = array_with_values(arr, values);

        Ok(unsafe {
            ArrayChunked::from_chunks_and_dtype_unchecked(
                self.name().clone(),
                vec![arr.into_boxed()],
                DataType::Array(Box::new(out.dtype().clone()), self.width()),
            )
        })
    }
}
