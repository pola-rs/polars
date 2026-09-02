//! Special fixed-size-list utility methods

mod iterator;

use std::borrow::Cow;

use arrow::bitmap::BitmapBuilder;
use either::Either;
use polars_array::builder::{PlArrayBuilder, builder_like};
use polars_array::concatenate::concatenate;

use crate::chunked_array::arrow_bridge::as_flat;
use crate::chunked_array::new_empty_chunk;
use crate::prelude::*;

/// The values `arr` is taken over: the values of every element, laid end to end.
///
/// A [`scalar`](polars_array::broadcast) array holds only the values of the single element every
/// one of them is, so it is written out; a flat array hands its values over as they are.
pub(crate) fn array_values(arr: &PlFixedSizeListArray) -> PlArrayRef {
    // TODO(polars-array-scalar): the callers read the values as one run per element, which a
    // scalar array has to be written out to hand over.
    as_flat(arr).values().to_boxed()
}

/// Returns `arr` with its values replaced, keeping its width and validity mask.
///
/// # Panics
/// Panics if `values` does not hold the width of every element, laid end to end.
pub(crate) fn array_with_values(
    arr: &PlFixedSizeListArray,
    values: PlArrayRef,
) -> PlFixedSizeListArray {
    let (width, length) = (arr.width(), arr.len());
    assert_eq!(values.len(), width * length);

    // SAFETY: just checked that the values hold `width` values for every element.
    unsafe { PlFixedSizeListArray::new_unchecked(values, width, length, None) }
        .with_validity(arr.validity().map(|v| v.to_flat_or_scalar()))
}

/// Lays `elements` out as the chunk of an [`ArrayChunked`] of `width` and `inner_dtype`.
///
/// Every element of a fixed size list array covers `width` values, so an element that is null
/// covers `width` of them too — they are the nulls this writes in its place. The width is not
/// read off the elements: it belongs to the `ArrayChunked`'s [`DataType`], which is the only thing
/// that has it when every element is null.
///
/// # Panics
/// Panics if any element is not `width` values long.
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
            has_nulls.then(|| validity.freeze()),
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

    /// # Panics
    /// Panics if the physical representation of `dtype` differs the physical
    /// representation of the existing inner `dtype`.
    pub fn set_inner_dtype(&mut self, dtype: DataType) {
        assert_eq!(dtype.to_physical(), self.inner_dtype().to_physical());
        let width = self.width();
        let field = Arc::make_mut(&mut self.field);
        field.coerce(DataType::Array(Box::new(dtype), width));
    }

    pub fn width(&self) -> usize {
        match self.dtype() {
            DataType::Array(_dt, size) => *size,
            _ => unreachable!(),
        }
    }

    /// # Safety
    /// The caller must ensure that the logical type given fits the physical type of the array.
    pub unsafe fn to_logical(&mut self, inner_dtype: DataType) {
        debug_assert_eq!(&inner_dtype.to_physical(), self.inner_dtype());
        let width = self.width();
        let fld = Arc::make_mut(&mut self.field);
        fld.coerce(DataType::Array(Box::new(inner_dtype), width))
    }

    /// Convert the datatype of the array into the physical datatype.
    pub fn to_physical_repr(&self) -> Cow<'_, ArrayChunked> {
        let Cow::Owned(physical_repr) = self.get_inner().to_physical_repr() else {
            return Cow::Borrowed(self);
        };

        let chunk_len_validity_iter =
            if physical_repr.chunks().len() == 1 && self.chunks().len() > 1 {
                // Physical repr got rechunked, rechunk our validity as well.
                Either::Left(std::iter::once((self.len(), self.rechunk_validity())))
            } else {
                // No rechunking, expect the same number of chunks.
                assert_eq!(self.chunks().len(), physical_repr.chunks().len());
                Either::Right(
                    self.chunks()
                        .iter()
                        .map(|c| (c.len(), c.validity().map(|v| v.to_flat_or_scalar()))),
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

        let chunks = self.downcast_iter().map(array_values).collect();

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

    /// Ignore the list indices and apply `func` to the inner type as [`Series`].
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
                vec![array_values(arr)],
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

    /// Recurse nested types until we are at the leaf array.
    pub fn get_leaf_array(&self) -> Series {
        let mut current = self.get_inner();
        while let Some(child_array) = current.try_array() {
            current = child_array.get_inner();
        }
        current
    }
}
