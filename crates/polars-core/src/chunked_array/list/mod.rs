//! Special list utility methods
pub(super) mod iterator;

use std::borrow::Cow;

use arrow::bitmap::BitmapBuilder;
use polars_array::concatenate::concatenate;
use polars_utils::itertools::Itertools;

use crate::chunked_array::new_empty_chunk;
use crate::prelude::*;

/// Lays `elements` out as the chunk of a [`ListChunked`] of `inner_dtype`.
///
/// The values are the elements laid end to end, so an element that is null contributes nothing to
/// them; `inner_dtype` is what says what the values are when every element is null, the chunks
/// carrying no logical type of their own.
pub(crate) fn collect_list_chunk(
    elements: Vec<Option<PlArrayRef>>,
    inner_dtype: &DataType,
) -> PlListArray {
    let length = elements.len();
    let mut offsets = Vec::with_capacity(length + 1);
    let mut validity = BitmapBuilder::with_capacity(length);
    let mut has_nulls = false;
    let mut total = 0u64;
    offsets.push(0);

    for element in &elements {
        total += element.as_ref().map_or(0, |values| values.len()) as u64;
        offsets.push(total);
        has_nulls |= element.is_none();
        validity.push(element.is_some());
    }

    let present = elements.iter().flatten().map(|v| &**v).collect::<Vec<_>>();
    let values = if present.is_empty() {
        // There is nothing to take the values from: every element is null, so the values are the
        // empty array the inner dtype describes.
        new_empty_chunk(inner_dtype)
    } else {
        concatenate(&present).expect("the elements of a list are all of the same type")
    };

    // SAFETY: the offsets were built from the lengths of the values, laid end to end.
    unsafe {
        PlListArray::new_unchecked(
            values,
            offsets.into(),
            length,
            has_nulls.then(|| validity.freeze()),
        )
    }
}

/// Returns `arr` with its values replaced, keeping its offsets and validity mask.
///
/// The offsets are handed over in whatever representation they are in — a
/// [`scalar`](polars_array::broadcast) list array holds the single range every element covers —
/// so this is `O(1)`.
///
/// # Panics
/// Panics if `values` is not as long as the values `arr` is taken over.
pub(crate) fn list_with_values(arr: &PlListArray, values: PlArrayRef) -> PlListArray {
    assert_eq!(arr.values().len(), values.len());
    let offsets_are_flat = arr.offsets_are_flat();
    let (_, offsets, length, validity) = arr.clone().into_inner();

    // SAFETY: only the values are replaced, by an array of the same length, so the offsets still
    // cover them and are in the representation they were taken out in.
    unsafe {
        if offsets_are_flat {
            PlListArray::new_unchecked(values, offsets, length, validity)
        } else {
            PlListArray::new_broadcast_unchecked(values, offsets, length, validity)
        }
    }
}

impl ListChunked {
    /// Get the inner data type of the list.
    pub fn inner_dtype(&self) -> &DataType {
        match self.dtype() {
            DataType::List(dt) => dt.as_ref(),
            _ => unreachable!(),
        }
    }

    /// # Panics
    /// Panics if the physical representation of `dtype` differs the physical
    /// representation of the existing inner `dtype`.
    pub fn set_inner_dtype(&mut self, dtype: DataType) {
        // A chunk carries no inner type, so a `ChunkedArray` built from one alone names `Null`
        // as its inner type until it is set here.
        assert!(
            self.inner_dtype().is_null() || dtype.to_physical() == self.inner_dtype().to_physical()
        );
        let field = Arc::make_mut(&mut self.field);
        field.set_dtype(DataType::List(Box::new(dtype)));
    }

    pub fn set_fast_explode(&mut self) {
        self.set_fast_explode_list(true)
    }

    pub fn _can_fast_explode(&self) -> bool {
        self.get_fast_explode_list()
    }

    /// Set the logical type of the [`ListChunked`].
    ///
    /// # Safety
    /// The caller must ensure that the logical type given fits the physical type of the array.
    pub unsafe fn to_logical(&mut self, inner_dtype: DataType) {
        // A chunk carries no inner type, so a `ChunkedArray` built from one alone names `Null`
        // as its inner type until it is set here.
        debug_assert!(
            self.inner_dtype().is_null() || &inner_dtype.to_physical() == self.inner_dtype()
        );
        let fld = Arc::make_mut(&mut self.field);
        fld.set_dtype(DataType::List(Box::new(inner_dtype)))
    }

    /// Convert the datatype of the list into the physical datatype.
    pub fn to_physical_repr(&self) -> Cow<'_, ListChunked> {
        let Cow::Owned(physical_repr) = self.get_inner().to_physical_repr() else {
            return Cow::Borrowed(self);
        };

        let ca = if physical_repr.chunks().len() == 1 && self.chunks().len() > 1 {
            // Physical repr got rechunked, rechunk self as well.
            self.rechunk()
        } else {
            Cow::Borrowed(self)
        };

        assert_eq!(ca.chunks().len(), physical_repr.chunks().len());

        let chunks: Vec<_> = ca
            .downcast_iter()
            .zip(physical_repr.into_chunks())
            .map(|(chunk, values)| list_with_values(chunk, values).into_boxed())
            .collect();

        let name = self.name().clone();
        let dtype = DataType::List(Box::new(self.inner_dtype().to_physical()));
        Cow::Owned(unsafe { ListChunked::from_chunks_and_dtype_unchecked(name, chunks, dtype) })
    }

    /// Convert a non-logical [`ListChunked`] back into a logical [`ListChunked`] without casting.
    ///
    /// # Safety
    ///
    /// This can lead to invalid memory access in downstream code.
    pub unsafe fn from_physical_unchecked(
        &self,
        to_inner_dtype: DataType,
    ) -> PolarsResult<ListChunked> {
        debug_assert!(!self.inner_dtype().is_logical());

        let inner_chunks = self
            .downcast_iter()
            .map(|chunk| chunk.values().to_boxed())
            .collect();

        let inner = unsafe {
            Series::from_chunks_and_dtype_unchecked(
                PlSmallStr::EMPTY,
                inner_chunks,
                self.inner_dtype(),
            )
        };
        let inner = unsafe { inner.from_physical_unchecked(&to_inner_dtype) }?;

        let chunks: Vec<_> = self
            .downcast_iter()
            .zip(inner.into_chunks())
            .map(|(chunk, values)| list_with_values(chunk, values).into_boxed())
            .collect();

        let name = self.name().clone();
        let dtype = DataType::List(Box::new(to_inner_dtype));
        Ok(unsafe { ListChunked::from_chunks_and_dtype_unchecked(name, chunks, dtype) })
    }

    /// Get the inner values as [`Series`], ignoring the list offsets.
    pub fn get_inner(&self) -> Series {
        let chunks: Vec<_> = self
            .downcast_iter()
            .map(|c| c.values().to_boxed())
            .collect();

        // SAFETY: Data type of arrays matches because they are chunks from the same array.
        unsafe {
            Series::from_chunks_and_dtype_unchecked(self.name().clone(), chunks, self.inner_dtype())
        }
    }

    pub fn inner_length(&self) -> usize {
        self.downcast_iter().map(|c| c.values().len()).sum()
    }

    /// Ignore the list indices and apply `func` to the inner type as [`Series`].
    pub fn apply_to_inner(
        &self,
        func: &dyn Fn(Series) -> PolarsResult<Series>,
    ) -> PolarsResult<ListChunked> {
        // generated Series will have wrong length otherwise.
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

        let arr = list_with_values(arr, values);

        // SAFETY: arr's inner dtype is derived from out dtype.
        Ok(unsafe {
            ListChunked::from_chunks_and_dtype_unchecked(
                ca.name().clone(),
                vec![arr.into_boxed()],
                DataType::List(Box::new(out.dtype().clone())),
            )
        })
    }

    pub fn with_inner_values(&self, values: &Series) -> ListChunked {
        if cfg!(debug_assertions) {
            assert_eq!(values.len(), self.inner_length());
        }

        // Align the chunks of the lists inner values and the values series.
        fn align_inner_chunks(ca: &'_ ListChunked, values: &'_ Series) -> Series {
            if ca.chunks().len() == values.chunks().len()
                && ca
                    .downcast_iter()
                    .map(|arr| arr.values().len())
                    .zip(values.chunks().iter().map(|arr| arr.len()))
                    .all_equal()
            {
                return values.clone();
            }

            let mut values = values.rechunk();
            let chunks = unsafe { values.chunks_mut() };
            let mut arr = chunks.pop().unwrap();
            chunks.extend(ca.downcast_iter().map(|ca_arr| {
                let length = ca_arr.values().len();
                let chunk = arr.sliced(0, length);
                arr = arr.sliced(length, arr.len() - length);
                chunk
            }));
            assert!(arr.is_empty());
            values
        }

        let values = align_inner_chunks(self, values);
        let values_dtype = values.dtype().clone();

        let chunks = self
            .downcast_iter()
            .zip(values.into_chunks())
            .map(|(ca_arr, v_arr)| list_with_values(ca_arr, v_arr).into_boxed())
            .collect::<Vec<_>>();

        // SAFETY: arr's inner dtype is derived from out dtype.
        unsafe {
            ListChunked::from_chunks_and_dtype_unchecked(
                self.name().clone(),
                chunks,
                DataType::List(Box::new(values_dtype)),
            )
        }
    }
}
