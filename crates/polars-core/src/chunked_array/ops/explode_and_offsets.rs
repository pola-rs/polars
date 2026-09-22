use polars_array::PlPrimitiveArrayBuilder;
use polars_array::arrow::export;
use polars_array::builder::StaticArrayBuilder;
use polars_arrow::bitmap::MutableBitmap;
use polars_arrow::bitmap::utils::set_bit_unchecked;
use polars_arrow::offset::OffsetsBuffer;
use polars_compute::gather::take_unchecked;

use super::*;
#[cfg(feature = "dtype-array")]
use crate::chunked_array::array::array_values;

impl ListChunked {
    /// Explodes a list array whose values are a single element repeated.
    ///
    /// Every slot the exploded array holds reads that same element, so only its *shape* is left
    /// to work out: how many slots each row contributes, and which of them are null. Both come
    /// out of one walk over the offsets, so no values are ever copied.
    fn explode_scalar_values(
        &self,
        listarr: &PlListArray,
        values: &dyn PlArray,
        offsets: &[i64],
        options: ExplodeOptions,
    ) -> (Series, OffsetsBuffer<i64>) {
        let validity = listarr.validity();
        let mut new_offsets = Vec::with_capacity(offsets.len());
        // The positions of the slots a null row or an empty row explodes to; empty for the
        // common shape, which is what keeps this path free of a bitmap allocation.
        let mut null_slots = Vec::new();
        let mut length = 0usize;
        let mut current_offset = 0i64;

        let mut iter = offsets.iter();
        if let Some(mut previous) = iter.next().copied() {
            new_offsets.push(current_offset);
            for (i, &offset) in iter.enumerate() {
                let len = offset - previous;
                // SAFETY: the mask covers one bit per row, and `offsets` holds one more.
                let valid = validity
                    .as_ref()
                    .is_none_or(|validity| unsafe { validity.get_unchecked(i) });

                if valid {
                    // An empty list explodes to a null, like a null row under `keep_nulls`.
                    if options.empty_as_null && len == 0 {
                        null_slots.push(length);
                        length += 1;
                    } else {
                        length += len as usize;
                    }
                    current_offset += len;
                } else if options.keep_nulls {
                    null_slots.push(length);
                    length += 1;
                }

                previous = offset;
                new_offsets.push(current_offset);
            }
        }

        // SAFETY: the values array is scalar, so it is not empty and element 0 stands for it.
        let mut chunk = unsafe { values.new_from_index_unchecked(0, length) };
        // A repeated null already reads null in every slot, punched out or not.
        if !null_slots.is_empty() && values.is_valid(0) {
            let mut validity = MutableBitmap::from_len_set(length);
            let validity_slice = validity.as_mut_slice();
            for i in null_slots {
                // SAFETY: every slot we recorded is one of the `length` we counted.
                unsafe { set_bit_unchecked(validity_slice, i, false) };
            }
            chunk.set_validity(Some(PlBitmap::from_bitmap(validity.into())));
        }

        // SAFETY: inner_dtype should be correct
        let s = unsafe {
            Series::from_chunks_and_dtype_unchecked(
                self.name().clone(),
                vec![chunk],
                &self.inner_dtype().to_physical(),
            )
        };
        // SAFETY: monotonically increasing
        let new_offsets = unsafe { OffsetsBuffer::new_unchecked(new_offsets.into()) };

        (s, new_offsets)
    }

    fn explode_specialized(
        &self,
        values: PlArrayRef,
        offsets: &[i64],
        offsets_buf: OffsetsBuffer<i64>,
        options: ExplodeOptions,
    ) -> (Series, OffsetsBuffer<i64>) {
        // SAFETY: inner_dtype should be correct
        let values = unsafe {
            Series::from_chunks_and_dtype_unchecked(
                self.name().clone(),
                vec![values],
                &self.inner_dtype().to_physical(),
            )
        };

        use crate::chunked_array::ops::explode::ExplodeByOffsets;

        let mut values = match values.dtype() {
            DataType::Boolean => {
                let t = values.bool().unwrap();
                ExplodeByOffsets::explode_by_offsets(t, offsets, options).into_series()
            },
            DataType::Null => {
                let t = values.null().unwrap();
                ExplodeByOffsets::explode_by_offsets(t, offsets, options).into_series()
            },
            dtype => {
                with_match_physical_numeric_polars_type!(dtype, |$T| {
                    let t: &ChunkedArray<$T> = values.as_ref().as_ref();
                    ExplodeByOffsets::explode_by_offsets(t, offsets, options).into_series()
                })
            },
        };

        // let mut values = values.explode_by_offsets(offsets);
        // restore logical type
        values = unsafe { values.from_physical_unchecked(self.inner_dtype()) }.unwrap();

        (values, offsets_buf)
    }
}

impl ChunkExplode for ListChunked {
    fn offsets(&self) -> PolarsResult<OffsetsBuffer<i64>> {
        let ca = self.rechunk();
        let listarr = ca.downcast_iter().next().unwrap().to_flat();
        Ok(export::offsets_to_arrow(listarr.offsets().clone()))
    }

    fn explode_and_offsets(
        &self,
        options: ExplodeOptions,
    ) -> PolarsResult<(Series, OffsetsBuffer<i64>)> {
        // A list array's memory layout is actually already 'exploded', so we can just take the
        // values array of the list. And we also return a slice of the offsets. This slice can be
        // used to find the old list layout or indexes to expand a DataFrame in the same manner as
        // the `explode` operation.
        let ca = self.rechunk();
        let listarr = ca.downcast_iter().next().unwrap().to_flat();
        let offsets_buf = export::offsets_to_arrow(listarr.offsets().clone());
        let offsets = offsets_buf.as_slice();
        let mut values = listarr.values().to_boxed();

        let (mut s, offsets) = if ca._can_fast_explode()
            && (!options.keep_nulls || !ca.has_nulls())
            && (!options.empty_as_null || !ca.has_empty_lists())
        {
            // ensure that the value array is sliced
            // as a list only slices its offsets on a slice operation

            // we only do this in fast-explode as for the other
            // branch the offsets must coincide with the values.
            if !offsets.is_empty() {
                let start = offsets[0] as usize;
                let len = offsets[offsets.len() - 1] as usize - start;
                // SAFETY:
                // we are in bounds
                values = unsafe { values.sliced_unchecked(start, len) };
            }
            // SAFETY: inner_dtype should be correct
            (
                unsafe {
                    Series::from_chunks_and_dtype_unchecked(
                        self.name().clone(),
                        vec![values],
                        &self.inner_dtype().to_physical(),
                    )
                },
                offsets_buf,
            )
        } else {
            // during tests
            // test that this code branch is not hit with list arrays that could be fast exploded
            #[cfg(test)]
            {
                let mut last = offsets[0];
                let mut has_empty = false;
                for &o in &offsets[1..] {
                    if o == last {
                        has_empty = true;
                    }
                    last = o;
                }
                if !has_empty && offsets[0] == 0 {
                    panic!("could have fast exploded")
                }
            }
            // An empty values array has no element 0 to stand for the rest -- and a null array
            // reads as scalar whatever its length -- so it is not one this path can answer.
            if !values.is_empty() && PlArray::is_scalar(&*values) {
                let (s, new_offsets) =
                    self.explode_scalar_values(&listarr, &*values, offsets, options);
                // SAFETY: inner_dtype should be correct
                let s = unsafe { s.from_physical_unchecked(self.inner_dtype()) }?;
                return Ok((s, new_offsets));
            }

            let (indices, new_offsets) = if listarr.null_count() == 0 {
                // SPECIALIZED path.
                let inner_phys = self.inner_dtype().to_physical();
                if inner_phys.is_primitive_numeric() || inner_phys.is_null() || inner_phys.is_bool()
                {
                    return Ok(self.explode_specialized(
                        values,
                        offsets_buf.as_slice(),
                        offsets_buf.clone(),
                        options,
                    ));
                }
                // Use gather
                let mut indices =
                    PlPrimitiveArrayBuilder::<IdxSize>::with_capacity(*offsets_buf.last() as usize);
                let mut new_offsets = Vec::with_capacity(listarr.len() + 1);
                let mut current_offset = 0i64;
                let mut iter = offsets.iter();
                if let Some(mut previous) = iter.next().copied() {
                    new_offsets.push(current_offset);
                    iter.for_each(|&offset| {
                        let len = offset - previous;
                        let start = previous as IdxSize;
                        let end = offset as IdxSize;

                        if options.empty_as_null && len == 0 {
                            indices.push_null();
                        } else {
                            indices.push_values(start..end);
                        }
                        current_offset += len;
                        previous = offset;
                        new_offsets.push(current_offset);
                    })
                }
                (indices, new_offsets)
            } else {
                // we have already ensure that validity is not none.
                let validity = listarr.validity().unwrap();

                let mut indices =
                    PlPrimitiveArrayBuilder::<IdxSize>::with_capacity(*offsets_buf.last() as usize);
                let mut new_offsets = Vec::with_capacity(listarr.len() + 1);
                let mut current_offset = 0i64;
                let mut iter = offsets.iter();
                if let Some(mut previous) = iter.next().copied() {
                    new_offsets.push(current_offset);
                    iter.enumerate().for_each(|(i, &offset)| {
                        let len = offset - previous;
                        let start = previous as IdxSize;
                        let end = offset as IdxSize;
                        // SAFETY: we are within bounds
                        if unsafe { validity.get_bit_unchecked(i) } {
                            // explode expects null value if sublist is empty.
                            if options.empty_as_null && len == 0 {
                                indices.push_null();
                            } else {
                                indices.push_values(start..end);
                            }
                            current_offset += len;
                        } else if options.keep_nulls {
                            indices.push_null();
                        }
                        previous = offset;
                        new_offsets.push(current_offset);
                    })
                }
                (indices, new_offsets)
            };

            let indices = indices.freeze();

            // SAFETY: the indices we generate are in bounds.
            let chunk = unsafe { take_unchecked(&*values, &indices) };
            // SAFETY: inner_dtype should be correct
            let s = unsafe {
                Series::from_chunks_and_dtype_unchecked(
                    self.name().clone(),
                    vec![chunk],
                    &self.inner_dtype().to_physical(),
                )
            };
            // SAFETY: monotonically increasing
            let new_offsets = unsafe { OffsetsBuffer::new_unchecked(new_offsets.into()) };
            (s, new_offsets)
        };
        debug_assert_eq!(s.name(), self.name());
        // restore logical type
        s = unsafe { s.from_physical_unchecked(self.inner_dtype()) }.unwrap();

        Ok((s, offsets))
    }
}

#[cfg(feature = "dtype-array")]
impl ChunkExplode for ArrayChunked {
    fn offsets(&self) -> PolarsResult<OffsetsBuffer<i64>> {
        // fast-path for non-null array.
        if self.null_count() == 0 {
            let width = self.width() as i64;
            let offsets = (0..self.len() + 1)
                .map(|i| {
                    let i = i as i64;
                    i * width
                })
                .collect::<Vec<_>>();
            // SAFETY: monotonically increasing
            let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets.into()) };

            return Ok(offsets);
        }

        let ca = self.rechunk();
        let arr = ca.downcast_iter().next().unwrap();
        // we have already ensure that validity is not none.
        let validity = arr.validity().unwrap();
        let width = arr.width();

        let mut current_offset = 0i64;
        let offsets = (0..=arr.len())
            .map(|i| {
                if i == 0 {
                    return current_offset;
                }
                // SAFETY: we are within bounds
                if unsafe { validity.get_unchecked(i - 1) } {
                    current_offset += width as i64
                }
                current_offset
            })
            .collect::<Vec<_>>();
        // SAFETY: monotonically increasing
        let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets.into()) };
        Ok(offsets)
    }

    fn explode_and_offsets(
        &self,
        options: ExplodeOptions,
    ) -> PolarsResult<(Series, OffsetsBuffer<i64>)> {
        if self.width() == 0 {
            let mut num_nulls = 0;
            if options.empty_as_null {
                num_nulls += self.len() - self.null_count();
            }
            if options.keep_nulls {
                num_nulls += self.null_count();
            }
            let offsets = (0..num_nulls as i64 + 1).collect::<Vec<i64>>();
            // SAFETY: monotonically increasing
            let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets.into()) };
            let s = Column::new_scalar(
                self.name().clone(),
                Scalar::null(self.inner_dtype().clone()),
                num_nulls,
            )
            .take_materialized_series();

            return Ok((s, offsets));
        }

        let ca = self.rechunk();
        let arr = ca.downcast_iter().next().unwrap();
        // fast-path for non-null array.
        if arr.null_count() == 0 {
            let s = unsafe {
                Series::from_chunks_and_dtype_unchecked(
                    self.name().clone(),
                    vec![array_values(arr)],
                    ca.inner_dtype(),
                )
            };
            let width = self.width() as i64;
            let offsets = (0..self.len() + 1)
                .map(|i| {
                    let i = i as i64;
                    i * width
                })
                .collect::<Vec<_>>();
            // SAFETY: monotonically increasing
            let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets.into()) };
            return Ok((s, offsets));
        }

        // we have already ensure that validity is not none.
        let validity = arr.validity().unwrap();
        let values = array_values(arr);
        let width = arr.width();

        let mut indices = PlPrimitiveArrayBuilder::<IdxSize>::with_capacity(
            values.len() - arr.null_count() * (width - 1),
        );
        let mut offsets = Vec::with_capacity(arr.len() + 1);
        let mut current_offset = 0i64;
        offsets.push(current_offset);
        (0..arr.len()).for_each(|i| {
            // SAFETY: we are within bounds
            if unsafe { validity.get_unchecked(i) } {
                let start = (i * width) as IdxSize;
                let end = start + width as IdxSize;
                indices.push_values(start..end);
                current_offset += width as i64;
            } else if options.keep_nulls {
                indices.push_null();
            }
            offsets.push(current_offset);
        });

        let indices = indices.freeze();

        // SAFETY: the indices we generate are in bounds
        let chunk = unsafe { take_unchecked(&*values, &indices) };
        // SAFETY: monotonically increasing
        let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets.into()) };

        Ok((
            // SAFETY: inner_dtype should be correct
            unsafe {
                Series::from_chunks_and_dtype_unchecked(
                    ca.name().clone(),
                    vec![chunk],
                    ca.inner_dtype(),
                )
            },
            offsets,
        ))
    }
}

#[cfg(test)]
mod test {
    use polars_array::{PlListArray, PlNullArray, PlPrimitiveArray};
    use polars_buffer::Buffer;

    use super::*;

    fn list_chunked(
        values: Box<dyn PlArray>,
        offsets: &[u64],
        validity: Option<&[bool]>,
    ) -> ListChunked {
        list_chunked_of(values, offsets, validity, DataType::Int32)
    }

    fn list_chunked_of(
        values: Box<dyn PlArray>,
        offsets: &[u64],
        validity: Option<&[bool]>,
        inner: DataType,
    ) -> ListChunked {
        let length = offsets.len() - 1;
        let validity = validity.map(|v| PlBitmap::from_bitmap(v.iter().copied().collect()));
        let arr = PlListArray::new(values, Buffer::from(offsets.to_vec()), length, validity);
        unsafe {
            ListChunked::from_chunks_and_dtype(
                PlSmallStr::from_static("a"),
                vec![arr.into_boxed()],
                DataType::List(Box::new(inner)),
            )
        }
    }

    /// A values array with no elements has no element the rest could stand for, whatever it
    /// answers about being scalar -- and a null array answers `true` at every length.
    #[test]
    fn test_explode_empty_values() -> PolarsResult<()> {
        // Three rows, all of them empty or null, over a values array of no elements at all.
        let offsets = [0u64, 0, 0, 0];
        let validity = [true, false, true];

        for (values, inner) in [
            (
                PlPrimitiveArray::<i32>::new_empty().into_boxed(),
                DataType::Int32,
            ),
            (PlNullArray::new(0).into_boxed(), DataType::Null),
        ] {
            for validity in [None, Some(validity.as_slice())] {
                let ca = list_chunked_of(values.clone(), &offsets, validity, inner.clone());

                for empty_as_null in [false, true] {
                    for keep_nulls in [false, true] {
                        let options = ExplodeOptions {
                            empty_as_null,
                            keep_nulls,
                        };
                        let (out, out_offsets) = ca.explode_and_offsets(options)?;
                        assert_eq!(out.len(), out.null_count(), "{inner:?}, {options:?}");
                        assert_eq!(out_offsets.len(), offsets.len());
                    }
                }
            }
        }

        Ok(())
    }

    /// A list array whose values read one repeated element explodes to what the same values
    /// explode to flat, whatever the offsets and the mask do to the rows.
    #[test]
    fn test_explode_scalar_values() -> PolarsResult<()> {
        // Rows of 3, 0, 2, 0 and 3 elements, the fourth of them null.
        let offsets = [0u64, 3, 3, 5, 5, 8];
        let masks = [None, Some([true, true, true, false, true].as_slice())];

        for value in [Some(7i32), None] {
            let scalar = match value {
                Some(value) => PlPrimitiveArray::new_scalar(value, 8),
                None => PlPrimitiveArray::<i32>::new_full_null(8),
            };
            let flat = PlPrimitiveArray::new(
                Buffer::from(vec![value.unwrap_or_default(); 8]),
                8,
                value
                    .is_none()
                    .then(|| PlBitmap::from_bitmap(polars_arrow::bitmap::Bitmap::new_zeroed(8))),
            );

            assert!(PlArray::is_scalar(&scalar));

            for validity in masks {
                let scalar_ca = list_chunked(scalar.clone().into_boxed(), &offsets, validity);
                let flat_ca = list_chunked(flat.clone().into_boxed(), &offsets, validity);
                assert!(!scalar_ca._can_fast_explode() && !flat_ca._can_fast_explode());

                for empty_as_null in [false, true] {
                    for keep_nulls in [false, true] {
                        let options = ExplodeOptions {
                            empty_as_null,
                            keep_nulls,
                        };
                        let (scalar_out, scalar_offsets) =
                            scalar_ca.explode_and_offsets(options)?;
                        let (flat_out, flat_offsets) = flat_ca.explode_and_offsets(options)?;

                        assert_eq!(
                            scalar_out, flat_out,
                            "value {value:?}, mask {validity:?}, {options:?}"
                        );
                        assert_eq!(scalar_offsets.as_slice(), flat_offsets.as_slice());
                    }
                }
            }
        }

        Ok(())
    }
}
