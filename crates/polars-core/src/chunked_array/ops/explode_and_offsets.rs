use arrow::compute::take::take_unchecked;
use arrow::offset::OffsetsBuffer;

use super::*;

impl ListChunked {
    fn specialized(
        &self,
        values: ArrayRef,
        offsets: &[i64],
        offsets_buf: OffsetsBuffer<i64>,
    ) -> (Series, OffsetsBuffer<i64>) {
        // SAFETY: inner_dtype should be correct
        let values = unsafe {
            Series::from_chunks_and_dtype_unchecked(
                self.name(),
                vec![values],
                &self.inner_dtype().to_physical(),
            )
        };

        use crate::chunked_array::ops::explode::ExplodeByOffsets;

        let mut values = match values.dtype() {
            DataType::Boolean => {
                let t = values.bool().unwrap();
                ExplodeByOffsets::explode_by_offsets(t, offsets).into_series()
            },
            DataType::Null => {
                let t = values.null().unwrap();
                ExplodeByOffsets::explode_by_offsets(t, offsets).into_series()
            },
            dtype => {
                with_match_physical_numeric_polars_type!(dtype, |$T| {
                    let t: &ChunkedArray<$T> = values.as_ref().as_ref();
                    ExplodeByOffsets::explode_by_offsets(t, offsets).into_series()
                })
            },
        };

        // let mut values = values.explode_by_offsets(offsets);
        // restore logical type
        unsafe {
            values = values.cast_unchecked(self.inner_dtype()).unwrap();
        }

        (values, offsets_buf)
    }
}

impl ChunkExplode for ListChunked {
    fn offsets(&self) -> PolarsResult<OffsetsBuffer<i64>> {
        let ca = self.rechunk();
        let listarr: &LargeListArray = ca.downcast_iter().next().unwrap();
        let offsets = listarr.offsets().clone();

        Ok(offsets)
    }

    fn explode_and_offsets(&self) -> PolarsResult<(Series, OffsetsBuffer<i64>)> {
        // A list array's memory layout is actually already 'exploded', so we can just take the
        // values array of the list. And we also return a slice of the offsets. This slice can be
        // used to find the old list layout or indexes to expand a DataFrame in the same manner as
        // the `explode` operation.
        let ca = self.rechunk();
        let listarr: &LargeListArray = ca.downcast_iter().next().unwrap();
        let offsets_buf = listarr.offsets().clone();
        let offsets = listarr.offsets().as_slice();
        let mut values = listarr.values().clone();

        let (mut s, offsets) = if ca._can_fast_explode() {
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
                        self.name(),
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
            let (indices, new_offsets) = if listarr.null_count() == 0 {
                // SPECIALIZED path.
                let inner_phys = self.inner_dtype().to_physical();
                if inner_phys.is_numeric() || inner_phys.is_null() || inner_phys.is_bool() {
                    return Ok(self.specialized(values, offsets, offsets_buf));
                }
                // Use gather
                let mut indices =
                    MutablePrimitiveArray::<IdxSize>::with_capacity(*offsets_buf.last() as usize);
                let mut new_offsets = Vec::with_capacity(listarr.len() + 1);
                let mut current_offset = 0i64;
                let mut iter = offsets.iter();
                if let Some(mut previous) = iter.next().copied() {
                    new_offsets.push(current_offset);
                    iter.for_each(|&offset| {
                        let len = offset - previous;
                        let start = previous as IdxSize;
                        let end = offset as IdxSize;

                        if len == 0 {
                            indices.push_null();
                        } else {
                            indices.extend_trusted_len_values(start..end);
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
                    MutablePrimitiveArray::<IdxSize>::with_capacity(*offsets_buf.last() as usize);
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
                            if len == 0 {
                                indices.push_null();
                            } else {
                                indices.extend_trusted_len_values(start..end);
                            }
                            current_offset += len;
                        } else {
                            indices.push_null();
                        }
                        previous = offset;
                        new_offsets.push(current_offset);
                    })
                }
                (indices, new_offsets)
            };

            // SAFETY: the indices we generate are in bounds
            let chunk = unsafe { take_unchecked(values.as_ref(), &indices.into()) };
            // SAFETY: inner_dtype should be correct
            let s = unsafe {
                Series::from_chunks_and_dtype_unchecked(
                    self.name(),
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
        unsafe {
            s = s.cast_unchecked(self.inner_dtype()).unwrap();
        }

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
        let width = arr.size();

        let mut current_offset = 0i64;
        let offsets = (0..=arr.len())
            .map(|i| {
                if i == 0 {
                    return current_offset;
                }
                // SAFETY: we are within bounds
                if unsafe { validity.get_bit_unchecked(i - 1) } {
                    current_offset += width as i64
                }
                current_offset
            })
            .collect::<Vec<_>>();
        // SAFETY: monotonically increasing
        let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets.into()) };
        Ok(offsets)
    }

    fn explode_and_offsets(&self) -> PolarsResult<(Series, OffsetsBuffer<i64>)> {
        let ca = self.rechunk();
        let arr = ca.downcast_iter().next().unwrap();
        // fast-path for non-null array.
        if arr.null_count() == 0 {
            let s = Series::try_from((self.name(), arr.values().clone()))
                .unwrap()
                .cast(ca.inner_dtype())?;
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
        let values = arr.values();
        let width = arr.size();

        let mut indices = MutablePrimitiveArray::<IdxSize>::with_capacity(
            values.len() - arr.null_count() * (width - 1),
        );
        let mut offsets = Vec::with_capacity(arr.len() + 1);
        let mut current_offset = 0i64;
        offsets.push(current_offset);
        (0..arr.len()).for_each(|i| {
            // SAFETY: we are within bounds
            if unsafe { validity.get_bit_unchecked(i) } {
                let start = (i * width) as IdxSize;
                let end = start + width as IdxSize;
                indices.extend_trusted_len_values(start..end);
                current_offset += width as i64;
            } else {
                indices.push_null();
            }
            offsets.push(current_offset);
        });

        // SAFETY: the indices we generate are in bounds
        let chunk = unsafe { take_unchecked(&**values, &indices.into()) };
        // SAFETY: monotonically increasing
        let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets.into()) };

        Ok((
            // SAFETY: inner_dtype should be correct
            unsafe {
                Series::from_chunks_and_dtype_unchecked(ca.name(), vec![chunk], ca.inner_dtype())
            },
            offsets,
        ))
    }
}
