use arrow::bitmap::MutableBitmap;

use super::*;

impl ChunkExplode for ListChunked {
    fn offsets(&self) -> PolarsResult<OffsetsBuffer<i64>> {
        let ca = self.rechunk();
        let listarr: &LargeListArray = ca.downcast_iter().next().unwrap();
        let offsets = listarr.offsets().clone();

        Ok(offsets)
    }

    fn explode_and_offsets(&self) -> PolarsResult<(Series, OffsetsBuffer<i64>)> {
        // A list array's memory layout is actually already 'exploded', so we can just take the values array
        // of the list. And we also return a slice of the offsets. This slice can be used to find the old
        // list layout or indexes to expand the DataFrame in the same manner as the 'explode' operation
        let ca = self.rechunk();
        let listarr: &LargeListArray = ca.downcast_iter().next().unwrap();
        let offsets_buf = listarr.offsets().clone();
        let offsets = listarr.offsets().as_slice();
        let mut values = listarr.values().clone();

        let mut s = if ca._can_fast_explode() {
            // ensure that the value array is sliced
            // as a list only slices its offsets on a slice operation

            // we only do this in fast-explode as for the other
            // branch the offsets must coincide with the values.
            if !offsets.is_empty() {
                let start = offsets[0] as usize;
                let len = offsets[offsets.len() - 1] as usize - start;
                // safety:
                // we are in bounds
                values = unsafe { values.sliced_unchecked(start, len) };
            }
            // safety: inner_dtype should be correct
            unsafe {
                Series::from_chunks_and_dtype_unchecked(
                    self.name(),
                    vec![values],
                    &self.inner_dtype().to_physical(),
                )
            }
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

            // safety: inner_dtype should be correct
            let values = unsafe {
                Series::from_chunks_and_dtype_unchecked(
                    self.name(),
                    vec![values],
                    &self.inner_dtype().to_physical(),
                )
            };
            values.explode_by_offsets(offsets)
        };
        debug_assert_eq!(s.name(), self.name());
        // restore logical type
        unsafe {
            s = s.cast_unchecked(&self.inner_dtype()).unwrap();
        }

        Ok((s, offsets_buf))
    }
}


#[cfg(feature = "dtype-array")]
impl ChunkExplode for ArrayChunked {
    fn offsets(&self) -> PolarsResult<OffsetsBuffer<i64>> {
        let width = self.width() as i64;
        let offsets = (0..self.len() + 1)
            .map(|i| {
                let i = i as i64;
                i * width
            })
            .collect::<Vec<_>>();
        // safety: monotonically increasing
        let offsets = unsafe { OffsetsBuffer::new_unchecked(offsets.into()) };

        Ok(offsets)
    }

    fn explode(&self) -> PolarsResult<Series> {
        let ca = self.rechunk();
        let arr = ca.downcast_iter().next().unwrap();
        Ok(Series::try_from((self.name(), arr.values().clone())).unwrap())
    }

    fn explode_and_offsets(&self) -> PolarsResult<(Series, OffsetsBuffer<i64>)> {
        let s = self.explode().unwrap();

        Ok((s, self.offsets()?))
    }
}
