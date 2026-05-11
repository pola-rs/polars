use arrow::array::BooleanArray;

use super::*;

pub struct IsEmptyReduce {
    is_empty: MutableBitmap,
    evicted_is_empty: BitmapBuilder,
    ignore_nulls: bool,
}

impl IsEmptyReduce {
    pub fn new(ignore_nulls: bool) -> Self {
        Self {
            is_empty: MutableBitmap::new(),
            evicted_is_empty: BitmapBuilder::new(),
            ignore_nulls,
        }
    }
}

impl GroupedReduction for IsEmptyReduce {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::new(self.ignore_nulls))
    }

    fn reserve(&mut self, additional: usize) {
        self.is_empty.reserve(additional);
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.is_empty.resize(num_groups as usize, true);
    }

    fn update_group(
        &mut self,
        values: &[&Column],
        group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        let &[values] = values else { unreachable!() };
        let is_empty = if self.ignore_nulls {
            values.is_full_null()
        } else {
            values.is_empty()
        };
        self.is_empty.and_pos(group_idx as usize, is_empty);
        Ok(())
    }

    unsafe fn update_groups_while_evicting(
        &mut self,
        values: &[&Column],
        subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        _seq_id: u64,
    ) -> PolarsResult<()> {
        let &[values] = values else { unreachable!() };
        assert!(subset.len() == group_idxs.len());
        let values = values.as_materialized_series(); // @scalar-opt
        let chunks = values.chunks();
        assert!(chunks.len() == 1);
        let arr = &*chunks[0];
        if arr.has_nulls() && !self.ignore_nulls {
            let valid = arr.validity().unwrap();
            for (i, g) in subset.iter().zip(group_idxs) {
                let mut is_empty = self.is_empty.get_unchecked(g.idx());
                if g.should_evict() {
                    self.evicted_is_empty.push(is_empty);
                    is_empty = true;
                }
                is_empty |= valid.get_bit_unchecked(*i as usize);
                self.is_empty.set_unchecked(g.idx(), is_empty);
            }
        } else {
            for (_, g) in subset.iter().zip(group_idxs) {
                if g.should_evict() {
                    self.evicted_is_empty
                        .push(self.is_empty.get_unchecked(g.idx()));
                }
                self.is_empty.set_unchecked(g.idx(), false);
            }
        }
        Ok(())
    }

    unsafe fn combine_subset(
        &mut self,
        other: &dyn GroupedReduction,
        subset: &[IdxSize],
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        assert!(subset.len() == group_idxs.len());
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                self.is_empty
                    .and_pos_unchecked(*g as usize, other.is_empty.get_unchecked(*i as usize));
            }
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            is_empty: core::mem::take(&mut self.evicted_is_empty).into_mut(),
            evicted_is_empty: BitmapBuilder::new(),
            ignore_nulls: self.ignore_nulls,
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let is_empty = core::mem::take(&mut self.is_empty);
        let arr = BooleanArray::from(is_empty.freeze()).boxed();
        Ok(unsafe {
            Series::from_chunks_and_dtype_unchecked(
                PlSmallStr::EMPTY,
                vec![arr],
                &DataType::Boolean,
            )
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
