use arrow::array::BooleanArray;

use super::*;

#[derive(Default)]
pub struct HasNullsReduce {
    has_nulls: MutableBitmap,
    evicted_has_nulls: BitmapBuilder,
}

impl HasNullsReduce {
    pub fn new() -> Self {
        Self::default()
    }
}

impl GroupedReduction for HasNullsReduce {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::default())
    }

    fn reserve(&mut self, additional: usize) {
        self.has_nulls.reserve(additional);
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.has_nulls.resize(num_groups as usize, false);
    }

    fn update_group(
        &mut self,
        values: &[&Column],
        group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        let &[values] = values else { unreachable!() };
        if !unsafe { self.has_nulls.get_unchecked(group_idx as usize) } && values.has_nulls() {
            self.has_nulls.set(group_idx as usize, true);
        }
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
        if arr.has_nulls() {
            let valid = arr.validity().unwrap();
            for (i, g) in subset.iter().zip(group_idxs) {
                let already_has_nulls = self.has_nulls.get_unchecked(g.idx());
                if g.should_evict() {
                    self.evicted_has_nulls.push(already_has_nulls);
                    let is_null = !valid.get_bit_unchecked(*i as usize);
                    self.has_nulls.set_unchecked(g.idx(), is_null);
                } else {
                    if !already_has_nulls {
                        let is_null = !valid.get_bit_unchecked(*i as usize);
                        self.has_nulls.set_unchecked(g.idx(), is_null);
                    }
                }
            }
        } else {
            for (_, g) in subset.iter().zip(group_idxs) {
                if g.should_evict() {
                    self.evicted_has_nulls
                        .push(self.has_nulls.get_unchecked(g.idx()));
                    self.has_nulls.set_unchecked(g.idx(), false);
                }
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
        for (i, g) in subset.iter().zip(group_idxs) {
            self.has_nulls
                .or_pos_unchecked(*g as usize, other.has_nulls.get_unchecked(*i as usize));
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            has_nulls: core::mem::take(&mut self.evicted_has_nulls).into_mut(),
            evicted_has_nulls: BitmapBuilder::new(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let v = core::mem::take(&mut self.has_nulls);
        let arr = BooleanArray::from(v.freeze());
        Ok(Series::from_array(PlSmallStr::EMPTY, arr))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
