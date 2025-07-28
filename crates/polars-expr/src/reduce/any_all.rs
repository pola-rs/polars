use arrow::array::BooleanArray;
use arrow::bitmap::binary_assign_mut;

use super::*;

pub fn new_any_reduction(ignore_nulls: bool) -> Box<dyn GroupedReduction> {
    if ignore_nulls {
        Box::new(AnyIgnoreNullGroupedReduction::default())
    } else {
        Box::new(AnyKleeneNullGroupedReduction::default())
    }
}

pub fn new_all_reduction(ignore_nulls: bool) -> Box<dyn GroupedReduction> {
    if ignore_nulls {
        Box::new(AllIgnoreNullGroupedReduction::default())
    } else {
        Box::new(AllKleeneNullGroupedReduction::default())
    }
}

#[derive(Default)]
struct AnyIgnoreNullGroupedReduction {
    values: MutableBitmap,
    evicted_values: BitmapBuilder,
}

impl GroupedReduction for AnyIgnoreNullGroupedReduction {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::default())
    }

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.values.resize(num_groups as usize, false);
    }

    fn update_group(
        &mut self,
        values: &Column,
        group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.dtype() == &DataType::Boolean);
        let values = values.as_materialized_series_maintain_scalar();
        let ca: &BooleanChunked = values.as_ref().as_ref();
        if ca.any() {
            self.values.set(group_idx as usize, true);
        }
        Ok(())
    }

    unsafe fn update_groups_while_evicting(
        &mut self,
        values: &Column,
        subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.dtype() == &DataType::Boolean);
        assert!(subset.len() == group_idxs.len());
        let values = values.as_materialized_series(); // @scalar-opt
        let ca: &BooleanChunked = values.as_ref().as_ref();
        let arr = ca.downcast_as_array();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                let ov = arr.get_unchecked(*i as usize);
                if g.should_evict() {
                    self.evicted_values.push(self.values.get_unchecked(g.idx()));
                    self.values.set_unchecked(g.idx(), ov.unwrap_or(false));
                } else {
                    self.values.or_pos_unchecked(g.idx(), ov.unwrap_or(false));
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
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                self.values
                    .or_pos_unchecked(*g as usize, other.values.get_unchecked(*i as usize));
            }
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            values: core::mem::take(&mut self.evicted_values).into_mut(),
            evicted_values: BitmapBuilder::new(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let v = core::mem::take(&mut self.values);
        let arr = BooleanArray::from(v.freeze());
        Ok(Series::from_array(PlSmallStr::EMPTY, arr))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Default)]
struct AllIgnoreNullGroupedReduction {
    values: MutableBitmap,
    evicted_values: BitmapBuilder,
}

impl GroupedReduction for AllIgnoreNullGroupedReduction {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::default())
    }

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.values.resize(num_groups as usize, true);
    }

    fn update_group(
        &mut self,
        values: &Column,
        group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.dtype() == &DataType::Boolean);
        let values = values.as_materialized_series_maintain_scalar();
        let ca: &BooleanChunked = values.as_ref().as_ref();
        if !ca.all() {
            self.values.set(group_idx as usize, false);
        }
        Ok(())
    }

    unsafe fn update_groups_while_evicting(
        &mut self,
        values: &Column,
        subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.dtype() == &DataType::Boolean);
        assert!(subset.len() == group_idxs.len());
        let values = values.as_materialized_series(); // @scalar-opt
        let ca: &BooleanChunked = values.as_ref().as_ref();
        let arr = ca.downcast_as_array();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                let ov = arr.get_unchecked(*i as usize);
                if g.should_evict() {
                    self.evicted_values.push(self.values.get_unchecked(g.idx()));
                    self.values.set_unchecked(g.idx(), ov.unwrap_or(true));
                } else {
                    self.values.and_pos_unchecked(g.idx(), ov.unwrap_or(true));
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
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                self.values
                    .and_pos_unchecked(*g as usize, other.values.get_unchecked(*i as usize));
            }
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            values: core::mem::take(&mut self.evicted_values).into_mut(),
            evicted_values: BitmapBuilder::new(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let v = core::mem::take(&mut self.values);
        let arr = BooleanArray::from(v.freeze());
        Ok(Series::from_array(PlSmallStr::EMPTY, arr))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Default)]
struct AnyKleeneNullGroupedReduction {
    seen_true: MutableBitmap,
    seen_null: MutableBitmap,
    evicted_values: BitmapBuilder,
    evicted_mask: BitmapBuilder,
}

impl GroupedReduction for AnyKleeneNullGroupedReduction {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::default())
    }

    fn reserve(&mut self, additional: usize) {
        self.seen_true.reserve(additional);
        self.seen_null.reserve(additional)
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.seen_true.resize(num_groups as usize, false);
        self.seen_null.resize(num_groups as usize, false);
    }

    fn update_group(
        &mut self,
        values: &Column,
        group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.dtype() == &DataType::Boolean);
        let values = values.as_materialized_series_maintain_scalar();
        let ca: &BooleanChunked = values.as_ref().as_ref();
        if ca.any() {
            self.seen_true.set(group_idx as usize, true);
        }
        if ca.len() != ca.null_count() {
            self.seen_null.set(group_idx as usize, true);
        }
        Ok(())
    }

    unsafe fn update_groups_while_evicting(
        &mut self,
        values: &Column,
        subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.dtype() == &DataType::Boolean);
        assert!(subset.len() == group_idxs.len());
        let values = values.as_materialized_series(); // @scalar-opt
        let ca: &BooleanChunked = values.as_ref().as_ref();
        let arr = ca.downcast_as_array();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                let ov = arr.get_unchecked(*i as usize);
                if g.should_evict() {
                    self.evicted_values
                        .push(self.seen_true.get_unchecked(g.idx()));
                    self.evicted_mask
                        .push(self.seen_null.get_unchecked(g.idx()));
                    self.seen_true.set_unchecked(g.idx(), ov.unwrap_or(false));
                    self.seen_null.set_unchecked(g.idx(), ov.is_none());
                } else {
                    self.seen_true
                        .or_pos_unchecked(g.idx(), ov.unwrap_or(false));
                    self.seen_null.or_pos_unchecked(g.idx(), ov.is_none());
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
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                self.seen_true
                    .or_pos_unchecked(*g as usize, other.seen_true.get_unchecked(*i as usize));
                self.seen_null
                    .or_pos_unchecked(*g as usize, other.seen_null.get_unchecked(*i as usize));
            }
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            seen_true: core::mem::take(&mut self.evicted_values).into_mut(),
            seen_null: core::mem::take(&mut self.evicted_mask).into_mut(),
            evicted_values: BitmapBuilder::new(),
            evicted_mask: BitmapBuilder::new(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let seen_true = core::mem::take(&mut self.seen_true);
        let mut mask = core::mem::take(&mut self.seen_null);
        binary_assign_mut(&mut mask, &seen_true, |mi: u64, ti: u64| mi & !ti);
        let arr = BooleanArray::from(seen_true.freeze())
            .with_validity(Some(mask.freeze()))
            .boxed();
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

#[derive(Default)]
struct AllKleeneNullGroupedReduction {
    seen_false: MutableBitmap,
    seen_null: MutableBitmap,
    evicted_values: BitmapBuilder,
    evicted_mask: BitmapBuilder,
}

impl GroupedReduction for AllKleeneNullGroupedReduction {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::default())
    }

    fn reserve(&mut self, additional: usize) {
        self.seen_false.reserve(additional);
        self.seen_null.reserve(additional)
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.seen_false.resize(num_groups as usize, false);
        self.seen_null.resize(num_groups as usize, false);
    }

    fn update_group(
        &mut self,
        values: &Column,
        group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.dtype() == &DataType::Boolean);
        let values = values.as_materialized_series_maintain_scalar();
        let ca: &BooleanChunked = values.as_ref().as_ref();
        if !ca.all() {
            self.seen_false.set(group_idx as usize, true);
        }
        if ca.len() != ca.null_count() {
            self.seen_null.set(group_idx as usize, true);
        }
        Ok(())
    }

    unsafe fn update_groups_while_evicting(
        &mut self,
        values: &Column,
        subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.dtype() == &DataType::Boolean);
        assert!(subset.len() == group_idxs.len());
        let values = values.as_materialized_series(); // @scalar-opt
        let ca: &BooleanChunked = values.as_ref().as_ref();
        let arr = ca.downcast_as_array();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                let ov = arr.get_unchecked(*i as usize);
                if g.should_evict() {
                    self.evicted_values
                        .push(self.seen_false.get_unchecked(g.idx()));
                    self.evicted_mask
                        .push(self.seen_null.get_unchecked(g.idx()));
                    self.seen_false.set_unchecked(g.idx(), !ov.unwrap_or(true));
                    self.seen_null.set_unchecked(g.idx(), ov.is_none());
                } else {
                    self.seen_false
                        .or_pos_unchecked(g.idx(), !ov.unwrap_or(true));
                    self.seen_null.or_pos_unchecked(g.idx(), ov.is_none());
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
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                self.seen_false
                    .or_pos_unchecked(*g as usize, other.seen_false.get_unchecked(*i as usize));
                self.seen_null
                    .or_pos_unchecked(*g as usize, other.seen_null.get_unchecked(*i as usize));
            }
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            seen_false: core::mem::take(&mut self.evicted_values).into_mut(),
            seen_null: core::mem::take(&mut self.evicted_mask).into_mut(),
            evicted_values: BitmapBuilder::new(),
            evicted_mask: BitmapBuilder::new(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let seen_false = core::mem::take(&mut self.seen_false);
        let mut mask = core::mem::take(&mut self.seen_null);
        binary_assign_mut(&mut mask, &seen_false, |mi: u64, fi: u64| mi & !fi);
        let arr = BooleanArray::from((!seen_false).freeze())
            .with_validity(Some(mask.freeze()))
            .boxed();
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
