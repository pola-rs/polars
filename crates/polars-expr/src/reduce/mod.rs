#![allow(unsafe_op_in_unsafe_fn)]
mod any_all;
#[cfg(feature = "approx_unique")]
mod approx_n_unique;
#[cfg(feature = "approx_quantile")]
mod approx_quantile;
#[cfg(feature = "bitwise")]
mod bitwise;
mod convert;
mod count;
#[cfg(feature = "cov")]
mod cov;
mod first_last;
mod first_last_nonnull;
mod has_nulls;
mod implode;
mod is_empty;
mod mean;
mod min_max;
mod min_max_by;
#[cfg(feature = "moment")]
mod skew_kurtosis;
mod sum;
mod var_std;

use std::any::Any;
use std::borrow::Cow;
use std::marker::PhantomData;

#[cfg(feature = "approx_quantile")]
pub use approx_quantile::{
    new_approx_quantile_merge_reduction, new_approx_quantile_state_reduction,
};
pub use convert::into_reduction;
pub use min_max::{new_max_reduction, new_min_reduction};
use polars_arrow::array::{Array, PrimitiveArray, StaticArray};
use polars_arrow::bitmap::utils::{get_bit_unchecked, set_bit_unchecked};
use polars_arrow::bitmap::{Bitmap, BitmapBuilder, MutableBitmap};
use polars_core::prelude::*;

use crate::EvictIdx;

/// For small grouped reductions we always allocate at least this many groups,
/// if there are no evictions. This way we can spread aggregations over multiple
/// lanes, reducing store-load stalls.
const LANE_SLOTS: usize = 64;
const LANE_MASK: usize = LANE_SLOTS - 1;

/// The layout of the per-group values of a grouped reduction: lane `l` of group
/// `g` is stored in slot `g + l * stride`. Update `r` goes to the slot at offset
/// `(r * stride) % LANE_SLOTS` from its group, which is always zero when there
/// is only one lane.
#[derive(Clone, Copy)]
struct LaneLayout {
    num_groups: usize,
    stride: usize,
}

impl LaneLayout {
    fn single(num_groups: usize) -> Self {
        Self {
            num_groups,
            stride: LANE_SLOTS,
        }
    }

    /// The layout that spreads the groups over lanes, if that is useful.
    fn new(num_groups: usize) -> Self {
        let stride = num_groups.next_power_of_two();
        if num_groups > 1 && stride < LANE_SLOTS {
            Self { num_groups, stride }
        } else {
            Self::single(num_groups)
        }
    }

    fn num_lanes(&self) -> usize {
        (LANE_SLOTS / self.stride).max(1)
    }

    fn num_slots(&self) -> usize {
        if self.num_lanes() > 1 {
            LANE_SLOTS
        } else {
            self.num_groups
        }
    }

    /// Combines all lanes into the first lane and truncates `values` to it.
    fn fold<V>(&self, values: &mut Vec<V>, combine: impl Fn(&mut V, &V)) {
        assert!(values.len() == self.num_slots());
        if self.num_lanes() > 1 {
            let (first, rest) = values.split_at_mut(self.stride);
            for lane in rest.chunks_exact(self.stride) {
                for (a, b) in first.iter_mut().zip(lane) {
                    combine(a, b);
                }
            }
        }
        values.truncate(self.num_groups);
    }

    /// Combines all lanes into the first lane and switches to a single lane.
    fn collapse<V>(&mut self, values: &mut Vec<V>, combine: impl Fn(&mut V, &V)) {
        self.fold(values, combine);
        *self = Self::single(self.num_groups);
    }

    /// Resizes to `num_groups` groups, collapsing the lanes if the number changes.
    fn resize<V: Clone>(
        &mut self,
        values: &mut Vec<V>,
        num_groups: usize,
        init: V,
        combine: impl Fn(&mut V, &V),
    ) {
        if self.num_lanes() > 1 && num_groups != self.num_groups {
            self.collapse(values, combine);
        }
        if self.num_lanes() == 1 {
            values.resize(num_groups, init);
        }
        self.num_groups = num_groups;
    }

    /// Spreads the groups over lanes if that is useful.
    fn expand<V: Clone>(&mut self, values: &mut Vec<V>, init: V) {
        let new = Self::new(self.num_groups);
        if new.stride != self.stride {
            // `resize` collapses the lanes when the number of groups changes.
            debug_assert!(self.num_lanes() == 1);
            values.resize(new.num_slots(), init);
            *self = new;
        }
    }
}

/// A reduction with groups.
///
/// Each group has its own reduction state that values can be aggregated into.
pub trait GroupedReduction: Any + Send + Sync {
    /// Returns a new empty reduction.
    fn new_empty(&self) -> Box<dyn GroupedReduction>;

    /// Reserves space in this GroupedReduction for an additional number of groups.
    fn reserve(&mut self, additional: usize);

    /// Resizes this GroupedReduction to the given number of groups.
    ///
    /// While not an actual member of the trait, the safety preconditions below
    /// refer to self.num_groups() as given by the last call of this function.
    fn resize(&mut self, num_groups: IdxSize);

    /// Updates the specified group with the given values.
    ///
    /// For order-sensitive grouped reductions, seq_id can be used to resolve
    /// order between calls/multiple reductions.
    fn update_group(
        &mut self,
        values: &[&Column],
        group_idx: IdxSize,
        seq_id: u64,
    ) -> PolarsResult<()>;

    /// Updates this GroupedReduction with new values. values[subset[i]] should
    /// be added to reduction self[group_idxs[i]]. For order-sensitive grouped
    /// reductions, seq_id can be used to resolve order between calls/multiple
    /// reductions.
    ///
    /// The column MUST consist of single chunk.
    ///
    /// # Safety
    /// The subset and group_idxs are in-bounds.
    unsafe fn update_groups_subset(
        &mut self,
        values: &[&Column],
        subset: &[IdxSize],
        group_idxs: &[IdxSize],
        seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.len() < (1 << (IdxSize::BITS - 1)));
        // SAFETY: EvictIdx is a wrapper for IdxSize and has same alignment.
        let evict_group_idxs = EvictIdx::cast_slice(group_idxs);
        self.update_groups_while_evicting(values, subset, evict_group_idxs, seq_id)
    }

    /// Updates this GroupedReduction with new values. values[subset[i]] should
    /// be added to reduction self[group_idxs[i]]. For order-sensitive grouped
    /// reductions, seq_id can be used to resolve order between calls/multiple
    /// reductions. If the group_idxs[i] has its evict bit set the current value
    /// in the group should be evicted and reset before updating.
    ///
    /// The column MUST consist of single chunk.
    ///
    /// # Safety
    /// The subset and group_idxs are in-bounds.
    unsafe fn update_groups_while_evicting(
        &mut self,
        values: &[&Column],
        subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        seq_id: u64,
    ) -> PolarsResult<()>;

    /// Combines this GroupedReduction with another. Group other[subset[i]]
    /// should be combined into group self[group_idxs[i]].
    ///
    /// # Safety
    /// subset[i] < other.num_groups() for all i.
    /// group_idxs[i] < self.num_groups() for all i.
    unsafe fn combine_subset(
        &mut self,
        other: &dyn GroupedReduction,
        subset: &[IdxSize],
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()>;

    /// Take the accumulated evicted groups.
    fn take_evictions(&mut self) -> Box<dyn GroupedReduction>;

    /// Returns the finalized value per group as a Series.
    ///
    /// After this operation the number of groups is reset to 0.
    fn finalize(&mut self) -> PolarsResult<Series>;

    /// Returns whether the given group is "done": its finalized value can no
    /// longer change no matter what further data is fed to this reduction.
    ///
    /// Used by the non-grouped streaming reduce to short-circuit and stop
    /// pulling input once every reduction has settled. The default is
    /// conservatively `false`; only reductions whose result is monotone /
    /// existential (e.g. `any`, `all`, `has_nulls`) override it.
    fn is_group_done(&self, _group_idx: IdxSize) -> bool {
        false
    }

    /// Returns this GroupedReduction as a dyn Any.
    fn as_any(&self) -> &dyn Any;
}

// Helper traits used in the VecGroupedReduction and VecMaskGroupedReduction to
// reduce code duplication.
pub trait Reducer: Send + Sync + Clone + 'static {
    type Dtype: PolarsPhysicalType;
    type Value: Clone + Send + Sync + 'static;
    /// Whether the values of a group may be reduced into several states that are
    /// combined afterwards, changing the result by at most floating-point rounding.
    /// If so, `init` must be an identity of `combine`.
    const ORDER_INDEPENDENT: bool = false;
    fn init(&self) -> Self::Value;
    #[inline(always)]
    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        Cow::Borrowed(s)
    }
    fn combine(&self, a: &mut Self::Value, b: &Self::Value);
    fn reduce_one(
        &self,
        a: &mut Self::Value,
        b: Option<<Self::Dtype as PolarsDataType>::Physical<'_>>,
        seq_id: u64,
    );
    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, seq_id: u64);
    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series>;
}

pub trait NumericReduction: Send + Sync + 'static {
    type Dtype: PolarsNumericType;
    fn init() -> <Self::Dtype as PolarsNumericType>::Native;
    fn combine(
        a: <Self::Dtype as PolarsNumericType>::Native,
        b: <Self::Dtype as PolarsNumericType>::Native,
    ) -> <Self::Dtype as PolarsNumericType>::Native;
    fn reduce_ca(
        ca: &ChunkedArray<Self::Dtype>,
    ) -> Option<<Self::Dtype as PolarsNumericType>::Native>;
}

struct NumReducer<R: NumericReduction>(PhantomData<R>);
impl<R: NumericReduction> NumReducer<R> {
    fn new() -> Self {
        Self(PhantomData)
    }
}
impl<R: NumericReduction> Clone for NumReducer<R> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<R: NumericReduction> Reducer for NumReducer<R> {
    type Dtype = <R as NumericReduction>::Dtype;
    type Value = <<R as NumericReduction>::Dtype as PolarsNumericType>::Native;
    const ORDER_INDEPENDENT: bool = true;

    #[inline(always)]
    fn init(&self) -> Self::Value {
        <R as NumericReduction>::init()
    }

    #[inline(always)]
    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    #[inline(always)]
    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        *a = <R as NumericReduction>::combine(*a, *b);
    }

    #[inline(always)]
    fn reduce_one(
        &self,
        a: &mut Self::Value,
        b: Option<<Self::Dtype as PolarsDataType>::Physical<'_>>,
        _seq_id: u64,
    ) {
        if let Some(b) = b {
            *a = <R as NumericReduction>::combine(*a, b);
        }
    }

    #[inline(always)]
    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        if let Some(r) = <R as NumericReduction>::reduce_ca(ca) {
            *v = <R as NumericReduction>::combine(*v, r);
        }
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        let arr = Box::new(PrimitiveArray::<Self::Value>::from_vec(v).with_validity(m));
        Ok(unsafe { Series::from_chunks_and_dtype_unchecked(PlSmallStr::EMPTY, vec![arr], dtype) })
    }
}

pub struct VecGroupedReduction<R: Reducer> {
    values: Vec<R::Value>,
    layout: LaneLayout,
    evicted_values: Vec<R::Value>,
    in_dtype: DataType,
    reducer: R,
}

impl<R: Reducer> VecGroupedReduction<R> {
    pub fn new(in_dtype: DataType, reducer: R) -> Self {
        Self {
            values: Vec::new(),
            layout: LaneLayout::single(0),
            evicted_values: Vec::new(),
            in_dtype,
            reducer,
        }
    }
}

impl<R> GroupedReduction for VecGroupedReduction<R>
where
    R: Reducer,
{
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::new(self.in_dtype.clone(), self.reducer.clone()))
    }

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.layout.resize(
            &mut self.values,
            num_groups as usize,
            self.reducer.init(),
            |a, b| self.reducer.combine(a, b),
        );
    }

    fn update_group(
        &mut self,
        values: &[&Column],
        group_idx: IdxSize,
        seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.len() == 1);
        let values = values[0];
        assert!(values.dtype() == &self.in_dtype);
        let seq_id = seq_id + 1; // So we can use 0 for 'none yet'.
        let values = values.as_materialized_series(); // @scalar-opt
        let values = self.reducer.cast_series(values);
        let ca: &ChunkedArray<R::Dtype> = values.as_ref().as_ref().as_ref();
        self.reducer
            .reduce_ca(&mut self.values[group_idx as usize], ca, seq_id);
        Ok(())
    }

    unsafe fn update_groups_subset(
        &mut self,
        values: &[&Column],
        subset: &[IdxSize],
        group_idxs: &[IdxSize],
        seq_id: u64,
    ) -> PolarsResult<()> {
        let &[values] = values else { unreachable!() };
        assert!(values.dtype() == &self.in_dtype);
        assert!(subset.len() == group_idxs.len());
        let seq_id = seq_id + 1; // So we can use 0 for 'none yet'.
        let values = values.as_materialized_series(); // @scalar-opt
        let values = self.reducer.cast_series(values);

        if R::ORDER_INDEPENDENT {
            self.layout.expand(&mut self.values, self.reducer.init());
        }
        let ca: &ChunkedArray<R::Dtype> = values.as_ref().as_ref().as_ref();
        let arr = ca.downcast_as_array();
        let stride = self.layout.stride;
        let slots = self.values.as_mut_slice();
        let mut offset = 0usize;
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            if values.has_nulls() {
                for (i, g) in subset.iter().zip(group_idxs) {
                    let grp = slots.get_unchecked_mut(*g as usize + (offset & LANE_MASK));
                    self.reducer
                        .reduce_one(grp, arr.get_unchecked(*i as usize), seq_id);
                    offset = offset.wrapping_add(stride);
                }
            } else {
                for (i, g) in subset.iter().zip(group_idxs) {
                    let grp = slots.get_unchecked_mut(*g as usize + (offset & LANE_MASK));
                    self.reducer
                        .reduce_one(grp, Some(arr.value_unchecked(*i as usize)), seq_id);
                    offset = offset.wrapping_add(stride);
                }
            }
        }
        Ok(())
    }

    unsafe fn update_groups_while_evicting(
        &mut self,
        values: &[&Column],
        subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        seq_id: u64,
    ) -> PolarsResult<()> {
        let &[values] = values else { unreachable!() };
        assert!(values.dtype() == &self.in_dtype);
        assert!(subset.len() == group_idxs.len());
        let seq_id = seq_id + 1; // So we can use 0 for 'none yet'.
        let values = values.as_materialized_series(); // @scalar-opt
        let values = self.reducer.cast_series(values);
        let reducer = &self.reducer;
        self.layout
            .collapse(&mut self.values, |a, b| reducer.combine(a, b));

        let ca: &ChunkedArray<R::Dtype> = values.as_ref().as_ref().as_ref();
        let arr = ca.downcast_as_array();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            if values.has_nulls() {
                for (i, g) in subset.iter().zip(group_idxs) {
                    let ov = arr.get_unchecked(*i as usize);
                    let grp = self.values.get_unchecked_mut(g.idx());
                    if g.should_evict() {
                        let old = core::mem::replace(grp, self.reducer.init());
                        self.evicted_values.push(old);
                    }
                    self.reducer.reduce_one(grp, ov, seq_id);
                }
            } else {
                for (i, g) in subset.iter().zip(group_idxs) {
                    let v = arr.value_unchecked(*i as usize);
                    let grp = self.values.get_unchecked_mut(g.idx());
                    if g.should_evict() {
                        let old = core::mem::replace(grp, self.reducer.init());
                        self.evicted_values.push(old);
                    }
                    self.reducer.reduce_one(grp, Some(v), seq_id);
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
        assert!(self.in_dtype == other.in_dtype);
        assert!(subset.len() == group_idxs.len());
        let other_lanes = other.layout.num_lanes();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                let grp = self.values.get_unchecked_mut(*g as usize);
                for l in 0..other_lanes {
                    let v = other
                        .values
                        .get_unchecked(*i as usize + l * other.layout.stride);
                    self.reducer.combine(grp, v);
                }
            }
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        let values = core::mem::take(&mut self.evicted_values);
        Box::new(Self {
            layout: LaneLayout::single(values.len()),
            values,
            evicted_values: Vec::new(),
            in_dtype: self.in_dtype.clone(),
            reducer: self.reducer.clone(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let reducer = &self.reducer;
        self.layout
            .fold(&mut self.values, |a, b| reducer.combine(a, b));
        self.layout = LaneLayout::single(0);
        let v = core::mem::take(&mut self.values);
        self.reducer.finish(v, None, &self.in_dtype)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Reduces the valid values of rows `subset` of `arr` into groups `group_idxs`, spreading
/// the updates over lanes with `stride`, and sets the bits of those groups in `mask`.
///
/// # Safety
/// The subset and group_idxs are in-bounds.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn update_masked_lanes<R: Reducer>(
    reducer: &R,
    slots: &mut [R::Value],
    mask: &mut [u8],
    arr: &<R::Dtype as PolarsDataType>::Array,
    subset: &[IdxSize],
    group_idxs: &[IdxSize],
    stride: usize,
    seq_id: u64,
) {
    let mut offset = 0usize;
    if !arr.has_nulls() && mask.len() <= 8 {
        // At most 64 groups, so the groups seen fit in a register.
        let mut seen = 0u64;
        for (i, g) in subset.iter().zip(group_idxs) {
            let g = *g as usize;
            let grp = slots.get_unchecked_mut(g + (offset & LANE_MASK));
            reducer.reduce_one(grp, Some(arr.value_unchecked(*i as usize)), seq_id);
            seen |= 1 << g;
            offset = offset.wrapping_add(stride);
        }
        for (byte, bits) in mask.iter_mut().zip(seen.to_le_bytes()) {
            *byte |= bits;
        }
        return;
    }

    for (i, g) in subset.iter().zip(group_idxs) {
        if let Some(v) = arr.get_unchecked(*i as usize) {
            let g = *g as usize;
            let grp = slots.get_unchecked_mut(g + (offset & LANE_MASK));
            reducer.reduce_one(grp, Some(v), seq_id);
            // Only store if needed, a store on each row would chain
            // the updates through the shared mask byte.
            if !get_bit_unchecked(mask, g) {
                set_bit_unchecked(mask, g, true);
            }
        }
        offset = offset.wrapping_add(stride);
    }
}

pub struct VecMaskGroupedReduction<R: Reducer> {
    values: Vec<R::Value>,
    layout: LaneLayout,
    mask: MutableBitmap,
    evicted_values: Vec<R::Value>,
    evicted_mask: BitmapBuilder,
    in_dtype: DataType,
    reducer: R,
}

impl<R: Reducer> VecMaskGroupedReduction<R> {
    fn new(in_dtype: DataType, reducer: R) -> Self {
        Self {
            values: Vec::new(),
            layout: LaneLayout::single(0),
            mask: MutableBitmap::new(),
            evicted_values: Vec::new(),
            evicted_mask: BitmapBuilder::new(),
            in_dtype,
            reducer,
        }
    }
}

impl<R> GroupedReduction for VecMaskGroupedReduction<R>
where
    R: Reducer,
{
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::new(self.in_dtype.clone(), self.reducer.clone()))
    }

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.mask.reserve(additional)
    }

    fn resize(&mut self, num_groups: IdxSize) {
        let reducer = &self.reducer;
        self.layout.resize(
            &mut self.values,
            num_groups as usize,
            reducer.init(),
            |a, b| reducer.combine(a, b),
        );
        self.mask.resize(num_groups as usize, false);
    }

    fn update_group(
        &mut self,
        values: &[&Column],
        group_idx: IdxSize,
        seq_id: u64,
    ) -> PolarsResult<()> {
        let &[values] = values else { unreachable!() };
        assert!(values.dtype() == &self.in_dtype);
        let seq_id = seq_id + 1; // So we can use 0 for 'none yet'.
        let values = values.as_materialized_series(); // @scalar-opt
        let values = self.reducer.cast_series(values);
        let ca: &ChunkedArray<R::Dtype> = values.as_ref().as_ref().as_ref();
        self.reducer
            .reduce_ca(&mut self.values[group_idx as usize], ca, seq_id);
        if ca.len() != ca.null_count() {
            self.mask.set(group_idx as usize, true);
        }
        Ok(())
    }

    unsafe fn update_groups_subset(
        &mut self,
        values: &[&Column],
        subset: &[IdxSize],
        group_idxs: &[IdxSize],
        seq_id: u64,
    ) -> PolarsResult<()> {
        let &[values] = values else { unreachable!() };
        assert!(values.dtype() == &self.in_dtype);
        assert!(subset.len() == group_idxs.len());
        let seq_id = seq_id + 1; // So we can use 0 for 'none yet'.
        let values = values.as_materialized_series(); // @scalar-opt
        let values = self.reducer.cast_series(values);

        if R::ORDER_INDEPENDENT {
            self.layout.expand(&mut self.values, self.reducer.init());
        }
        let ca: &ChunkedArray<R::Dtype> = values.as_ref().as_ref().as_ref();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            update_masked_lanes(
                &self.reducer,
                self.values.as_mut_slice(),
                self.mask.as_mut_slice(),
                ca.downcast_as_array(),
                subset,
                group_idxs,
                self.layout.stride,
                seq_id,
            );
        }
        Ok(())
    }

    unsafe fn update_groups_while_evicting(
        &mut self,
        values: &[&Column],
        subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        seq_id: u64,
    ) -> PolarsResult<()> {
        let &[values] = values else { unreachable!() };
        assert!(values.dtype() == &self.in_dtype);
        assert!(subset.len() == group_idxs.len());
        let seq_id = seq_id + 1; // So we can use 0 for 'none yet'.
        let values = values.as_materialized_series(); // @scalar-opt
        let values = self.reducer.cast_series(values);
        let reducer = &self.reducer;
        self.layout
            .collapse(&mut self.values, |a, b| reducer.combine(a, b));

        let ca: &ChunkedArray<R::Dtype> = values.as_ref().as_ref().as_ref();
        let arr = ca.downcast_as_array();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                let ov = arr.get_unchecked(*i as usize);
                let grp = self.values.get_unchecked_mut(g.idx());
                if g.should_evict() {
                    self.evicted_values
                        .push(core::mem::replace(grp, self.reducer.init()));
                    self.evicted_mask.push(self.mask.get_unchecked(g.idx()));
                    self.mask.set_unchecked(g.idx(), false);
                }
                if let Some(v) = ov {
                    self.reducer.reduce_one(grp, Some(v), seq_id);
                    self.mask.set_unchecked(g.idx(), true);
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
        assert!(self.in_dtype == other.in_dtype);
        assert!(subset.len() == group_idxs.len());
        let other_lanes = other.layout.num_lanes();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                let o = other.mask.get_unchecked(*i as usize);
                if o {
                    let grp = self.values.get_unchecked_mut(*g as usize);
                    for l in 0..other_lanes {
                        let v = other
                            .values
                            .get_unchecked(*i as usize + l * other.layout.stride);
                        self.reducer.combine(grp, v);
                    }
                    self.mask.set_unchecked(*g as usize, true);
                }
            }
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        let values = core::mem::take(&mut self.evicted_values);
        Box::new(Self {
            layout: LaneLayout::single(values.len()),
            values,
            mask: core::mem::take(&mut self.evicted_mask).into_mut(),
            evicted_values: Vec::new(),
            evicted_mask: BitmapBuilder::new(),
            in_dtype: self.in_dtype.clone(),
            reducer: self.reducer.clone(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let reducer = &self.reducer;
        self.layout
            .fold(&mut self.values, |a, b| reducer.combine(a, b));
        self.layout = LaneLayout::single(0);
        let v = core::mem::take(&mut self.values);
        let m = core::mem::take(&mut self.mask);
        self.reducer.finish(v, Some(m.freeze()), &self.in_dtype)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Clone)]
pub struct NullGroupedReduction {
    num_groups: IdxSize,
    num_evictions: IdxSize,
    output: Scalar,
}

impl NullGroupedReduction {
    pub fn new(output: Scalar) -> Self {
        Self {
            num_groups: 0,
            num_evictions: 0,
            output,
        }
    }
}

impl GroupedReduction for NullGroupedReduction {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::new(self.output.clone()))
    }

    fn reserve(&mut self, _additional: usize) {}

    fn resize(&mut self, num_groups: IdxSize) {
        self.num_groups = num_groups;
    }

    fn update_group(
        &mut self,
        values: &[&Column],
        _group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(!values.is_empty());
        assert!(values.iter().any(|v| v.dtype().is_null()));
        Ok(())
    }

    unsafe fn update_groups_while_evicting(
        &mut self,
        values: &[&Column],
        subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(!values.is_empty());
        assert!(values.iter().any(|v| v.dtype().is_null()));
        assert!(subset.len() == group_idxs.len());
        for g in group_idxs {
            self.num_evictions += g.should_evict() as IdxSize;
        }
        Ok(())
    }

    unsafe fn combine_subset(
        &mut self,
        other: &dyn GroupedReduction,
        subset: &[IdxSize],
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()> {
        let _other = other.as_any().downcast_ref::<Self>().unwrap();
        assert!(subset.len() == group_idxs.len());
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            num_groups: core::mem::replace(&mut self.num_evictions, 0),
            num_evictions: 0,
            output: self.output.clone(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let length = core::mem::replace(&mut self.num_groups, 0) as usize;
        let s = self.output.clone().into_series(PlSmallStr::EMPTY);
        Ok(s.new_from_index(0, length))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
