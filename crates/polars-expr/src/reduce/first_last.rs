use std::marker::PhantomData;

use polars_core::frame::row::AnyValueBufferTrusted;

use super::*;

pub fn new_first_reduction(dtype: DataType) -> Box<dyn GroupedReduction> {
    new_reduction_with_policy::<First>(dtype)
}

pub fn new_last_reduction(dtype: DataType) -> Box<dyn GroupedReduction> {
    new_reduction_with_policy::<Last>(dtype)
}

fn new_reduction_with_policy<P: Policy + 'static>(dtype: DataType) -> Box<dyn GroupedReduction> {
    Box::new(GenericFirstLastGroupedReduction::<P>::new(dtype))
}

trait Policy {
    fn index(len: usize) -> usize;
    fn should_replace(new: u64, old: u64) -> bool;
}

struct First;
impl Policy for First {
    fn index(_len: usize) -> usize {
        0
    }

    fn should_replace(new: u64, old: u64) -> bool {
        // Subtracting 1 with wrapping leaves all order unchanged, except it
        // makes 0 (no value) the largest possible.
        new.wrapping_sub(1) < old.wrapping_sub(1)
    }
}

struct Last;
impl Policy for Last {
    fn index(len: usize) -> usize {
        len - 1
    }

    fn should_replace(new: u64, old: u64) -> bool {
        new > old
    }
}

#[expect(dead_code)]
struct Arbitrary;
impl Policy for Arbitrary {
    fn index(_len: usize) -> usize {
        0
    }

    fn should_replace(_new: u64, old: u64) -> bool {
        old == 0
    }
}

pub struct GenericFirstLastGroupedReduction<P> {
    dtype: DataType,
    values: Vec<AnyValue<'static>>,
    seqs: Vec<u64>,
    policy: PhantomData<fn() -> P>,
}

impl<P> GenericFirstLastGroupedReduction<P> {
    fn new(dtype: DataType) -> Self {
        Self {
            dtype,
            values: Vec::new(),
            seqs: Vec::new(),
            policy: PhantomData,
        }
    }
}

impl<P: Policy + 'static> GroupedReduction for GenericFirstLastGroupedReduction<P> {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            dtype: self.dtype.clone(),
            values: Vec::new(),
            seqs: Vec::new(),
            policy: PhantomData,
        })
    }

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.seqs.reserve(additional);
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.values.resize(num_groups as usize, AnyValue::Null);
        self.seqs.resize(num_groups as usize, 0);
    }

    fn update_group(
        &mut self,
        values: &Series,
        group_idx: IdxSize,
        seq_id: u64,
    ) -> PolarsResult<()> {
        if values.len() > 0 {
            let seq_id = seq_id + 1; // We use 0 for 'no value'.
            if P::should_replace(seq_id, self.seqs[group_idx as usize]) {
                self.values[group_idx as usize] = values.get(P::index(values.len()))?.into_static();
                self.seqs[group_idx as usize] = seq_id;
            }
        }
        Ok(())
    }

    unsafe fn update_groups(
        &mut self,
        values: &Series,
        group_idxs: &[IdxSize],
        seq_id: u64,
    ) -> PolarsResult<()> {
        let seq_id = seq_id + 1; // We use 0 for 'no value'.
        for (i, g) in group_idxs.iter().enumerate() {
            if P::should_replace(seq_id, *self.seqs.get_unchecked(*g as usize)) {
                *self.values.get_unchecked_mut(*g as usize) = values.get_unchecked(i).into_static();
                *self.seqs.get_unchecked_mut(*g as usize) = seq_id;
            }
        }
        Ok(())
    }

    unsafe fn combine(
        &mut self,
        other: &dyn GroupedReduction,
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        for (i, g) in group_idxs.iter().enumerate() {
            if P::should_replace(
                *other.seqs.get_unchecked(i),
                *self.seqs.get_unchecked(*g as usize),
            ) {
                *self.values.get_unchecked_mut(*g as usize) = other.values.get_unchecked(i).clone();
                *self.seqs.get_unchecked_mut(*g as usize) = *other.seqs.get_unchecked(i);
            }
        }
        Ok(())
    }

    unsafe fn gather_combine(
        &mut self,
        other: &dyn GroupedReduction,
        subset: &[IdxSize],
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        for (i, g) in group_idxs.iter().enumerate() {
            let si = *subset.get_unchecked(i) as usize;
            if P::should_replace(
                *other.seqs.get_unchecked(si),
                *self.seqs.get_unchecked(*g as usize),
            ) {
                *self.values.get_unchecked_mut(*g as usize) =
                    other.values.get_unchecked(si).clone();
                *self.seqs.get_unchecked_mut(*g as usize) = *other.seqs.get_unchecked(si);
            }
        }
        Ok(())
    }

    unsafe fn partition(
        self: Box<Self>,
        partition_sizes: &[IdxSize],
        partition_idxs: &[IdxSize],
    ) -> Vec<Box<dyn GroupedReduction>> {
        let values = partition::partition_vec(self.values, partition_sizes, partition_idxs);
        let seqs = partition::partition_vec(self.seqs, partition_sizes, partition_idxs);
        std::iter::zip(values, seqs)
            .map(|(values, seqs)| {
                Box::new(Self {
                    dtype: self.dtype.clone(),
                    values,
                    seqs,
                    policy: PhantomData,
                }) as _
            })
            .collect()
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        self.seqs.clear();
        unsafe {
            let mut buf = AnyValueBufferTrusted::new(&self.dtype, self.values.len());
            for v in core::mem::take(&mut self.values) {
                buf.add_unchecked_owned_physical(&v);
            }
            Ok(buf.into_series())
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
