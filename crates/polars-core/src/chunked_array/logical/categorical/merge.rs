use std::borrow::Cow;

use super::*;
use crate::series::IsSorted;
use crate::utils::align_chunks_binary;

fn slots_to_mut(slots: &Utf8ViewArray) -> MutablePlString {
    slots.clone().make_mut()
}

struct State {
    map: PlHashMap<u32, u32>,
    slots: MutablePlString,
}

#[derive(Default)]
pub struct GlobalRevMapMerger {
    id: u32,
    original: Arc<RevMapping>,
    // only initiate state when
    // we encounter a rev-map from a different source,
    // but from the same string cache
    state: Option<State>,
}

impl GlobalRevMapMerger {
    pub fn new(rev_map: Arc<RevMapping>) -> Self {
        let RevMapping::Global(_, _, id) = rev_map.as_ref() else {
            unreachable!()
        };

        GlobalRevMapMerger {
            state: None,
            id: *id,
            original: rev_map,
        }
    }

    fn init_state(&mut self) {
        let RevMapping::Global(map, slots, _) = self.original.as_ref() else {
            unreachable!()
        };
        self.state = Some(State {
            map: (*map).clone(),
            slots: slots_to_mut(slots),
        })
    }

    pub fn merge_map(&mut self, rev_map: &Arc<RevMapping>) -> PolarsResult<()> {
        // happy path they come from the same source
        if Arc::ptr_eq(&self.original, rev_map) {
            return Ok(());
        }

        let RevMapping::Global(map, slots, id) = rev_map.as_ref() else {
            polars_bail!(string_cache_mismatch)
        };
        polars_ensure!(*id == self.id, string_cache_mismatch);

        if self.state.is_none() {
            self.init_state()
        }
        let state = self.state.as_mut().unwrap();

        for (cat, idx) in map.iter() {
            state.map.entry(*cat).or_insert_with(|| {
                // SAFETY:
                // within bounds
                let str_val = unsafe { slots.value_unchecked(*idx as usize) };
                let new_idx = state.slots.len() as u32;
                state.slots.push(Some(str_val));

                new_idx
            });
        }
        Ok(())
    }

    pub fn finish(self) -> Arc<RevMapping> {
        match self.state {
            None => self.original,
            Some(state) => {
                let new_rev = RevMapping::Global(state.map, state.slots.into(), self.id);
                Arc::new(new_rev)
            },
        }
    }
}

fn merge_local_rhs_categorical<'a>(
    categories: &'a Utf8ViewArray,
    ca_right: &'a CategoricalChunked,
) -> Result<(UInt32Chunked, Arc<RevMapping>), PolarsError> {
    // Counterpart of the GlobalRevmapMerger.
    // In case of local categorical we also need to change the physicals not only the revmap

    polars_warn!(
        CategoricalRemappingWarning,
        "Local categoricals have different encodings, expensive re-encoding is done \
    to perform this merge operation. Consider using a StringCache or an Enum type \
    if the categories are known in advance"
    );

    let RevMapping::Local(cats_right, _) = &**ca_right.get_rev_map() else {
        unreachable!()
    };

    let cats_left_hashmap = PlHashMap::from_iter(
        categories
            .values_iter()
            .enumerate()
            .map(|(k, v)| (v, k as u32)),
    );
    let mut new_categories = slots_to_mut(categories);
    let mut idx_mapping = PlHashMap::with_capacity(cats_right.len());

    for (idx, s) in cats_right.values_iter().enumerate() {
        if let Some(v) = cats_left_hashmap.get(&s) {
            idx_mapping.insert(idx as u32, *v);
        } else {
            idx_mapping.insert(idx as u32, new_categories.len() as u32);
            new_categories.push(Some(s));
        }
    }
    let new_rev_map = Arc::new(RevMapping::build_local(new_categories.into()));
    Ok((
        ca_right
            .physical
            .apply(|opt_v| opt_v.map(|v| *idx_mapping.get(&v).unwrap())),
        new_rev_map,
    ))
}

pub trait CategoricalMergeOperation {
    fn finish(self, lhs: &UInt32Chunked, rhs: &UInt32Chunked) -> PolarsResult<UInt32Chunked>;
}

// Make the right categorical compatible with the left while applying the merge operation
pub fn call_categorical_merge_operation<I: CategoricalMergeOperation>(
    cat_left: &CategoricalChunked,
    cat_right: &CategoricalChunked,
    merge_ops: I,
) -> PolarsResult<CategoricalChunked> {
    let rev_map_left = cat_left.get_rev_map();
    let rev_map_right = cat_right.get_rev_map();
    let (new_physical, new_rev_map) = match (&**rev_map_left, &**rev_map_right) {
        (RevMapping::Global(_, _, idl), RevMapping::Global(_, _, idr)) if idl == idr => {
            let mut rev_map_merger = GlobalRevMapMerger::new(rev_map_left.clone());
            rev_map_merger.merge_map(rev_map_right)?;
            (
                merge_ops.finish(cat_left.physical(), cat_right.physical())?,
                rev_map_merger.finish(),
            )
        },
        (RevMapping::Local(_, idl), RevMapping::Local(_, idr))
            if idl == idr && cat_left.is_enum() == cat_right.is_enum() =>
        {
            (
                merge_ops.finish(cat_left.physical(), cat_right.physical())?,
                rev_map_left.clone(),
            )
        },
        (RevMapping::Local(categorical, _), RevMapping::Local(_, _))
            if !cat_left.is_enum() && !cat_right.is_enum() =>
        {
            let (rhs_physical, rev_map) = merge_local_rhs_categorical(categorical, cat_right)?;
            (
                merge_ops.finish(cat_left.physical(), &rhs_physical)?,
                rev_map,
            )
        },
        (RevMapping::Local(_, _), RevMapping::Local(_, _))
            if cat_left.is_enum() | cat_right.is_enum() =>
        {
            polars_bail!(ComputeError: "can not merge incompatible Enum types")
        },
        _ => polars_bail!(string_cache_mismatch),
    };
    // SAFETY: physical and rev map are correctly constructed above
    unsafe {
        Ok(CategoricalChunked::from_cats_and_rev_map_unchecked(
            new_physical,
            new_rev_map,
            cat_left.is_enum(),
            cat_left.get_ordering(),
        ))
    }
}

struct DoNothing;
impl CategoricalMergeOperation for DoNothing {
    fn finish(self, _lhs: &UInt32Chunked, rhs: &UInt32Chunked) -> PolarsResult<UInt32Chunked> {
        Ok(rhs.clone())
    }
}

// Make the right categorical compatible with the left
pub fn make_categoricals_compatible(
    ca_left: &CategoricalChunked,
    ca_right: &CategoricalChunked,
) -> PolarsResult<(CategoricalChunked, CategoricalChunked)> {
    let new_ca_right = call_categorical_merge_operation(ca_left, ca_right, DoNothing)?;

    // Alter rev map of left
    let mut new_ca_left = ca_left.clone();
    // SAFETY: We just made both rev maps compatible only appended categories
    unsafe {
        new_ca_left.set_rev_map(
            new_ca_right.get_rev_map().clone(),
            ca_left.get_rev_map().len() == new_ca_right.get_rev_map().len(),
        )
    };

    Ok((new_ca_left, new_ca_right))
}

pub fn make_list_categoricals_compatible(
    mut list_ca_left: ListChunked,
    list_ca_right: ListChunked,
) -> PolarsResult<(ListChunked, ListChunked)> {
    // Make categoricals compatible

    let cat_left = list_ca_left.get_inner();
    let cat_right = list_ca_right.get_inner();
    let (cat_left, cat_right) =
        make_categoricals_compatible(cat_left.categorical()?, cat_right.categorical()?)?;

    // we only appended categories to the rev_map at the end, so only change the inner dtype
    list_ca_left.set_inner_dtype(cat_left.dtype().clone());

    // We changed the physicals and the rev_map, offsets and validity buffers are still good
    let (list_ca_right, cat_physical): (Cow<ListChunked>, Cow<UInt32Chunked>) =
        align_chunks_binary(&list_ca_right, cat_right.physical());
    let mut list_ca_right = list_ca_right.into_owned();
    // SAFETY:
    // Chunks are aligned, length / dtype remains correct
    unsafe {
        list_ca_right
            .downcast_iter_mut()
            .zip(cat_physical.chunks())
            .for_each(|(arr, new_phys)| {
                *arr = ListArray::new(
                    arr.data_type().clone(),
                    arr.offsets().clone(),
                    new_phys.clone(),
                    arr.validity().cloned(),
                )
            });
    }
    // reset the sorted flag and add extra categories back in
    list_ca_right.set_sorted_flag(IsSorted::Not);
    list_ca_right.set_inner_dtype(cat_right.dtype().clone());
    Ok((list_ca_left, list_ca_right))
}
