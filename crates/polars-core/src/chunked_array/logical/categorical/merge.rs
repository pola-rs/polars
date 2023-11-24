use std::sync::Arc;

use arrow::bitmap::MutableBitmap;
use arrow::offset::Offsets;

use super::*;

fn slots_to_mut(slots: &Utf8Array<i64>) -> MutableUtf8Array<i64> {
    // safety: invariants don't change, just the type
    let offset_buf = unsafe { Offsets::new_unchecked(slots.offsets().as_slice().to_vec()) };
    let values_buf = slots.values().as_slice().to_vec();

    let validity_buf = if let Some(validity) = slots.validity() {
        let mut validity_buf = MutableBitmap::new();
        let (b, offset, len) = validity.as_slice();
        validity_buf.extend_from_slice(b, offset, len);
        Some(validity_buf)
    } else {
        None
    };

    // Safety
    // all offsets are valid and the u8 data is valid utf8
    unsafe {
        MutableUtf8Array::new_unchecked(
            DataType::Utf8.to_arrow(),
            offset_buf,
            values_buf,
            validity_buf,
        )
    }
}

struct State {
    map: PlHashMap<u32, u32>,
    slots: MutableUtf8Array<i64>,
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
                // Safety
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
    categories: &'a Utf8Array<i64>,
    ca_right: &'a CategoricalChunked,
) -> Result<(UInt32Chunked, Arc<RevMapping>), PolarsError> {
    // Counterpart of the GlobalRevmapMerger.
    // In case of local categorical we also need to change the physicals not only the revmap

    polars_warn!(
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
        (RevMapping::Local(_, idl), RevMapping::Local(_, idr)) if idl == idr => (
            merge_ops.finish(cat_left.physical(), cat_right.physical())?,
            rev_map_left.clone(),
        ),
        (RevMapping::Local(categorical, _), RevMapping::Local(_, _)) => {
            let (rhs_physical, rev_map) = merge_local_rhs_categorical(categorical, cat_right)?;
            (
                merge_ops.finish(cat_left.physical(), &rhs_physical)?,
                rev_map,
            )
        },
        _ => polars_bail!(string_cache_mismatch),
    };
    // Safety: physical and rev map are correctly constructed above
    unsafe {
        Ok(CategoricalChunked::from_cats_and_rev_map_unchecked(
            new_physical,
            new_rev_map,
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
    // Safety: We just made both rev maps compatible only appended categories
    unsafe {
        new_ca_left.set_rev_map(
            new_ca_right.get_rev_map().clone(),
            ca_left.get_rev_map().len() == new_ca_right.get_rev_map().len(),
        )
    };

    Ok((new_ca_left, new_ca_right))
}

#[cfg(test)]
#[cfg(feature = "single_thread")]
mod test {
    use super::*;
    use crate::chunked_array::categorical::CategoricalChunkedBuilder;
    use crate::{disable_string_cache, enable_string_cache, StringCacheHolder};

    #[test]
    fn test_merge_rev_map() {
        let _lock = SINGLE_LOCK.lock();
        disable_string_cache();
        let _sc = StringCacheHolder::hold();

        let mut builder1 = CategoricalChunkedBuilder::new("foo", 10);
        let mut builder2 = CategoricalChunkedBuilder::new("foo", 10);
        builder1.drain_iter(vec![None, Some("hello"), Some("vietnam")]);
        builder2.drain_iter(vec![Some("hello"), None, Some("world"), Some("bar")].into_iter());
        let ca1 = builder1.finish();
        let ca2 = builder2.finish();
        let rev_map = ca1._merge_categorical_map(&ca2).unwrap();

        let mut ca = UInt32Chunked::new("", &[0, 1, 2, 3]);
        ca.categorical_map = Some(rev_map);
        let s = ca
            .cast(&DataType::Categorical)
            .unwrap()
            .cast(&DataType::Utf8)
            .unwrap();
        let ca = s.utf8().unwrap();
        let vals = ca.into_no_null_iter().collect::<Vec<_>>();
        assert_eq!(vals, &["hello", "vietnam", "world", "bar"]);
    }
}
