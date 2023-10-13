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
pub(crate) struct RevMapMerger {
    id: Option<u32>,
    original: Arc<RevMapping>,
    // only initiate state when
    // we encounter a rev-map from a different source,
    // but from the same string cache
    state: Option<State>,
}

impl RevMapMerger {
    pub(crate) fn new(rev_map: Arc<RevMapping>) -> Self {
        let id = if let RevMapping::Global(_, _, id) = rev_map.as_ref() {
            Some(*id)
        } else {
            None
        };
        RevMapMerger {
            state: None,
            id,
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

    pub(crate) fn merge_map(&mut self, rev_map: &Arc<RevMapping>) -> PolarsResult<()> {
        // happy path
        // they come from the same source
        if Arc::ptr_eq(&self.original, rev_map) {
            return Ok(());
        }
        let msg = "categoricals don't originate from the same string cache\n\
    try setting a global string cache or increase the scope of the local string cache";
        let RevMapping::Global(map, slots, id) = rev_map.as_ref() else {
            polars_bail!(ComputeError: msg)
        };
        polars_ensure!(Some(*id) == self.id, ComputeError: msg);

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

    pub(crate) fn finish(self) -> Arc<RevMapping> {
        match self.state {
            None => self.original,
            Some(state) => {
                let new_rev = RevMapping::Global(state.map, state.slots.into(), self.id.unwrap());
                Arc::new(new_rev)
            },
        }
    }
}

pub(crate) fn merge_rev_map(
    left: &Arc<RevMapping>,
    right: &Arc<RevMapping>,
) -> PolarsResult<Arc<RevMapping>> {
    match (&**left, &**right) {
        (RevMapping::Global(_, _, _), RevMapping::Global(_, _, _)) => {
            let mut merger = RevMapMerger::new(left.clone());
            merger.merge_map(right)?;
            Ok(merger.finish())
        },
        (RevMapping::Local(arr_l), RevMapping::Local(arr_r)) => {
            // they are from the same source, just clone
            if std::ptr::eq(arr_l, arr_r) {
                return Ok(left.clone());
            }

            let arr = arrow::compute::concatenate::concatenate(&[arr_l, arr_r]).unwrap();
            let arr = arr
                .as_any()
                .downcast_ref::<Utf8Array<i64>>()
                .unwrap()
                .clone();

            Ok(Arc::new(RevMapping::Local(arr)))
        },
        _ => polars_bail!(
            ComputeError:
            "unable to merge categorical under a global string cache with a non-cached one"
        ),
    }
}

impl CategoricalChunked {
    pub fn _merge_categorical_map(&self, other: &Self) -> PolarsResult<Arc<RevMapping>> {
        merge_rev_map(self.get_rev_map(), other.get_rev_map())
    }
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
