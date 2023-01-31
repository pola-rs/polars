use std::sync::Arc;

use arrow::bitmap::MutableBitmap;
use arrow::offset::Offsets;

use super::*;

pub(crate) fn merge_categorical_map(
    left: &Arc<RevMapping>,
    right: &Arc<RevMapping>,
) -> PolarsResult<Arc<RevMapping>> {
    match (&**left, &**right) {
        (RevMapping::Global(l_map, l_slots, l_id), RevMapping::Global(r_map, r_slots, r_id)) => {
            if l_id != r_id {
                return Err(PolarsError::ComputeError("The two categorical arrays are not created under the same global string cache. They cannot be merged. Hint: set a global StringCache.".into()));
            }
            let mut new_map = (*l_map).clone();

            // safety: invariants don't change, just the type
            let offset_buf =
                unsafe { Offsets::new_unchecked(l_slots.offsets().as_slice().to_vec()) };
            let values_buf = l_slots.values().as_slice().to_vec();

            let validity_buf = if let Some(validity) = l_slots.validity() {
                let mut validity_buf = MutableBitmap::new();
                let (b, offset, len) = validity.as_slice();
                validity_buf.extend_from_slice(b, offset, len);
                Some(validity_buf)
            } else {
                None
            };

            // Safety
            // all offsets are valid and the u8 data is valid utf8
            let mut new_slots = unsafe {
                MutableUtf8Array::new_unchecked(
                    DataType::Utf8.to_arrow(),
                    offset_buf,
                    values_buf,
                    validity_buf,
                )
            };

            for (cat, idx) in r_map.iter() {
                new_map.entry(*cat).or_insert_with(|| {
                    // Safety
                    // within bounds
                    let str_val = unsafe { r_slots.value_unchecked(*idx as usize) };
                    let new_idx = new_slots.len() as u32;
                    new_slots.push(Some(str_val));

                    new_idx
                });
            }
            let new_rev = RevMapping::Global(new_map, new_slots.into(), *l_id);
            Ok(Arc::new(new_rev))
        }
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
        }
        _ => Err(PolarsError::ComputeError(
            "cannot combine categorical under a global string cache with a non cached categorical"
                .into(),
        )),
    }
}

impl CategoricalChunked {
    pub(crate) fn merge_categorical_map(&self, other: &Self) -> PolarsResult<Arc<RevMapping>> {
        merge_categorical_map(self.get_rev_map(), other.get_rev_map())
    }
}

#[cfg(test)]
#[cfg(feature = "single_thread")]
mod test {
    use super::*;
    use crate::chunked_array::categorical::CategoricalChunkedBuilder;
    use crate::{reset_string_cache, toggle_string_cache};

    #[test]
    fn test_merge_rev_map() {
        let _lock = SINGLE_LOCK.lock();
        reset_string_cache();
        toggle_string_cache(true);

        let mut builder1 = CategoricalChunkedBuilder::new("foo", 10);
        let mut builder2 = CategoricalChunkedBuilder::new("foo", 10);
        builder1.drain_iter(vec![None, Some("hello"), Some("vietnam")]);
        builder2.drain_iter(vec![Some("hello"), None, Some("world"), Some("bar")].into_iter());
        let ca1 = builder1.finish();
        let ca2 = builder2.finish();
        let rev_map = ca1.merge_categorical_map(&ca2).unwrap();

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
