use super::*;
use arrow::bitmap::MutableBitmap;
use std::sync::Arc;

impl CategoricalChunked {
    pub(crate) fn merge_categorical_map(&self, other: &Self) -> Arc<RevMapping> {
        match (
            &**self.get_rev_map(),
            &**other.get_rev_map()
        ) {
            (
                RevMapping::Global(l_map, l_slots, l_id),
                RevMapping::Global(r_map, r_slots, r_id),
            ) => {
                if l_id != r_id {
                    panic!("The two categorical arrays are not created under the same global string cache. They cannot be merged")
                }
                let mut new_map = (*l_map).clone();

                let offset_buf = l_slots.offsets().as_slice().to_vec();
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
                    MutableUtf8Array::from_data_unchecked(
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
                Arc::new(new_rev)
            }
            (RevMapping::Local(arr_l), RevMapping::Local(arr_r)) => {
                // they are from the same source, just clone
                if std::ptr::eq(arr_l, arr_r) {
                    return self.get_rev_map().clone()
                }

                let arr = arrow::compute::concatenate::concatenate(&[arr_l, arr_r]).unwrap();
                let arr = arr.as_any().downcast_ref::<Utf8Array<i64>>().unwrap().clone();

                Arc::new(RevMapping::Local(arr))
            }
            _ => panic!("cannot combine categorical under a global string cache with a non cached categorical")
        }
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
        reset_string_cache();
        toggle_string_cache(true);

        let mut builder1 = CategoricalChunkedBuilder::new("foo", 10);
        let mut builder2 = CategoricalChunkedBuilder::new("foo", 10);
        builder1.drain_iter(vec![None, Some("hello"), Some("vietnam")]);
        builder2.drain_iter(vec![Some("hello"), None, Some("world"), Some("bar")].into_iter());
        let ca1 = builder1.finish();
        let ca2 = builder2.finish();
        let rev_map = ca1.merge_categorical_map(&ca2);

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
