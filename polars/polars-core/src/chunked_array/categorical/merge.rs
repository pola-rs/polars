use super::*;
use arrow::bitmap::MutableBitmap;
use arrow::buffer::MutableBuffer;
use std::sync::Arc;

impl CategoricalChunked {
    pub(crate) fn merge_categorical_map(&self, other: &Self) -> Arc<RevMapping> {
        match (
            self.categorical_map.as_deref(),
            other.categorical_map.as_deref(),
        ) {
            (
                Some(RevMapping::Global(l_map, l_slots, l_id)),
                Some(RevMapping::Global(r_map, r_slots, r_id)),
            ) => {
                if l_id != r_id {
                    panic!("The two categorical arrays are not created under the same global string cache. They cannot be merged")
                }
                let mut new_map = (*l_map).clone();

                let mut offset_buf = MutableBuffer::new();
                offset_buf.extend_from_slice(l_slots.offsets().as_slice());

                let mut values_buf = MutableBuffer::new();
                values_buf.extend_from_slice(l_slots.values().as_slice());

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
            _ => {
                // pass for now. Still need to do some checks for local maps that are equal
                self.categorical_map.as_ref().unwrap().clone()
            }
        }
    }
}
