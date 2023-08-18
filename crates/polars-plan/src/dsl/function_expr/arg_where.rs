use polars_core::utils::arrow::bitmap::utils::SlicesIterator;

use super::*;

pub(super) fn arg_where(s: &mut [Series]) -> PolarsResult<Option<Series>> {
    let predicate = s[0].bool()?;

    if predicate.is_empty() {
        Ok(Some(Series::full_null(predicate.name(), 0, &IDX_DTYPE)))
    } else {
        let capacity = predicate.sum().unwrap();
        let mut out = Vec::with_capacity(capacity as usize);
        let mut total_offset = 0;

        predicate.downcast_iter().for_each(|arr| {
            let values = match arr.validity() {
                Some(validity) if validity.unset_bits() > 0 => validity & arr.values(),
                _ => arr.values().clone(),
            };

            for (offset, len) in SlicesIterator::new(&values) {
                // law of small numbers optimization
                if len == 1 {
                    out.push((total_offset + offset) as IdxSize)
                } else {
                    let offset = (offset + total_offset) as IdxSize;
                    let len = len as IdxSize;
                    let iter = offset..offset + len;
                    out.extend(iter)
                }
            }

            total_offset += arr.len();
        });
        let ca = IdxCa::with_chunk(predicate.name(), IdxArr::from_vec(out));
        Ok(Some(ca.into_series()))
    }
}
