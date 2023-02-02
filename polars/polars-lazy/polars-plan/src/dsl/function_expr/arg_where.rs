use polars_arrow::trusted_len::PushUnchecked;

use super::*;

pub(super) fn arg_where(s: &mut [Series]) -> PolarsResult<Option<Series>> {
    let predicate = s[0].bool()?;

    if predicate.is_empty() {
        Ok(Some(Series::full_null(predicate.name(), 0, &IDX_DTYPE)))
    } else {
        let capacity = predicate.sum().unwrap();
        let mut out = Vec::with_capacity(capacity as usize);
        let mut cnt = 0 as IdxSize;

        predicate.downcast_iter().for_each(|arr| {
            let values = match arr.validity() {
                Some(validity) => validity & arr.values(),
                None => arr.values().clone(),
            };

            // todo! could use chunkiter from arrow here
            for bit in values.iter() {
                if bit {
                    // safety:
                    // we allocated enough slots
                    unsafe { out.push_unchecked(cnt) }
                }
                cnt += 1;
            }
        });
        let arr = Box::new(IdxArr::from_vec(out)) as ArrayRef;
        unsafe {
            Ok(Some(
                IdxCa::from_chunks(predicate.name(), vec![arr]).into_series(),
            ))
        }
    }
}
