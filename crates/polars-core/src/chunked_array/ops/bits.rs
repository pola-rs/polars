use super::BooleanChunked;

fn first_true_idx_impl(ca: &BooleanChunked, invert: bool) -> Option<usize> {
    let null_count = ca.null_count();
    if null_count == ca.len() {
        return None;
    }

    if (ca.is_sorted_ascending_flag() && invert) || (ca.is_sorted_descending_flag() && !invert) {
        return ca.first_non_null();
    }

    let invert_mask = if invert { u64::MAX } else { 0 };
    let mut offset = 0;
    for arr in ca.downcast_iter() {
        let values = arr.values();
        if let Some(validity) = arr.validity() {
            let mut x_it = values.fast_iter_u56();
            let mut v_it = validity.fast_iter_u56();
            for (x, v) in x_it.by_ref().zip(v_it.by_ref()) {
                let n = ((x ^ invert_mask) & v).trailing_zeros() as usize;
                if n < 56 {
                    return Some(offset + n);
                }
                offset += 56;
            }

            let (x, rest_len) = x_it.remainder();
            let (v, _rest_len) = v_it.remainder();
            let n = ((x ^ invert_mask) & v).trailing_zeros() as usize;
            if n < rest_len {
                return Some(offset + n);
            }
            offset += rest_len;
        } else {
            let n = if invert {
                values.leading_ones()
            } else {
                values.leading_zeros()
            };
            if n < values.len() {
                return Some(offset + n);
            }
            offset += values.len();
        }
    }

    None
}

impl BooleanChunked {
    pub fn num_trues(&self) -> usize {
        self.downcast_iter()
            .map(|arr| match arr.validity() {
                None => arr.values().set_bits(),
                Some(validity) => arr.values().num_intersections_with(validity),
            })
            .sum()
    }

    pub fn num_falses(&self) -> usize {
        self.downcast_iter()
            .map(|arr| match arr.validity() {
                None => arr.values().unset_bits(),
                Some(validity) => (!arr.values()).num_intersections_with(validity),
            })
            .sum()
    }

    pub fn first_true_idx(&self) -> Option<usize> {
        first_true_idx_impl(self, false)
    }

    pub fn first_false_idx(&self) -> Option<usize> {
        first_true_idx_impl(self, true)
    }
}
