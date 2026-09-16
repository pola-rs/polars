use polars_error::{PolarsResult, polars_bail};

// Never intended to be in-scope or used, but allows us to be generic over types
// we can put into broadcast_len, and provide better error messages if they're
// not just usize lengths.
pub trait BroadcastLength {
    fn _broadcast_len(&self) -> usize;
    fn _column_name(&self) -> Option<&str>;
}

impl<T: BroadcastLength> BroadcastLength for &T {
    fn _broadcast_len(&self) -> usize {
        (*self)._broadcast_len()
    }

    fn _column_name(&self) -> Option<&str> {
        (*self)._column_name()
    }
}

impl BroadcastLength for usize {
    fn _broadcast_len(&self) -> usize {
        *self
    }

    fn _column_name(&self) -> Option<&str> {
        None
    }
}

// Calculates the common length a set of lengths should be broadcast to. This
// is the shared non-unit length of the set. If there are multiple different
// non-unit lengths this returns an error.
//
// Returns 0 if the iterator is empty.
pub fn broadcast_len(iter: impl IntoIterator<Item = impl BroadcastLength>) -> PolarsResult<usize> {
    let mut iter = iter.into_iter();
    let Some(first) = iter.next() else {
        return Ok(0);
    };

    let mut broadcast_len = first._broadcast_len();
    let mut broadcast_val = first;
    for val in iter {
        let len = val._broadcast_len();
        if broadcast_len == 1 {
            broadcast_len = len;
            broadcast_val = val;
        } else if len != broadcast_len && len != 1 {
            let fmt_opt_name = |opt_n: Option<&str>| {
                opt_n
                    .filter(|n| !n.is_empty())
                    .map(|n| format!(" (column '{n}')"))
                    .unwrap_or_default()
            };
            let our_info = fmt_opt_name(val._column_name());
            let broadcast_info = fmt_opt_name(broadcast_val._column_name());
            polars_bail!(
                ShapeMismatch:
                "can't compute broadcast length, found incompatible lengths {len}{our_info} and {broadcast_len}{broadcast_info}"
            )
        }
    }

    Ok(broadcast_len)
}
