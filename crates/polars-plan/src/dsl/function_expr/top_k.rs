use polars_core::POOL;
use polars_core::chunked_array::ops::sort::arg_bottom_k::_arg_bottom_k;

use super::*;

pub(super) fn top_k(args: &[Column], descending: bool) -> PolarsResult<Column> {
    fn extract_target_and_k(s: &[Column]) -> PolarsResult<(usize, &Column)> {
        let k_s = &s[1];
        polars_ensure!(
            k_s.len() == 1,
            ComputeError: "`k` must be a single value for `top_k`."
        );

        let Some(k) = k_s.cast(&IDX_DTYPE)?.idx()?.get(0) else {
            polars_bail!(ComputeError: "`k` must be set for `top_k`")
        };

        let src = &s[0];
        Ok((k as usize, src))
    }

    let (k, src) = extract_target_and_k(args)?;

    if src.is_empty() {
        return Ok(src.clone());
    }

    let sorted_flag = src.is_sorted_flag();
    let is_sorted = match src.is_sorted_flag() {
        IsSorted::Ascending => true,
        IsSorted::Descending => true,
        IsSorted::Not => false,
    };
    if is_sorted {
        let out_len = k.min(src.len());
        let ignored_len = src.len() - out_len;
        let slice_at_start = (sorted_flag == IsSorted::Ascending) == descending;
        let nulls_at_start = src.get(0).unwrap() == AnyValue::Null;
        let offset = if nulls_at_start == slice_at_start {
            src.null_count().min(ignored_len)
        } else {
            0
        };

        return if slice_at_start {
            Ok(src.slice(offset as i64, out_len))
        } else {
            Ok(src.slice(-(offset as i64) - (out_len as i64), out_len))
        };
    }

    src.top_k(k, descending)
}

pub(super) fn top_k_by(args: &[Column], descending: Vec<bool>) -> PolarsResult<Column> {
    /// Return (k, src, by)
    fn extract_parameters(args: &[Column]) -> PolarsResult<(usize, &Column, &[Column])> {
        let k_s = &args[1];

        polars_ensure!(
            k_s.len() == 1,
            ComputeError: "`k` must be a single value for `top_k`."
        );

        let Some(k) = k_s.cast(&IDX_DTYPE)?.idx()?.get(0) else {
            polars_bail!(ComputeError: "`k` must be set for `top_k`")
        };

        let src = &args[0];

        let by = &args[2..];

        Ok((k as usize, src, by))
    }

    let (k, src, by) = extract_parameters(args)?;

    if src.is_empty() {
        return Ok(src.clone());
    }

    if by.first().map(|x| x.is_empty()).unwrap_or(false) {
        return Ok(src.clone());
    }

    for s in by {
        if s.len() != src.len() {
            polars_bail!(ComputeError: "`by` column's ({}) length ({}) should have the same length as the source column length ({}) in `top_k`", s.name(), s.len(), src.len())
        }
    }

    let multithreaded = k >= 10000 && POOL.current_num_threads() > 1;
    let mut sort_options = SortMultipleOptions {
        descending: descending.into_iter().map(|x| !x).collect(),
        nulls_last: vec![true; by.len()],
        multithreaded,
        maintain_order: false,
        limit: None,
    };

    let idx = _arg_bottom_k(k, by, &mut sort_options)?;

    let result = unsafe {
        src.as_materialized_series()
            .take_unchecked(&idx.into_inner())
    };
    Ok(result.into())
}
