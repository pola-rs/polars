use polars_core::chunked_array::ops::search_sorted::{SearchSortedSide, binary_search_ca};
use polars_core::prelude::row_encode::_get_rows_encoded_ca;
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::total_ord::TotalOrd;
use search_sorted::binary_search_ca_with_overrides;

pub fn search_sorted(
    s: &Series,
    search_values: &Series,
    side: SearchSortedSide,
    descending: bool,
) -> PolarsResult<IdxCa> {
    let original_dtype = s.dtype();

    if s.dtype().is_categorical() {
        // See https://github.com/pola-rs/polars/issues/20171
        polars_bail!(InvalidOperation: "'search_sorted' is not supported on dtype: {}", s.dtype())
    }

    print!("Original series {} needle {}", s, search_values);
    let s = s.to_physical_repr();
    let phys_dtype = s.dtype();

    match phys_dtype {
        DataType::String => {
            let ca = s.str().unwrap();
            let ca = ca.as_binary();
            let search_values = search_values.str()?;
            let search_values = search_values.as_binary();
            let idx = binary_search_ca(&ca, search_values.iter(), side, descending);
            Ok(IdxCa::new_vec(s.name().clone(), idx))
        },
        DataType::Boolean => {
            let ca = s.bool().unwrap();
            let search_values = search_values.bool()?;

            let mut none_pos = None;
            let mut false_pos = None;
            let mut true_pos = None;
            let idxs = search_values
                .iter()
                .map(|v| {
                    let cache = match v {
                        None => &mut none_pos,
                        Some(false) => &mut false_pos,
                        Some(true) => &mut true_pos,
                    };
                    *cache.get_or_insert_with(|| {
                        binary_search_ca(ca, [v].into_iter(), side, descending)[0]
                    })
                })
                .collect();
            Ok(IdxCa::new_vec(s.name().clone(), idxs))
        },
        DataType::Binary => {
            let ca = s.binary().unwrap();

            let idx = match search_values.dtype() {
                DataType::BinaryOffset => {
                    let search_values = search_values.binary_offset().unwrap();
                    binary_search_ca(ca, search_values.iter(), side, descending)
                },
                DataType::Binary => {
                    let search_values = search_values.binary().unwrap();
                    binary_search_ca(ca, search_values.iter(), side, descending)
                },
                _ => unreachable!(),
            };

            Ok(IdxCa::new_vec(s.name().clone(), idx))
        },
        DataType::BinaryOffset => {
            let ca = s.binary_offset().unwrap();
            let search_values = search_values.binary_offset().unwrap();
            let idx = binary_search_ca(ca, search_values.iter(), side, descending);
            Ok(IdxCa::new_vec(s.name().clone(), idx))
        },

        dt if dt.is_primitive_numeric() => {
            let search_values = search_values.to_physical_repr();

            let idx = with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                let search_values: &ChunkedArray<$T> = search_values.as_ref().as_ref().as_ref();
                binary_search_ca(ca, search_values.iter(), side, descending)
            });
            Ok(IdxCa::new_vec(s.name().clone(), idx))
        },
        dt if dt.is_nested() => {
            // We need to know if nulls are last, and nesting introduces the
            // problem that nulls may be inside arbitrarily nested values, so we
            // can just look at the first value to see if it's null. Instead, we
            // figure out which presumptive nulls location preserves the sorting
            // invariant.
            //
            // In particular, a row-encoded first value should always be >= or
            // <= depending on whether this is ascending or descending sort,
            // respectively. So if we get the wrong order for nulls_last = true,
            // that means nulls_last should be false.
            let nulls_last = if s.len() < 2 {
                true // doesn't matter, really
            } else {
                let mut first_and_last = s.slice(0, 1);
                first_and_last.append(&s.slice(-1, 1))?;
                // Assume nulls_last is true:
                let first_and_last = _get_rows_encoded_ca(
                    "".into(),
                    &[first_and_last.into_column()],
                    &[descending],
                    &[true],
                )?;
                let first = first_and_last.first().unwrap();
                let last = first_and_last.last().unwrap();
                if descending {
                    // if nulls_last really is true, encoding should have preserved the descending invariant:
                    first.tot_ge(&last)
                } else {
                    first.tot_le(&last)
                }
            };

            // This is O(N), whereas typically search_sorted would be O(logN).
            // Ideally the implementation would only row-encode values that are
            // actively being looked up, instead of all of them...
            let ca = _get_rows_encoded_ca(
                "".into(),
                &[s.as_ref().clone().into_column()],
                &[descending],
                &[nulls_last],
            )?;

            let search_values = _get_rows_encoded_ca(
                "".into(),
                &[search_values.clone().into_column()],
                &[descending],
                &[nulls_last],
            )?;
            let idx = binary_search_ca_with_overrides(
                &ca,
                search_values.iter(),
                side,
                false,
                nulls_last,
            );
            Ok(IdxCa::new_vec(s.name().clone(), idx))
        },
        _ => polars_bail!(opq = search_sorted, original_dtype),
    }
}
