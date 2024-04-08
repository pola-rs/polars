use arrow::array::Array;
use polars_core::chunked_array::ops::search_sorted::{binary_search_array, SearchSortedSide};
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;

fn search_sorted_ca_array<T>(
    ca: &ChunkedArray<T>,
    search_values: &ChunkedArray<T>,
    side: SearchSortedSide,
    descending: bool,
) -> Vec<IdxSize>
where
    T: PolarsNumericType,
{
    let ca = ca.rechunk();
    let arr = ca.downcast_iter().next().unwrap();

    let mut out = Vec::with_capacity(search_values.len());

    for search_arr in search_values.downcast_iter() {
        if search_arr.null_count() == 0 {
            for search_value in search_arr.values_iter() {
                out.push(binary_search_array(side, arr, *search_value, descending))
            }
        } else {
            for opt_v in search_arr.into_iter() {
                match opt_v {
                    None => out.push(0),
                    Some(search_value) => {
                        out.push(binary_search_array(side, arr, *search_value, descending))
                    },
                }
            }
        }
    }
    out
}

fn search_sorted_bin_array_with_binary_offset(
    ca: &BinaryChunked,
    search_values: &BinaryOffsetChunked,
    side: SearchSortedSide,
    descending: bool,
) -> Vec<IdxSize> {
    let ca = ca.rechunk();
    let arr = ca.downcast_iter().next().unwrap();

    let mut out = Vec::with_capacity(search_values.len());

    for search_arr in search_values.downcast_iter() {
        if search_arr.null_count() == 0 {
            for search_value in search_arr.values_iter() {
                out.push(binary_search_array(side, arr, search_value, descending))
            }
        } else {
            for opt_v in search_arr.into_iter() {
                match opt_v {
                    None => out.push(0),
                    Some(search_value) => {
                        out.push(binary_search_array(side, arr, search_value, descending))
                    },
                }
            }
        }
    }
    out
}

fn search_sorted_bin_array(
    ca: &BinaryChunked,
    search_values: &BinaryChunked,
    side: SearchSortedSide,
    descending: bool,
) -> Vec<IdxSize> {
    let ca = ca.rechunk();
    let arr = ca.downcast_iter().next().unwrap();

    let mut out = Vec::with_capacity(search_values.len());

    for search_arr in search_values.downcast_iter() {
        if search_arr.null_count() == 0 {
            for search_value in search_arr.values_iter() {
                out.push(binary_search_array(side, arr, search_value, descending))
            }
        } else {
            for opt_v in search_arr.into_iter() {
                match opt_v {
                    None => out.push(0),
                    Some(search_value) => {
                        out.push(binary_search_array(side, arr, search_value, descending))
                    },
                }
            }
        }
    }
    out
}

pub fn search_sorted(
    s: &Series,
    search_values: &Series,
    side: SearchSortedSide,
    descending: bool,
) -> PolarsResult<IdxCa> {
    let original_dtype = s.dtype();
    let s = s.to_physical_repr();
    let phys_dtype = s.dtype();

    match phys_dtype {
        DataType::String => {
            let ca = s.str().unwrap();
            let ca = ca.as_binary();
            let search_values = search_values.str()?;
            let search_values = search_values.as_binary();
            let idx = search_sorted_bin_array(&ca, &search_values, side, descending);

            Ok(IdxCa::new_vec(s.name(), idx))
        },
        DataType::Binary => {
            let ca = s.binary().unwrap();

            let idx = match search_values.dtype() {
                DataType::BinaryOffset => {
                    let search_values = search_values.binary_offset().unwrap();
                    search_sorted_bin_array_with_binary_offset(ca, search_values, side, descending)
                },
                DataType::Binary => {
                    let search_values = search_values.binary().unwrap();
                    search_sorted_bin_array(ca, search_values, side, descending)
                },
                _ => unreachable!(),
            };

            Ok(IdxCa::new_vec(s.name(), idx))
        },
        dt if dt.is_numeric() => {
            let search_values = search_values.to_physical_repr();

            let idx = with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                let search_values: &ChunkedArray<$T> = search_values.as_ref().as_ref().as_ref();

                search_sorted_ca_array(ca, search_values, side, descending)
            });
            Ok(IdxCa::new_vec(s.name(), idx))
        },
        _ => polars_bail!(opq = search_sorted, original_dtype),
    }
}
