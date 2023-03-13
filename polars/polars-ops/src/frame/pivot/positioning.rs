use std::hash::Hash;

use polars_core::prelude::*;

use super::*;

pub(super) fn position_aggregates(
    n_rows: usize,
    n_cols: usize,
    row_locations: &[IdxSize],
    col_locations: &[IdxSize],
    value_agg_phys: &Series,
    logical_type: &DataType,
    headers: &Utf8Chunked,
) -> Vec<Series> {
    let mut buf = vec![AnyValue::Null; n_rows * n_cols];
    let start_ptr = buf.as_mut_ptr() as usize;

    let n_threads = POOL.current_num_threads();
    let split = _split_offsets(row_locations.len(), n_threads);

    POOL.install(|| {
        split.into_par_iter().for_each(|(offset, len)| {
            let start_ptr = start_ptr as *mut AnyValue;
            let row_locations = &row_locations[offset..offset + len];
            let col_locations = &col_locations[offset..offset + len];
            let value_agg_phys = value_agg_phys.slice(offset as i64, len);

            for ((row_idx, col_idx), val) in row_locations
                .iter()
                .zip(col_locations)
                .zip(value_agg_phys.phys_iter())
            {
                // Safety:
                // in bounds
                unsafe {
                    let idx = *row_idx as usize + *col_idx as usize * n_rows;
                    debug_assert!(idx < buf.len());
                    let pos = start_ptr.add(idx);
                    std::ptr::write(pos, val)
                }
            }
        });

        let headers_iter = headers.par_iter_indexed();

        (0..n_cols)
            .into_par_iter()
            .zip(headers_iter)
            .map(|(i, opt_name)| {
                let offset = i * n_rows;
                let avs = &buf[offset..offset + n_rows];
                let name = opt_name.unwrap_or("null");
                let out = Series::new(name, avs);
                unsafe { out.cast_unchecked(logical_type).unwrap() }
            })
            .collect::<Vec<_>>()
    })
}

pub(super) fn position_aggregates_numeric<T>(
    n_rows: usize,
    n_cols: usize,
    row_locations: &[IdxSize],
    col_locations: &[IdxSize],
    value_agg_phys: &ChunkedArray<T>,
    logical_type: &DataType,
    headers: &Utf8Chunked,
) -> Vec<Series>
where
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
{
    let mut buf = vec![None; n_rows * n_cols];
    let start_ptr = buf.as_mut_ptr() as usize;

    let n_threads = POOL.current_num_threads();

    let split = _split_offsets(row_locations.len(), n_threads);

    POOL.install(|| {
        split.into_par_iter().for_each(|(offset, len)| {
            let start_ptr = start_ptr as *mut Option<T::Native>;
            let row_locations = &row_locations[offset..offset + len];
            let col_locations = &col_locations[offset..offset + len];
            let value_agg_phys = value_agg_phys.slice(offset as i64, len);

            for ((row_idx, col_idx), val) in row_locations
                .iter()
                .zip(col_locations)
                .zip(value_agg_phys.into_iter())
            {
                // Safety:
                // in bounds
                unsafe {
                    let idx = *row_idx as usize + *col_idx as usize * n_rows;
                    debug_assert!(idx < buf.len());
                    let pos = start_ptr.add(idx);
                    std::ptr::write(pos, val)
                }
            }
        });
        let headers_iter = headers.par_iter_indexed();

        (0..n_cols)
            .into_par_iter()
            .zip(headers_iter)
            .map(|(i, opt_name)| {
                let offset = i * n_rows;
                let opt_values = &buf[offset..offset + n_rows];
                let name = opt_name.unwrap_or("null");
                let out = ChunkedArray::<T>::from_slice_options(name, opt_values).into_series();
                unsafe { out.cast_unchecked(logical_type).unwrap() }
            })
            .collect::<Vec<_>>()
    })
}

fn compute_col_idx_numeric<T>(column_agg_physical: &ChunkedArray<T>) -> Vec<IdxSize>
where
    T: PolarsNumericType,
    T::Native: Hash + Eq,
{
    let mut col_to_idx = PlHashMap::with_capacity(HASHMAP_INIT_SIZE);
    let mut idx = 0 as IdxSize;
    column_agg_physical
        .into_iter()
        .map(|v| {
            let idx = *col_to_idx.entry(v).or_insert_with(|| {
                let old_idx = idx;
                idx += 1;
                old_idx
            });
            idx
        })
        .collect()
}

pub(super) fn compute_col_idx(
    pivot_df: &DataFrame,
    column: &str,
    groups: &GroupsProxy,
) -> PolarsResult<(Vec<IdxSize>, Series)> {
    let column_s = pivot_df.column(column)?;
    let column_agg = unsafe { column_s.agg_first(groups) };
    let column_agg_physical = column_agg.to_physical_repr();

    use DataType::*;
    let col_locations = match column_agg_physical.dtype() {
        Int32 | UInt32 | Float32 => {
            let ca = column_agg_physical.bit_repr_small();
            compute_col_idx_numeric(&ca)
        }
        Int64 | UInt64 | Float64 => {
            let ca = column_agg_physical.bit_repr_large();
            compute_col_idx_numeric(&ca)
        }
        _ => {
            let mut col_to_idx = PlHashMap::with_capacity(HASHMAP_INIT_SIZE);
            let mut idx = 0 as IdxSize;
            column_agg_physical
                .phys_iter()
                .map(|v| {
                    let idx = *col_to_idx.entry(v).or_insert_with(|| {
                        let old_idx = idx;
                        idx += 1;
                        old_idx
                    });
                    idx
                })
                .collect()
        }
    };

    Ok((col_locations, column_agg))
}

fn compute_row_idx_numeric<T>(
    index: &[String],
    index_agg_physical: &ChunkedArray<T>,
    count: usize,
    logical_type: &DataType,
) -> (Vec<IdxSize>, usize, Option<Vec<Series>>)
where
    T: PolarsNumericType,
    T::Native: Hash + Eq,
    ChunkedArray<T>: IntoSeries,
{
    let mut row_to_idx =
        PlIndexMap::with_capacity_and_hasher(HASHMAP_INIT_SIZE, Default::default());
    let mut idx = 0 as IdxSize;
    let row_locations = index_agg_physical
        .into_iter()
        .map(|v| {
            let idx = *row_to_idx.entry(v).or_insert_with(|| {
                let old_idx = idx;
                idx += 1;
                old_idx
            });
            idx
        })
        .collect::<Vec<_>>();

    let row_index = match count {
        0 => {
            let mut s = row_to_idx
                .into_iter()
                .map(|(k, _)| k)
                .collect::<ChunkedArray<T>>()
                .into_series();
            s.rename(&index[0]);
            let s = restore_logical_type(&s, logical_type);
            Some(vec![s])
        }
        _ => None,
    };

    (row_locations, idx as usize, row_index)
}

// TODO! Also create a specialized version for numerics.
pub(super) fn compute_row_idx(
    pivot_df: &DataFrame,
    index: &[String],
    groups: &GroupsProxy,
    count: usize,
) -> PolarsResult<(Vec<IdxSize>, usize, Option<Vec<Series>>)> {
    let (row_locations, n_rows, row_index) = if index.len() == 1 {
        let index_s = pivot_df.column(&index[0])?;
        let index_agg = unsafe { index_s.agg_first(groups) };
        let index_agg_physical = index_agg.to_physical_repr();

        use DataType::*;
        match index_agg_physical.dtype() {
            Int32 | UInt32 | Float32 => {
                let ca = index_agg_physical.bit_repr_small();
                compute_row_idx_numeric(index, &ca, count, index_s.dtype())
            }
            Int64 | UInt64 | Float64 => {
                let ca = index_agg_physical.bit_repr_large();
                compute_row_idx_numeric(index, &ca, count, index_s.dtype())
            }
            _ => {
                let mut row_to_idx =
                    PlIndexMap::with_capacity_and_hasher(HASHMAP_INIT_SIZE, Default::default());
                let mut idx = 0 as IdxSize;
                let row_locations = index_agg_physical
                    .phys_iter()
                    .map(|v| {
                        let idx = *row_to_idx.entry(v).or_insert_with(|| {
                            let old_idx = idx;
                            idx += 1;
                            old_idx
                        });
                        idx
                    })
                    .collect::<Vec<_>>();

                let row_index = match count {
                    0 => {
                        let s = Series::new(
                            &index[0],
                            row_to_idx.into_iter().map(|(k, _)| k).collect::<Vec<_>>(),
                        );
                        let s = restore_logical_type(&s, index_s.dtype());
                        Some(vec![s])
                    }
                    _ => None,
                };

                (row_locations, idx as usize, row_index)
            }
        }
    } else {
        let index_s = pivot_df.columns(index)?;
        let index_agg_physical = index_s
            .iter()
            .map(|s| unsafe { s.agg_first(groups).to_physical_repr().into_owned() })
            .collect::<Vec<_>>();
        let mut iters = index_agg_physical
            .iter()
            .map(|s| s.phys_iter())
            .collect::<Vec<_>>();
        let mut row_to_idx =
            PlIndexMap::with_capacity_and_hasher(HASHMAP_INIT_SIZE, Default::default());
        let mut idx = 0 as IdxSize;

        let mut row_locations = Vec::with_capacity(groups.len());
        loop {
            match iters
                .iter_mut()
                .map(|it| it.next())
                .collect::<Option<Vec<_>>>()
            {
                None => break,
                Some(items) => {
                    let idx = *row_to_idx.entry(items).or_insert_with(|| {
                        let old_idx = idx;
                        idx += 1;
                        old_idx
                    });
                    row_locations.push(idx)
                }
            }
        }
        let row_index = match count {
            0 => Some(
                index
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        let s = Series::new(
                            name,
                            row_to_idx
                                .iter()
                                .map(|(k, _)| {
                                    debug_assert!(i < k.len());
                                    unsafe { k.get_unchecked(i).clone() }
                                })
                                .collect::<Vec<_>>(),
                        );
                        restore_logical_type(&s, index_s[i].dtype())
                    })
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        };

        (row_locations, idx as usize, row_index)
    };

    Ok((row_locations, n_rows, row_index))
}
