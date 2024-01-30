use std::hash::Hash;

use arrow::legacy::trusted_len::TrustedLenPush;
use polars_core::prelude::*;
use polars_utils::sync::SyncPtr;

use super::*;

pub(super) fn position_aggregates(
    n_rows: usize,
    n_cols: usize,
    row_locations: &[IdxSize],
    col_locations: &[IdxSize],
    value_agg_phys: &Series,
    logical_type: &DataType,
    headers: &StringChunked,
) -> Vec<Series> {
    let mut buf = vec![AnyValue::Null; n_rows * n_cols];
    let start_ptr = buf.as_mut_ptr() as usize;

    let n_threads = POOL.current_num_threads();
    let split = _split_offsets(row_locations.len(), n_threads);

    // ensure the slice series are not dropped
    // so the AnyValues are referencing correct data, if they reference arrays (struct)
    let n_splits = split.len();
    let mut arrays: Vec<Series> = Vec::with_capacity(n_splits);

    // every thread will only write to their partition
    let array_ptr = unsafe { SyncPtr::new(arrays.as_mut_ptr()) };

    POOL.install(|| {
        split
            .into_par_iter()
            .enumerate()
            .for_each(|(i, (offset, len))| {
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
                // ensure the `values_agg_phys` stays alive
                let array_ptr = array_ptr.clone().get();
                unsafe { std::ptr::write(array_ptr.add(i), value_agg_phys) }
            });
        // ensure the content of the arrays are dropped
        unsafe {
            arrays.set_len(n_splits);
        }

        let headers_iter = headers.par_iter_indexed();
        let phys_type = logical_type.to_physical();

        (0..n_cols)
            .into_par_iter()
            .zip(headers_iter)
            .map(|(i, opt_name)| {
                let offset = i * n_rows;
                let avs = &buf[offset..offset + n_rows];
                let name = opt_name.unwrap_or("null");
                let out = match &phys_type {
                    #[cfg(feature = "dtype-struct")]
                    DataType::Struct(_) => {
                        // we know we can trust this data, so we use the explicit builder
                        use polars_core::frame::row::AnyValueBufferTrusted;
                        let mut buf = AnyValueBufferTrusted::new(&phys_type, avs.len());
                        for av in avs {
                            unsafe {
                                buf.add_unchecked_borrowed_physical(av);
                            }
                        }
                        let mut out = buf.into_series();
                        out.rename(name);
                        out
                    },
                    _ => Series::from_any_values_and_dtype(name, avs, &phys_type, false).unwrap(),
                };
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
    headers: &StringChunked,
) -> Vec<Series>
where
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
{
    let mut buf = vec![None; n_rows * n_cols];
    let start_ptr = buf.as_mut_ptr() as usize;

    let n_threads = POOL.current_num_threads();

    let split = _split_offsets(row_locations.len(), n_threads);
    let n_splits = split.len();
    // ensure the arrays are not dropped
    // so the AnyValues are referencing correct data, if they reference arrays (struct)
    let mut arrays: Vec<ChunkedArray<T>> = Vec::with_capacity(n_splits);

    // every thread will only write to their partition
    let array_ptr = unsafe { SyncPtr::new(arrays.as_mut_ptr()) };

    POOL.install(|| {
        split
            .into_par_iter()
            .enumerate()
            .for_each(|(i, (offset, len))| {
                let start_ptr = start_ptr as *mut Option<T::Native>;
                let row_locations = &row_locations[offset..offset + len];
                let col_locations = &col_locations[offset..offset + len];
                let value_agg_phys = value_agg_phys.slice(offset as i64, len);

                // todo! remove lint silencing
                #[allow(clippy::useless_conversion)]
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
                // ensure the `values_agg_phys` stays alive
                let array_ptr = array_ptr.clone().get();
                unsafe { std::ptr::write(array_ptr.add(i), value_agg_phys) }
            });
        // ensure the content of the arrays are dropped
        unsafe {
            arrays.set_len(n_splits);
        }
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
    let mut out = Vec::with_capacity(column_agg_physical.len());

    for arr in column_agg_physical.downcast_iter() {
        for opt_v in arr.into_iter() {
            let idx = *col_to_idx.entry(opt_v).or_insert_with(|| {
                let old_idx = idx;
                idx += 1;
                old_idx
            });
            // SAFETY:
            // we pre-allocated
            unsafe { out.push_unchecked(idx) };
        }
    }
    out
}

fn compute_col_idx_gen<'a, T>(column_agg_physical: &'a ChunkedArray<T>) -> Vec<IdxSize>
where
    T: PolarsDataType,
    &'a T::Array: IntoIterator<Item = Option<T::Physical<'a>>>,
    T::Physical<'a>: Hash + Eq,
{
    let mut col_to_idx = PlHashMap::with_capacity(HASHMAP_INIT_SIZE);
    let mut idx = 0 as IdxSize;
    let mut out = Vec::with_capacity(column_agg_physical.len());

    for arr in column_agg_physical.downcast_iter() {
        for opt_v in arr.into_iter() {
            let idx = *col_to_idx.entry(opt_v).or_insert_with(|| {
                let old_idx = idx;
                idx += 1;
                old_idx
            });
            // SAFETY:
            // we pre-allocated
            unsafe { out.push_unchecked(idx) };
        }
    }
    out
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
        },
        Int64 | UInt64 | Float64 => {
            let ca = column_agg_physical.bit_repr_large();
            compute_col_idx_numeric(&ca)
        },
        Struct(_) => {
            let ca = column_agg_physical.struct_().unwrap();
            let ca = ca.rows_encode()?;
            compute_col_idx_gen(&ca)
        },
        String => {
            let ca = column_agg_physical.str().unwrap();
            let ca = ca.as_binary();
            compute_col_idx_gen(&ca)
        },
        Binary => {
            let ca = column_agg_physical.binary().unwrap();
            compute_col_idx_gen(ca)
        },
        Boolean => {
            let ca = column_agg_physical.bool().unwrap();
            compute_col_idx_gen(ca)
        },
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
        },
    };

    Ok((col_locations, column_agg))
}

fn compute_row_index<'a, T>(
    index: &[String],
    index_agg_physical: &'a ChunkedArray<T>,
    count: usize,
    logical_type: &DataType,
) -> (Vec<IdxSize>, usize, Option<Vec<Series>>)
where
    T: PolarsDataType,
    T::Physical<'a>: Hash + Eq + Copy,
    ChunkedArray<T>: FromIterator<Option<T::Physical<'a>>>,
    ChunkedArray<T>: IntoSeries,
{
    let mut row_to_idx =
        PlIndexMap::with_capacity_and_hasher(HASHMAP_INIT_SIZE, Default::default());
    let mut idx = 0 as IdxSize;

    let mut row_locations = Vec::with_capacity(index_agg_physical.len());
    for arr in index_agg_physical.downcast_iter() {
        for opt_v in arr.iter() {
            let idx = *row_to_idx.entry(opt_v).or_insert_with(|| {
                let old_idx = idx;
                idx += 1;
                old_idx
            });

            // SAFETY:
            // we pre-allocated
            unsafe {
                row_locations.push_unchecked(idx);
            }
        }
    }
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
        },
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
                compute_row_index(index, &ca, count, index_s.dtype())
            },
            Int64 | UInt64 | Float64 => {
                let ca = index_agg_physical.bit_repr_large();
                compute_row_index(index, &ca, count, index_s.dtype())
            },
            Boolean => {
                let ca = index_agg_physical.bool().unwrap();
                compute_row_index(index, ca, count, index_s.dtype())
            },
            String => {
                let ca = index_agg_physical.str().unwrap();
                compute_row_index(index, ca, count, index_s.dtype())
            },
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
                    },
                    _ => None,
                };

                (row_locations, idx as usize, row_index)
            },
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
                },
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
