use std::hash::Hash;

use arrow::legacy::trusted_len::TrustedLenPush;
use polars_core::prelude::*;
use polars_core::series::BitRepr;
use polars_utils::sync::SyncPtr;
use polars_utils::total_ord::{ToTotalOrd, TotalEq, TotalHash};

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn position_aggregates(
    n_rows: usize,
    n_cols: usize,
    row_locations: &[IdxSize],
    col_locations: &[IdxSize],
    value_agg_phys: &Series,
    logical_type: &DataType,
    headers: &StringChunked,
    default_val: &AnyValue,
) -> Vec<Column> {
    let mut buf = vec![default_val.clone(); n_rows * n_cols];
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
                    // SAFETY:
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
                let name = opt_name
                    .map(PlSmallStr::from_str)
                    .unwrap_or_else(|| PlSmallStr::from_static("null"));
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
                unsafe { out.from_physical_unchecked(logical_type).unwrap() }.into()
            })
            .collect::<Vec<_>>()
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn position_aggregates_numeric<T: PolarsNumericType>(
    n_rows: usize,
    n_cols: usize,
    row_locations: &[IdxSize],
    col_locations: &[IdxSize],
    value_agg_phys: &ChunkedArray<T>,
    logical_type: &DataType,
    headers: &StringChunked,
    default_val: Option<T::Native>,
) -> Vec<Column> {
    let mut buf = vec![default_val; n_rows * n_cols];
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
                    // SAFETY:
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
                let name = opt_name
                    .map(PlSmallStr::from_str)
                    .unwrap_or_else(|| PlSmallStr::from_static("null"));
                let out = ChunkedArray::<T>::from_slice_options(name, opt_values).into_series();
                unsafe { out.from_physical_unchecked(logical_type).unwrap() }.into()
            })
            .collect::<Vec<_>>()
    })
}

fn compute_col_idx_numeric<T>(column_agg_physical: &ChunkedArray<T>) -> Vec<IdxSize>
where
    T: PolarsNumericType,
    T::Native: TotalHash + TotalEq + ToTotalOrd,
    <T::Native as ToTotalOrd>::TotalOrdItem: Hash + Eq,
{
    let mut col_to_idx = PlHashMap::with_capacity(HASHMAP_INIT_SIZE);
    let mut idx = 0 as IdxSize;
    let mut out = Vec::with_capacity(column_agg_physical.len());

    for opt_v in column_agg_physical.iter() {
        let opt_v = opt_v.to_total_ord();
        let idx = *col_to_idx.entry(opt_v).or_insert_with(|| {
            let old_idx = idx;
            idx += 1;
            old_idx
        });
        // SAFETY:
        // we pre-allocated
        unsafe { out.push_unchecked(idx) };
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
    groups: &GroupsType,
) -> PolarsResult<(Vec<IdxSize>, Column)> {
    let column_s = pivot_df.column(column)?;
    let column_agg = unsafe { column_s.agg_first(groups) };
    let column_agg_physical = column_agg.to_physical_repr();

    use DataType as T;
    let col_locations = match column_agg_physical.dtype() {
        T::Int32 | T::UInt32 => {
            let Some(BitRepr::U32(ca)) = column_agg_physical.bit_repr() else {
                polars_bail!(ComputeError: "Expected 32-bit representation to be available; this should never happen");
            };
            compute_col_idx_numeric(&ca)
        },
        T::Int64 | T::UInt64 => {
            let Some(BitRepr::U64(ca)) = column_agg_physical.bit_repr() else {
                polars_bail!(ComputeError: "Expected 64-bit representation to be available; this should never happen");
            };
            compute_col_idx_numeric(&ca)
        },
        T::Float64 => {
            let ca: &ChunkedArray<Float64Type> = column_agg_physical
                .as_materialized_series()
                .as_ref()
                .as_ref();
            compute_col_idx_numeric(ca)
        },
        T::Float32 => {
            let ca: &ChunkedArray<Float32Type> = column_agg_physical
                .as_materialized_series()
                .as_ref()
                .as_ref();
            compute_col_idx_numeric(ca)
        },
        T::Struct(_) => {
            let ca = column_agg_physical.struct_().unwrap();
            let ca = ca.get_row_encoded(Default::default())?;
            compute_col_idx_gen(&ca)
        },
        T::String => {
            let ca = column_agg_physical.str().unwrap();
            let ca = ca.as_binary();
            compute_col_idx_gen(&ca)
        },
        T::Binary => {
            let ca = column_agg_physical.binary().unwrap();
            compute_col_idx_gen(ca)
        },
        T::Boolean => {
            let ca = column_agg_physical.bool().unwrap();
            compute_col_idx_gen(ca)
        },
        _ => {
            let mut col_to_idx = PlHashMap::with_capacity(HASHMAP_INIT_SIZE);
            let mut idx = 0 as IdxSize;
            column_agg_physical
                .as_materialized_series()
                .phys_iter()
                .map(|v| {
                    *col_to_idx.entry(v).or_insert_with(|| {
                        let old_idx = idx;
                        idx += 1;
                        old_idx
                    })
                })
                .collect()
        },
    };

    Ok((col_locations, column_agg))
}

fn compute_row_index<'a, T>(
    index: &[PlSmallStr],
    index_agg_physical: &'a ChunkedArray<T>,
    count: usize,
    logical_type: &DataType,
) -> (Vec<IdxSize>, usize, Option<Vec<Column>>)
where
    T: PolarsPhysicalType,
    T::Physical<'a>: TotalHash + TotalEq + Copy + ToTotalOrd,
    <Option<T::Physical<'a>> as ToTotalOrd>::TotalOrdItem: Hash + Eq,
    ChunkedArray<T>: FromIterator<Option<T::Physical<'a>>>,
{
    let mut row_to_idx =
        PlIndexMap::with_capacity_and_hasher(HASHMAP_INIT_SIZE, Default::default());
    let mut idx = 0 as IdxSize;

    let mut row_locations = Vec::with_capacity(index_agg_physical.len());
    for opt_v in index_agg_physical.iter() {
        let opt_v = opt_v.to_total_ord();
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
    let row_index = match count {
        0 => {
            let mut s = row_to_idx
                .into_iter()
                .map(|(k, _)| Option::<T::Physical<'a>>::peel_total_ord(k))
                .collect::<ChunkedArray<T>>()
                .into_series();
            s.rename(index[0].clone());
            let s = restore_logical_type(&s, logical_type);
            Some(vec![s.into()])
        },
        _ => None,
    };

    (row_locations, idx as usize, row_index)
}

fn compute_row_index_struct(
    index: &[PlSmallStr],
    index_agg: &Series,
    index_agg_physical: &BinaryOffsetChunked,
    count: usize,
) -> (Vec<IdxSize>, usize, Option<Vec<Column>>) {
    let mut row_to_idx =
        PlIndexMap::with_capacity_and_hasher(HASHMAP_INIT_SIZE, Default::default());
    let mut idx = 0 as IdxSize;

    let mut row_locations = Vec::with_capacity(index_agg_physical.len());
    let mut unique_indices = Vec::with_capacity(index_agg_physical.len());
    let mut row_number: IdxSize = 0;
    for arr in index_agg_physical.downcast_iter() {
        for opt_v in arr.iter() {
            let idx = *row_to_idx.entry(opt_v).or_insert_with(|| {
                // SAFETY: we pre-allocated
                unsafe { unique_indices.push_unchecked(row_number) };
                let old_idx = idx;
                idx += 1;
                old_idx
            });
            row_number += 1;

            // SAFETY:
            // we pre-allocated
            unsafe {
                row_locations.push_unchecked(idx);
            }
        }
    }
    let row_index = match count {
        0 => {
            // SAFETY: `unique_indices` is filled with elements between
            // 0 and `index_agg.len() - 1`.
            let mut s = unsafe { index_agg.take_slice_unchecked(&unique_indices) };
            s.rename(index[0].clone());
            Some(vec![s.into()])
        },
        _ => None,
    };

    (row_locations, idx as usize, row_index)
}

// TODO! Also create a specialized version for numerics.
pub(super) fn compute_row_idx(
    pivot_df: &DataFrame,
    index: &[PlSmallStr],
    groups: &GroupsType,
    count: usize,
) -> PolarsResult<(Vec<IdxSize>, usize, Option<Vec<Column>>)> {
    let (row_locations, n_rows, row_index) = if index.len() == 1 {
        let index_s = pivot_df.column(&index[0])?;
        let index_agg = unsafe { index_s.agg_first(groups) };
        let index_agg_physical = index_agg.to_physical_repr();

        use DataType as T;
        match index_agg_physical.dtype() {
            T::Int32 | T::UInt32 => {
                let Some(BitRepr::U32(ca)) = index_agg_physical.bit_repr() else {
                    polars_bail!(ComputeError: "Expected 32-bit representation to be available; this should never happen");
                };
                compute_row_index(index, &ca, count, index_s.dtype())
            },
            T::Int64 | T::UInt64 => {
                let Some(BitRepr::U64(ca)) = index_agg_physical.bit_repr() else {
                    polars_bail!(ComputeError: "Expected 64-bit representation to be available; this should never happen");
                };
                compute_row_index(index, &ca, count, index_s.dtype())
            },
            T::Float64 => {
                let ca: &ChunkedArray<Float64Type> = index_agg_physical
                    .as_materialized_series()
                    .as_ref()
                    .as_ref();
                compute_row_index(index, ca, count, index_s.dtype())
            },
            T::Float32 => {
                let ca: &ChunkedArray<Float32Type> = index_agg_physical
                    .as_materialized_series()
                    .as_ref()
                    .as_ref();
                compute_row_index(index, ca, count, index_s.dtype())
            },
            T::Boolean => {
                let ca = index_agg_physical.bool().unwrap();
                compute_row_index(index, ca, count, index_s.dtype())
            },
            T::Struct(_) => {
                let ca = index_agg_physical.struct_().unwrap();
                let ca = ca.get_row_encoded(Default::default())?;
                compute_row_index_struct(index, index_agg.as_materialized_series(), &ca, count)
            },
            T::String => {
                let ca = index_agg_physical.str().unwrap();
                compute_row_index(index, ca, count, index_s.dtype())
            },
            _ => {
                let mut row_to_idx =
                    PlIndexMap::with_capacity_and_hasher(HASHMAP_INIT_SIZE, Default::default());
                let mut idx = 0 as IdxSize;
                let row_locations = index_agg_physical
                    .as_materialized_series()
                    .phys_iter()
                    .map(|v| {
                        *row_to_idx.entry(v).or_insert_with(|| {
                            let old_idx = idx;
                            idx += 1;
                            old_idx
                        })
                    })
                    .collect::<Vec<_>>();

                let row_index = match count {
                    0 => {
                        let s = Series::new(
                            index[0].clone(),
                            row_to_idx.into_iter().map(|(k, _)| k).collect::<Vec<_>>(),
                        );
                        let s = restore_logical_type(&s, index_s.dtype());
                        Some(vec![Column::from(s)])
                    },
                    _ => None,
                };

                (row_locations, idx as usize, row_index)
            },
        }
    } else {
        let binding = pivot_df.select(index.iter().cloned())?;
        let fields = binding.get_columns();
        let index_struct_series = StructChunked::from_columns(
            PlSmallStr::from_static("placeholder"),
            fields[0].len(),
            fields,
        )?
        .into_series();
        let index_agg = unsafe { index_struct_series.agg_first(groups) };
        let index_agg_physical = index_agg.to_physical_repr();
        let ca = index_agg_physical.struct_()?;
        let ca = ca.get_row_encoded(Default::default())?;
        let (row_locations, n_rows, row_index) =
            compute_row_index_struct(index, &index_agg, &ca, count);
        let row_index = row_index.map(|x| {
             let ca = x.first().unwrap()
                .struct_().unwrap();

            polars_ensure!(ca.null_count() == 0, InvalidOperation: "outer nullability in struct pivot not yet supported");

            // @scalar-opt
            Ok(ca.fields_as_series().into_iter().map(Column::from).collect())
        }).transpose()?;
        (row_locations, n_rows, row_index)
    };

    Ok((row_locations, n_rows, row_index))
}
