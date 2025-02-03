use std::borrow::Cow;
use std::collections::VecDeque;
use std::ops::Range;

use arrow::array::BooleanArray;
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::datatypes::ArrowSchemaRef;
use polars_core::chunked_array::builder::NullChunkedBuilder;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::utils::{accumulate_dataframes_vertical, split_df};
use polars_core::{config, POOL};
use polars_parquet::read::{
    self, ColumnChunkMetadata, FileMetadata, Filter, PredicateFilter, RowGroupMetadata,
};
use rayon::prelude::*;

#[cfg(feature = "cloud")]
use super::async_impl::FetchRowGroupsFromObjectStore;
use super::mmap::{mmap_columns, ColumnStore};
use super::predicates::read_this_row_group;
use super::utils::materialize_empty_df;
use super::{mmap, ParallelStrategy};
use crate::hive::{self, materialize_hive_partitions};
use crate::mmap::{MmapBytesReader, ReaderBytes};
use crate::parquet::metadata::FileMetadataRef;
use crate::parquet::read::ROW_COUNT_OVERFLOW_ERR;
use crate::predicates::{apply_predicate, ColumnPredicateExpr, ScanIOPredicate};
use crate::utils::get_reader_bytes;
use crate::utils::slice::split_slice_at_file;
use crate::RowIndex;

#[cfg(debug_assertions)]
// Ensure we get the proper polars types from schema inference
// This saves unneeded casts.
fn assert_dtypes(dtype: &ArrowDataType) {
    use ArrowDataType as D;

    match dtype {
        // These should all be cast to the BinaryView / Utf8View variants
        D::Utf8 | D::Binary | D::LargeUtf8 | D::LargeBinary => unreachable!(),

        // These should be cast to Float32
        D::Float16 => unreachable!(),

        // This should have been converted to a LargeList
        D::List(_) => unreachable!(),

        // This should have been converted to a LargeList(Struct(_))
        D::Map(_, _) => unreachable!(),

        // Recursive checks
        D::Dictionary(_, dtype, _) => assert_dtypes(dtype),
        D::Extension(ext) => assert_dtypes(&ext.inner),
        D::LargeList(inner) => assert_dtypes(&inner.dtype),
        D::FixedSizeList(inner, _) => assert_dtypes(&inner.dtype),
        D::Struct(fields) => fields.iter().for_each(|f| assert_dtypes(f.dtype())),

        _ => {},
    }
}

fn should_copy_sortedness(dtype: &DataType) -> bool {
    // @NOTE: For now, we are a bit conservative with this.
    use DataType as D;

    matches!(
        dtype,
        D::Int8 | D::Int16 | D::Int32 | D::Int64 | D::UInt8 | D::UInt16 | D::UInt32 | D::UInt64
    )
}

pub fn try_set_sorted_flag(
    series: &mut Series,
    col_idx: usize,
    sorting_map: &PlHashMap<usize, IsSorted>,
) {
    if let Some(is_sorted) = sorting_map.get(&col_idx) {
        if should_copy_sortedness(series.dtype()) {
            if config::verbose() {
                eprintln!(
                    "Parquet conserved SortingColumn for column chunk of '{}' to {is_sorted:?}",
                    series.name()
                );
            }

            series.set_sorted_flag(*is_sorted);
        }
    }
}

pub fn create_sorting_map(md: &RowGroupMetadata) -> PlHashMap<usize, IsSorted> {
    let capacity = md.sorting_columns().map_or(0, |s| s.len());
    let mut sorting_map = PlHashMap::with_capacity(capacity);

    if let Some(sorting_columns) = md.sorting_columns() {
        for sorting in sorting_columns {
            let prev_value = sorting_map.insert(
                sorting.column_idx as usize,
                if sorting.descending {
                    IsSorted::Descending
                } else {
                    IsSorted::Ascending
                },
            );

            debug_assert!(prev_value.is_none());
        }
    }

    sorting_map
}

fn column_idx_to_series(
    column_i: usize,
    // The metadata belonging to this column
    field_md: &[&ColumnChunkMetadata],
    filter: Option<Filter>,
    file_schema: &ArrowSchema,
    store: &mmap::ColumnStore,
) -> PolarsResult<(Series, Bitmap)> {
    let field = file_schema.get_at_index(column_i).unwrap().1;

    #[cfg(debug_assertions)]
    {
        assert_dtypes(field.dtype())
    }
    let columns = mmap_columns(store, field_md);
    let (array, pred_true_mask) = mmap::to_deserializer(columns, field.clone(), filter)?;
    let series = Series::try_from((field, array))?;

    Ok((series, pred_true_mask))
}

#[allow(clippy::too_many_arguments)]
fn rg_to_dfs(
    store: &mmap::ColumnStore,
    previous_row_count: &mut IdxSize,
    row_group_start: usize,
    row_group_end: usize,
    slice: (usize, usize),
    file_metadata: &FileMetadata,
    schema: &ArrowSchemaRef,
    predicate: Option<&ScanIOPredicate>,
    row_index: Option<RowIndex>,
    parallel: ParallelStrategy,
    projection: &[usize],
    use_statistics: bool,
    hive_partition_columns: Option<&[Series]>,
) -> PolarsResult<Vec<DataFrame>> {
    if config::verbose() {
        eprintln!("parquet scan with parallel = {parallel:?}");
    }

    // If we are only interested in the row_index, we take a little special path here.
    if projection.is_empty() {
        if let Some(row_index) = row_index {
            let placeholder =
                NullChunkedBuilder::new(PlSmallStr::from_static("__PL_TMP"), slice.1).finish();
            return Ok(vec![DataFrame::new(vec![placeholder
                .into_series()
                .into_column()])?
            .with_row_index(
                row_index.name.clone(),
                Some(row_index.offset + IdxSize::try_from(slice.0).unwrap()),
            )?
            .select(std::iter::once(row_index.name))?]);
        }
    }

    use ParallelStrategy as S;

    if parallel == S::Prefiltered {
        if let Some(predicate) = predicate {
            if !predicate.live_columns.is_empty() {
                return rg_to_dfs_prefiltered(
                    store,
                    previous_row_count,
                    row_group_start,
                    row_group_end,
                    file_metadata,
                    schema,
                    predicate,
                    row_index,
                    projection,
                    use_statistics,
                    hive_partition_columns,
                );
            }
        }
    }

    match parallel {
        S::Columns | S::None => rg_to_dfs_optionally_par_over_columns(
            store,
            previous_row_count,
            row_group_start,
            row_group_end,
            slice,
            file_metadata,
            schema,
            predicate,
            row_index,
            parallel,
            projection,
            use_statistics,
            hive_partition_columns,
        ),
        _ => rg_to_dfs_par_over_rg(
            store,
            row_group_start,
            row_group_end,
            previous_row_count,
            slice,
            file_metadata,
            schema,
            predicate,
            row_index,
            projection,
            use_statistics,
            hive_partition_columns,
        ),
    }
}

/// Load several Parquet row groups as DataFrames while filtering predicate items.
///
/// This strategy works as follows:
///
/// ```text
/// For each Row Group:
///     1. Skip this row group if statistics already filter it out
///     2. Load all the data for the columns needed for the predicate (i.e. the live columns)
///     3. Create a predicate mask.
///     4. Load the filtered data for the columns not in the predicate (i.e. the dead columns)
///     5. Merge the columns into the right DataFrame
/// ```
#[allow(clippy::too_many_arguments)]
fn rg_to_dfs_prefiltered(
    store: &mmap::ColumnStore,
    previous_row_count: &mut IdxSize,
    row_group_start: usize,
    row_group_end: usize,
    file_metadata: &FileMetadata,
    schema: &ArrowSchemaRef,
    predicate: &ScanIOPredicate,
    row_index: Option<RowIndex>,
    projection: &[usize],
    use_statistics: bool,
    hive_partition_columns: Option<&[Series]>,
) -> PolarsResult<Vec<DataFrame>> {
    if row_group_end > u32::MAX as usize {
        polars_bail!(ComputeError: "Parquet file contains too many row groups (> {})", u32::MAX);
    }

    let mut row_offset = *previous_row_count;
    let rg_offsets: Vec<IdxSize> = match row_index {
        None => Vec::new(),
        Some(_) => (row_group_start..row_group_end)
            .map(|index| {
                let md = &file_metadata.row_groups[index];

                let current_offset = row_offset;
                let current_row_count = md.num_rows() as IdxSize;
                row_offset += current_row_count;

                current_offset
            })
            .collect(),
    };

    // Get the number of live columns
    let num_live_columns = predicate.live_columns.len();
    let num_dead_columns =
        projection.len() + hive_partition_columns.map_or(0, |x| x.len()) - num_live_columns;

    if config::verbose() {
        eprintln!("parquet live columns = {num_live_columns}, dead columns = {num_dead_columns}");
    }

    // We create two look-up tables that map indexes offsets into the live- and dead-set onto
    // column indexes of the schema.
    // Note: This may contain less than `num_live_columns` if there are hive columns involved.
    let mut live_idx_to_col_idx = Vec::with_capacity(num_live_columns);
    let mut dead_idx_to_col_idx: Vec<usize> = Vec::with_capacity(num_dead_columns);
    for &i in projection.iter() {
        let name = schema.get_at_index(i).unwrap().0.as_str();

        if predicate.live_columns.contains(name) {
            live_idx_to_col_idx.push(i);
        } else {
            dead_idx_to_col_idx.push(i);
        }
    }

    let do_parquet_expr = std::env::var("POLARS_PARQUET_EXPR").as_deref() == Ok("1")
        && predicate.live_columns.len() == 1 // Only do it with one column for now
        && hive_partition_columns.is_none_or(|hc| {
            !hc.iter()
                .any(|c| c.name().as_str() == predicate.live_columns[0].as_str())
        }) // No hive columns
        && !schema
            .get(predicate.live_columns[0].as_str())
            .unwrap()
            .dtype()
            .is_nested(); // No nested columns
    let column_exprs = do_parquet_expr.then(|| {
        predicate
            .live_columns
            .iter()
            .map(|name| {
                let (p, specialized) = predicate.predicate.isolate_column_expr(name.as_str())?;

                let p = ColumnPredicateExpr::new(
                    name.clone(),
                    DataType::from_arrow_field(schema.get(name).unwrap()),
                    p,
                    specialized,
                );

                let eq_scalar = p.to_eq_scalar().cloned();
                let predicate = Arc::new(p) as _;

                Some((
                    PredicateFilter {
                        predicate,
                        include_values: eq_scalar.is_none(),
                    },
                    eq_scalar,
                ))
            })
            .collect::<Vec<_>>()
    });

    let mask_setting = PrefilterMaskSetting::init_from_env();
    let projected_schema = schema.try_project_indices(projection).unwrap();

    let dfs: Vec<Option<DataFrame>> = POOL.install(move || {
        // Set partitioned fields to prevent quadratic behavior.
        // Ensure all row groups are partitioned.

        (row_group_start..row_group_end)
            .into_par_iter()
            .map(|rg_idx| {
                let md = &file_metadata.row_groups[rg_idx];

                if use_statistics {
                    match read_this_row_group(Some(predicate), md, schema) {
                        Ok(false) => return Ok(None),
                        Ok(true) => {},
                        Err(e) => return Err(e),
                    }
                }

                let sorting_map = create_sorting_map(md);

                // Collect the data for the live columns
                let (live_columns, filters) = (0..live_idx_to_col_idx.len())
                    .into_par_iter()
                    .map(|i| {
                        let col_idx = live_idx_to_col_idx[i];

                        let (name, field) = schema.get_at_index(col_idx).unwrap();

                        let Some(iter) = md.columns_under_root_iter(name) else {
                            return Ok((
                                Column::full_null(
                                    name.clone(),
                                    md.num_rows(),
                                    &DataType::from_arrow_field(field),
                                ),
                                None,
                            ));
                        };

                        let part = iter.collect::<Vec<_>>();

                        let (filter, equals_scalar) = match column_exprs.as_ref() {
                            None => (None, None),
                            Some(column_expr) => match column_expr.get(i) {
                                Some(Some((p, s))) => {
                                    (Some(Filter::Predicate(p.clone())), s.clone())
                                },
                                _ => (None, None),
                            },
                        };

                        let (mut series, pred_true_mask) =
                            column_idx_to_series(col_idx, part.as_slice(), filter, schema, store)?;

                        debug_assert!(
                            pred_true_mask.is_empty() || pred_true_mask.len() == md.num_rows()
                        );
                        match equals_scalar {
                            None => {
                                try_set_sorted_flag(&mut series, col_idx, &sorting_map);
                                Ok((
                                    series.into_column(),
                                    (!pred_true_mask.is_empty()).then_some(pred_true_mask),
                                ))
                            },
                            Some(sc) => Ok((
                                Column::new_scalar(name.clone(), sc, pred_true_mask.set_bits()),
                                Some(pred_true_mask),
                            )),
                        }
                    })
                    .collect::<PolarsResult<(Vec<_>, Vec<_>)>>()?;

                // Apply the predicate to the live columns and save the dataframe and the bitmask
                let md = &file_metadata.row_groups[rg_idx];
                let filter_mask: Bitmap;
                let mut df: DataFrame;

                if let Some(Some(f)) = filters.first() {
                    if f.set_bits() == 0 {
                        if config::verbose() {
                            eprintln!("parquet filter mask found that row group can be skipped");
                        }

                        return Ok(None);
                    }

                    if let Some(rc) = &row_index {
                        df = unsafe { DataFrame::new_no_checks(md.num_rows(), vec![]) };
                        df.with_row_index_mut(
                            rc.name.clone(),
                            Some(rg_offsets[rg_idx] + rc.offset),
                        );
                        df = df.filter(&BooleanChunked::from_chunk_iter(
                            PlSmallStr::EMPTY,
                            [BooleanArray::new(ArrowDataType::Boolean, f.clone(), None)],
                        ))?;
                        unsafe { df.column_extend_unchecked(live_columns) }
                    } else {
                        df = DataFrame::new(live_columns).unwrap();
                    }

                    filter_mask = f.clone();
                } else {
                    df = unsafe { DataFrame::new_no_checks(md.num_rows(), live_columns.clone()) };

                    materialize_hive_partitions(
                        &mut df,
                        schema.as_ref(),
                        hive_partition_columns,
                        md.num_rows(),
                    );
                    let s = predicate.predicate.evaluate_io(&df)?;
                    let mask = s.bool().expect("filter predicates was not of type boolean");

                    // Create without hive columns - the first merge phase does not handle hive partitions. This also saves
                    // some unnecessary filtering.
                    df = unsafe { DataFrame::new_no_checks(md.num_rows(), live_columns) };

                    if let Some(rc) = &row_index {
                        df.with_row_index_mut(
                            rc.name.clone(),
                            Some(rg_offsets[rg_idx] + rc.offset),
                        );
                    }
                    df = df.filter(mask)?;

                    let mut mut_filter_mask = BitmapBuilder::with_capacity(mask.len());

                    // We need to account for the validity of the items
                    for chunk in mask.downcast_iter() {
                        match chunk.validity() {
                            None => mut_filter_mask.extend_from_bitmap(chunk.values()),
                            Some(validity) => {
                                mut_filter_mask.extend_from_bitmap(&(validity & chunk.values()))
                            },
                        }
                    }

                    filter_mask = mut_filter_mask.freeze();
                }

                debug_assert_eq!(md.num_rows(), filter_mask.len());
                debug_assert_eq!(df.height(), filter_mask.set_bits());

                if filter_mask.set_bits() == 0 {
                    if config::verbose() {
                        eprintln!("parquet filter mask found that row group can be skipped");
                    }

                    return Ok(None);
                }

                // We don't need to do any further work if there are no dead columns
                if dead_idx_to_col_idx.is_empty() {
                    materialize_hive_partitions(
                        &mut df,
                        schema.as_ref(),
                        hive_partition_columns,
                        md.num_rows(),
                    );

                    return Ok(Some(df));
                }

                let prefilter_cost = matches!(mask_setting, PrefilterMaskSetting::Auto)
                    .then(|| calc_prefilter_cost(&filter_mask))
                    .unwrap_or_default();

                // #[cfg(debug_assertions)]
                // {
                //     let md = &file_metadata.row_groups[rg_idx];
                //     debug_assert_eq!(md.num_rows(), mask.len());
                // }

                let n_rows_in_result = filter_mask.set_bits();

                let dead_columns = (0..dead_idx_to_col_idx.len())
                    .into_par_iter()
                    .map(|i| {
                        let col_idx = dead_idx_to_col_idx[i];

                        let (name, field) = schema.get_at_index(col_idx).unwrap();

                        let Some(iter) = md.columns_under_root_iter(name) else {
                            return Ok(Column::full_null(
                                name.clone(),
                                n_rows_in_result,
                                &DataType::from_arrow_field(field),
                            ));
                        };

                        let field_md = iter.collect::<Vec<_>>();

                        let pre = || {
                            let (array, _) = column_idx_to_series(
                                col_idx,
                                field_md.as_slice(),
                                Some(Filter::new_masked(filter_mask.clone())),
                                schema,
                                store,
                            )?;

                            PolarsResult::Ok(array)
                        };
                        let post = || {
                            let (array, _) = column_idx_to_series(
                                col_idx,
                                field_md.as_slice(),
                                None,
                                schema,
                                store,
                            )?;

                            debug_assert_eq!(array.len(), md.num_rows());

                            let mask_arr = BooleanArray::new(
                                ArrowDataType::Boolean,
                                filter_mask.clone(),
                                None,
                            );
                            let mask_arr = BooleanChunked::from(mask_arr);
                            array.filter(&mask_arr)
                        };

                        let mut series = if mask_setting.should_prefilter(
                            prefilter_cost,
                            &schema.get_at_index(col_idx).unwrap().1.dtype,
                        ) {
                            pre()?
                        } else {
                            post()?
                        };

                        debug_assert_eq!(series.len(), filter_mask.set_bits());

                        try_set_sorted_flag(&mut series, col_idx, &sorting_map);

                        Ok(series.into_column())
                    })
                    .collect::<PolarsResult<Vec<Column>>>()?;

                debug_assert!(dead_columns.iter().all(|v| v.len() == df.height()));

                let height = df.height();
                let live_columns = df.take_columns();

                assert_eq!(live_columns.len() + dead_columns.len(), projection.len());

                let mut merged = Vec::with_capacity(live_columns.len() + dead_columns.len());

                // * All hive columns are always in `live_columns` if there are any.
                // * `materialize_hive_partitions()` guarantees `live_columns` is sorted by their appearance in `reader_schema`.

                // We re-use `hive::merge_sorted_to_schema_order()` as it performs most of the merge operation we want.
                // But we take out the `row_index` column as it isn't on the right side.

                if row_index.is_some() {
                    merged.push(live_columns[0].clone());
                };

                hive::merge_sorted_to_schema_order(
                    &mut dead_columns.into_iter(), // df_columns
                    &mut live_columns.into_iter().skip(row_index.is_some() as usize), // hive_columns
                    &projected_schema,
                    &mut merged,
                );

                // SAFETY: This is completely based on the schema so all column names are unique
                // and the length is given by the parquet file which should always be the same.
                let mut df = unsafe { DataFrame::new_no_checks(height, merged) };

                materialize_hive_partitions(
                    &mut df,
                    schema.as_ref(),
                    hive_partition_columns,
                    md.num_rows(),
                );

                PolarsResult::Ok(Some(df))
            })
            .collect::<PolarsResult<Vec<Option<DataFrame>>>>()
    })?;

    let dfs: Vec<DataFrame> = dfs.into_iter().flatten().collect();

    let row_count: usize = dfs.iter().map(|df| df.height()).sum();
    let row_count = IdxSize::try_from(row_count).map_err(|_| ROW_COUNT_OVERFLOW_ERR)?;
    *previous_row_count = previous_row_count
        .checked_add(row_count)
        .ok_or(ROW_COUNT_OVERFLOW_ERR)?;

    Ok(dfs)
}

#[allow(clippy::too_many_arguments)]
// might parallelize over columns
fn rg_to_dfs_optionally_par_over_columns(
    store: &mmap::ColumnStore,
    previous_row_count: &mut IdxSize,
    row_group_start: usize,
    row_group_end: usize,
    slice: (usize, usize),
    file_metadata: &FileMetadata,
    schema: &ArrowSchemaRef,
    predicate: Option<&ScanIOPredicate>,
    row_index: Option<RowIndex>,
    parallel: ParallelStrategy,
    projection: &[usize],
    use_statistics: bool,
    hive_partition_columns: Option<&[Series]>,
) -> PolarsResult<Vec<DataFrame>> {
    let mut dfs = Vec::with_capacity(row_group_end - row_group_start);

    let mut n_rows_processed: usize = (0..row_group_start)
        .map(|i| file_metadata.row_groups[i].num_rows())
        .sum();
    let slice_end = slice.0 + slice.1;

    for rg_idx in row_group_start..row_group_end {
        let md = &file_metadata.row_groups[rg_idx];

        let rg_slice =
            split_slice_at_file(&mut n_rows_processed, md.num_rows(), slice.0, slice_end);
        let current_row_count = md.num_rows() as IdxSize;

        if use_statistics
            && !read_this_row_group(predicate, &file_metadata.row_groups[rg_idx], schema)?
        {
            *previous_row_count += rg_slice.1 as IdxSize;
            continue;
        }

        let sorting_map = create_sorting_map(md);

        let f = |column_i: &usize| {
            let (name, field) = schema.get_at_index(*column_i).unwrap();

            let Some(iter) = md.columns_under_root_iter(name) else {
                return Ok(Column::full_null(
                    name.clone(),
                    rg_slice.1,
                    &DataType::from_arrow_field(field),
                ));
            };

            let part = iter.collect::<Vec<_>>();

            let (mut series, _) = column_idx_to_series(
                *column_i,
                part.as_slice(),
                Some(Filter::new_ranged(rg_slice.0, rg_slice.0 + rg_slice.1)),
                schema,
                store,
            )?;

            try_set_sorted_flag(&mut series, *column_i, &sorting_map);
            Ok(series.into_column())
        };

        let columns = if let ParallelStrategy::Columns = parallel {
            POOL.install(|| {
                projection
                    .par_iter()
                    .map(f)
                    .collect::<PolarsResult<Vec<_>>>()
            })?
        } else {
            projection.iter().map(f).collect::<PolarsResult<Vec<_>>>()?
        };

        let mut df = unsafe { DataFrame::new_no_checks(rg_slice.1, columns) };
        if let Some(rc) = &row_index {
            df.with_row_index_mut(
                rc.name.clone(),
                Some(*previous_row_count + rc.offset + rg_slice.0 as IdxSize),
            );
        }

        materialize_hive_partitions(&mut df, schema.as_ref(), hive_partition_columns, rg_slice.1);
        apply_predicate(
            &mut df,
            predicate.as_ref().map(|p| p.predicate.as_ref()),
            true,
        )?;

        *previous_row_count = previous_row_count.checked_add(current_row_count).ok_or_else(||
            polars_err!(
                ComputeError: "Parquet file produces more than pow(2, 32) rows; \
                consider compiling with polars-bigidx feature (polars-u64-idx package on python), \
                or set 'streaming'"
            ),
        )?;
        dfs.push(df);

        if *previous_row_count as usize >= slice_end {
            break;
        }
    }

    Ok(dfs)
}

#[allow(clippy::too_many_arguments)]
// parallelizes over row groups
fn rg_to_dfs_par_over_rg(
    store: &mmap::ColumnStore,
    row_group_start: usize,
    row_group_end: usize,
    rows_read: &mut IdxSize,
    slice: (usize, usize),
    file_metadata: &FileMetadata,
    schema: &ArrowSchemaRef,
    predicate: Option<&ScanIOPredicate>,
    row_index: Option<RowIndex>,
    projection: &[usize],
    use_statistics: bool,
    hive_partition_columns: Option<&[Series]>,
) -> PolarsResult<Vec<DataFrame>> {
    // compute the limits per row group and the row count offsets
    let mut row_groups = Vec::with_capacity(row_group_end - row_group_start);

    let mut n_rows_processed: usize = (0..row_group_start)
        .map(|i| file_metadata.row_groups[i].num_rows())
        .sum();
    let slice_end = slice.0 + slice.1;

    // rows_scanned is the number of rows that have been scanned so far when checking for overlap with the slice.
    // rows_read is the number of rows found to overlap with the slice, and thus the number of rows that will be
    // read into a dataframe.
    let mut rows_scanned: IdxSize;

    if row_group_start > 0 {
        // In the case of async reads, we need to account for the fact that row_group_start may be greater than
        // zero due to earlier processing.
        // For details, see: https://github.com/pola-rs/polars/pull/20508#discussion_r1900165649
        rows_scanned = (0..row_group_start)
            .map(|i| file_metadata.row_groups[i].num_rows() as IdxSize)
            .sum();
    } else {
        rows_scanned = 0;
    }

    for i in row_group_start..row_group_end {
        let row_count_start = rows_scanned;
        let rg_md = &file_metadata.row_groups[i];
        let n_rows_this_file = rg_md.num_rows();
        let rg_slice =
            split_slice_at_file(&mut n_rows_processed, n_rows_this_file, slice.0, slice_end);
        rows_scanned = rows_scanned
            .checked_add(n_rows_this_file as IdxSize)
            .ok_or(ROW_COUNT_OVERFLOW_ERR)?;

        *rows_read += rg_slice.1 as IdxSize;

        if rg_slice.1 == 0 {
            continue;
        }

        row_groups.push((rg_md, rg_slice, row_count_start));
    }

    let dfs = POOL.install(|| {
        // Set partitioned fields to prevent quadratic behavior.
        // Ensure all row groups are partitioned.
        row_groups
            .into_par_iter()
            .map(|(md, slice, row_count_start)| {
                if slice.1 == 0 || use_statistics && !read_this_row_group(predicate, md, schema)? {
                    return Ok(None);
                }
                // test we don't read the parquet file if this env var is set
                #[cfg(debug_assertions)]
                {
                    assert!(std::env::var("POLARS_PANIC_IF_PARQUET_PARSED").is_err())
                }

                let sorting_map = create_sorting_map(md);

                let columns = projection
                    .iter()
                    .map(|column_i| {
                        let (name, field) = schema.get_at_index(*column_i).unwrap();

                        let Some(iter) = md.columns_under_root_iter(name) else {
                            return Ok(Column::full_null(
                                name.clone(),
                                md.num_rows(),
                                &DataType::from_arrow_field(field),
                            ));
                        };

                        let part = iter.collect::<Vec<_>>();

                        let (mut series, _) = column_idx_to_series(
                            *column_i,
                            part.as_slice(),
                            Some(Filter::new_ranged(slice.0, slice.0 + slice.1)),
                            schema,
                            store,
                        )?;

                        try_set_sorted_flag(&mut series, *column_i, &sorting_map);
                        Ok(series.into_column())
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;

                let mut df = unsafe { DataFrame::new_no_checks(slice.1, columns) };

                if let Some(rc) = &row_index {
                    df.with_row_index_mut(
                        rc.name.clone(),
                        Some(row_count_start as IdxSize + rc.offset + slice.0 as IdxSize),
                    );
                }

                materialize_hive_partitions(
                    &mut df,
                    schema.as_ref(),
                    hive_partition_columns,
                    slice.1,
                );
                apply_predicate(
                    &mut df,
                    predicate.as_ref().map(|p| p.predicate.as_ref()),
                    false,
                )?;

                Ok(Some(df))
            })
            .collect::<PolarsResult<Vec<_>>>()
    })?;
    Ok(dfs.into_iter().flatten().collect())
}

#[allow(clippy::too_many_arguments)]
pub fn read_parquet<R: MmapBytesReader>(
    mut reader: R,
    slice: (usize, usize),
    projection: Option<&[usize]>,
    reader_schema: &ArrowSchemaRef,
    metadata: Option<FileMetadataRef>,
    predicate: Option<&ScanIOPredicate>,
    mut parallel: ParallelStrategy,
    row_index: Option<RowIndex>,
    use_statistics: bool,
    hive_partition_columns: Option<&[Series]>,
) -> PolarsResult<DataFrame> {
    // Fast path.
    if slice.1 == 0 {
        return Ok(materialize_empty_df(
            projection,
            reader_schema,
            hive_partition_columns,
            row_index.as_ref(),
        ));
    }

    let file_metadata = metadata
        .map(Ok)
        .unwrap_or_else(|| read::read_metadata(&mut reader).map(Arc::new))?;
    let n_row_groups = file_metadata.row_groups.len();

    // if there are multiple row groups and categorical data
    // we need a string cache
    // we keep it alive until the end of the function
    let _sc = if n_row_groups > 1 {
        #[cfg(feature = "dtype-categorical")]
        {
            Some(polars_core::StringCacheHolder::hold())
        }
        #[cfg(not(feature = "dtype-categorical"))]
        {
            Some(0u8)
        }
    } else {
        None
    };

    let materialized_projection = projection
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned((0usize..reader_schema.len()).collect::<Vec<_>>()));

    if let Some(predicate) = predicate {
        let prefilter_env = std::env::var("POLARS_PARQUET_PREFILTER");
        let prefilter_env = prefilter_env.as_deref();

        let num_live_variables = predicate.live_columns.len();
        let mut do_prefilter = false;

        do_prefilter |= prefilter_env == Ok("1"); // Force enable
        do_prefilter |= matches!(parallel, ParallelStrategy::Auto)
            && num_live_variables * n_row_groups >= POOL.current_num_threads()
            && materialized_projection.len() >= num_live_variables;

        do_prefilter &= prefilter_env != Ok("0"); // Force disable

        if do_prefilter {
            parallel = ParallelStrategy::Prefiltered;
        }
    }
    if ParallelStrategy::Auto == parallel {
        if n_row_groups > materialized_projection.len() || n_row_groups > POOL.current_num_threads()
        {
            parallel = ParallelStrategy::RowGroups;
        } else {
            parallel = ParallelStrategy::Columns;
        }
    }

    if let (ParallelStrategy::Columns, true) = (parallel, materialized_projection.len() == 1) {
        parallel = ParallelStrategy::None;
    }

    let reader = ReaderBytes::from(&mut reader);
    let store = mmap::ColumnStore::Local(unsafe {
        std::mem::transmute::<ReaderBytes<'_>, ReaderBytes<'static>>(reader).to_memslice()
    });

    let dfs = rg_to_dfs(
        &store,
        &mut 0,
        0,
        n_row_groups,
        slice,
        &file_metadata,
        reader_schema,
        predicate,
        row_index.clone(),
        parallel,
        &materialized_projection,
        use_statistics,
        hive_partition_columns,
    )?;

    if dfs.is_empty() {
        Ok(materialize_empty_df(
            projection,
            reader_schema,
            hive_partition_columns,
            row_index.as_ref(),
        ))
    } else {
        accumulate_dataframes_vertical(dfs)
    }
}

pub struct FetchRowGroupsFromMmapReader(ReaderBytes<'static>);

impl FetchRowGroupsFromMmapReader {
    pub fn new(mut reader: Box<dyn MmapBytesReader>) -> PolarsResult<Self> {
        // SAFETY: we will keep ownership on the struct and reference the bytes on the heap.
        // this should not work with passed bytes so we check if it is a file
        assert!(reader.to_file().is_some());
        let reader_ptr = unsafe {
            std::mem::transmute::<&mut dyn MmapBytesReader, &'static mut dyn MmapBytesReader>(
                reader.as_mut(),
            )
        };
        let reader_bytes = get_reader_bytes(reader_ptr)?;
        Ok(FetchRowGroupsFromMmapReader(reader_bytes))
    }

    fn fetch_row_groups(&mut self, _row_groups: Range<usize>) -> PolarsResult<ColumnStore> {
        // @TODO: we can something smarter here with mmap
        Ok(mmap::ColumnStore::Local(self.0.to_memslice()))
    }
}

// We couldn't use a trait as async trait gave very hard HRT lifetime errors.
// Maybe a puzzle for another day.
pub enum RowGroupFetcher {
    #[cfg(feature = "cloud")]
    ObjectStore(FetchRowGroupsFromObjectStore),
    Local(FetchRowGroupsFromMmapReader),
}

#[cfg(feature = "cloud")]
impl From<FetchRowGroupsFromObjectStore> for RowGroupFetcher {
    fn from(value: FetchRowGroupsFromObjectStore) -> Self {
        RowGroupFetcher::ObjectStore(value)
    }
}

impl From<FetchRowGroupsFromMmapReader> for RowGroupFetcher {
    fn from(value: FetchRowGroupsFromMmapReader) -> Self {
        RowGroupFetcher::Local(value)
    }
}

impl RowGroupFetcher {
    async fn fetch_row_groups(&mut self, _row_groups: Range<usize>) -> PolarsResult<ColumnStore> {
        match self {
            RowGroupFetcher::Local(f) => f.fetch_row_groups(_row_groups),
            #[cfg(feature = "cloud")]
            RowGroupFetcher::ObjectStore(f) => f.fetch_row_groups(_row_groups).await,
        }
    }
}

pub(super) fn compute_row_group_range(
    row_group_start: usize,
    row_group_end: usize,
    slice: (usize, usize),
    row_groups: &[RowGroupMetadata],
) -> std::ops::Range<usize> {
    let mut start = row_group_start;
    let mut cum_rows: usize = (0..row_group_start).map(|i| row_groups[i].num_rows()).sum();
    let row_group_end = row_groups.len().min(row_group_end);

    loop {
        if start == row_group_end {
            break;
        }

        cum_rows += row_groups[start].num_rows();

        if cum_rows >= slice.0 {
            break;
        }

        start += 1;
    }

    let slice_end = slice.0 + slice.1;
    let mut end = (1 + start).min(row_group_end);

    loop {
        if end == row_group_end {
            break;
        }

        if cum_rows >= slice_end {
            break;
        }

        cum_rows += row_groups[end].num_rows();
        end += 1;
    }

    start..end
}

pub struct BatchedParquetReader {
    // use to keep ownership
    #[allow(dead_code)]
    row_group_fetcher: RowGroupFetcher,
    slice: (usize, usize),
    projection: Arc<[usize]>,
    schema: ArrowSchemaRef,
    metadata: FileMetadataRef,
    predicate: Option<ScanIOPredicate>,
    row_index: Option<RowIndex>,
    rows_read: IdxSize,
    row_group_offset: usize,
    n_row_groups: usize,
    chunks_fifo: VecDeque<DataFrame>,
    parallel: ParallelStrategy,
    chunk_size: usize,
    use_statistics: bool,
    hive_partition_columns: Option<Arc<[Series]>>,
    include_file_path: Option<Column>,
    /// Has returned at least one materialized frame.
    has_returned: bool,
}

impl BatchedParquetReader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        row_group_fetcher: RowGroupFetcher,
        metadata: FileMetadataRef,
        schema: ArrowSchemaRef,
        slice: (usize, usize),
        projection: Option<Vec<usize>>,
        predicate: Option<ScanIOPredicate>,
        row_index: Option<RowIndex>,
        chunk_size: usize,
        use_statistics: bool,
        hive_partition_columns: Option<Vec<Series>>,
        include_file_path: Option<(PlSmallStr, Arc<str>)>,
        mut parallel: ParallelStrategy,
    ) -> PolarsResult<Self> {
        let n_row_groups = metadata.row_groups.len();
        let projection = projection
            .map(Arc::from)
            .unwrap_or_else(|| (0usize..schema.len()).collect::<Arc<[_]>>());

        parallel = match parallel {
            ParallelStrategy::Auto => {
                if n_row_groups > projection.len() || n_row_groups > POOL.current_num_threads() {
                    ParallelStrategy::RowGroups
                } else {
                    ParallelStrategy::Columns
                }
            },
            _ => parallel,
        };

        if let (ParallelStrategy::Columns, true) = (parallel, projection.len() == 1) {
            parallel = ParallelStrategy::None;
        }

        Ok(BatchedParquetReader {
            row_group_fetcher,
            slice,
            projection,
            schema,
            metadata,
            row_index,
            rows_read: 0,
            predicate,
            row_group_offset: 0,
            n_row_groups,
            chunks_fifo: VecDeque::with_capacity(POOL.current_num_threads()),
            parallel,
            chunk_size,
            use_statistics,
            hive_partition_columns: hive_partition_columns.map(Arc::from),
            include_file_path: include_file_path.map(|(col, path)| {
                Column::new_scalar(
                    col,
                    Scalar::new(
                        DataType::String,
                        AnyValue::StringOwned(path.as_ref().into()),
                    ),
                    1,
                )
            }),
            has_returned: false,
        })
    }

    pub fn schema(&self) -> &ArrowSchemaRef {
        &self.schema
    }

    pub fn is_finished(&self) -> bool {
        self.row_group_offset >= self.n_row_groups
    }

    pub fn finishes_this_batch(&self, n: usize) -> bool {
        self.row_group_offset + n > self.n_row_groups
    }

    #[cfg(feature = "async")]
    pub async fn next_batches(&mut self, n: usize) -> PolarsResult<Option<Vec<DataFrame>>> {
        if self.rows_read as usize == self.slice.0 + self.slice.1 && self.has_returned {
            return if self.chunks_fifo.is_empty() {
                Ok(None)
            } else {
                // the range end point must not be greater than the length of the deque
                let n_drainable = std::cmp::min(n, self.chunks_fifo.len());
                Ok(Some(self.chunks_fifo.drain(..n_drainable).collect()))
            };
        }

        let mut skipped_all_rgs = false;
        // fill up fifo stack
        if (self.rows_read as usize) < self.slice.0 + self.slice.1
            && self.row_group_offset < self.n_row_groups
            && self.chunks_fifo.len() < n
        {
            // Ensure we apply the limit on the metadata, before we download the row-groups.
            let row_group_range = compute_row_group_range(
                self.row_group_offset,
                self.row_group_offset + n,
                self.slice,
                &self.metadata.row_groups,
            );

            let store = self
                .row_group_fetcher
                .fetch_row_groups(row_group_range.clone())
                .await?;

            let prev_rows_read = self.rows_read;

            let mut dfs = {
                // Spawn the decoding and decompression of the bytes on a rayon task.
                // This will ensure we don't block the async thread.

                // Make everything 'static.
                let mut rows_read = self.rows_read;
                let row_index = self.row_index.clone();
                let predicate = self.predicate.clone();
                let schema = self.schema.clone();
                let metadata = self.metadata.clone();
                let parallel = self.parallel;
                let projection = self.projection.clone();
                let use_statistics = self.use_statistics;
                let hive_partition_columns = self.hive_partition_columns.clone();
                let slice = self.slice;

                let func = move || {
                    let dfs = rg_to_dfs(
                        &store,
                        &mut rows_read,
                        row_group_range.start,
                        row_group_range.end,
                        slice,
                        &metadata,
                        &schema,
                        predicate.as_ref(),
                        row_index,
                        parallel,
                        &projection,
                        use_statistics,
                        hive_partition_columns.as_deref(),
                    );

                    dfs.map(|x| (x, rows_read))
                };

                let (dfs, rows_read) = crate::pl_async::get_runtime().spawn_rayon(func).await?;

                self.rows_read = rows_read;
                dfs
            };

            if let Some(column) = self.include_file_path.as_ref() {
                if dfs.first().is_some_and(|x| x.width() > 0) {
                    for df in &mut dfs {
                        unsafe { df.with_column_unchecked(column.new_from_index(0, df.height())) };
                    }
                } else {
                    let (offset, len) = self.slice;
                    let end = offset + len;

                    debug_assert_eq!(dfs.len(), 1);
                    dfs.get_mut(0).unwrap().insert_column(
                        0,
                        column.new_from_index(
                            0,
                            (self.rows_read.min(end.try_into().unwrap_or(IdxSize::MAX))
                                - prev_rows_read)
                                .try_into()
                                .unwrap(),
                        ),
                    )?;
                }
            }

            self.row_group_offset += n;

            // case where there is no data in the file
            // the streaming engine needs at least a single chunk
            if self.rows_read == 0 && dfs.is_empty() {
                let mut df = materialize_empty_df(
                    Some(self.projection.as_ref()),
                    &self.schema,
                    self.hive_partition_columns.as_deref(),
                    self.row_index.as_ref(),
                );

                if let Some(ca) = &self.include_file_path {
                    unsafe {
                        df.with_column_unchecked(ca.clear().into_column());
                    }
                };

                return Ok(Some(vec![df]));
            }

            // TODO! this is slower than it needs to be
            // we also need to parallelize over row groups here.

            skipped_all_rgs |= dfs.is_empty();
            for mut df in dfs {
                // make sure that the chunks are not too large
                let n = df.height() / self.chunk_size;
                if n > 1 {
                    for df in split_df(&mut df, n, false) {
                        self.chunks_fifo.push_back(df)
                    }
                } else {
                    self.chunks_fifo.push_back(df)
                }
            }
        } else {
            skipped_all_rgs = !self.has_returned;
        };

        if self.chunks_fifo.is_empty() {
            if skipped_all_rgs {
                self.has_returned = true;
                let mut df = materialize_empty_df(
                    Some(self.projection.as_ref()),
                    &self.schema,
                    self.hive_partition_columns.as_deref(),
                    self.row_index.as_ref(),
                );

                if let Some(ca) = &self.include_file_path {
                    unsafe {
                        df.with_column_unchecked(ca.clear().into_column());
                    }
                };

                Ok(Some(vec![df]))
            } else {
                Ok(None)
            }
        } else {
            let mut chunks = Vec::with_capacity(n);
            let mut i = 0;
            while let Some(df) = self.chunks_fifo.pop_front() {
                chunks.push(df);
                i += 1;
                if i == n {
                    break;
                }
            }

            self.has_returned = true;
            Ok(Some(chunks))
        }
    }

    /// Turn the batched reader into an iterator.
    #[cfg(feature = "async")]
    pub fn iter(self, batches_per_iter: usize) -> BatchedParquetIter {
        BatchedParquetIter {
            batches_per_iter,
            inner: self,
            current_batch: vec![].into_iter(),
        }
    }
}

#[cfg(feature = "async")]
pub struct BatchedParquetIter {
    batches_per_iter: usize,
    inner: BatchedParquetReader,
    current_batch: std::vec::IntoIter<DataFrame>,
}

#[cfg(feature = "async")]
impl BatchedParquetIter {
    // todo! implement stream
    pub(crate) async fn next_(&mut self) -> Option<PolarsResult<DataFrame>> {
        match self.current_batch.next() {
            Some(df) => Some(Ok(df)),
            None => match self.inner.next_batches(self.batches_per_iter).await {
                Err(e) => Some(Err(e)),
                Ok(opt_batch) => {
                    let batch = opt_batch?;
                    self.current_batch = batch.into_iter();
                    self.current_batch.next().map(Ok)
                },
            },
        }
    }
}

pub fn calc_prefilter_cost(mask: &arrow::bitmap::Bitmap) -> f64 {
    let num_edges = mask.num_edges() as f64;
    let rg_len = mask.len() as f64;

    // @GB: I did quite some analysis on this.
    //
    // Pre-filtered and Post-filtered can both be faster in certain scenarios.
    //
    // - Pre-filtered is faster when there is some amount of clustering or
    // sorting involved or if the number of values selected is small.
    // - Post-filtering is faster when the predicate selects a somewhat random
    // elements throughout the row group.
    //
    // The following is a heuristic value to try and estimate which one is
    // faster. Essentially, it sees how many times it needs to switch between
    // skipping items and collecting items and compares it against the number
    // of values that it will collect.
    //
    // Closer to 0: pre-filtering is probably better.
    // Closer to 1: post-filtering is probably better.
    (num_edges / rg_len).clamp(0.0, 1.0)
}

pub enum PrefilterMaskSetting {
    Auto,
    Pre,
    Post,
}

impl PrefilterMaskSetting {
    pub fn init_from_env() -> Self {
        std::env::var("POLARS_PQ_PREFILTERED_MASK").map_or(Self::Auto, |v| match &v[..] {
            "auto" => Self::Auto,
            "pre" => Self::Pre,
            "post" => Self::Post,
            _ => panic!("Invalid `POLARS_PQ_PREFILTERED_MASK` value '{v}'."),
        })
    }

    pub fn should_prefilter(&self, prefilter_cost: f64, dtype: &ArrowDataType) -> bool {
        match self {
            Self::Auto => {
                // Prefiltering is only expensive for nested types so we make the cut-off quite
                // high.
                let is_nested = dtype.is_nested();

                // We empirically selected these numbers.
                is_nested && prefilter_cost <= 0.01
            },
            Self::Pre => true,
            Self::Post => false,
        }
    }
}
