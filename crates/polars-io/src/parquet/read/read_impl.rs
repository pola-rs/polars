use std::borrow::Cow;
use std::collections::VecDeque;
use std::ops::{Deref, Range};

use arrow::array::BooleanArray;
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::ArrowSchemaRef;
use polars_core::prelude::*;
use polars_core::utils::{accumulate_dataframes_vertical, split_df};
use polars_core::POOL;
use polars_parquet::parquet::error::ParquetResult;
use polars_parquet::parquet::statistics::Statistics;
use polars_parquet::read::{
    self, ColumnChunkMetaData, FileMetaData, Filter, PhysicalType, RowGroupMetaData,
};
use polars_utils::mmap::MemSlice;
use polars_utils::vec::inplace_zip_filtermap;
use rayon::prelude::*;

#[cfg(feature = "cloud")]
use super::async_impl::FetchRowGroupsFromObjectStore;
use super::mmap::{mmap_columns, ColumnStore};
use super::predicates::read_this_row_group;
use super::to_metadata::ToMetadata;
use super::utils::materialize_empty_df;
use super::{mmap, ParallelStrategy};
use crate::hive::materialize_hive_partitions;
use crate::mmap::{MmapBytesReader, ReaderBytes};
use crate::parquet::metadata::FileMetaDataRef;
use crate::parquet::read::metadata::PartitionedColumnChunkMD;
use crate::parquet::read::ROW_COUNT_OVERFLOW_ERR;
use crate::predicates::{apply_predicate, PhysicalIoExpr};
use crate::utils::get_reader_bytes;
use crate::utils::slice::split_slice_at_file;
use crate::RowIndex;

#[cfg(debug_assertions)]
// Ensure we get the proper polars types from schema inference
// This saves unneeded casts.
fn assert_dtypes(data_type: &ArrowDataType) {
    use ArrowDataType as D;

    match data_type {
        // These should all be casted to the BinaryView / Utf8View variants
        D::Utf8 | D::Binary | D::LargeUtf8 | D::LargeBinary => unreachable!(),

        // This should have been converted to a LargeList
        D::List(_) => unreachable!(),

        // This should have been converted to a LargeList(Struct(_))
        D::Map(_, _) => unreachable!(),

        // Recursive checks
        D::Dictionary(_, data_type, _) => assert_dtypes(data_type),
        D::Extension(_, data_type, _) => assert_dtypes(data_type),
        D::LargeList(inner) => assert_dtypes(&inner.data_type),
        D::FixedSizeList(inner, _) => assert_dtypes(&inner.data_type),
        D::Struct(fields) => fields.iter().for_each(|f| assert_dtypes(f.data_type())),

        _ => {},
    }
}

fn column_idx_to_series(
    column_i: usize,
    // The metadata belonging to this column
    field_md: &[&ColumnChunkMetaData],
    filter: Option<Filter>,
    file_schema: &ArrowSchema,
    store: &mmap::ColumnStore,
) -> PolarsResult<Series> {
    let field = &file_schema.fields[column_i];

    #[cfg(debug_assertions)]
    {
        assert_dtypes(field.data_type())
    }
    let columns = mmap_columns(store, field_md);
    let stats = columns
        .iter()
        .map(|(col_md, _)| col_md.statistics().transpose())
        .collect::<ParquetResult<Vec<Option<Statistics>>>>();
    let array = mmap::to_deserializer(columns, field.clone(), filter)?;
    let mut series = Series::try_from((field, array))?;

    // We cannot really handle nested metadata at the moment. Just skip it.
    use ArrowDataType as AD;
    match field.data_type() {
        AD::List(_) | AD::LargeList(_) | AD::Struct(_) | AD::FixedSizeList(_, _) => {
            return Ok(series)
        },
        _ => {},
    }

    // See if we can find some statistics for this series. If we cannot find anything just return
    // the series as is.
    let Ok(Some(stats)) = stats.map(|mut s| s.pop().flatten()) else {
        return Ok(series);
    };

    let series_trait = series.as_ref();

    macro_rules! match_dtypes_into_metadata {
        ($(($dtype:pat, $phystype:pat) => ($stats:ident, $pldtype:ty),)+) => {
            match (series_trait.dtype(), stats.physical_type()) {
                $(
                ($dtype, $phystype) => {
                    series.try_set_metadata(
                        ToMetadata::<$pldtype>::to_metadata(stats.$stats())
                    );
                })+
                _ => {},
            }
        };
    }

    // Match the data types used by the Series and by the Statistics. If we find a match, set some
    // Metadata for the underlying ChunkedArray.
    use {DataType as D, PhysicalType as P};
    match_dtypes_into_metadata! {
        (D::Boolean, P::Boolean  ) => (expect_as_boolean, BooleanType),
        (D::UInt8,   P::Int32    ) => (expect_as_int32,   UInt8Type  ),
        (D::UInt16,  P::Int32    ) => (expect_as_int32,   UInt16Type ),
        (D::UInt32,  P::Int32    ) => (expect_as_int32,   UInt32Type ),
        (D::UInt64,  P::Int64    ) => (expect_as_int64,   UInt64Type ),
        (D::Int8,    P::Int32    ) => (expect_as_int32,   Int8Type   ),
        (D::Int16,   P::Int32    ) => (expect_as_int32,   Int16Type  ),
        (D::Int32,   P::Int32    ) => (expect_as_int32,   Int32Type  ),
        (D::Int64,   P::Int64    ) => (expect_as_int64,   Int64Type  ),
        (D::Float32, P::Float    ) => (expect_as_float,   Float32Type),
        (D::Float64, P::Double   ) => (expect_as_double,  Float64Type),
        (D::String,  P::ByteArray) => (expect_as_binary,  StringType ),
        (D::Binary,  P::ByteArray) => (expect_as_binary,  BinaryType ),
    }

    Ok(series)
}

#[allow(clippy::too_many_arguments)]
fn rg_to_dfs(
    store: &mmap::ColumnStore,
    previous_row_count: &mut IdxSize,
    row_group_start: usize,
    row_group_end: usize,
    slice: (usize, usize),
    file_metadata: &FileMetaData,
    schema: &ArrowSchemaRef,
    predicate: Option<&dyn PhysicalIoExpr>,
    row_index: Option<RowIndex>,
    parallel: ParallelStrategy,
    projection: &[usize],
    use_statistics: bool,
    hive_partition_columns: Option<&[Series]>,
) -> PolarsResult<Vec<DataFrame>> {
    use ParallelStrategy as S;

    if parallel == S::Prefiltered {
        if let Some(predicate) = predicate {
            if let Some(live_variables) = predicate.live_variables() {
                return rg_to_dfs_prefiltered(
                    store,
                    previous_row_count,
                    row_group_start,
                    row_group_end,
                    file_metadata,
                    schema,
                    live_variables,
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

/// Collect a HashSet of the projected columns.
///  Returns `None` if all columns are projected.
fn projected_columns_set<'a>(
    schema: &'a ArrowSchema,
    projection: &[usize],
) -> Option<PlHashSet<&'a str>> {
    if projection.len() == schema.len() {
        None
    } else {
        Some(
            projection
                .iter()
                .map(|i| schema.fields[*i].name.as_str())
                .collect::<PlHashSet<_>>(),
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn rg_to_dfs_prefiltered(
    store: &mmap::ColumnStore,
    previous_row_count: &mut IdxSize,
    row_group_start: usize,
    row_group_end: usize,
    file_metadata: &FileMetaData,
    schema: &ArrowSchemaRef,
    live_variables: Vec<Arc<str>>,
    predicate: &dyn PhysicalIoExpr,
    row_index: Option<RowIndex>,
    projection: &[usize],
    use_statistics: bool,
    hive_partition_columns: Option<&[Series]>,
) -> PolarsResult<Vec<DataFrame>> {
    struct RowGroupInfo {
        index: u32,
        row_offset: IdxSize,
    }

    if row_group_end > u32::MAX as usize {
        polars_bail!(ComputeError: "Parquet file contains too many row groups (> {})", u32::MAX);
    }

    let projected_columns = projected_columns_set(schema, projection);

    let part_mds = POOL.install(|| {
        file_metadata
            .row_groups
            .par_iter()
            .map(|rg| {
                let mut part_md = PartitionedColumnChunkMD::new(rg);
                part_md.set_partitions(projected_columns.as_ref());
                part_md
            })
            .collect::<Vec<_>>()
    });

    let mut row_offset = *previous_row_count;
    let mut row_groups: Vec<RowGroupInfo> = (row_group_start..row_group_end)
        .filter_map(|index| {
            let part_md = &part_mds[index];
            let md = &file_metadata.row_groups[index];

            let current_offset = row_offset;
            let current_row_count = md.num_rows() as IdxSize;
            row_offset += current_row_count;

            if use_statistics {
                match read_this_row_group(Some(predicate), part_md, schema) {
                    Ok(false) => return None,
                    Ok(true) => {},
                    Err(e) => return Some(Err(e)),
                }
            }

            Some(Ok(RowGroupInfo {
                index: index as u32,
                row_offset: current_offset,
            }))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    // Deduplicate the live variables
    let live_variables = live_variables
        .iter()
        .map(Deref::deref)
        .collect::<PlHashSet<_>>();

    // Get the number of live columns
    let num_live_columns = live_variables.len();
    let num_dead_columns = projection.len() - num_live_columns;

    // We create two look-up tables that map indexes offsets into the live- and dead-set onto
    // column indexes of the schema.
    let mut live_idx_to_col_idx = Vec::with_capacity(num_live_columns);
    let mut dead_idx_to_col_idx = Vec::with_capacity(num_dead_columns);
    for (i, field) in schema.fields.iter().enumerate() {
        if live_variables.contains(&field.name[..]) {
            live_idx_to_col_idx.push(i);
        } else {
            dead_idx_to_col_idx.push(i);
        }
    }

    debug_assert_eq!(live_idx_to_col_idx.len(), num_live_columns);
    debug_assert_eq!(dead_idx_to_col_idx.len(), num_dead_columns);

    enum MaskSetting {
        Auto,
        Pre,
        Post,
    }

    let mask_setting =
        std::env::var("POLARS_PQ_PREFILTERED_MASK").map_or(MaskSetting::Auto, |v| match &v[..] {
            "auto" => MaskSetting::Auto,
            "pre" => MaskSetting::Pre,
            "post" => MaskSetting::Post,
            _ => panic!("Invalid `POLARS_PQ_PREFILTERED_MASK` value '{v}'."),
        });

    POOL.install(|| {
        // Set partitioned fields to prevent quadratic behavior.
        // Ensure all row groups are partitioned.

        // Collect the data for the live columns
        let mut live_columns = (0..row_groups.len() * num_live_columns)
            .into_par_iter()
            .map(|i| {
                let col_idx = live_idx_to_col_idx[i % num_live_columns];

                let name = &schema.fields[col_idx].name;
                let rg_idx = row_groups[i / num_live_columns].index;
                let field_md = part_mds[rg_idx as usize].get_partitions(name).unwrap();

                column_idx_to_series(col_idx, field_md.as_slice(), None, schema, store)
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        // Apply the predicate to the live columns and save the dataframe and the bitmask
        let mut dfs = live_columns
            .par_chunks_exact_mut(num_live_columns)
            .enumerate()
            .map(|(i, columns)| {
                let rg = &row_groups[i];
                let rg_idx = rg.index as usize;

                let columns = columns.iter_mut().map(std::mem::take).collect::<Vec<_>>();

                let md = &file_metadata.row_groups[rg_idx];
                let mut df = unsafe { DataFrame::new_no_checks(columns) };

                materialize_hive_partitions(
                    &mut df,
                    schema.as_ref(),
                    hive_partition_columns,
                    md.num_rows(),
                );
                let s = predicate.evaluate_io(&df)?;
                let mask = s.bool().expect("filter predicates was not of type boolean");

                if let Some(rc) = &row_index {
                    df.with_row_index_mut(&rc.name, Some(rg.row_offset + rc.offset));
                }
                df = df.filter(mask)?;

                let mut bitmap = MutableBitmap::with_capacity(mask.len());

                // We need to account for the validity of the items
                for chunk in mask.downcast_iter() {
                    match chunk.validity() {
                        None => bitmap.extend_from_bitmap(chunk.values()),
                        Some(validity) => bitmap.extend_from_bitmap(&(validity & chunk.values())),
                    }
                }

                let bitmap = bitmap.freeze();

                debug_assert_eq!(md.num_rows(), bitmap.len());
                debug_assert_eq!(df.height(), bitmap.set_bits());

                Ok((bitmap, df))
            })
            .collect::<PolarsResult<Vec<(Bitmap, DataFrame)>>>()?;

        // Filter out the row-groups that do not include any rows that match the predicate.
        inplace_zip_filtermap(&mut dfs, &mut row_groups, |(mask, df), rg| {
            (mask.set_bits() > 0).then_some(((mask, df), rg))
        });

        for (_, df) in &dfs {
            let height = IdxSize::try_from(df.height()).map_err(|_| ROW_COUNT_OVERFLOW_ERR)?;
            *previous_row_count = previous_row_count
                .checked_add(height)
                .ok_or(ROW_COUNT_OVERFLOW_ERR)?;
        }

        // We don't need to do any further work if there are no dead columns
        if num_dead_columns == 0 {
            return Ok(dfs.into_iter().map(|(_, df)| df).collect());
        }

        let rg_prefilter_costs = matches!(mask_setting, MaskSetting::Auto)
            .then(|| {
                dfs.par_iter()
                    .map(|(mask, _)| {
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
                        // Closer to 0: post-filtering is probably better.
                        // Closer to 1: pre-filtering is probably better.
                        (num_edges / rg_len).clamp(0.0, 1.0)
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let mut rg_columns = (0..dfs.len() * num_dead_columns)
            .into_par_iter()
            .map(|i| {
                let col_idx = dead_idx_to_col_idx[i % num_dead_columns];
                let name = &schema.fields[col_idx].name;

                let (mask, _) = &dfs[i / num_dead_columns];

                let rg_idx = row_groups[i / num_dead_columns].index;

                #[cfg(debug_assertions)]
                {
                    let md = &file_metadata.row_groups[rg_idx as usize];
                    debug_assert_eq!(md.num_rows(), mask.len());
                }
                let field_md = part_mds[rg_idx as usize].get_partitions(name).unwrap();

                let pre = || {
                    column_idx_to_series(
                        col_idx,
                        field_md.as_slice(),
                        Some(Filter::new_masked(mask.clone())),
                        schema,
                        store,
                    )
                };
                let post = || {
                    let array =
                        column_idx_to_series(col_idx, field_md.as_slice(), None, schema, store)?;

                    debug_assert_eq!(array.len(), mask.len());

                    let mask_arr = BooleanArray::new(ArrowDataType::Boolean, mask.clone(), None);
                    let mask_arr = BooleanChunked::from(mask_arr);
                    array.filter(&mask_arr)
                };

                let array = match mask_setting {
                    MaskSetting::Auto => {
                        // Prefiltering is more expensive for nested types so we make the cut-off
                        // higher.
                        let is_nested = schema.fields[col_idx].data_type.is_nested();
                        let prefilter_cost = rg_prefilter_costs[i / num_dead_columns];

                        // We empirically selected these numbers.
                        let do_prefilter = (is_nested && prefilter_cost <= 0.01)
                            || (!is_nested && prefilter_cost <= 0.02);

                        if do_prefilter {
                            pre()?
                        } else {
                            post()?
                        }
                    },
                    MaskSetting::Pre => pre()?,
                    MaskSetting::Post => post()?,
                };

                debug_assert_eq!(array.len(), mask.set_bits());

                Ok(array)
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        let Some(df) = dfs.first().map(|(_, df)| df) else {
            return Ok(Vec::new());
        };
        let mut rearranged_schema = df.schema();
        rearranged_schema.merge(Schema::from(schema));

        rg_columns
            .par_chunks_exact_mut(num_dead_columns)
            .zip(dfs)
            .map(|(rg_cols, (_, mut df))| {
                let rg_cols = rg_cols.iter_mut().map(std::mem::take).collect::<Vec<_>>();

                debug_assert!(rg_cols.iter().all(|v| v.len() == df.height()));

                // We first add the columns with the live columns at the start. Then, we do a
                // projections that puts the columns at the right spot.
                df._add_columns(rg_cols, &rearranged_schema)?;
                let df = df.select(schema.get_names())?;

                PolarsResult::Ok(df)
            })
            .collect::<PolarsResult<Vec<DataFrame>>>()
    })
}

#[allow(clippy::too_many_arguments)]
// might parallelize over columns
fn rg_to_dfs_optionally_par_over_columns(
    store: &mmap::ColumnStore,
    previous_row_count: &mut IdxSize,
    row_group_start: usize,
    row_group_end: usize,
    slice: (usize, usize),
    file_metadata: &FileMetaData,
    schema: &ArrowSchemaRef,
    predicate: Option<&dyn PhysicalIoExpr>,
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

        // Set partitioned fields to prevent quadratic behavior.
        let projected_columns = projected_columns_set(schema, projection);
        let mut part_md = PartitionedColumnChunkMD::new(md);
        part_md.set_partitions(projected_columns.as_ref());

        let rg_slice =
            split_slice_at_file(&mut n_rows_processed, md.num_rows(), slice.0, slice_end);
        let current_row_count = md.num_rows() as IdxSize;

        if use_statistics && !read_this_row_group(predicate, &part_md, schema)? {
            *previous_row_count += rg_slice.1 as IdxSize;
            continue;
        }
        // test we don't read the parquet file if this env var is set
        #[cfg(debug_assertions)]
        {
            assert!(std::env::var("POLARS_PANIC_IF_PARQUET_PARSED").is_err())
        }

        let columns = if let ParallelStrategy::Columns = parallel {
            POOL.install(|| {
                projection
                    .par_iter()
                    .map(|column_i| {
                        let name = &schema.fields[*column_i].name;
                        let part = part_md.get_partitions(name).unwrap();

                        column_idx_to_series(
                            *column_i,
                            part.as_slice(),
                            Some(Filter::new_ranged(rg_slice.0, rg_slice.0 + rg_slice.1)),
                            schema,
                            store,
                        )
                    })
                    .collect::<PolarsResult<Vec<_>>>()
            })?
        } else {
            projection
                .iter()
                .map(|column_i| {
                    let name = &schema.fields[*column_i].name;
                    let part = part_md.get_partitions(name).unwrap();

                    column_idx_to_series(
                        *column_i,
                        part.as_slice(),
                        Some(Filter::new_ranged(rg_slice.0, rg_slice.0 + rg_slice.1)),
                        schema,
                        store,
                    )
                })
                .collect::<PolarsResult<Vec<_>>>()?
        };

        let mut df = unsafe { DataFrame::new_no_checks(columns) };
        if let Some(rc) = &row_index {
            df.with_row_index_mut(&rc.name, Some(*previous_row_count + rc.offset));
        }

        materialize_hive_partitions(&mut df, schema.as_ref(), hive_partition_columns, rg_slice.1);
        apply_predicate(&mut df, predicate, true)?;

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
    previous_row_count: &mut IdxSize,
    slice: (usize, usize),
    file_metadata: &FileMetaData,
    schema: &ArrowSchemaRef,
    predicate: Option<&dyn PhysicalIoExpr>,
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

    for i in row_group_start..row_group_end {
        let row_count_start = *previous_row_count;
        let rg_md = &file_metadata.row_groups[i];
        let rg_slice =
            split_slice_at_file(&mut n_rows_processed, rg_md.num_rows(), slice.0, slice_end);
        *previous_row_count = previous_row_count
            .checked_add(rg_slice.1 as IdxSize)
            .ok_or(ROW_COUNT_OVERFLOW_ERR)?;

        if rg_slice.1 == 0 {
            continue;
        }

        row_groups.push((i, rg_md, rg_slice, row_count_start));
    }

    let dfs = POOL.install(|| {
        // Set partitioned fields to prevent quadratic behavior.
        // Ensure all row groups are partitioned.
        let part_mds = {
            let projected_columns = projected_columns_set(schema, projection);
            row_groups
                .par_iter()
                .map(|(_, rg, _, _)| {
                    let mut ccmd = PartitionedColumnChunkMD::new(rg);
                    ccmd.set_partitions(projected_columns.as_ref());
                    ccmd
                })
                .collect::<Vec<_>>()
        };

        row_groups
            .into_par_iter()
            .enumerate()
            .map(|(iter_idx, (_rg_idx, _md, slice, row_count_start))| {
                let part_md = &part_mds[iter_idx];

                if slice.1 == 0
                    || use_statistics && !read_this_row_group(predicate, part_md, schema)?
                {
                    return Ok(None);
                }
                // test we don't read the parquet file if this env var is set
                #[cfg(debug_assertions)]
                {
                    assert!(std::env::var("POLARS_PANIC_IF_PARQUET_PARSED").is_err())
                }

                let columns = projection
                    .iter()
                    .map(|column_i| {
                        let name = &schema.fields[*column_i].name;
                        let field_md = part_md.get_partitions(name).unwrap();

                        column_idx_to_series(
                            *column_i,
                            field_md.as_slice(),
                            Some(Filter::new_ranged(slice.0, slice.0 + slice.1)),
                            schema,
                            store,
                        )
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;

                let mut df = unsafe { DataFrame::new_no_checks(columns) };

                if let Some(rc) = &row_index {
                    df.with_row_index_mut(&rc.name, Some(row_count_start as IdxSize + rc.offset));
                }

                materialize_hive_partitions(
                    &mut df,
                    schema.as_ref(),
                    hive_partition_columns,
                    slice.1,
                );
                apply_predicate(&mut df, predicate, false)?;

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
    metadata: Option<FileMetaDataRef>,
    predicate: Option<&dyn PhysicalIoExpr>,
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

    if let ParallelStrategy::Auto = parallel {
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
    let store = mmap::ColumnStore::Local(
        unsafe { std::mem::transmute::<ReaderBytes<'_>, ReaderBytes<'static>>(reader) }
            .into_mem_slice(),
    );

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
        Ok(mmap::ColumnStore::Local(MemSlice::from_vec(
            self.0.deref().to_vec(),
        )))
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
    row_groups: &[RowGroupMetaData],
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
    metadata: FileMetaDataRef,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    row_index: Option<RowIndex>,
    rows_read: IdxSize,
    row_group_offset: usize,
    n_row_groups: usize,
    chunks_fifo: VecDeque<DataFrame>,
    parallel: ParallelStrategy,
    chunk_size: usize,
    use_statistics: bool,
    hive_partition_columns: Option<Arc<[Series]>>,
    include_file_path: Option<StringChunked>,
    /// Has returned at least one materialized frame.
    has_returned: bool,
}

impl BatchedParquetReader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        row_group_fetcher: RowGroupFetcher,
        metadata: FileMetaDataRef,
        schema: ArrowSchemaRef,
        slice: (usize, usize),
        projection: Option<Vec<usize>>,
        predicate: Option<Arc<dyn PhysicalIoExpr>>,
        row_index: Option<RowIndex>,
        chunk_size: usize,
        use_statistics: bool,
        hive_partition_columns: Option<Vec<Series>>,
        include_file_path: Option<(Arc<str>, Arc<str>)>,
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
            include_file_path: include_file_path
                .map(|(col, path)| StringChunked::full(&col, &path, 1)),
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

            let mut dfs = match store {
                ColumnStore::Local(_) => rg_to_dfs(
                    &store,
                    &mut self.rows_read,
                    row_group_range.start,
                    row_group_range.end,
                    self.slice,
                    &self.metadata,
                    &self.schema,
                    self.predicate.as_deref(),
                    self.row_index.clone(),
                    self.parallel,
                    &self.projection,
                    self.use_statistics,
                    self.hive_partition_columns.as_deref(),
                ),
                #[cfg(feature = "async")]
                ColumnStore::Fetched(b) => {
                    // This branch we spawn the decoding and decompression of the bytes on a rayon task.
                    // This will ensure we don't block the async thread.

                    // Reconstruct as that makes it a 'static.
                    let store = ColumnStore::Fetched(b);
                    let (tx, rx) = tokio::sync::oneshot::channel();

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

                    let f = move || {
                        let dfs = rg_to_dfs(
                            &store,
                            &mut rows_read,
                            row_group_range.start,
                            row_group_range.end,
                            slice,
                            &metadata,
                            &schema,
                            predicate.as_deref(),
                            row_index,
                            parallel,
                            &projection,
                            use_statistics,
                            hive_partition_columns.as_deref(),
                        );

                        // Don't unwrap send attempt - async task could be cancelled.
                        let _ = tx.send((dfs, rows_read));
                    };

                    // Spawn the task and wait on it asynchronously.
                    if POOL.current_thread_index().is_some() {
                        // We are a rayon thread, so we can't use POOL.spawn as it would mean we spawn a task and block until
                        // another rayon thread executes it - we would deadlock if all rayon threads did this.
                        // Safety: The tokio runtime flavor is multi-threaded.
                        tokio::task::block_in_place(f);
                    } else {
                        POOL.spawn(f);
                    };

                    let (dfs, rows_read) = rx.await.unwrap();
                    self.rows_read = rows_read;
                    dfs
                },
            }?;

            if let Some(ca) = self.include_file_path.as_mut() {
                let mut max_len = 0;

                if self.projection.is_empty() {
                    max_len = self.metadata.num_rows;
                } else {
                    for df in &dfs {
                        max_len = std::cmp::max(max_len, df.height());
                    }
                }

                // Re-use the same ChunkedArray
                if ca.len() < max_len {
                    *ca = ca.new_from_index(0, max_len);
                }

                for df in &mut dfs {
                    unsafe {
                        df.with_column_unchecked(
                            ca.slice(
                                0,
                                if !self.projection.is_empty() {
                                    df.height()
                                } else {
                                    self.metadata.num_rows
                                },
                            )
                            .into_series(),
                        )
                    };
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
                        df.with_column_unchecked(ca.clear().into_series());
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
                        df.with_column_unchecked(ca.clear().into_series());
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
