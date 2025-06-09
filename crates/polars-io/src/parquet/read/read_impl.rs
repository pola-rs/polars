use std::borrow::Cow;

use arrow::bitmap::Bitmap;
use arrow::datatypes::ArrowSchemaRef;
use polars_core::chunked_array::builder::NullChunkedBuilder;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::utils::accumulate_dataframes_vertical;
use polars_core::{POOL, config};
use polars_parquet::read::{self, ColumnChunkMetadata, FileMetadata, Filter, RowGroupMetadata};
use rayon::prelude::*;

use super::mmap::mmap_columns;
use super::utils::materialize_empty_df;
use super::{ParallelStrategy, mmap};
use crate::RowIndex;
use crate::hive::materialize_hive_partitions;
use crate::mmap::{MmapBytesReader, ReaderBytes};
use crate::parquet::metadata::FileMetadataRef;
use crate::parquet::read::ROW_COUNT_OVERFLOW_ERR;
use crate::utils::slice::split_slice_at_file;

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
    pre_slice: (usize, usize),
    file_metadata: &FileMetadata,
    schema: &ArrowSchemaRef,
    row_index: Option<RowIndex>,
    parallel: ParallelStrategy,
    projection: &[usize],
    hive_partition_columns: Option<&[Series]>,
) -> PolarsResult<Vec<DataFrame>> {
    if config::verbose() {
        eprintln!("parquet scan with parallel = {parallel:?}");
    }

    // If we are only interested in the row_index, we take a little special path here.
    if projection.is_empty() {
        if let Some(row_index) = row_index {
            let placeholder =
                NullChunkedBuilder::new(PlSmallStr::from_static("__PL_TMP"), pre_slice.1).finish();
            return Ok(vec![
                DataFrame::new(vec![placeholder.into_series().into_column()])?
                    .with_row_index(
                        row_index.name.clone(),
                        Some(row_index.offset + IdxSize::try_from(pre_slice.0).unwrap()),
                    )?
                    .select(std::iter::once(row_index.name))?,
            ]);
        }
    }

    use ParallelStrategy as S;

    match parallel {
        S::Columns | S::None => rg_to_dfs_optionally_par_over_columns(
            store,
            previous_row_count,
            row_group_start,
            row_group_end,
            pre_slice,
            file_metadata,
            schema,
            row_index,
            parallel,
            projection,
            hive_partition_columns,
        ),
        _ => rg_to_dfs_par_over_rg(
            store,
            row_group_start,
            row_group_end,
            previous_row_count,
            pre_slice,
            file_metadata,
            schema,
            row_index,
            projection,
            hive_partition_columns,
        ),
    }
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
    row_index: Option<RowIndex>,
    parallel: ParallelStrategy,
    projection: &[usize],
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
            unsafe {
                df.with_row_index_mut(
                    rc.name.clone(),
                    Some(*previous_row_count + rc.offset + rg_slice.0 as IdxSize),
                )
            };
        }

        materialize_hive_partitions(&mut df, schema.as_ref(), hive_partition_columns);

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
    row_index: Option<RowIndex>,
    projection: &[usize],
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
                if slice.1 == 0 {
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
                    unsafe {
                        df.with_row_index_mut(
                            rc.name.clone(),
                            Some(row_count_start as IdxSize + rc.offset + slice.0 as IdxSize),
                        )
                    };
                }

                materialize_hive_partitions(&mut df, schema.as_ref(), hive_partition_columns);

                Ok(Some(df))
            })
            .collect::<PolarsResult<Vec<_>>>()
    })?;
    Ok(dfs.into_iter().flatten().collect())
}

#[allow(clippy::too_many_arguments)]
pub fn read_parquet<R: MmapBytesReader>(
    mut reader: R,
    pre_slice: (usize, usize),
    projection: Option<&[usize]>,
    reader_schema: &ArrowSchemaRef,
    metadata: Option<FileMetadataRef>,
    mut parallel: ParallelStrategy,
    row_index: Option<RowIndex>,
    hive_partition_columns: Option<&[Series]>,
) -> PolarsResult<DataFrame> {
    // Fast path.
    if pre_slice.1 == 0 {
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
        pre_slice,
        &file_metadata,
        reader_schema,
        row_index.clone(),
        parallel,
        &materialized_projection,
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

#[derive(Clone, Copy)]
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
                !is_nested && prefilter_cost <= 0.01
            },
            Self::Pre => true,
            Self::Post => false,
        }
    }
}
