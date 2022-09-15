use std::borrow::Cow;
use std::convert::TryFrom;
use std::ops::Deref;
use std::sync::Arc;

use arrow::array::new_empty_array;
use arrow::io::parquet::read;
use arrow::io::parquet::read::{ArrayIter, FileMetaData, RowGroupMetaData};
use polars_core::prelude::*;
use polars_core::utils::accumulate_dataframes_vertical;
use polars_core::POOL;
use rayon::prelude::*;

use crate::aggregations::{apply_aggregations, ScanAggregation};
use crate::mmap::{MmapBytesReader, ReaderBytes};
use crate::parquet::mmap::mmap_columns;
use crate::parquet::predicates::read_this_row_group;
use crate::parquet::{mmap, ParallelStrategy};
use crate::predicates::{apply_predicate, arrow_schema_to_empty_df, PhysicalIoExpr};
use crate::utils::apply_projection;
use crate::RowCount;

fn column_idx_to_series(
    column_i: usize,
    md: &RowGroupMetaData,
    remaining_rows: usize,
    schema: &ArrowSchema,
    bytes: &[u8],
    chunk_size: usize,
) -> PolarsResult<Series> {
    let field = &schema.fields[column_i];
    let columns = mmap_columns(bytes, md.columns(), &field.name);
    let iter = mmap::to_deserializer(columns, field.clone(), remaining_rows, Some(chunk_size))?;

    if remaining_rows < md.num_rows() {
        array_iter_to_series(iter, field, Some(remaining_rows))
    } else {
        array_iter_to_series(iter, field, None)
    }
}

fn array_iter_to_series(
    iter: ArrayIter,
    field: &ArrowField,
    num_rows: Option<usize>,
) -> PolarsResult<Series> {
    let mut total_count = 0;
    let chunks = match num_rows {
        None => iter.collect::<arrow::error::Result<Vec<_>>>()?,
        Some(n) => {
            let mut out = Vec::with_capacity(2);

            for arr in iter {
                let arr = arr?;
                let len = arr.len();
                out.push(arr);

                total_count += len;
                if total_count >= n {
                    break;
                }
            }
            out
        }
    };
    if chunks.is_empty() {
        let arr = new_empty_array(field.data_type.clone());
        Series::try_from((field.name.as_str(), arr))
    } else {
        Series::try_from((field.name.as_str(), chunks))
    }
}

#[allow(clippy::too_many_arguments)]
// might parallelize over columns
fn rg_to_dfs(
    bytes: &[u8],
    n_row_groups: usize,
    limit: usize,
    file_metadata: &FileMetaData,
    schema: &ArrowSchema,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    aggregate: Option<&[ScanAggregation]>,
    row_count: Option<RowCount>,
    parallel: ParallelStrategy,
    projection: &[usize],
) -> PolarsResult<Vec<DataFrame>> {
    let mut dfs = Vec::with_capacity(n_row_groups);

    let mut remaining_rows = limit;

    let mut previous_row_count = 0;
    for rg in 0..n_row_groups {
        let md = &file_metadata.row_groups[rg];
        let current_row_count = md.num_rows() as IdxSize;

        if !read_this_row_group(predicate.as_ref(), file_metadata, schema, rg)? {
            previous_row_count += current_row_count;
            continue;
        }
        // test we don't read the parquet file if this env var is set
        #[cfg(debug_assertions)]
        {
            assert!(std::env::var("POLARS_PANIC_IF_PARQUET_PARSED").is_err())
        }

        let chunk_size = md.num_rows() as usize;
        let columns = if let ParallelStrategy::Columns = parallel {
            POOL.install(|| {
                projection
                    .par_iter()
                    .map(|column_i| {
                        column_idx_to_series(
                            *column_i,
                            md,
                            remaining_rows,
                            schema,
                            bytes,
                            chunk_size,
                        )
                    })
                    .collect::<PolarsResult<Vec<_>>>()
            })?
        } else {
            projection
                .iter()
                .map(|column_i| {
                    column_idx_to_series(*column_i, md, remaining_rows, schema, bytes, chunk_size)
                })
                .collect::<PolarsResult<Vec<_>>>()?
        };

        remaining_rows =
            remaining_rows.saturating_sub(file_metadata.row_groups[rg].num_rows() as usize);

        let mut df = DataFrame::new_no_checks(columns);
        if let Some(rc) = &row_count {
            df.with_row_count_mut(&rc.name, Some(previous_row_count + rc.offset));
        }

        apply_predicate(&mut df, predicate.as_deref(), true)?;
        apply_aggregations(&mut df, aggregate)?;

        previous_row_count += current_row_count;
        dfs.push(df);

        if remaining_rows == 0 {
            break;
        }
    }
    Ok(dfs)
}

#[allow(clippy::too_many_arguments)]
// parallelizes over row groups
fn rg_to_dfs_par(
    bytes: &[u8],
    limit: usize,
    file_metadata: &FileMetaData,
    schema: &ArrowSchema,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    aggregate: Option<&[ScanAggregation]>,
    row_count: Option<RowCount>,
    projection: &[usize],
) -> PolarsResult<Vec<DataFrame>> {
    let mut remaining_rows = limit;
    let mut previous_row_count = 0;

    // compute the limits per row group and the row count offsets
    let row_groups = file_metadata
        .row_groups
        .iter()
        .enumerate()
        .map(|(rg_idx, rg_md)| {
            let row_count_start = previous_row_count;
            let num_rows = rg_md.num_rows();
            previous_row_count += num_rows;
            let local_limit = remaining_rows;
            remaining_rows = remaining_rows.saturating_sub(num_rows);

            (rg_idx, rg_md, local_limit, row_count_start)
        })
        .collect::<Vec<_>>();

    let dfs = row_groups
        .into_par_iter()
        .map(|(rg_idx, md, local_limit, row_count_start)| {
            if local_limit == 0
                || !read_this_row_group(predicate.as_ref(), file_metadata, schema, rg_idx)?
            {
                return Ok(None);
            }
            // test we don't read the parquet file if this env var is set
            #[cfg(debug_assertions)]
            {
                assert!(std::env::var("POLARS_PANIC_IF_PARQUET_PARSED").is_err())
            }

            let chunk_size = md.num_rows() as usize;
            let columns = projection
                .iter()
                .map(|column_i| {
                    column_idx_to_series(*column_i, md, local_limit, schema, bytes, chunk_size)
                })
                .collect::<PolarsResult<Vec<_>>>()?;

            let mut df = DataFrame::new_no_checks(columns);

            if let Some(rc) = &row_count {
                df.with_row_count_mut(&rc.name, Some(row_count_start as IdxSize + rc.offset));
            }

            apply_predicate(&mut df, predicate.as_deref(), false)?;
            apply_aggregations(&mut df, aggregate)?;

            Ok(Some(df))
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    Ok(dfs.into_iter().flatten().collect())
}

#[allow(clippy::too_many_arguments)]
pub fn read_parquet<R: MmapBytesReader>(
    mut reader: R,
    limit: usize,
    projection: Option<&[usize]>,
    schema: &ArrowSchema,
    metadata: Option<FileMetaData>,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    aggregate: Option<&[ScanAggregation]>,
    mut parallel: ParallelStrategy,
    row_count: Option<RowCount>,
    low_memory: bool,
) -> PolarsResult<DataFrame> {
    let file_metadata = metadata
        .map(Ok)
        .unwrap_or_else(|| read::read_metadata(&mut reader))?;
    let row_group_len = file_metadata.row_groups.len();

    let projection = projection
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned((0usize..schema.fields.len()).collect::<Vec<_>>()));

    if let ParallelStrategy::Auto = parallel {
        if row_group_len > projection.len() || row_group_len > POOL.current_num_threads() {
            parallel = ParallelStrategy::RowGroups;
        } else {
            parallel = ParallelStrategy::Columns;
        }
    }

    if let (ParallelStrategy::Columns, true) = (parallel, projection.len() == 1) {
        parallel = ParallelStrategy::None;
    }

    let reader = ReaderBytes::from(&reader);
    let bytes = reader.deref();
    let dfs = match parallel {
        ParallelStrategy::Columns | ParallelStrategy::None => rg_to_dfs(
            bytes,
            row_group_len,
            limit,
            &file_metadata,
            schema,
            predicate,
            aggregate,
            row_count,
            parallel,
            &projection,
        )?,
        ParallelStrategy::RowGroups => rg_to_dfs_par(
            bytes,
            limit,
            &file_metadata,
            schema,
            predicate,
            aggregate,
            row_count,
            &projection,
        )?,
        // auto should already be replaced by Columns or RowGroups
        ParallelStrategy::Auto => unimplemented!(),
    };

    if dfs.is_empty() {
        let schema = if let Cow::Borrowed(_) = projection {
            Cow::Owned(apply_projection(schema, &projection))
        } else {
            Cow::Borrowed(schema)
        };
        Ok(arrow_schema_to_empty_df(&schema))
    } else {
        let mut df = accumulate_dataframes_vertical(dfs.into_iter())?;
        apply_aggregations(&mut df, aggregate)?;
        Ok(if low_memory {
            df._slice_and_realloc(0, limit)
        } else {
            df.slice_par(0, limit)
        })
    }
}
