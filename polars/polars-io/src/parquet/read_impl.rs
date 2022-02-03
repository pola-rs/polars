use crate::aggregations::{apply_aggregations, ScanAggregation};
use crate::mmap::{MmapBytesReader, ReaderBytes};
use crate::parquet::predicates::collect_statistics;
use crate::predicates::{apply_predicate, arrow_schema_to_empty_df, PhysicalIoExpr};
use crate::utils::apply_projection;
use arrow::array::ArrayRef;
use arrow::io::parquet::read;
use arrow::io::parquet::read::{FileMetaData, MutStreamingIterator};
use polars_core::prelude::*;
use polars_core::utils::accumulate_dataframes_vertical;
use polars_core::POOL;
use polars_utils::contention_pool::LowContentionPool;
use rayon::prelude::*;
use std::borrow::Cow;
use std::convert::TryFrom;
use std::io::Cursor;
use std::ops::Deref;
use std::sync::Arc;

pub fn read_parquet<R: MmapBytesReader>(
    reader: R,
    limit: usize,
    projection: Option<&[usize]>,
    schema: &ArrowSchema,
    metadata: Option<FileMetaData>,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    aggregate: Option<&[ScanAggregation]>,
) -> Result<DataFrame> {
    let reader = ReaderBytes::from(&reader);
    let bytes = reader.deref();
    let mut reader = Cursor::new(bytes);

    let file_metadata = metadata
        .map(Ok)
        .unwrap_or_else(|| read::read_metadata(&mut reader))?;
    let row_group_len = file_metadata.row_groups.len();

    let projection = projection
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned((0usize..schema.fields.len()).collect::<Vec<_>>()));

    let mut dfs = Vec::with_capacity(row_group_len);

    let mut buf_1 = Vec::with_capacity(1024);
    let mut buf_2 = Vec::with_capacity(1024);

    let mut remaining_rows = limit;

    for rg in 0..row_group_len {
        let md = &file_metadata.row_groups[rg];
        if let Some(pred) = &predicate {
            if let Some(pred) = pred.as_stats_evaluator() {
                if let Some(stats) = collect_statistics(md.columns(), schema)? {
                    let should_read = pred.should_read(&stats);
                    // a parquet file may not have statistics of all columns
                    if matches!(should_read, Ok(false)) {
                        continue;
                    } else if !matches!(should_read, Err(PolarsError::NotFound(_))) {
                        let _ = should_read?;
                    }
                }
            }
        }
        // test we don't read the parquet file if this env var is set
        #[cfg(debug_assertions)]
        {
            assert!(std::env::var("POLARS_PANIC_IF_PARQUET_PARSED").is_err())
        }

        let columns = projection
            .clone()
            .iter()
            .map(|column_i| {
                let b1 = std::mem::take(&mut buf_1);
                let b2 = std::mem::take(&mut buf_2);

                // the get_column_iterator is an iterator of columns, each column contains compressed pages.
                // get_column_iterator yields `Vec<Vec<CompressedPage>>`:
                // outer `Vec` is len 1 for primitive types,
                // inner `Vec` is whatever number of pages the chunk contains.
                let column_iter =
                    read::get_column_iterator(&mut reader, &file_metadata, rg, *column_i, None, b1);
                let fld = &schema.fields[*column_i];
                let (mut array, b1, b2) = read::column_iter_to_array(column_iter, fld, b2)?;

                if array.len() > remaining_rows {
                    array = array.slice(0, remaining_rows);
                }

                buf_1 = b1;
                buf_2 = b2;

                Series::try_from((fld.name.as_str(), Arc::from(array)))
            })
            .collect::<Result<Vec<_>>>()?;

        remaining_rows = file_metadata.row_groups[rg].num_rows() as usize;

        let mut df = DataFrame::new_no_checks(columns);

        apply_predicate(&mut df, predicate.as_deref())?;
        apply_aggregations(&mut df, aggregate)?;

        dfs.push(df)
    }

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
        Ok(df.slice(0, limit))
    }
}

pub(crate) fn parallel_read<R: MmapBytesReader>(
    reader: R,
    limit: usize,
    projection: Option<&[usize]>,
    arrow_schema: &ArrowSchema,
    metadata: Option<FileMetaData>,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    aggregate: Option<&[ScanAggregation]>,
) -> Result<DataFrame> {
    let reader = ReaderBytes::from(&reader);
    let bytes = reader.deref();
    let mut reader = Cursor::new(bytes);

    let file_metadata = metadata
        .map(Ok)
        .unwrap_or_else(|| read::read_metadata(&mut reader))?;

    let parq_fields = if let Some(projection) = projection {
        let parq_fields = file_metadata.schema().fields();
        Cow::Owned(
            projection
                .iter()
                .map(|i| parq_fields[*i].clone())
                .collect::<Vec<_>>(),
        )
    } else {
        Cow::Borrowed(file_metadata.schema().fields())
    };

    let n_groups = file_metadata.row_groups.len();
    let mut dfs = Vec::with_capacity(n_groups);

    // we need to store two buffers to be reused, so the contention pool size
    // is the number of threads * 3 b1, b2, and column_chunks
    let pool_size = POOL.current_num_threads() * 2 + 1;

    let cont_pool = LowContentionPool::<Vec<u8>>::new(pool_size);

    for row_group in 0..n_groups {
        let md = &file_metadata.row_groups[row_group];
        if let Some(pred) = &predicate {
            if let Some(pred) = pred.as_stats_evaluator() {
                if let Some(stats) = collect_statistics(md.columns(), arrow_schema)? {
                    let should_read = pred.should_read(&stats);
                    // a parquet file may not have statistics of all columns
                    if matches!(should_read, Ok(false)) {
                        continue;
                    } else if !matches!(should_read, Err(PolarsError::NotFound(_))) {
                        let _ = should_read?;
                    }
                }
            }
        }
        // test we don't read the parquet file if this env var is set
        #[cfg(debug_assertions)]
        {
            assert!(std::env::var("POLARS_PANIC_IF_PARQUET_PARSED").is_err())
        }

        let columns = POOL.install(|| {
            parq_fields
                .par_iter()
                .enumerate()
                .map(|(mut field_i, field)| {
                    if let Some(projection) = projection {
                        field_i = projection[field_i];
                    }

                    // <IO bounded>

                    // create a new reader
                    let reader = Cursor::new(bytes);

                    let b1 = cont_pool.get();
                    // get compressed column pages
                    let mut columns = read::get_column_iterator(
                        reader,
                        &file_metadata,
                        row_group,
                        field_i,
                        None,
                        b1,
                    );

                    let mut column_chunks = Vec::with_capacity(64);
                    while let read::State::Some(mut new_iter) = columns.advance().unwrap() {
                        if let Some((pages, metadata)) = new_iter.get() {
                            let pages = pages.collect::<Vec<_>>();

                            column_chunks.push((pages, metadata.clone()));
                        }
                        columns = new_iter;
                    }

                    // <CPU bounded>
                    let columns = read::ReadColumnIterator::new(field.clone(), column_chunks);
                    let field = &arrow_schema.fields[field_i];

                    let b2 = cont_pool.get();
                    let (arr, b1, b2) = read::column_iter_to_array(columns, field, b2)?;
                    cont_pool.set(b1);
                    cont_pool.set(b2);
                    Series::try_from((field.name.as_str(), Arc::from(arr) as ArrayRef))
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let mut df = DataFrame::new_no_checks(columns);
        apply_predicate(&mut df, predicate.as_deref())?;
        apply_aggregations(&mut df, aggregate)?;

        dfs.push(df)
    }

    if dfs.is_empty() {
        let schema = if let Some(proj) = projection {
            Cow::Owned(apply_projection(arrow_schema, proj))
        } else {
            Cow::Borrowed(arrow_schema)
        };
        Ok(arrow_schema_to_empty_df(&schema))
    } else {
        let mut df = accumulate_dataframes_vertical(dfs.into_iter())?;
        apply_aggregations(&mut df, aggregate)?;
        Ok(df.slice(0, limit))
    }
}
