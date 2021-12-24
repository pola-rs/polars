use crate::mmap::{MmapBytesReader, ReaderBytes};
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

pub(crate) fn parallel_read<R: MmapBytesReader>(
    reader: R,
    limit: usize,
    projection: Option<&[usize]>,
    arrow_schema: &ArrowSchema,
    metadata: Option<FileMetaData>,
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
                    let field = &arrow_schema.fields()[field_i];

                    let b2 = cont_pool.get();
                    let (arr, b1, b2) = read::column_iter_to_array(columns, field, b2)?;
                    cont_pool.set(b1);
                    cont_pool.set(b2);
                    Series::try_from((field.name().as_str(), Arc::from(arr) as ArrayRef))
                })
                .collect::<Result<Vec<_>>>()
        })?;
        dfs.push(DataFrame::new_no_checks(columns))
    }

    accumulate_dataframes_vertical(dfs.into_iter()).map(|df| df.slice(0, limit))
}
