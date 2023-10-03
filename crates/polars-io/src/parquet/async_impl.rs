//! Read parquet files in parallel from the Object Store without a third party crate.
use std::ops::Range;
use std::sync::Arc;

use arrow::io::parquet::read::{
    self as parquet2_read, get_field_columns, ColumnChunkMetaData, RowGroupMetaData,
};
use arrow::io::parquet::write::FileMetaData;
use bytes::Bytes;
use futures::future::try_join_all;
use futures::TryFutureExt;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use polars_core::config::verbose;
use polars_core::datatypes::PlHashMap;
use polars_core::error::{to_compute_err, PolarsResult};
use polars_core::prelude::*;
use polars_core::schema::Schema;
use smartstring::alias::String as SmartString;

use super::cloud::{build_object_store, CloudLocation, CloudReader};
use super::mmap;
use super::mmap::ColumnStore;
use crate::cloud::CloudOptions;

pub struct ParquetObjectStore {
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    length: Option<u64>,
    metadata: Option<FileMetaData>,
}

impl ParquetObjectStore {
    pub async fn from_uri(uri: &str, options: Option<&CloudOptions>) -> PolarsResult<Self> {
        let (CloudLocation { prefix, .. }, store) = build_object_store(uri, options).await?;

        Ok(ParquetObjectStore {
            store,
            path: ObjectPath::from_url_path(prefix).map_err(to_compute_err)?,
            length: None,
            metadata: None,
        })
    }

    /// Initialize the length property of the object, unless it has already been fetched.
    async fn initialize_length(&mut self) -> PolarsResult<()> {
        if self.length.is_some() {
            return Ok(());
        }
        self.length = Some(
            self.store
                .head(&self.path)
                .await
                .map_err(to_compute_err)?
                .size as u64,
        );
        Ok(())
    }

    pub async fn schema(&mut self) -> PolarsResult<Schema> {
        let metadata = self.get_metadata().await?;

        let arrow_schema = parquet2_read::infer_schema(metadata)?;

        Ok(Schema::from_iter(&arrow_schema.fields))
    }

    /// Number of rows in the parquet file.
    pub async fn num_rows(&mut self) -> PolarsResult<usize> {
        let metadata = self.get_metadata().await?;
        Ok(metadata.num_rows)
    }

    /// Fetch the metadata of the parquet file, do not memoize it.
    async fn fetch_metadata(&mut self) -> PolarsResult<FileMetaData> {
        self.initialize_length().await?;
        let object_store = self.store.clone();
        let path = self.path.clone();
        let length = self.length;
        let mut reader = CloudReader::new(length, object_store, path);
        parquet2_read::read_metadata_async(&mut reader)
            .await
            .map_err(to_compute_err)
    }

    /// Fetch and memoize the metadata of the parquet file.
    pub async fn get_metadata(&mut self) -> PolarsResult<&FileMetaData> {
        self.initialize_length().await?;
        if self.metadata.is_none() {
            self.metadata = Some(self.fetch_metadata().await?);
        }
        Ok(self.metadata.as_ref().unwrap())
    }
}

/// A vector of downloaded RowGroups.
/// A RowGroup will have 1 or more columns, for each column we store:
///   - a reference to its metadata
///   - the actual content as downloaded from object storage (generally cloud).
type RowGroupChunks = Vec<Vec<(u64, Bytes)>>;

async fn read_single_column_async(
    async_reader: &ParquetObjectStore,
    meta: &ColumnChunkMetaData,
) -> PolarsResult<(u64, Bytes)> {
    let (start, length) = meta.byte_range();
    let start = start as usize;
    let length = length as usize;
    let chunk = async_reader
        .store
        .get_range(&async_reader.path, start..start + length)
        .await
        .map_err(to_compute_err)?;
    Ok((start as u64, chunk))
}

// async fn download_row_group<'a>(
//     async_reader: &ParquetObjectStore,
//     columns: &'a [ColumnChunkMetaData],
//     field_name: &str,
// ) {
//
//     {
//         let ranges = get_field_columns(columns, field_name).iter().map(|meta| {
//             let (start, len) = meta.byte_range();
//             (start, start + len)
//         }).collect::<Vec<_>>();
//
//         let mut clustered = Vec::with_capacity(ranges.len());
//         let first_range = ranges[0];
//         let start = first_range.0;
//         let mut previous_end = first_range.1;
//         clustered.push((start, previous_end, vec![first_range]));
//         for (start, end) in ranges.iter().copied() {
//             let this_range = (start, end);
//
//             if start == previous_end {
//                 let (_start, total_end, members) = clustered.last_mut().unwrap();
//                 *total_end += end;
//                 members.push(this_range);
//             } else {
//                 clustered.push((start, end, vec![this_range]))
//             }
//             previous_end = end;
//         }
//         dbg!(clustered);
//     }
//
// }
//
// fn cluster_ranges(ranges: &[(u64, u64)]) -> Vec<(u64, u64, Vec<(u64, u64)>)> {
//
//     let mut clustered = Vec::with_capacity(ranges.len());
//     let first_range = ranges[0];
//     let start = first_range.0;
//     let mut previous_end = first_range.1;
//     clustered.push((start, previous_end, vec![first_range]));
//     for (start, end) in ranges.iter().copied().skip(1) {
//         let this_range = (start, end);
//
//         if start == previous_end {
//             let (_start, total_end, members) = clustered.last_mut().unwrap();
//             *total_end += end;
//             members.push(this_range);
//         } else {
//             clustered.push((start, end, vec![this_range]))
//         }
//         previous_end = end;
//     }
//     dbg!(clustered)
// }

async fn read_columns_async<'a>(
    async_reader: &ParquetObjectStore,
    columns: &'a [ColumnChunkMetaData],
    field_name: &str,
) -> PolarsResult<Vec<(u64, Bytes)>> {
    let futures = get_field_columns(columns, field_name)
        .into_iter()
        .map(|meta| async { read_single_column_async(async_reader, meta).await });

    try_join_all(futures).await
}

// async fn download_projection_rg<'a: 'b, 'b>(
//     fields: &[SmartString],
//     projection: &[usize],
//     row_group: &'a RowGroupMetaData,
//     async_reader: &'b ParquetObjectStore,
// ) -> PolarsResult<RowGroupChunks<'a>> {
//
//     let mut ranges = vec![];
//
//     // - index of the buffer
//     // - start
//     // - len
//     let mut mapping = vec![];
//
//     let mut current_range = (0, 0);
//
//     let mut buf_idx = 0usize;
//     let mut previous_col_idx = projection[0].overflowing_sub(1).0;
//     // let mut first_iter = true;
//     for (field_name, &col_idx) in fields.iter().zip(projection) {
//         let columns = row_group.columns();
//         let col_chunk_meta = get_field_columns(columns, field_name);
//         let start = col_chunk_meta.iter().map(|meta| meta.byte_range().0).next().unwrap();
//         let end = col_chunk_meta.iter().map(|meta| {
//             let (start, len) = meta.byte_range();
//             let end = start + len;
//             end
//         } ).last().unwrap();
//
//         mapping.push((buf_idx, start - current_range.0, end));
//
//         // Same cluster;
//         if col_idx == previous_col_idx.overflowing_add(1).0 {
//             current_range.1 = end;
//         // New cluster
//         } else {
//             buf_idx += 1;
//             current_range = (start, end);
//         }
//
//         previous_col_idx = col_idx;
//     }
//     ranges.push(current_range);
//     dbg!(ranges, mapping);
//
//     todo!()
// }

/// Download rowgroups for the column whose indexes are given in `projection`.
/// We concurrently download the columns for each field.
async fn download_projection(
    fields: &[SmartString],
    row_groups: &[RowGroupMetaData],
    async_reader: &ParquetObjectStore,
) -> PolarsResult<Vec<Vec<(u64, Bytes)>>> {

    // Build the cartesian product of the fields and the row groups.
    let product_futures = fields
        .iter()
        .flat_map(|name| row_groups.iter().map(move |r| (name.clone(), r)))
        .map(|(name, row_group)| async move {
            let columns = row_group.columns();
            read_columns_async(async_reader, columns, name.as_ref())
                .map_err(to_compute_err)
                .await
        });

    // Download concurrently
    futures::future::try_join_all(product_futures).await
}

// async fn download_projection2<'a: 'b, 'b>(
//     fields: &[smartstring],
//     row_groups: &'a [RowGroupMetaData],
//     async_reader: &'b ParquetObjectStore,
// ) -> PolarsResult<RowGroupChunks<'a>> {
//
//     // Build the cartesian product of the fields and the row groups.
//     let product_futures = fields
//         .iter()
//         .flat_map(|name| row_groups.iter().map(move |r| (name.clone(), r)))
//         .map(|(name, row_group)| async move {
//             let columns = row_group.columns();
//
//             let handle = tokio::spawn(async move {
//                 read_columns_async(async_reader, columns, name.as_ref())
//                     .map_err(to_compute_err).await
//             });
//             handle.await.unwrap()
//         });
//
//     // Download concurrently
//     futures::future::try_join_all(product_futures).await
// }

pub struct FetchRowGroupsFromObjectStore {
    reader: ParquetObjectStore,
    row_groups_metadata: Vec<RowGroupMetaData>,
    projected_fields: Vec<SmartString>,
    projection: Vec<usize>,
    logging: bool,
    schema: ArrowSchema,
}

impl FetchRowGroupsFromObjectStore {
    pub fn new(
        reader: ParquetObjectStore,
        metadata: &FileMetaData,
        projection: &Option<Vec<usize>>,
    ) -> PolarsResult<Self> {
        // TODO! schema should already be known
        let schema = parquet2_read::schema::infer_schema(metadata)?;
        let logging = verbose();

        let projection = projection.to_owned()
            .unwrap_or_else(|| (0usize..schema.fields.len()).collect());

        let projected_fields = projection
            .iter()
            .map(|i| (&schema.fields[*i].name).into())
            .collect::<Vec<_>>();


        Ok(FetchRowGroupsFromObjectStore {
            reader,
            row_groups_metadata: metadata.row_groups.to_owned(),
            projected_fields,
            projection,
            logging,
            schema,
        })
    }

    pub(crate) async fn fetch_row_groups(
        &mut self,
        row_groups: Range<usize>,
    ) -> PolarsResult<ColumnStore> {
        // Fetch the required row groups.
        let row_groups = &self
            .row_groups_metadata
            .get(row_groups.clone())
            .map_or_else(
                || Err(polars_err!(
                    ComputeError: "cannot access slice {0}..{1}", row_groups.start, row_groups.end,
                )),
                Ok,
            )?;

        // Package in the format required by ColumnStore.
        let downloaded =
            download_projection(&self.projected_fields, row_groups, &self.reader).await?;

        // let downloaded =
        //     download_projection_rg(&self.projected_fields, &self.projection, &row_groups[0], &self.reader).await?;

        if self.logging {
            eprintln!(
                "BatchedParquetReader: fetched {} row_groups for {} fields, yielding {} column chunks.",
                row_groups.len(),
                self.projected_fields.len(),
                downloaded.len(),
            );
        }
        let downloaded_per_filepos = downloaded
            .into_iter()
            .flat_map(|rg| {
                rg.into_iter()
            })
            .collect::<PlHashMap<_, _>>();

        if self.logging {
            eprintln!(
                "BatchedParquetReader: column chunks start & len: {}.",
                downloaded_per_filepos
                    .iter()
                    .map(|(pos, data)| format!("({}, {})", pos, data.len()))
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }

        Ok(mmap::ColumnStore::Fetched(downloaded_per_filepos))
    }
}
