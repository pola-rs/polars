use arrow::io::ipc::read;
use arrow::io::ipc::read::{Dictionaries, FileMetadata};
use arrow::mmap::{mmap_dictionaries_unchecked, mmap_unchecked};
use arrow::record_batch::RecordBatch;
use polars_core::prelude::*;

use super::ipc_file::IpcReader;
use crate::mmap::{MMapSemaphore, MmapBytesReader};
use crate::predicates::PhysicalIoExpr;
use crate::shared::{finish_reader, ArrowReader};
use crate::utils::{apply_projection, columns_to_projection};

impl<R: MmapBytesReader> IpcReader<R> {
    pub(super) fn finish_memmapped(
        &mut self,
        predicate: Option<Arc<dyn PhysicalIoExpr>>,
    ) -> PolarsResult<DataFrame> {
        #[cfg(target_family = "unix")]
        use std::os::unix::fs::MetadataExt;
        match self.reader.to_file() {
            Some(file) => {
                #[cfg(target_family = "unix")]
                let metadata = file.metadata()?;
                let mmap = unsafe { memmap::Mmap::map(file).unwrap() };
                #[cfg(target_family = "unix")]
                let semaphore = MMapSemaphore::new(metadata.dev(), metadata.ino(), mmap);
                #[cfg(not(target_family = "unix"))]
                let semaphore = MMapSemaphore::new(mmap);
                let metadata =
                    read::read_file_metadata(&mut std::io::Cursor::new(semaphore.as_ref()))?;

                if let Some(columns) = &self.columns {
                    let schema = &metadata.schema;
                    let prj = columns_to_projection(columns, schema)?;
                    self.projection = Some(prj);
                }

                let schema = if let Some(projection) = &self.projection {
                    Arc::new(apply_projection(&metadata.schema, projection))
                } else {
                    metadata.schema.clone()
                };

                let reader = MMapChunkIter::new(Arc::new(semaphore), metadata, &self.projection)?;

                finish_reader(
                    reader,
                    // don't rechunk, that would trigger a read.
                    false,
                    self.n_rows,
                    predicate,
                    &schema,
                    self.row_index.clone(),
                )
            },
            None => polars_bail!(ComputeError: "cannot memory-map, you must provide a file"),
        }
    }
}

struct MMapChunkIter<'a> {
    dictionaries: Dictionaries,
    metadata: FileMetadata,
    mmap: Arc<MMapSemaphore>,
    idx: usize,
    end: usize,
    projection: &'a Option<Vec<usize>>,
}

impl<'a> MMapChunkIter<'a> {
    fn new(
        mmap: Arc<MMapSemaphore>,
        metadata: FileMetadata,
        projection: &'a Option<Vec<usize>>,
    ) -> PolarsResult<Self> {
        let end = metadata.blocks.len();
        // mmap the dictionaries
        let dictionaries = unsafe { mmap_dictionaries_unchecked(&metadata, mmap.clone())? };

        Ok(Self {
            dictionaries,
            metadata,
            mmap,
            idx: 0,
            end,
            projection,
        })
    }
}

impl ArrowReader for MMapChunkIter<'_> {
    fn next_record_batch(&mut self) -> PolarsResult<Option<RecordBatch>> {
        if self.idx < self.end {
            let chunk = unsafe {
                mmap_unchecked(
                    &self.metadata,
                    &self.dictionaries,
                    self.mmap.clone(),
                    self.idx,
                )
            }?;
            self.idx += 1;
            let chunk = match &self.projection {
                None => chunk,
                Some(proj) => {
                    let cols = chunk.into_arrays();
                    let arrays = proj.iter().map(|i| cols[*i].clone()).collect();
                    RecordBatch::new(arrays)
                },
            };
            Ok(Some(chunk))
        } else {
            Ok(None)
        }
    }
}
