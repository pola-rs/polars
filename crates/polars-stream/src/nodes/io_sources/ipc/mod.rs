use std::cmp::Reverse;
use std::fs::File;
use std::io::Cursor;
use std::ops::Range;
use std::sync::Arc;

use arrow::array::TryExtend;
use arrow::io::ipc::read::{Dictionaries, read_dictionary_block};
use async_trait::async_trait;

use polars_core::frame::DataFrame;
use polars_core::prelude::DataType;
use polars_core::schema::{Schema, SchemaExt};
use polars_core::utils::arrow::io::ipc::read::{
    BlockReader, FileMetadata, ProjectionInfo, get_row_count_from_blocks, prepare_projection,
    read_batch, read_file_metadata,
};
use polars_error::{ErrString, PolarsError, PolarsResult, feature_gated, polars_err};
use polars_io::cloud::CloudOptions;
use polars_io::utils::byte_source::{
    ByteSource, DynByteSource, DynByteSourceBuilder, MemSliceByteSource,
};
use polars_io::{RowIndex, pl_async};
use polars_plan::dsl::{ScanSource, ScanSourceRef};
use polars_utils::IdxSize;
use polars_utils::aliases::{InitHashMaps, PlHashMap};
use polars_utils::mem::prefetch::get_memory_prefetch_func;
use polars_utils::mmap::MemSlice;
use polars_utils::plpath::{PlPath, PlPathRef};
use polars_utils::priority::Priority;
use polars_utils::slice_enum::Slice;

use super::multi_scan::reader_interface::output::FileReaderOutputRecv;
use super::multi_scan::reader_interface::{BeginReadArgs, calc_row_position_after_slice};
use crate::async_primitives::oneshot_channel::Sender;
use crate::async_executor::{self, AbortOnDropHandle, JoinHandle, TaskPriority, spawn}; //kdn TODO cleanup
use crate::async_primitives::distributor_channel::distributor_channel;
use crate::async_primitives::linearizer::Linearizer;
use crate::morsel::{Morsel, MorselSeq, SourceToken, get_ideal_morsel_size};
use crate::nodes::io_sources::multi_scan::reader_interface::output::FileReaderOutputSend;
use crate::nodes::io_sources::multi_scan::reader_interface::{
    FileReader, FileReaderCallbacks, Projection,
};
use crate::utils::task_handles_ext;
use crate::{DEFAULT_DISTRIBUTOR_BUFFER_SIZE, DEFAULT_LINEARIZER_BUFFER_SIZE};

pub mod builder;
mod init;
mod record_batch_data_fetch;
mod record_batch_decode;

// kdn TODO TOGGLE
// mod ipc_bkp;

const ROW_COUNT_OVERFLOW_ERR: PolarsError = PolarsError::ComputeError(ErrString::new_static(
    "\
IPC file produces more than 2^32 rows; \
consider compiling with polars-bigidx feature (pip install polars[rt64])",
));

struct IpcFileReader {
    scan_source: ScanSource,
    cloud_options: Option<Arc<CloudOptions>>,
    metadata: Option<Arc<FileMetadata>>,
    byte_source_builder: DynByteSourceBuilder,
    verbose: bool,

    init_data: Option<InitializedState>,
}

#[derive(Clone)]
struct InitializedState_OLD {
    memslice: MemSlice,
    file_metadata: Arc<FileMetadata>,
    // Lazily initialized - getting this involves iterating record batches.
    n_rows_in_file: Option<IdxSize>,
}

#[derive(Clone)]
struct InitializedState {
    file_metadata: Arc<FileMetadata>,
    byte_source: Arc<DynByteSource>,
    dictionaries: Arc<Option<Dictionaries>>,
    // Lazily initialized - getting this involves iterating record batches.
    n_rows_in_file: Option<IdxSize>,
}

/// Move `slice` forward by `n` and return the slice until then.
fn slice_take(slice: &mut Range<usize>, n: usize) -> Range<usize> {
    let offset = slice.start;
    let length = slice.len();

    assert!(offset <= n);

    let chunk_length = (n - offset).min(length);
    let rng = offset..offset + chunk_length;
    *slice = 0..length - chunk_length;

    rng
}

fn get_max_morsel_size() -> usize {
    std::env::var("POLARS_STREAMING_IPC_SOURCE_MAX_MORSEL_SIZE")
        .map_or_else(
            |_| get_ideal_morsel_size(),
            |v| {
                v.parse::<usize>().expect(
                    "POLARS_STREAMING_IPC_SOURCE_MAX_MORSEL_SIZE does not contain valid size",
                )
            },
        )
        .max(1)
}

#[async_trait]
impl FileReader for IpcFileReader {
    async fn initialize(&mut self) -> PolarsResult<()> {
        dbg!("start initialize for impl FileReader for IpcFileReader"); //kdn
        if self.init_data.is_some() {
            return Ok(());
        }

        // =============
        let verbose = self.verbose;
        let scan_source = self.scan_source.clone();
        let byte_source_builder = self.byte_source_builder.clone();
        let cloud_options = self.cloud_options.clone();

        let byte_source = pl_async::get_runtime()
            .spawn(async move {
                scan_source
                    .as_scan_source_ref()
                    .to_dyn_byte_source(&byte_source_builder, cloud_options.as_deref())
                    .await
            })
            .await
            .unwrap()?;

        let byte_source = Arc::new(byte_source);

        let file_metadata = if let Some(v) = self.metadata.clone() {
            v
        } else {
            dbg!("get file_metadata in initialize");
            //kdn TODO - move to function
            let file_metadata = match self.scan_source.as_scan_source_ref() {
                ScanSourceRef::Path(path) => match path {
                    PlPathRef::Cloud(_) => {
                        feature_gated!("cloud", {
                            pl_async::get_runtime().block_on(async {
                                polars_io::ipc::IpcReaderAsync::from_uri(
                                    path,
                                    self.cloud_options.as_deref(),
                                )
                                .await?
                                .metadata()
                                .await
                            })?
                        })
                    },
                    PlPathRef::Local(path) => arrow::io::ipc::read::read_file_metadata(
                        &mut std::io::BufReader::new(polars_utils::open_file(path)?),
                    )?,
                },
                ScanSourceRef::File(file) => {
                    arrow::io::ipc::read::read_file_metadata(&mut std::io::BufReader::new(file))?
                },
                ScanSourceRef::Buffer(buff) => {
                    arrow::io::ipc::read::read_file_metadata(&mut std::io::Cursor::new(buff))?
                },
            };
            Arc::new(file_metadata)
        };

        // kdn TODO review clone and async
        let byte_source_async = byte_source.clone();
        let metadata_async = file_metadata.clone();
        let dictionaries = pl_async::get_runtime()
            .spawn(
                async move { read_dictionaries(&byte_source_async, metadata_async, verbose).await },
            )
            .await
            .unwrap()?;
        let dictionaries = Arc::new(Some(dictionaries));

        self.init_data = Some(InitializedState {
            file_metadata,
            byte_source,
            dictionaries,
            n_rows_in_file: None,
        });
        dbg!("done initialize for impl FileReader for IpcFileReader"); //kdn
        Ok(())
    }

    fn begin_read(
        &mut self,
        args: BeginReadArgs,
    ) -> PolarsResult<(FileReaderOutputRecv, JoinHandle<PolarsResult<()>>)> {
        // let kdn_select_old = false;
        // if kdn_select_old {
        //     return self.begin_read_old(args);
        // }

        // NEW =======
        dbg!("start begin_read"); //kdn
        let verbose = self.verbose;

        // Initialize.
        let InitializedState {
            file_metadata,
            byte_source,
            dictionaries,
            n_rows_in_file: _,
        } = self.init_data.clone().unwrap();

        let BeginReadArgs {
            projection: Projection::Plain(projected_schema),
            row_index,
            pre_slice: pre_slice_arg,
            predicate: None,
            cast_columns_policy: _,
            num_pipelines,
            callbacks:
                FileReaderCallbacks {
                    file_schema_tx,
                    n_rows_in_file_tx,
                    row_position_on_end_tx,
                },
        } = args
        else {
            panic!("unsupported args: {:?}", &args)
        };

        let file_schema_pl = std::cell::LazyCell::new(|| {
            Arc::new(Schema::from_arrow_schema(file_metadata.schema.as_ref()))
        });

        //kdn TODO - this is expensive and requires mmap
        dbg!(&pre_slice_arg);
        // let normalized_pre_slice = if let Some(pre_slice) = pre_slice_arg.clone() {
        //     Some(pre_slice.restrict_to_bounds(usize::try_from(self._n_rows_in_file()?).unwrap()))
        // } else {
        //     None
        // };
        let normalized_pre_slice = pre_slice_arg.clone(); //kdn SHORTCUT FOR NOW
        dbg!(&normalized_pre_slice);

        // Handle callbacks that are ready now.
        if let Some(file_schema_tx) = file_schema_tx {
            dbg!("has callback file_schema_tx"); //kdn
            _ = file_schema_tx.send(dbg!(file_schema_pl.clone())); //kdn
        }

        if normalized_pre_slice.as_ref().is_some_and(|x| x.len() == 0) {
            let (_, rx) = FileReaderOutputSend::new_serial();

            if verbose {
                eprintln!(
                    "[IpcFileReader]: early return: \
                    n_rows_in_file: {}, \
                    pre_slice: {:?}, \
                    resolved_pre_slice: {:?} \
                    ",
                    self._n_rows_in_file()?,
                    pre_slice_arg,
                    normalized_pre_slice
                )
            }

            return Ok((rx, spawn(TaskPriority::Low, std::future::ready(Ok(())))));
        }

        // Prepare parameters for tasks

        // Always create a slice. If no slice was given, just make the biggest slice possible.

        let slice_range: Range<usize> = normalized_pre_slice
            .clone()
            .map_or(0..usize::MAX, Range::<usize>::from);
        dbg!(&slice_range);

        // kdn TODO fixup slice_range vs slice

        // Avoid materializing projection info if we are projecting all the columns of this file.
        let projection_indices: Option<Vec<usize>> = if let Some(first_mismatch_idx) =
            (0..file_metadata.schema.len().min(projected_schema.len())).find(|&i| {
                file_metadata.schema.get_at_index(i).unwrap().0
                    != projected_schema.get_at_index(i).unwrap().0
            }) {
            let mut out = Vec::with_capacity(file_metadata.schema.len());

            out.extend(0..first_mismatch_idx);

            out.extend(
                (first_mismatch_idx..projected_schema.len()).filter_map(|i| {
                    file_metadata
                        .schema
                        .index_of(projected_schema.get_at_index(i).unwrap().0)
                }),
            );

            Some(out)
        } else if file_metadata.schema.len() > projected_schema.len() {
            // Names match up to projected schema len.
            Some((0..projected_schema.len()).collect::<Vec<_>>())
        } else {
            // Name order matches up to `file_metadata.schema.len()`, we are projecting all columns
            // in this file.
            None
        };

        if verbose {
            eprintln!(
                "[IpcFileReader]: \
                project: {} / {}, \
                pre_slice: {:?}, \
                resolved_pre_slice: {:?} \
                ",
                projection_indices
                    .as_ref()
                    .map_or(file_metadata.schema.len(), |x| x.len()),
                file_metadata.schema.len(),
                pre_slice_arg,
                normalized_pre_slice
            )
        }

        let projection_info: Option<ProjectionInfo> =
            projection_indices.map(|indices| prepare_projection(&file_metadata.schema, indices));
        let projection_info = Arc::new(projection_info);

        // Prepare parameters for Prefetch
        let memory_prefetch_func = get_memory_prefetch_func(verbose);
        let record_batch_prefetch_size = polars_core::config::get_record_batch_prefetch_size();

        // This can be set to 1 to force column-per-thread parallelism, e.g. for bug reproduction.
        let target_values_per_thread = std::env::var("POLARS_IPC_DECODE_TARGET_VALUES_PER_THREAD")
            .map(|x| x.parse::<usize>().expect("integer").max(1))
            .unwrap_or(16_777_216);

        let config = Config {
            num_pipelines,
            record_batch_prefetch_size,
            target_values_per_thread,
        };

        // Create IPCReadImpl and run.
        let (output_recv, handle) = IPCReadImpl {
            byte_source,
            metadata: file_metadata,
            dictionaries,
            slice_range,
            projection_info,
            row_index,
            n_rows_in_file_tx,
            row_position_on_end_tx,
            config,
            verbose,
            memory_prefetch_func,
        }
        .run();

        Ok((
            output_recv,
            async_executor::spawn(TaskPriority::Low, async move { handle.await.unwrap() }),
        ))
    }

    async fn n_rows_in_file(&mut self) -> PolarsResult<IdxSize> {
        self._n_rows_in_file()
    }

    async fn row_position_after_slice(
        &mut self,
        pre_slice: Option<Slice>,
    ) -> PolarsResult<IdxSize> {
        self._row_position_after_slice(pre_slice)
    }
}

impl IpcFileReader {
    fn _n_rows_in_file(&mut self) -> PolarsResult<IdxSize> {
        dbg!("start _n_rows_in_file"); //kdn
        let InitializedState {
            file_metadata,
            byte_source,
            dictionaries: _,
            n_rows_in_file,
        } = self.init_data.as_mut().unwrap();

        //kdn TODO
        let DynByteSource::MemSlice(MemSliceByteSource(memslice)) = &**byte_source else {
            panic!("Unsupported DynByteSource variant")
        };

        if n_rows_in_file.is_none() {
            let n_rows: i64 = get_row_count_from_blocks(
                &mut std::io::Cursor::new(memslice.as_ref()),
                &file_metadata.blocks,
            )?;

            let n_rows = IdxSize::try_from(n_rows)
                .map_err(|_| polars_err!(bigidx, ctx = "ipc file", size = n_rows))?;

            *n_rows_in_file = Some(n_rows);
        }

        Ok(n_rows_in_file.unwrap())
    }

    fn _row_position_after_slice(&mut self, pre_slice: Option<Slice>) -> PolarsResult<IdxSize> {
        Ok(calc_row_position_after_slice(
            self._n_rows_in_file()?,
            pre_slice,
        ))
    }
}

type AsyncTaskData = (
    FileReaderOutputRecv,
    task_handles_ext::AbortOnDropHandle<PolarsResult<()>>,
);

struct IPCReadImpl {
    // Source info
    byte_source: Arc<DynByteSource>,
    metadata: Arc<FileMetadata>,
    // Lazily initialized.
    dictionaries: Arc<Option<Dictionaries>>, //kdn TODO do we need Option?
    // Query parameters
    slice_range: Range<usize>,
    projection_info: Arc<Option<ProjectionInfo>>,
    row_index: Option<RowIndex>,
    // ? is_full_projection: bool,
    // ? predicate: Option<ScanIOPredicate>,
    // ? options: ParquetOptions,
    // Call-backs
    n_rows_in_file_tx: Option<Sender<IdxSize>>,
    row_position_on_end_tx: Option<Sender<IdxSize>>,
    // Run-time vars
    config: Config,
    verbose: bool,
    memory_prefetch_func: fn(&[u8]) -> (),
}

#[derive(Debug)]
struct Config {
    num_pipelines: usize,
    /// Number of record batches to prefetch concurrently, this can be across files
    record_batch_prefetch_size: usize,
    /// Minimum number of values for a parallel spawned task to process to amortize
    /// parallelism overhead.
    target_values_per_thread: usize,
}

async fn read_dictionaries(
    byte_source: &DynByteSource,
    file_metadata: Arc<FileMetadata>,
    verbose: bool,
) -> PolarsResult<Dictionaries> {
    //kdn TODO verbose

    let blocks = if let Some(blocks) = &file_metadata.dictionaries {
        blocks
    } else {
        return Ok(Dictionaries::default());
    };

    let mut dictionaries = Dictionaries::default();

    for block in blocks {
        // get bytes
        let range = block.offset as usize
            ..block.offset as usize + block.meta_data_length as usize + block.body_length as usize; //kdn TODO i32?
        let bytes = byte_source.get_range(range).await?;

        // IpcBlockReader
        let mut reader = BlockReader::new(
            Cursor::new(bytes.as_ref()),
            file_metadata.as_ref().clone(),
            None,
        );

        // kdn TODO: fix this? - we are not re-using scratches

        let mut message_scratch = Vec::new();
        let mut dictionary_scratch = Vec::new();

        // reader.set_scratches((
        //     std::mem::take(&mut data_scratch),
        //     std::mem::take(&mut message_scratch),
        // ));

        read_dictionary_block(
            &mut reader.reader,
            file_metadata.as_ref(),
            block,
            true,
            &mut dictionaries,
            &mut message_scratch,
            &mut dictionary_scratch,
        )?;

        // let chunk = read_dictionary_batch(
        //     &mut reader.reader,
        //     &file_metadata.clone(),
        //     &mut message_scratch,
        //     &mut data_scratch,
        // );

        // read_block (see read_batch)

        // get message
    }

    Ok(dictionaries)
}
