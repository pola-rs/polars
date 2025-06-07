use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use polars_core::config;
use polars_core::schema::{SchemaExt, SchemaRef};
use polars_error::{PolarsResult, polars_bail};
use polars_io::cloud::CloudOptions;
use polars_io::fwf::{decode_fwf, get_field_width, Endianness, FwfReadOptions};
use polars_plan::prelude::ScanSource;
use polars_utils::IdxSize;
use polars_utils::mmap::MemSlice;
use polars_utils::slice_enum::Slice;

use super::multi_file_reader::reader_interface::output::{
    FileReaderOutputRecv, FileReaderOutputSend,
};
use super::multi_file_reader::reader_interface::{
    BeginReadArgs, FileReader, FileReaderCallbacks, calc_row_position_after_slice,
};
use crate::async_executor::{AbortOnDropHandle, spawn};
use crate::morsel::{SourceToken, get_ideal_morsel_size};
use crate::nodes::compute_node_prelude::*;
use crate::nodes::io_sources::multi_file_reader::reader_interface::builder::FileReaderBuilder;
use crate::nodes::io_sources::multi_file_reader::reader_interface::capabilities::ReaderCapabilities;



impl FileReaderBuilder for Arc<FwfReadOptions> {
    fn reader_name(&self) -> &str {
        "fwf"
    }

    fn reader_capabilities(&self) -> ReaderCapabilities {
        use ReaderCapabilities as RC;
        // Supports row index, positive pre-slice, negative pre-slice, and full filter
        RC::ROW_INDEX | RC::PRE_SLICE | RC::NEGATIVE_PRE_SLICE
    }

    fn build_file_reader(
        &self,
        source: ScanSource,
        cloud_options: Option<Arc<CloudOptions>>,
        _scan_source_idx: usize,
    ) -> Box<dyn FileReader> {
        let scan_source = source;
        let schema = self.schema();
        let verbose = config::verbose();
        let mut offsets = Vec::with_capacity(schema.len());
        let mut widths = Vec::with_capacity(schema.len());
        let mut offset = 0;
        for field in schema.iter_fields() {
            let width = get_field_width(&field).unwrap();
            offsets.push(offset);
            widths.push(width);
            offset += width;
        }
        let row_width = offset;

        let reader = FwfFileReader {
            scan_source,
            cloud_options,
            schema,
            endianness: self.endianness(),
            verbose,
            row_width,
            offsets,
            widths,
            cached_bytes: None,
            total_rows: None,
        };

        Box::new(reader) as Box<dyn FileReader>
    }
}

pub struct FwfFileReader {
    scan_source: ScanSource,
    #[expect(unused)] // Will be used when implementing cloud streaming.
    cloud_options: Option<Arc<CloudOptions>>,
    schema: SchemaRef,
    endianness: Arc<Vec<Endianness>>,
    row_width: usize,
    offsets: Vec<usize>,
    widths: Vec<usize>,
    // Cached on first access - we may be called multiple times e.g. on negative slice.
    cached_bytes: Option<MemSlice>,
    total_rows: Option<usize>,
    verbose: bool,
}

#[async_trait]
impl FileReader for FwfFileReader {
    async fn initialize(&mut self) -> PolarsResult<()> {
        let memslice = self
            .scan_source
            .as_scan_source_ref()
            .to_memslice_async_assume_latest(self.scan_source.run_async())?;
        if memslice.len() % self.row_width != 0 {
            polars_bail!(ComputeError: "File size is not a multiple of row size")
        }
        self.total_rows = Some(memslice.len() / self.row_width);
        // Note: We do not decompress in `initialize()`.
        self.cached_bytes = Some(memslice);
        Ok(())
    }

    fn begin_read(
        &mut self,
        args: BeginReadArgs,
    ) -> PolarsResult<(FileReaderOutputRecv, JoinHandle<PolarsResult<()>>)> {
        let verbose = self.verbose;

        let schema = self.schema.clone();
        let BeginReadArgs {
            projected_schema,
            row_index,
            pre_slice,
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

        if let Some(mut n_rows_in_file_tx) = n_rows_in_file_tx {
            _ = n_rows_in_file_tx.try_send(self._n_rows_in_file()?);
        }

        if let Some(mut row_position_on_end_tx) = row_position_on_end_tx {
            _ = row_position_on_end_tx.try_send(self._row_position_after_slice(pre_slice.clone())?);
        }

        if let Some(mut file_schema_tx) = file_schema_tx {
            _ = file_schema_tx.try_send(schema.clone());
        }

        let slice = self.normalize_slice(pre_slice.clone());
        if slice.len() == 0 {
            let (_, rx) = FileReaderOutputSend::new_serial();
            if verbose {
                eprintln!(
                    "[FwfFileReader]: early return: \
                    n_rows_in_file: {} \
                    pre_slice: {:?} \
                    resolved_pre_slice: {:?} \
                    ",
                    self._n_rows_in_file()?,
                    pre_slice,
                    slice
                )
            }
            return Ok((rx, spawn(TaskPriority::Low, std::future::ready(Ok(())))));
        }

        let projection: Vec<usize> = self.get_projection(&projected_schema);
        let endians = &self.endianness;
        if verbose {
            eprintln!(
                "[FwfFileReader]: project: {} / {}, slice: {:?}, row_index: {:?}",
                projection.len(),
                schema.len(),
                slice,
                row_index,
            )
        }
        let n_tasks = num_pipelines.max(1);
        let row_ranges = split_range(slice.start, slice.end, n_tasks);
        let morsel_size: usize = get_ideal_morsel_size();

        // Async channel for output
        let (morsel_senders, rx) = FileReaderOutputSend::new_parallel(n_tasks);
        let memslice = self.cached_bytes.clone().unwrap();
        let schema = schema.clone();
        let projection = projection.clone();
        let endians = endians.clone();

        // Spawn all decoding tasks
        let handles: Vec<_> = row_ranges
            .into_iter()
            .zip(morsel_senders)
            .map(|(range, mut morsel_sender)| {
                let buffer = memslice.clone();
                let schema = schema.clone();
                let projection = projection.clone();
                let endians = endians.clone();
                let offsets = self.offsets.clone();
                let widths = self.widths.clone();
                let row_width = self.row_width.clone();
                let source_token = SourceToken::new();
                AbortOnDropHandle::new(spawn(TaskPriority::Low, async move {
                    let mut offset = range.start;
                    let mut seq = ((range.start - slice.start) / morsel_size) as u64;
                    while offset < range.end {
                        let end = (offset + morsel_size).min(range.end);
                        if offset == end {
                            break;
                        }
                        let df = decode_fwf(
                            buffer.as_ref(),
                            schema.clone(),
                            offset..end,
                            &projection,
                            &offsets,
                            &widths,
                            &row_width,
                            &endians,
                        )?;
                        let morsel = Morsel::new(df, MorselSeq::new(seq), source_token.clone());
                        if morsel_sender.send_morsel(morsel).await.is_err() {
                            break;
                        }
                        offset = end;
                        seq += 1;
                    }
                    PolarsResult::Ok(())
                }))
            })
            .collect();

        // Join all handles
        let join_handle = spawn(TaskPriority::Low, async move {
            for handle in handles {
                handle.await?;
            }
            Ok(())
        });

        Ok((rx, join_handle))
    }
    async fn file_schema(&mut self) -> PolarsResult<SchemaRef> {
        Ok(self.schema.clone())
    }
    async fn n_rows_in_file(&mut self) -> PolarsResult<IdxSize> {
        self._n_rows_in_file()
    }
    async fn fast_n_rows_in_file(&mut self) -> PolarsResult<Option<IdxSize>> {
        Ok(Some(self._n_rows_in_file()?))
    }
    async fn row_position_after_slice(
        &mut self,
        pre_slice: Option<Slice>,
    ) -> PolarsResult<IdxSize> {
        self._row_position_after_slice(pre_slice)
    }
}
impl FwfFileReader {
    fn _n_rows_in_file(&mut self) -> PolarsResult<IdxSize> {
        Ok(self.total_rows.unwrap() as u32)
    }

    fn _row_position_after_slice(&mut self, pre_slice: Option<Slice>) -> PolarsResult<IdxSize> {
        Ok(calc_row_position_after_slice(
            self._n_rows_in_file()?,
            pre_slice,
        ))
    }
    fn get_projection(&self, projected_schema: &SchemaRef) -> Vec<usize> {
        projected_schema
            .iter_names()
            .filter_map(|name| self.schema.index_of(name))
            .collect()
    }
    pub fn normalize_slice(&self, slice: Option<Slice>) -> Range<usize> {
        if let Some(slice) = slice {
            let restricted_slice = slice.restrict_to_bounds(self.total_rows.unwrap());
            restricted_slice.into()
        } else {
            0..self.total_rows.unwrap()
        }
    }
}

fn split_range(start: usize, end: usize, n_slices: usize) -> Vec<std::ops::Range<usize>> {
    let total = end - start;
    let mut out = Vec::with_capacity(n_slices);
    let mut begin = start;
    for i in 0..n_slices {
        let chunk = total / n_slices + if i < total % n_slices { 1 } else { 0 };
        let next = begin + chunk;
        if begin < next {
            out.push(begin..next);
        }
        begin = next;
    }
    out
}