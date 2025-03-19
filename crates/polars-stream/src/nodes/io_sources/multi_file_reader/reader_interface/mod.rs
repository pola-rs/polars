//! Interface for single-file readers

pub mod builder;
pub mod output;

use async_trait::async_trait;
use output::FileReaderOutputRecv;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_utils::IdxSize;
use polars_utils::slice_enum::Slice;

use super::extra_ops::apply::ApplyExtraOps;
use crate::async_executor::JoinHandle;

/// Interface to read a single file
#[async_trait]
pub trait FileReader: Send + Sync {
    /// Initialize this FileReader. Intended to allow the reader to pre-fetch metadata.
    ///
    /// This must be called before calling any other functions of the FileReader.
    ///
    /// Returns the schema of the morsels that this FileReader will return.
    async fn initialize(&mut self) -> PolarsResult<()>;

    /// Begin reading the file into morsels.
    fn begin_read(
        &self,
        // Note: This may contain more columns that what exist in the file. The reader should project
        // the ones that it finds. The remaining ones will be handled in post.
        projected_schema: &SchemaRef,
        extra_ops: ApplyExtraOps,

        num_pipelines: usize,
        callbacks: FileReaderCallbacks,
        // TODO
        // We could introduce dynamic `Option<Box<dyn Any>>` for the reader to use. That would help
        // with e.g. synchronizing row group prefetches across multiple files in Parquet. Currently
        // every reader started concurrently will prefetch up to the row group prefetch limit.
    ) -> PolarsResult<(FileReaderOutputRecv, JoinHandle<PolarsResult<()>>)>;

    /// This FileReader must be initialized before calling this.
    ///
    /// Note: The default implementation of this dispatches to `begin_read`, so should not be
    /// called from there.
    async fn n_rows_in_file(&self) -> PolarsResult<IdxSize> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        let (morsel_receivers, handle) = self.begin_read(
            &Default::default(), // pass empty schema
            ApplyExtraOps::Noop,
            1,
            FileReaderCallbacks {
                n_rows_in_file_tx: Some(tx),
                ..Default::default()
            },
        )?;

        drop(morsel_receivers);

        match rx.await {
            Ok(v) => Ok(v),
            // It is an impl error if nothing is returned on a non-error case.
            Err(_) => Err(handle.await.unwrap_err()),
        }
    }

    /// Returns the row position after applying a slice.
    ///
    /// This is essentially `n_rows_in_file`, but potentially with early stopping.
    async fn row_position_after_slice(&self, pre_slice: Option<Slice>) -> PolarsResult<IdxSize> {
        if pre_slice.is_none() {
            return self.n_rows_in_file().await;
        }

        let (tx, rx) = tokio::sync::oneshot::channel();

        let (mut morsel_receivers, handle) = self.begin_read(
            &Default::default(), // pass empty schema
            ApplyExtraOps::Noop,
            1,
            FileReaderCallbacks {
                row_position_on_end_tx: Some(tx),
                ..Default::default()
            },
        )?;

        // We are using the `row_position_on_end` callback, this means we must fully consume all of
        // the morsels sent by the reader.
        //
        // Note:
        // * Readers that don't rely on this should implement this function.
        // * It is not correct to rely on summing the morsel heights here, as the reader could begin
        //   from a non-zero offset.
        while morsel_receivers.recv().await.is_ok() {}

        match rx.await {
            Ok(v) => Ok(v),
            // It is an impl error if nothing is returned on a non-error case.
            Err(_) => Err(handle.await.unwrap_err()),
        }
    }
}

#[derive(Default)]
/// We have this to avoid a footgun of accidentally swapping the arguments.
pub struct FileReaderCallbacks {
    /// Full file schema
    pub file_schema_tx: Option<tokio::sync::oneshot::Sender<SchemaRef>>,

    /// Callback for full row count. Avoid using this as it can trigger a full row count depending
    /// on the source. Prefer instead to use `row_position_on_end`, which can be much faster.
    ///
    /// Notes:
    /// * Some readers will only send this after their output morsels to be fully consumed (or if
    ///   their output port is dropped), so you should not block morsel consumption on waiting for this.
    /// * All readers must ensure that this count is sent if requested, even if the output port
    ///   closes prematurely.
    pub n_rows_in_file_tx: Option<tokio::sync::oneshot::Sender<IdxSize>>,

    /// Returns the row position reached by this reader.
    ///
    /// The returned value indicates how much of a requested slice was consumed (or the full row
    /// count if there was none). This is only guaranteed to be the case if the morsels of this
    /// reader are fully consumed (i.e. no premature stopping), and the reader finishes with an Ok(()) result.
    ///
    /// In all other cases, the behavior is subject to reader-specific implementation details and is
    /// generally not well-defined - i.e. it could be any of:
    /// * An upfront calculated position based on the total row count
    /// * An arbitrary position in the file where the reader is currently positioned
    /// * Not sent at all
    /// * etc.
    pub row_position_on_end_tx: Option<tokio::sync::oneshot::Sender<IdxSize>>,
}

/// Calculate from a known total row count.
pub fn calc_row_position_after_slice(n_rows_in_file: IdxSize, pre_slice: Option<Slice>) -> IdxSize {
    let n_rows_in_file = usize::try_from(n_rows_in_file).unwrap();

    let out = match pre_slice {
        None => n_rows_in_file,

        Some(Slice::Positive { offset, len }) => {
            let slice_end = offset.saturating_add(len);
            n_rows_in_file.min(slice_end)
        },

        Some(Slice::Negative {
            offset_from_end,
            len,
        }) => {
            let n_from_end = offset_from_end.saturating_sub(len);
            n_rows_in_file.saturating_sub(n_from_end)
        },
    };

    IdxSize::try_from(out).unwrap_or(IdxSize::MAX)
}
