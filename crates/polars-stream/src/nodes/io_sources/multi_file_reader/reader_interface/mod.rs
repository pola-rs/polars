//! Interface for single-file readers

pub mod builder;
pub mod capabilities;
pub mod output;

use async_trait::async_trait;
use output::FileReaderOutputRecv;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::RowIndex;
use polars_io::predicates::ScanIOPredicate;
use polars_utils::IdxSize;
use polars_utils::slice_enum::Slice;

use super::extra_ops::cast_columns::CastColumnsPolicy;
use super::extra_ops::missing_columns::MissingColumnsPolicy;
use crate::async_executor::JoinHandle;

/// Interface to read a single file
#[async_trait]
pub trait FileReader: Send + Sync {
    /// Initialize this FileReader. Intended to allow the reader to pre-fetch metadata.
    ///
    /// This must be called before calling any other functions of the FileReader.
    async fn initialize(&mut self) -> PolarsResult<()>;

    /// Begin reading the file into morsels.
    fn begin_read(
        &self,
        args: BeginReadArgs,
    ) -> PolarsResult<(FileReaderOutputRecv, JoinHandle<PolarsResult<()>>)>;

    /// This FileReader must be initialized before calling this.
    ///
    /// Note: The default implementation of this dispatches to `begin_read`, so should not be
    /// called from there.
    async fn n_rows_in_file(&self) -> PolarsResult<IdxSize> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        let (morsel_receivers, handle) = self.begin_read(BeginReadArgs {
            // Passing 0-0 slice indicates to the reader that we want the full row count, but it can
            // skip actually reading the data if it is able to.
            pre_slice: Some(Slice::Positive { offset: 0, len: 0 }),
            callbacks: FileReaderCallbacks {
                n_rows_in_file_tx: Some(tx),
                ..Default::default()
            },
            ..Default::default()
        })?;

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

        let (mut morsel_receivers, handle) = self.begin_read(BeginReadArgs {
            pre_slice,
            callbacks: FileReaderCallbacks {
                row_position_on_end_tx: Some(tx),
                ..Default::default()
            },
            ..Default::default()
        })?;

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

#[derive(Debug)]
pub struct BeginReadArgs {
    /// Columns to project from the file.
    pub projected_schema: SchemaRef,

    pub row_index: Option<RowIndex>,
    pub pre_slice: Option<Slice>,
    pub predicate: Option<ScanIOPredicate>,

    /// User-configured policy for when datatypes do not match.
    ///
    /// A reader may wish to use this if it is applying predicates.
    ///
    /// This can be ignored by the reader, as the policy is also applied in post.
    #[expect(unused)]
    pub cast_columns_policy: CastColumnsPolicy,
    /// User-configured policy for when columns are not found in the file.
    ///
    /// A reader may wish to use this if it is applying predicates.
    ///
    /// This can be ignored by the reader, as the policy is also applied in post.
    pub missing_columns_policy: MissingColumnsPolicy,

    pub num_pipelines: usize,
    pub callbacks: FileReaderCallbacks,
    // TODO
    // We could introduce dynamic `Option<Box<dyn Any>>` for the reader to use. That would help
    // with e.g. synchronizing row group prefetches across multiple files in Parquet. Currently
    // every reader started concurrently will prefetch up to the row group prefetch limit.
}

impl Default for BeginReadArgs {
    fn default() -> Self {
        BeginReadArgs {
            projected_schema: SchemaRef::default(),
            row_index: None,
            pre_slice: None,
            predicate: None,
            // TODO: Use less restrictive default
            cast_columns_policy: CastColumnsPolicy::ErrorOnMismatch,
            missing_columns_policy: MissingColumnsPolicy::Insert,
            num_pipelines: 1,
            callbacks: FileReaderCallbacks::default(),
        }
    }
}

#[derive(Debug, Default)]
/// We have this to avoid a footgun of accidentally swapping the arguments.
pub struct FileReaderCallbacks {
    /// Full file schema
    pub file_schema_tx: Option<tokio::sync::oneshot::Sender<SchemaRef>>,

    /// Callback for full row count. Avoid using this as it can trigger a full row count depending
    /// on the source. Prefer instead to use `row_position_on_end`, which can be much faster.
    ///
    /// Notes:
    /// * All readers must ensure that this count is sent if requested, even if the output port
    ///   closes prematurely, or a slice is sent.
    /// * Some readers will only send this after their output morsels to be fully consumed (or if
    ///   their output port is dropped), so you should not block morsel consumption on waiting for this.
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
        Some(v) => v.restrict_to_bounds(n_rows_in_file).end_position(),
    };

    IdxSize::try_from(out).unwrap_or(IdxSize::MAX)
}
