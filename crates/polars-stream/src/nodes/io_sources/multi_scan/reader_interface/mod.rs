//! Interface for single-file readers

pub mod builder;
pub mod capabilities;
pub mod output;

use arrow::datatypes::ArrowSchemaRef;
use async_trait::async_trait;
use output::FileReaderOutputRecv;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::RowIndex;
use polars_io::predicates::ScanIOPredicate;
use polars_plan::dsl::CastColumnsPolicy;
use polars_utils::IdxSize;
use polars_utils::slice_enum::Slice;

use crate::async_executor::JoinHandle;
use crate::async_primitives::oneshot_channel;
pub use crate::nodes::io_sources::multi_scan::components::projection::Projection;

/// Interface to read a single file
#[async_trait]
pub trait FileReader: Send + Sync {
    /// Initialize this FileReader. Intended to allow the reader to pre-fetch metadata.
    ///
    /// This must be called before calling any other functions of the FileReader.
    async fn initialize(&mut self) -> PolarsResult<()>;

    /// When reading a files list, `prepare_read()` will be called sequentially for each reader in
    /// the order in which the files are read.
    ///
    /// This can be used e.g. to synchronize data fetches such that they happen in order.
    ///
    /// This is not guaranteed to always to be called.
    fn prepare_read(&mut self) -> PolarsResult<()> {
        Ok(())
    }

    /// Begin reading the file into morsels.
    fn begin_read(
        &mut self,
        args: BeginReadArgs,
    ) -> PolarsResult<(FileReaderOutputRecv, JoinHandle<PolarsResult<()>>)>;

    /// Schema of the file.
    async fn file_schema(&mut self) -> PolarsResult<SchemaRef> {
        // Currently only gets called if the reader is taking predicates.
        unimplemented!()
    }

    /// Returns the arrow schema for file formats that contain one. Returns None otherwise.
    async fn file_arrow_schema(&mut self) -> PolarsResult<Option<ArrowSchemaRef>> {
        Ok(None)
    }

    /// This FileReader must be initialized before calling this.
    ///
    /// Note: The default implementation of this dispatches to `begin_read`, so should not be
    /// called from there.
    async fn n_rows_in_file(&mut self) -> PolarsResult<IdxSize> {
        let (tx, rx) = oneshot_channel::channel();

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

        match rx.recv().await {
            Ok(v) => Ok(v),
            // It is an impl error if nothing is returned on a non-error case.
            Err(_) => Err(handle.await.unwrap_err()),
        }
    }

    /// Returns `Some(_)` if the row count is cheaply retrievable.
    async fn fast_n_rows_in_file(&mut self) -> PolarsResult<Option<IdxSize>> {
        Ok(None)
    }

    /// Returns the row position after applying a slice.
    ///
    /// This is essentially `n_rows_in_file`, but potentially with early stopping.
    async fn row_position_after_slice(
        &mut self,
        pre_slice: Option<Slice>,
    ) -> PolarsResult<IdxSize> {
        let Some(pre_slice) = pre_slice else {
            return self.n_rows_in_file().await;
        };

        let (tx, rx) = oneshot_channel::channel();

        let (mut morsel_receivers, handle) = self.begin_read(BeginReadArgs {
            pre_slice: Some(match pre_slice {
                // Normalize positive slices, as some row-skipping codepaths are single-threaded (e.g. NDJSON).
                v @ Slice::Positive { .. } => Slice::Positive {
                    offset: 0,
                    len: v.end_position(),
                },
                v @ Slice::Negative { .. } => v,
            }),
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

        match rx.recv().await {
            Ok(v) => Ok(v),
            // It is an impl error if nothing is returned on a non-error case.
            Err(_) => Err(handle.await.unwrap_err()),
        }
    }
}

/// Configuration for sampling at the file reader level.
#[derive(Clone, Debug)]
pub struct SampleConfig {
    /// Fraction of rows to sample.
    /// For Bernoulli (with_replacement=false): 0.0 to 1.0
    /// For Poisson (with_replacement=true): can be > 1.0
    pub fraction: f64,
    /// If true, uses Poisson sampling (rows can appear multiple times).
    /// If false, uses Bernoulli sampling (each row included with probability `fraction`).
    pub with_replacement: bool,
    /// Seed for deterministic sampling.
    pub seed: u64,
}

impl SampleConfig {
    /// Create a seeded RNG for a specific row group/batch.
    /// Combines the global seed with the row offset for deterministic results.
    fn create_rng(&self, row_offset: u64) -> rand::rngs::SmallRng {
        use rand::SeedableRng;
        // Combine seed with row_offset for deterministic per-batch randomness
        rand::rngs::SmallRng::seed_from_u64(self.seed.wrapping_add(row_offset))
    }

    /// Generate a Bernoulli sample mask for a batch of rows.
    /// Each row is independently included with probability `fraction`.
    pub fn generate_bernoulli_mask(&self, num_rows: usize, row_offset: u64) -> Vec<bool> {
        use rand::prelude::Distribution;
        use rand_distr::Bernoulli;

        let mut rng = self.create_rng(row_offset);
        let dist = Bernoulli::new(self.fraction).unwrap();

        (0..num_rows).map(|_| dist.sample(&mut rng)).collect()
    }

    /// Generate Poisson counts for a batch of rows.
    /// Each row appears Poisson(fraction) times on average.
    /// Returns (mask, counts) where mask[i] = counts[i] > 0.
    pub fn generate_poisson_counts(&self, num_rows: usize, row_offset: u64) -> (Vec<bool>, Vec<u32>) {
        use rand::prelude::Distribution;
        use rand_distr::Poisson;

        let mut rng = self.create_rng(row_offset);
        let dist = Poisson::new(self.fraction).unwrap();

        let counts: Vec<u32> = (0..num_rows)
            .map(|_| dist.sample(&mut rng) as u32)
            .collect();
        let mask: Vec<bool> = counts.iter().map(|&c| c > 0).collect();

        (mask, counts)
    }
}

#[derive(Debug)]
pub struct BeginReadArgs {
    /// Columns to project from the file.
    pub projection: Projection,

    pub row_index: Option<RowIndex>,
    pub pre_slice: Option<Slice>,
    pub predicate: Option<ScanIOPredicate>,

    /// Optional sample configuration for pre-filtered decode optimization.
    /// When set, the reader should use hash-based sampling to only decode sampled rows.
    pub sample: Option<SampleConfig>,

    /// User-configured policy for when datatypes do not match.
    ///
    /// A reader may wish to use this if it is applying predicates.
    ///
    /// This can be ignored by the reader, as the policy is also applied in post.
    pub cast_columns_policy: CastColumnsPolicy,

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
            projection: Projection::Plain(SchemaRef::default()),
            row_index: None,
            pre_slice: None,
            predicate: None,
            sample: None,
            // TODO: Use less restrictive default
            cast_columns_policy: CastColumnsPolicy::ERROR_ON_MISMATCH,
            num_pipelines: 1,
            callbacks: FileReaderCallbacks::default(),
        }
    }
}

/// Note, these are oneshot, but we are using the connector as tokio's oneshot channel deadlocks
/// on our async runtime. Not sure exactly why - a task wakeup seems to be lost somewhere.
/// (See https://github.com/pola-rs/polars/pull/21916).
#[derive(Default)]
pub struct FileReaderCallbacks {
    /// Full file schema
    pub file_schema_tx: Option<oneshot_channel::Sender<SchemaRef>>,

    /// Callback for full physical row count. Avoid using this as it can trigger a full row count
    /// on the source. Prefer instead to use `row_position_on_end_tx`, which can be much faster.
    ///
    /// Notes:
    /// * The reader must ensure that this count is sent if requested, even if the output port
    ///   closes prematurely, or a slice is sent. This is unless the reader encounters an error.
    /// * Some readers will only send this after their output morsels to be fully consumed (or if
    ///   their output port is dropped), so you should not block morsel consumption on waiting for this.
    pub n_rows_in_file_tx: Option<oneshot_channel::Sender<IdxSize>>,

    /// Callback for the physical (i.e. without accounting for deleted rows) row position this
    /// reader will have reached upon finishing.
    ///
    /// This callback should be sent as soon as possible, as it can be a serial dependency for the
    /// next reader. The returned value is allowed to exceed the slice limit (if provided), but must
    /// not exceed the total number of rows in the file.
    ///
    /// Readers that know their total row count upfront can simply send this value immediately.
    /// Readers that don't have this information may instead track their position in the file during
    /// reading and send this value after finishing.
    ///
    /// The returned value is useful for determining how much of a requested slice is consumed by a reader.
    /// It is more efficient than `n_rows_in_file_tx`, as it does not unnecessarily require the reader to
    /// fully consume the file.
    pub row_position_on_end_tx: Option<oneshot_channel::Sender<IdxSize>>,
}

impl std::fmt::Debug for FileReaderCallbacks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let FileReaderCallbacks {
            file_schema_tx,
            n_rows_in_file_tx,
            row_position_on_end_tx,
        } = self;

        f.write_str(&format!(
            "\
            FileReaderCallbacks: \
            file_schema_tx: {:?}, \
            n_rows_in_file_tx: {:?}, \
            row_position_on_end_tx: {:?} \
            ",
            file_schema_tx.as_ref().map(|_| ""),
            n_rows_in_file_tx.as_ref().map(|_| ""),
            row_position_on_end_tx.as_ref().map(|_| "")
        ))
    }
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
