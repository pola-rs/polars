use std::str::FromStr;

/// The number of partitions per pipeline when memory does not ask for more.
const PARTITIONS_PER_PIPELINE: usize = 16;
/// The most partitions chosen without forcing, unless `PARTITIONS_PER_PIPELINE` asks for more.
const MAX_PARTITIONS: usize = 1024;
/// The fewest output ranges per pipeline when memory does not ask for more.
const RANGES_PER_PIPELINE: usize = 4;
/// Below this many rows per partition, fewer partitions are used.
const DEFAULT_MIN_ROWS_PER_PARTITION: usize = 4096;
/// The smallest budget the sizes are derived from. Spilling uses the configured budget.
const MIN_SIZING_BUDGET: usize = 64 << 20;

fn env_var<T: FromStr>(name: &str) -> Option<T> {
    let v = std::env::var(name).ok()?;
    let parsed = v
        .parse()
        .unwrap_or_else(|_| panic!("invalid value for {name}: {v}"));
    Some(parsed)
}

fn budget() -> usize {
    let budget = polars_config::config().ooc_memory_budget_bytes();
    usize::try_from(budget)
        .unwrap_or(usize::MAX)
        .max(MIN_SIZING_BUDGET)
}

/// The estimated bytes per row of the read columns and of the window columns.
pub struct RowSizes {
    pub read: usize,
    pub window: usize,
}

impl RowSizes {
    /// The working memory of evaluating one row: the gathered row and a sorted copy, the sort
    /// index with an 8-byte key, the group id and offset, and the window values with a
    /// reordered copy.
    pub fn work(&self) -> usize {
        2 * self.read + 32 + 2 * self.window
    }
}

/// Tuning knobs of the window node, read from the environment at node construction so that each
/// query sees the current values.
pub struct WindowTuning {
    min_rows_per_partition: usize,
    num_partitions: Option<usize>,
    batch_bytes: Option<usize>,
    range_bytes: Option<usize>,
    max_partition_bytes: Option<usize>,
}

impl WindowTuning {
    pub fn from_env() -> Self {
        Self {
            min_rows_per_partition: env_var("POLARS_WINDOW_MIN_ROWS_PER_PARTITION")
                .unwrap_or(DEFAULT_MIN_ROWS_PER_PARTITION)
                .max(1),
            num_partitions: env_var::<usize>("POLARS_WINDOW_NUM_PARTITIONS").map(|n| n.max(1)),
            batch_bytes: env_var("POLARS_WINDOW_BATCH_BYTES"),
            range_bytes: env_var("POLARS_WINDOW_RANGE_BYTES"),
            max_partition_bytes: env_var("POLARS_WINDOW_MAX_PARTITION_BYTES"),
        }
    }

    /// The size of the input frames loaded together to split them into partitions.
    pub fn batch_bytes(&self) -> usize {
        self.batch_bytes.unwrap_or_else(|| budget() / 8)
    }

    /// The size of the input morsels whose window columns are put back together, given the size
    /// of the whole output. There are at least a few ranges per pipeline, so that the pipelines
    /// prepare them in parallel. The ranges prepared ahead of the output are limited to
    /// [`Self::output_ahead_bytes`] in total.
    pub fn range_bytes(&self, num_pipelines: usize, total_bytes: usize) -> usize {
        let t = num_pipelines.max(1);
        self.range_bytes
            .unwrap_or_else(|| (budget() / (8 * t)).min(total_bytes / (RANGES_PER_PIPELINE * t)))
    }

    pub fn output_ahead_bytes(&self) -> usize {
        self.range_bytes.unwrap_or_else(|| budget() / 8)
    }

    /// The most working memory the partitions evaluated at the same time may need together.
    /// A partition that needs more on its own is split again.
    pub fn max_partition_bytes(&self) -> usize {
        self.max_partition_bytes.unwrap_or_else(|| budget() / 2)
    }

    /// The number of partitions for `num_rows` rows that each need `bytes_per_row` of working
    /// memory: enough that the work of `num_pipelines` partitions fits in half the budget.
    pub fn num_partitions(
        &self,
        num_rows: usize,
        bytes_per_row: usize,
        num_pipelines: usize,
    ) -> usize {
        if let Some(n) = self.num_partitions {
            return n;
        }
        let t = num_pipelines.max(1);
        let min_partitions = PARTITIONS_PER_PIPELINE * t;
        let work = num_rows.saturating_mul(bytes_per_row);
        let n_memory = work.saturating_mul(2 * t).div_ceil(budget());
        let n = n_memory
            .div_ceil(t)
            .saturating_mul(t)
            .clamp(min_partitions, MAX_PARTITIONS.max(min_partitions));
        n.min((num_rows / self.min_rows_per_partition).max(1))
    }
}
