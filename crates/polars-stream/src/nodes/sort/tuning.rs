use std::str::FromStr;

/// Partition when the buffered input exceeds this many bytes. Below it a
/// single in-memory sort is faster.
const DEFAULT_PARTITION_THRESHOLD_BYTES: u64 = 32 << 20;
/// Target size of a single bucket.
const DEFAULT_TARGET_BUCKET_BYTES: u64 = 64 << 20;
const DEFAULT_MAX_BUCKETS: usize = 1024;
/// Memory for all bucket builders together while partitioning, at most a
/// quarter of the memory budget. Builders can't be spilled, so each is frozen
/// into a `SpillFrame` once it holds its share.
const DEFAULT_BUILDER_MEMORY_BYTES: u64 = 2 << 30;
const DEFAULT_SAMPLE_ROWS: usize = 1024;

fn budget() -> u64 {
    polars_config::config().ooc_memory_budget_bytes()
}

fn env_var<T: FromStr>(name: &str, default: T) -> T {
    let Ok(v) = std::env::var(name) else {
        return default;
    };
    v.parse()
        .unwrap_or_else(|_| panic!("invalid value for {name}: {v}"))
}

/// Tuning knobs of the sort node, read from the environment at node
/// construction so that each query sees the current values.
pub struct SortTuning {
    pub partition_threshold_bytes: u64,
    pub target_bucket_bytes: u64,
    /// Always a power of two and at least two.
    pub max_buckets: usize,
    pub builder_memory_bytes: u64,
    pub sample_rows: usize,
}

impl SortTuning {
    pub fn from_env() -> Self {
        let max_buckets = env_var("POLARS_SORT_MAX_BUCKETS", DEFAULT_MAX_BUCKETS).max(2);

        Self {
            partition_threshold_bytes: env_var(
                "POLARS_SORT_PARTITION_THRESHOLD_BYTES",
                DEFAULT_PARTITION_THRESHOLD_BYTES,
            ),
            target_bucket_bytes: env_var(
                "POLARS_SORT_TARGET_BUCKET_BYTES",
                DEFAULT_TARGET_BUCKET_BYTES,
            )
            .max(1),
            max_buckets: 1 << max_buckets.ilog2(),
            builder_memory_bytes: env_var(
                "POLARS_SORT_BUILDER_MEMORY_BYTES",
                DEFAULT_BUILDER_MEMORY_BYTES,
            ),
            sample_rows: env_var("POLARS_SORT_SAMPLE_ROWS", DEFAULT_SAMPLE_ROWS).max(1),
        }
    }

    /// The buffered size above which the input is partitioned instead of
    /// sorted in memory.
    pub fn partition_threshold(&self) -> u64 {
        self.partition_threshold_bytes.min(budget() / 2)
    }

    /// The number of buckets that may be sorted or waiting to be emitted at
    /// once while flushing: two per pipeline, as long as they take at most a
    /// quarter of the memory budget.
    pub fn flush_ahead(&self, bucket_bytes: u64, num_pipelines: usize) -> usize {
        let fit = budget() / 4 / bucket_bytes.max(1);
        usize::try_from(fit)
            .unwrap_or(usize::MAX)
            .clamp(1, 2 * num_pipelines)
    }

    /// The number of value buckets to partition into, always a power of two.
    ///
    /// `sample_len` is the number of non-null sampled keys; it caps the number
    /// of buckets when too few distinct keys are available to split on.
    pub fn bucket_count(&self, total_bytes: u64, sample_len: usize, num_pipelines: usize) -> usize {
        // Enough buckets that every pipeline gets several to sort.
        let min_buckets = (4 * num_pipelines as u64).next_power_of_two();
        let buckets_wanted = (total_bytes / self.target_bucket_bytes).max(min_buckets);
        let b = (1u64 << buckets_wanted.ilog2()).clamp(2, self.max_buckets as u64) as usize;
        if sample_len < b - 1 {
            1 << (sample_len + 1).ilog2()
        } else {
            b
        }
    }

    /// The number of rows after which a bucket builder is frozen, given
    /// `num_builders` builders in total.
    pub fn flush_rows(&self, total_bytes: u64, total_rows: usize, num_builders: usize) -> usize {
        let avg_row_bytes = (total_bytes / total_rows.max(1) as u64).max(1);
        let builder_memory = self.builder_memory_bytes.min(budget() / 4);
        let flush_bytes = builder_memory / num_builders.max(1) as u64;
        let rows = (flush_bytes / avg_row_bytes).max(1);
        usize::try_from(rows).unwrap_or(usize::MAX)
    }
}
