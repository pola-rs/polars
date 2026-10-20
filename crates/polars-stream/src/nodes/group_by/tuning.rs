use std::str::FromStr;

/// Below this many rows of work per partition, fewer partitions than pipelines are used.
const DEFAULT_MIN_ROWS_PER_PARTITION: u64 = 4096;
/// The smallest frame worth storing for one partition from one batch, in bytes.
const MIN_STORED_FRAME_BYTES: u64 = 256 * 1024;
/// The most partitions chosen without forcing, unless that is fewer than two per pipeline.
const MAX_PARTITIONS: u64 = 1024;
/// The estimated state of one group in one reduction, in bytes, used when there are no cold
/// rows to estimate the size of a group from.
const REDUCTION_STATE_BYTES: u64 = 16;

pub fn budget() -> u64 {
    polars_config::config().ooc_memory_budget_bytes()
}

fn env_var<T: FromStr>(name: &str) -> Option<T> {
    let v = std::env::var(name).ok()?;
    let parsed = v
        .parse()
        .unwrap_or_else(|_| panic!("invalid value for {name}: {v}"));
    Some(parsed)
}

/// What is known about the received input when choosing the number of partitions.
#[derive(Default)]
pub struct InputStats {
    pub cold_rows: u64,
    pub cold_bytes: u64,
    pub pre_agg_groups: u64,
    pub pre_agg_key_bytes: u64,
    pub num_reductions: u64,
    pub estimated_groups: u64,
}

impl InputStats {
    fn work(&self) -> u64 {
        self.cold_rows + self.pre_agg_groups
    }

    /// The estimated memory of one group of the result, in bytes.
    fn group_bytes(&self) -> u64 {
        self.cold_bytes
            .checked_div(self.cold_rows)
            .unwrap_or_else(|| {
                self.pre_agg_key_bytes / self.pre_agg_groups.max(1)
                    + REDUCTION_STATE_BYTES * self.num_reductions
            })
    }
}

/// Tuning knobs of the group-by node, read from the environment at node construction so that
/// each query sees the current values.
pub struct GroupByTuning {
    min_rows_per_partition: u64,
    /// Forces the number of partitions, unless the input is empty.
    num_partitions: Option<usize>,
    /// Forces the size of a batch of cold frames.
    batch_bytes: Option<usize>,
}

impl GroupByTuning {
    pub fn from_env() -> Self {
        Self {
            min_rows_per_partition: env_var("POLARS_GROUP_BY_MIN_ROWS_PER_PARTITION")
                .unwrap_or(DEFAULT_MIN_ROWS_PER_PARTITION)
                .max(1),
            num_partitions: env_var::<usize>("POLARS_GROUP_BY_NUM_PARTITIONS").map(|n| n.max(1)),
            batch_bytes: env_var("POLARS_GROUP_BY_BATCH_BYTES"),
        }
    }

    /// The size of the cold frames loaded together: an eighth of the memory budget.
    pub fn batch_bytes(&self) -> usize {
        self.batch_bytes
            .unwrap_or_else(|| usize::try_from(budget() / 8).unwrap_or(usize::MAX))
    }

    /// The number of partitions: enough for every pipeline to have work, and enough that the
    /// results of any `num_pipelines` partitions fit in memory. Above `num_pipelines` it is a
    /// multiple of `num_pipelines`.
    pub fn num_partitions(&self, stats: &InputStats, num_pipelines: usize) -> usize {
        let work = stats.work();
        if work == 0 {
            return 1;
        }
        if let Some(n) = self.num_partitions {
            return n;
        }

        let t = num_pipelines.max(1) as u64;
        let n_rows = (work / self.min_rows_per_partition).clamp(1, t);
        let result_bytes = (2 * t)
            .saturating_mul(stats.estimated_groups)
            .saturating_mul(stats.group_bytes());
        let n_memory = result_bytes.div_ceil((budget() / 2).max(1));
        if n_memory <= t {
            return n_rows.max(n_memory) as usize;
        }

        let max_partitions = MAX_PARTITIONS.min(self.batch_bytes() as u64 / MIN_STORED_FRAME_BYTES);
        let n_cap = (max_partitions / t * t).max(2 * t);
        let n = n_memory.div_ceil(t).saturating_mul(t).min(n_cap);
        usize::try_from(n).unwrap()
    }
}
