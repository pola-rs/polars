use std::str::FromStr;

use polars_utils::IdxSize;

/// Below this many rows of work per partition, fewer partitions than pipelines are used.
const DEFAULT_MIN_ROWS_PER_PARTITION: u64 = 4096;
/// The smallest frame worth storing for one partition from one batch, in bytes.
const MIN_STORED_FRAME_BYTES: u64 = 256 * 1024;
/// The most partitions chosen without forcing, unless that is fewer than two per pipeline.
const MAX_PARTITIONS: u64 = 1024;
/// The estimated state of one group in one reduction, in bytes, used when there are no cold
/// rows to estimate the size of a group from.
const REDUCTION_STATE_BYTES: u64 = 16;
/// The memory for the partitions being built and sent is at least the memory budget divided by
/// this, unless the memory grant is smaller.
const MIN_PARTITION_MEMORY_DIVISOR: u64 = 16;
/// The largest share of the memory grant pool held by one group-by.
const DEFAULT_MAX_GRANT_FRACTION: f64 = 2.0 / 3.0;

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
    /// The pre-aggregate groups, or the values they hold if that is more.
    pub pre_agg_work: u64,
    pub pre_agg_key_bytes: u64,
    /// The size of the states of reductions that grow with the values of a group.
    pub pre_agg_state_bytes: u64,
    /// The number of states of the other reductions, summed over the pre-aggregate groups.
    pub pre_agg_fixed_states: u64,
    /// The number of fixed-size reduction states of one group.
    pub states_per_group: u64,
    /// Whether a reduction has states that grow with the values of a group.
    pub has_growing_states: bool,
    pub estimated_groups: u64,
}

impl InputStats {
    fn work(&self) -> u64 {
        self.cold_rows + self.pre_agg_work
    }

    /// The estimated memory of the whole result, in bytes.
    fn result_bytes(&self) -> u64 {
        let result_bytes = self.estimated_groups.saturating_mul(self.group_bytes());
        if self.has_growing_states {
            // A group can hold as many values as it received, so its size is not known from
            // the number of groups.
            let received_bytes =
                self.cold_bytes + self.pre_agg_key_bytes + self.pre_agg_state_bytes;
            result_bytes.max(received_bytes)
        } else {
            result_bytes
        }
    }

    /// The estimated memory of the pre-aggregates and their split into partitions, which is kept
    /// until all partitions are built.
    fn pre_agg_bytes(&self) -> u64 {
        let state_bytes =
            self.pre_agg_state_bytes + self.pre_agg_fixed_states * REDUCTION_STATE_BYTES;
        let split_bytes = self.pre_agg_groups * size_of::<IdxSize>() as u64;
        self.pre_agg_key_bytes + state_bytes + split_bytes
    }

    /// The estimated memory of the whole result with a factor 2 margin, in bytes.
    fn result_bytes_with_margin(&self) -> u64 {
        self.result_bytes().saturating_mul(2)
    }

    /// The estimated memory of the result of one partition, with a factor 2 margin.
    pub fn partition_grant_bytes(&self, num_partitions: usize) -> u64 {
        self.result_bytes_with_margin() / num_partitions.max(1) as u64
    }

    /// The estimated memory of one group of the result, in bytes.
    fn group_bytes(&self) -> u64 {
        self.cold_bytes
            .checked_div(self.cold_rows)
            .unwrap_or_else(|| {
                self.pre_agg_key_bytes / self.pre_agg_groups.max(1)
                    + REDUCTION_STATE_BYTES * self.states_per_group
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
    /// The largest share of the memory grant pool held by one group-by.
    pub max_grant_fraction: f64,
}

impl GroupByTuning {
    pub fn from_env() -> Self {
        Self {
            min_rows_per_partition: env_var("POLARS_GROUP_BY_MIN_ROWS_PER_PARTITION")
                .unwrap_or(DEFAULT_MIN_ROWS_PER_PARTITION)
                .max(1),
            num_partitions: env_var::<usize>("POLARS_GROUP_BY_NUM_PARTITIONS").map(|n| n.max(1)),
            batch_bytes: env_var("POLARS_GROUP_BY_BATCH_BYTES"),
            max_grant_fraction: env_var("POLARS_GROUP_BY_MEMORY_GRANT_FRACTION")
                .unwrap_or(DEFAULT_MAX_GRANT_FRACTION)
                .clamp(0.0, 1.0),
        }
    }

    /// The size of the cold frames loaded together: an eighth of the memory budget.
    pub fn batch_bytes(&self) -> usize {
        self.batch_bytes
            .unwrap_or_else(|| usize::try_from(budget() / 8).unwrap_or(usize::MAX))
    }

    /// The largest memory grant, in bytes.
    pub fn max_grant_bytes(&self) -> u64 {
        let pool_bytes = polars_config::config().ooc_memory_grant_budget_bytes();
        (pool_bytes as f64 * self.max_grant_fraction).round() as u64
    }

    /// The memory grant to ask for: the whole result with a factor 2 margin, up to the largest
    /// grant.
    pub fn wanted_grant_bytes(&self, stats: &InputStats) -> u64 {
        stats.result_bytes_with_margin().min(self.max_grant_bytes())
    }

    /// The number of partitions: enough for every pipeline to have work, and enough that the
    /// results of any `2 * num_pipelines` partitions fit in `memory_limit`, or in what the
    /// pre-aggregates leave of the memory budget if that is less. Above `num_pipelines` it is a
    /// multiple of `num_pipelines`.
    pub fn num_partitions(
        &self,
        stats: &InputStats,
        num_pipelines: usize,
        memory_limit: u64,
    ) -> usize {
        let work = stats.work();
        if work == 0 {
            return 1;
        }
        if let Some(n) = self.num_partitions {
            return n;
        }

        let t = num_pipelines.max(1) as u64;
        let n_rows = (work / self.min_rows_per_partition).clamp(1, t);
        let result_bytes = (2 * t).saturating_mul(stats.result_bytes());
        let partition_memory = budget()
            .saturating_sub(stats.pre_agg_bytes())
            .max(budget() / MIN_PARTITION_MEMORY_DIVISOR)
            .min(memory_limit);
        let n_memory = result_bytes.div_ceil(partition_memory.max(1));
        if n_memory <= t {
            return n_rows.max(n_memory) as usize;
        }

        let max_partitions = MAX_PARTITIONS.min(self.batch_bytes() as u64 / MIN_STORED_FRAME_BYTES);
        let n_cap = (max_partitions / t * t).max(2 * t);
        let n = n_memory.div_ceil(t).saturating_mul(t).min(n_cap);
        usize::try_from(n).unwrap()
    }
}
