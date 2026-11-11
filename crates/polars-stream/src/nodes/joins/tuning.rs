use std::str::FromStr;

use super::utils::table_bytes;

/// The largest share of the memory grant pool held by one join. Tables up to this size are kept
/// in memory.
const DEFAULT_MAX_GRANT_FRACTION: f64 = 2.0 / 3.0;
/// The smallest frame worth storing for one partition from one pipeline, in bytes.
const MIN_STORED_FRAME_BYTES: u64 = 256 * 1024;
/// The most partitions chosen without forcing, unless that is fewer than two per pipeline.
const MAX_PARTITIONS: u64 = 1024;

fn budget() -> u64 {
    polars_config::config().ooc_memory_budget_bytes()
}

fn env_var<T: FromStr>(name: &str) -> Option<T> {
    let v = std::env::var(name).ok()?;
    let parsed = v
        .parse()
        .unwrap_or_else(|_| panic!("invalid value for {name}: {v}"));
    Some(parsed)
}

/// What is known about the build side once it is received.
#[derive(Default)]
pub struct BuildStats {
    pub rows: u64,
    /// The rows with a null key, unless nulls are valid keys.
    pub null_rows: u64,
    /// The estimated memory of the payload and keys of all rows, in bytes.
    pub bytes: u64,
    /// The estimated memory of the keys of all rows, in bytes.
    pub key_bytes: u64,
    pub estimated_keys: u64,
}

impl BuildStats {
    /// Adds the row and byte counts of `other`.
    pub fn add(&mut self, other: &BuildStats) {
        self.rows += other.rows;
        self.null_rows += other.null_rows;
        self.bytes += other.bytes;
        self.key_bytes += other.key_bytes;
    }

    /// The estimated keys among `rows` rows.
    pub fn keys_in(&self, rows: u64) -> u64 {
        (self.estimated_keys as f64 * rows as f64 / self.rows.max(1) as f64) as u64
    }

    /// The estimated memory of a table of `rows` rows whose payload and keys take `bytes` bytes,
    /// in bytes.
    pub fn table_bytes(&self, rows: u64, bytes: u64) -> u64 {
        let key_width = self.key_bytes as f64 / self.rows.max(1) as f64;
        let row_width = bytes as f64 / rows.max(1) as f64;
        table_bytes(self.keys_in(rows) as f64, key_width, rows as f64, row_width) as u64
    }

    /// The estimated memory of all tables, in bytes. The rows with a null key are in a table only
    /// if `with_null_rows`.
    pub fn all_tables_bytes(&self, with_null_rows: bool) -> u64 {
        let rows = if with_null_rows {
            self.rows
        } else {
            self.rows - self.null_rows
        };
        let bytes = self.bytes as f64 * rows as f64 / self.rows.max(1) as f64;
        self.table_bytes(rows, bytes as u64)
    }
}

/// Tuning knobs of the join nodes, read from the environment at node construction so that each
/// query sees the current values.
pub struct JoinTuning {
    /// Forces the size of a batch of build frames.
    batch_bytes: Option<usize>,
    /// Forces the number of partitions, and storing partitions.
    num_partitions: Option<usize>,
    /// Forces the number of partitions kept in memory when partitions are stored.
    in_memory_partitions: Option<usize>,
    /// The largest share of the memory grant pool held by one join.
    pub max_grant_fraction: f64,
}

impl JoinTuning {
    pub fn from_env() -> Self {
        Self {
            batch_bytes: env_var("POLARS_JOIN_BATCH_BYTES"),
            num_partitions: env_var::<usize>("POLARS_JOIN_NUM_PARTITIONS").map(|n| n.max(1)),
            in_memory_partitions: env_var("POLARS_JOIN_IN_MEMORY_PARTITIONS"),
            max_grant_fraction: env_var("POLARS_JOIN_MEMORY_GRANT_FRACTION")
                .unwrap_or(DEFAULT_MAX_GRANT_FRACTION)
                .clamp(0.0, 1.0),
        }
    }

    /// The size of the build frames loaded together: an eighth of the memory budget.
    pub fn batch_bytes(&self) -> usize {
        self.batch_bytes
            .unwrap_or_else(|| usize::try_from(budget() / 8).unwrap_or(usize::MAX))
    }

    /// The largest memory grant, in bytes. Tables up to this size are kept in memory.
    pub fn max_grant_bytes(&self) -> u64 {
        let pool_bytes = polars_config::config().ooc_memory_grant_budget_bytes();
        (pool_bytes as f64 * self.max_grant_fraction).round() as u64
    }

    /// The number of partitions to store the build side in, or `None` to build every table in
    /// memory: when the tables fit in the grant taken. Stored, the tables of any
    /// `2 * num_pipelines` partitions fit in half the memory budget, the partitions are a multiple
    /// of `num_pipelines`, and each pipeline writes frames of at least `MIN_STORED_FRAME_BYTES` for
    /// every partition into a quarter of the budget.
    pub fn num_partitions(
        &self,
        table_bytes: u64,
        grant_bytes: u64,
        num_pipelines: usize,
    ) -> Option<usize> {
        if let Some(n) = self.num_partitions {
            return Some(n);
        }
        if table_bytes <= grant_bytes {
            return None;
        }

        let t = num_pipelines.max(1) as u64;
        let n_memory = (2 * t)
            .saturating_mul(table_bytes)
            .div_ceil((budget() / 2).max(1));
        let max_partitions = MAX_PARTITIONS.min(budget() / 4 / (t * MIN_STORED_FRAME_BYTES));
        let n_cap = (max_partitions / t * t).max(2 * t);
        let n = n_memory.div_ceil(t).saturating_mul(t).clamp(2 * t, n_cap);
        Some(usize::try_from(n).unwrap())
    }

    /// The number of stored partitions whose tables are kept in memory, given the estimated sizes
    /// of their tables from small to large: as many as fit in the grant.
    pub fn num_in_memory_partitions(&self, table_bytes: &[u64], grant_bytes: u64) -> usize {
        if let Some(n) = self.in_memory_partitions {
            return n.min(table_bytes.len());
        }
        let mut total = 0;
        table_bytes
            .iter()
            .take_while(|bytes| {
                total += **bytes;
                total <= grant_bytes
            })
            .count()
    }

    /// The most bytes a pipeline writes into a frame of one of `num_stored` stored partitions,
    /// so that the frames being written take at most a quarter of the memory budget.
    pub fn writer_frame_bytes(&self, num_pipelines: usize, num_stored: usize) -> usize {
        let writers = (num_pipelines * num_stored).max(1) as u64;
        usize::try_from(budget() / 4 / writers).unwrap_or(usize::MAX)
    }
}
