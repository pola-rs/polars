use std::str::FromStr;

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

/// Tuning knobs of the join nodes, read from the environment at node construction so that each
/// query sees the current values.
pub struct JoinTuning {
    /// Forces the size of a batch of build frames.
    batch_bytes: Option<usize>,
}

impl JoinTuning {
    pub fn from_env() -> Self {
        Self {
            batch_bytes: env_var("POLARS_JOIN_BATCH_BYTES"),
        }
    }

    /// The size of the build frames loaded together: an eighth of the memory budget.
    pub fn batch_bytes(&self) -> usize {
        self.batch_bytes
            .unwrap_or_else(|| usize::try_from(budget() / 8).unwrap_or(usize::MAX))
    }
}
