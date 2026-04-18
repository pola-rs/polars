use std::sync::{LazyLock, Mutex};

use rand::prelude::*;

static POLARS_GLOBAL_RNG_STATE: LazyLock<Mutex<SmallRng>> =
    LazyLock::new(|| Mutex::new(SmallRng::from_os_rng()));

pub(crate) fn get_global_random_u64() -> u64 {
    POLARS_GLOBAL_RNG_STATE.lock().unwrap().next_u64()
}

/// Draw `n` u64 seeds from the global RNG in a single locked, sequential pass.
/// Used by group-aware randomised expressions to pre-compute per-group seeds
/// before parallel dispatch, preserving deterministic seed assignment regardless
/// of scheduling.
pub fn draw_n_global_seeds(n: usize) -> Vec<u64> {
    let mut rng = POLARS_GLOBAL_RNG_STATE.lock().unwrap();
    (0..n).map(|_| rng.next_u64()).collect()
}

pub fn set_global_random_seed(seed: u64) {
    *POLARS_GLOBAL_RNG_STATE.lock().unwrap() = SmallRng::seed_from_u64(seed);
}
