use std::sync::Mutex;

use once_cell::sync::Lazy;
use rand::prelude::*;

static POLARS_GLOBAL_RNG_STATE: Lazy<Mutex<SmallRng>> =
    Lazy::new(|| Mutex::new(SmallRng::from_entropy()));

pub(crate) fn get_global_random_u64() -> u64 {
    POLARS_GLOBAL_RNG_STATE.lock().unwrap().next_u64()
}

pub fn set_global_random_seed(seed: u64) {
    *POLARS_GLOBAL_RNG_STATE.lock().unwrap() = SmallRng::seed_from_u64(seed);
}
