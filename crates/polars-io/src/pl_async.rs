use std::ops::Deref;

use once_cell::sync::Lazy;
use polars_core::POOL;
use tokio::runtime::{Builder, Runtime};

static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    Builder::new_multi_thread()
        .worker_threads(std::cmp::max(POOL.current_num_threads() / 2, 4))
        .enable_io()
        .enable_time()
        .build()
        .unwrap()
});

pub fn get_runtime() -> &'static Runtime {
    RUNTIME.deref()
}
