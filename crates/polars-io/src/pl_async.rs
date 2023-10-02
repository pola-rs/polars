use std::ops::Deref;

use once_cell::sync::Lazy;
use tokio::runtime::{Builder, Runtime};

static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    Builder::new_multi_thread()
        .enable_io()
        .enable_time()
        .build()
        .unwrap()
});

pub fn get_runtime() -> &'static Runtime {
    RUNTIME.deref()
}
