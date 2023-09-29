use tokio::runtime::{Builder, Runtime};

pub fn get_runtime() -> Runtime {
    Builder::new_current_thread()
        .enable_io()
        .enable_time()
        .build()
        .unwrap()
}
