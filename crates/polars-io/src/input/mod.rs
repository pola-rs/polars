use std::future::Future;

use futures::StreamExt;
use once_cell::sync::Lazy;
use polars_error::{polars_err, PolarsResult};
use tokio::runtime::{Handle, Runtime};

pub mod file_format;
pub mod file_listing;

pub(crate) static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| polars_err!(ComputeError:"failed to create async runtime {}", e))
        .unwrap()
});

pub(crate) fn try_blocking_io<F: Future>(f: F) -> PolarsResult<F::Output> {
    let _guard = RUNTIME.enter();
    let handle = Handle::try_current().map_err(|e| {
        polars_err!(ComputeError:
            "failed to create current async handle {}", e)
    })?;

    Ok(handle.block_on(f))
}

pub fn try_map_async<I, O, F>(input: Vec<I>, buffer: usize, f: F) -> PolarsResult<Vec<O>>
where
    F: Fn(I) -> O,
{
    let iter = futures::stream::iter(input);

    try_blocking_io(async {
        iter.map(|v| async { f(v) })
            .buffered(buffer)
            .collect::<Vec<O>>()
            .await
    })
}
