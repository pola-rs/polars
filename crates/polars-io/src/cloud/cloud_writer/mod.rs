mod bufferer;
mod internal_writer;
mod io_trait_wrap;
mod multipart_upload;
mod writer;

use std::time::Duration;

use futures::future::Either;
pub use io_trait_wrap::CloudWriterIoTraitWrap;
use polars_async::primitives::wait_group::{TokenHeld, WaitToken};
use polars_core::runtime::ASYNC;
use polars_error::PolarsResult;
pub use writer::CloudWriter;

/// How long cleanup may run once its writer is gone; requests have no timeout otherwise.
const CLEANUP_TIMEOUT: Duration = Duration::from_secs(30);

/// Holds `token`, if any, until `fut` is dropped.
fn hold<F: Future>(token: Option<WaitToken>, fut: F) -> Either<TokenHeld<F>, F> {
    match token {
        Some(token) => Either::Left(token.hold_until_drop(fut)),
        None => Either::Right(fut),
    }
}

/// Runs `fut` in a task of its own, so that a request that was sent also finishes when the
/// caller is dropped. Once the caller is gone, `fut` is dropped after [`CLEANUP_TIMEOUT`].
async fn run_to_completion<T: Send + 'static>(
    token: Option<WaitToken>,
    fut: impl Future<Output = PolarsResult<T>> + Send + 'static,
) -> PolarsResult<T> {
    let (_caller_alive, caller_gone) = tokio::sync::oneshot::channel::<()>();
    let give_up = async move {
        let _ = caller_gone.await;
        tokio::time::sleep(CLEANUP_TIMEOUT).await;
    };
    let task = async move {
        match futures::future::select(std::pin::pin!(fut), std::pin::pin!(give_up)).await {
            Either::Left((out, _)) => Some(out),
            Either::Right(_) => {
                if polars_core::config::verbose() {
                    eprintln!(
                        "[CloudWriter]: dropped a request {CLEANUP_TIMEOUT:?} after its writer"
                    );
                }
                None
            },
        }
    };
    ASYNC.spawn(hold(token, task)).await.unwrap().unwrap()
}
