pub mod in_memory_linearize;
pub mod late_materialized_df;
pub mod task_handles_ext;

use std::future::Future;
use std::sync::LazyLock;
use std::time::Duration;

use futures::future::{Either, Lazy};
use polars_io::pl_async;

pub trait TracedAwait: Sized + Future + Send {
    #[track_caller]
    fn traced_await(self) -> impl Future<Output = Self::Output> + Send {
        static TIMEOUT_REPOLL: LazyLock<bool> =
            LazyLock::new(|| std::env::var("POLARS_AWAIT_TIMEOUT_REPOLL").as_deref() == Ok("1"));

        let caller_location = std::panic::Location::caller();

        let slf = Box::pin(self);

        async move {
            let file_loc = caller_location.file();

            let mut timed_out_fut = None;

            // Important: Sleep is ordered first, because select() polls the left future first on
            // wakeups. We don't want to re-poll the problematic future with a wakeup intended for
            // the sleep.
            match futures::future::select(
                pl_async::get_runtime()
                    .spawn(async { tokio::time::sleep(Duration::from_secs(5)).await }),
                slf,
            )
            .await
            {
                Either::Left((_, fut)) => {
                    timed_out_fut = Some(fut);
                    eprintln!("await timed out {}:{}", file_loc, caller_location.line());
                },
                Either::Right((out, _)) => return out,
            };

            // If we poll again here, we will unblock ourselves from the deadlock.
            if *TIMEOUT_REPOLL {
                eprintln!("repolling in 2 seconds");
                pl_async::get_runtime()
                    .spawn(async { tokio::time::sleep(Duration::from_secs(2)).await })
                    .await
                    .unwrap();
                return timed_out_fut.unwrap().await;
            } else {
                // Just leave ourselves hanging for the dev to see
                pl_async::get_runtime()
                    .spawn(async {
                        tokio::time::sleep(Duration::from_secs(9999)).await;
                    })
                    .await
                    .unwrap();

                panic!();
            }
        }
    }
}

impl<F: Sized + Future + Send> TracedAwait for F {}
