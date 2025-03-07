use std::future::Future;
use std::sync::atomic::AtomicUsize;
use std::sync::LazyLock;

pub trait TracedAwait: Sized + Future {
    #[track_caller]
    fn traced_await(self) -> impl Future<Output = Self::Output> {
        static ID: AtomicUsize = AtomicUsize::new(0);
        static ENABLE: LazyLock<bool> =
            LazyLock::new(|| std::env::var("POLARS_TRACED_AWAIT").as_deref() == Ok("1"));

        let enable = *ENABLE;
        let caller_location = std::panic::Location::caller();

        let local_id = if enable {
            ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        } else {
            0
        };

        async move {
            if enable {
                eprintln!(
                    "begin_await {} {}:{}",
                    local_id,
                    caller_location.file(),
                    caller_location.line()
                );
            }

            let out = self.await;

            if enable {
                eprintln!(
                    "finish_await {} {}:{}",
                    local_id,
                    caller_location.file(),
                    caller_location.line()
                );
            }

            out
        }
    }
}

impl<F: Sized + Future> TracedAwait for F {}
