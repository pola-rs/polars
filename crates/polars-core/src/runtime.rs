use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::LazyLock;
use std::sync::mpsc::{TryRecvError, sync_channel};

use polars_utils::with_drop::WithDrop;
use rayon::{ThreadPool, ThreadPoolBuilder, Yield};
use tokio::runtime::{Builder, Runtime};

pub struct RAYON;

// Thread locals to allow disabling threading for specific threads.
#[cfg(any(target_os = "emscripten", not(target_family = "wasm")))]
thread_local! {
    static NOOP_POOL: RefCell<ThreadPool> = RefCell::new(
        ThreadPoolBuilder::new()
            .use_current_thread()
            .num_threads(1)
            .build()
            .expect("could not create no-op thread pool")
    );
}

impl RAYON {
    pub fn install<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        #[cfg(not(any(target_os = "emscripten", not(target_family = "wasm"))))]
        {
            op()
        }

        #[cfg(any(target_os = "emscripten", not(target_family = "wasm")))]
        {
            self.with(|p| p.install(op))
        }
    }

    pub fn join<A, B, RA, RB>(&self, oper_a: A, oper_b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        self.install(|| rayon::join(oper_a, oper_b))
    }

    pub fn scope<'scope, OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce(&rayon::Scope<'scope>) -> R + Send,
        R: Send,
    {
        self.install(|| rayon::scope(op))
    }

    pub fn spawn<OP>(&self, op: OP)
    where
        OP: FnOnce() + Send + 'static,
    {
        #[cfg(not(any(target_os = "emscripten", not(target_family = "wasm"))))]
        {
            rayon::spawn(op)
        }

        #[cfg(any(target_os = "emscripten", not(target_family = "wasm")))]
        {
            self.with(|p| {
                p.spawn(op);
                if p.current_num_threads() == 1 {
                    p.yield_now();
                }
            })
        }
    }

    pub fn spawn_fifo<OP>(&self, op: OP)
    where
        OP: FnOnce() + Send + 'static,
    {
        #[cfg(not(any(target_os = "emscripten", not(target_family = "wasm"))))]
        {
            rayon::spawn_fifo(op)
        }

        #[cfg(any(target_os = "emscripten", not(target_family = "wasm")))]
        {
            self.with(|p| {
                p.spawn_fifo(op);
                if p.current_num_threads() == 1 {
                    p.yield_now();
                }
            })
        }
    }

    pub fn current_thread_has_pending_tasks(&self) -> Option<bool> {
        #[cfg(not(any(target_os = "emscripten", not(target_family = "wasm"))))]
        {
            None
        }

        #[cfg(any(target_os = "emscripten", not(target_family = "wasm")))]
        {
            self.with(|p| p.current_thread_has_pending_tasks())
        }
    }

    pub fn current_thread_index(&self) -> Option<usize> {
        #[cfg(not(any(target_os = "emscripten", not(target_family = "wasm"))))]
        {
            rayon::current_thread_index()
        }

        #[cfg(any(target_os = "emscripten", not(target_family = "wasm")))]
        {
            self.with(|p| p.current_thread_index())
        }
    }

    pub fn current_num_threads(&self) -> usize {
        #[cfg(not(any(target_os = "emscripten", not(target_family = "wasm"))))]
        {
            rayon::current_num_threads()
        }

        #[cfg(any(target_os = "emscripten", not(target_family = "wasm")))]
        {
            self.with(|p| p.current_num_threads())
        }
    }

    #[cfg(any(target_os = "emscripten", not(target_family = "wasm")))]
    pub fn with<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce(&ThreadPool) -> R + Send,
        R: Send,
    {
        if polars_async::executor::ALLOW_RAYON_THREADS.get()
            || THREAD_POOL.current_thread_index().is_some()
        {
            op(&THREAD_POOL)
        } else {
            NOOP_POOL.with(|v| op(&v.borrow()))
        }
    }

    /// Calls a blocking function without blocking the rayon thread pool by
    /// moving it to a different thread.
    ///
    /// If this thread isn't a rayon thread this simply calls f directly.
    pub fn block_on<R: Send, F: FnOnce() -> R + Send>(&self, f: F) -> R {
        if THREAD_POOL.current_thread_index().is_some() {
            let (send, recv) = sync_channel(1);
            let mut opt_f: Option<F> = Some(f);
            let mut wrap_f = || {
                let f = AssertUnwindSafe(opt_f.take().unwrap());
                send.send(catch_unwind(f)).unwrap();
            };

            // SAFETY: we always await the future to completion before returning from here, meaning
            // wrap_f stays alive for as long as it needs to. If for whatever reason we unwind we
            // abort.
            let abort = WithDrop::new((), |()| std::process::abort());
            let ref_wrap_f: &mut (dyn Send + FnMut()) = &mut wrap_f;
            let static_wrap_f: &'static mut (dyn Send + FnMut() + 'static) =
                unsafe { core::mem::transmute(ref_wrap_f) };
            ASYNC.spawn_blocking(static_wrap_f);

            loop {
                match recv.try_recv() {
                    Ok(r) => {
                        WithDrop::dismiss(abort);
                        match r {
                            Ok(v) => return v,
                            Err(panic) => std::panic::resume_unwind(panic),
                        }
                    },
                    Err(TryRecvError::Empty) => match rayon::yield_now() {
                        Some(Yield::Executed) => {},
                        Some(Yield::Idle) => std::thread::yield_now(),
                        None => unreachable!(),
                    },
                    Err(TryRecvError::Disconnected) => unreachable!(),
                }
            }
        } else {
            f()
        }
    }
}

// this is re-exported in utils for polars child crates
#[cfg(not(target_family = "wasm"))] // only use this on non wasm targets
pub static THREAD_POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    let thread_name = std::env::var("POLARS_THREAD_NAME").unwrap_or_else(|_| "polars".to_string());
    ThreadPoolBuilder::new()
        .num_threads(polars_config::config().max_threads())
        .thread_name(move |i| format!("{thread_name}-{i}"))
        .build()
        .expect("could not spawn threads")
});

#[cfg(all(target_os = "emscripten", target_family = "wasm"))] // Use 1 rayon thread on emscripten
pub static THREAD_POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    ThreadPoolBuilder::new()
        .num_threads(1)
        .use_current_thread()
        .build()
        .expect("could not create pool")
});

pub use polars_async::ASYNC;
