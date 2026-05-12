use std::cell::{Cell, RefCell};
use std::sync::LazyLock;

use rayon::{ThreadPool, ThreadPoolBuilder};
use tokio::runtime::{Builder, Runtime};

pub struct POOL;

// Thread locals to allow disabling threading for specific threads.
#[cfg(any(target_os = "emscripten", not(target_family = "wasm")))]
thread_local! {
    pub static ALLOW_RAYON_THREADS: Cell<bool> = const { Cell::new(true) };
    static NOOP_POOL: RefCell<ThreadPool> = RefCell::new(
        ThreadPoolBuilder::new()
            .use_current_thread()
            .num_threads(1)
            .build()
            .expect("could not create no-op thread pool")
    );
}

impl POOL {
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
        if ALLOW_RAYON_THREADS.get() || THREAD_POOL.current_thread_index().is_some() {
            op(&THREAD_POOL)
        } else {
            NOOP_POOL.with(|v| op(&v.borrow()))
        }
    }
}

// this is re-exported in utils for polars child crates
#[cfg(not(target_family = "wasm"))] // only use this on non wasm targets
pub static THREAD_POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    let thread_name = std::env::var("POLARS_THREAD_NAME").unwrap_or_else(|_| "polars".to_string());
    ThreadPoolBuilder::new()
        .num_threads(
            std::env::var("POLARS_MAX_THREADS")
                .map(|s| s.parse::<usize>().expect("integer"))
                .unwrap_or_else(|_| {
                    std::thread::available_parallelism()
                        .unwrap_or(std::num::NonZeroUsize::new(1).unwrap())
                        .get()
                }),
        )
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

pub struct AsyncRuntime {
    rt: Runtime,
}

impl AsyncRuntime {
    fn new() -> Self {
        let n_threads = std::env::var("POLARS_ASYNC_THREAD_COUNT")
            .map(|x| x.parse::<usize>().expect("integer"))
            .unwrap_or(usize::min(POOL.current_num_threads(), 32));

        let max_blocking = std::env::var("POLARS_MAX_BLOCKING_THREAD_COUNT")
            .map(|x| x.parse::<usize>().expect("integer"))
            .unwrap_or(512);

        if crate::config::verbose() {
            eprintln!("async thread count: {n_threads}");
            eprintln!("blocking thread count: {max_blocking}");
        }

        let rt = Builder::new_multi_thread()
            .worker_threads(n_threads)
            .max_blocking_threads(max_blocking)
            .enable_io()
            .enable_time()
            .build()
            .unwrap();

        Self { rt }
    }

    /// Forcibly blocks this thread to evaluate the given future. This can be
    /// dangerous and lead to deadlocks if called re-entrantly on an async
    /// worker thread as the entire thread pool can end up blocking, leading to
    /// a deadlock. If you want to prevent this use block_on, which will panic
    /// if called from an async thread.
    pub fn block_in_place_on<F>(&self, future: F) -> F::Output
    where
        F: Future,
    {
        tokio::task::block_in_place(|| self.rt.block_on(future))
    }

    /// Blocks this thread to evaluate the given future. Panics if the current
    /// thread is an async runtime worker thread.
    pub fn block_on<F>(&self, future: F) -> F::Output
    where
        F: Future,
    {
        self.rt.block_on(future)
    }

    /// Spawns a future onto the Tokio runtime (see [`tokio::runtime::Runtime::spawn`]).
    pub fn spawn<F>(&self, future: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.rt.spawn(future)
    }

    // See [`tokio::runtime::Runtime::spawn_blocking`].
    pub fn spawn_blocking<F, R>(&self, f: F) -> tokio::task::JoinHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.rt.spawn_blocking(f)
    }
}

pub static ASYNC: LazyLock<AsyncRuntime> = LazyLock::new(AsyncRuntime::new);
