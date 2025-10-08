#[cfg(feature = "async")]
use polars_io::pl_async;
use polars_utils::unique_id::UniqueId;

use super::*;

pub struct CachePrefill {
    input: Box<dyn Executor>,
    id: UniqueId,
    hit_count: u32,
    /// Signals that this is a scan executed async in the streaming engine and needs extra handling
    is_new_streaming_scan: bool,
}

impl CachePrefill {
    pub fn new_cache(input: Box<dyn Executor>, id: UniqueId) -> Self {
        Self {
            input,
            id,
            hit_count: 0,
            is_new_streaming_scan: false,
        }
    }

    pub fn new_scan(input: Box<dyn Executor>) -> Self {
        Self {
            input,
            id: UniqueId::new(),
            hit_count: 0,
            is_new_streaming_scan: true,
        }
    }

    pub fn new_sink(input: Box<dyn Executor>) -> Self {
        Self {
            input,
            id: UniqueId::new(),
            hit_count: 0,
            is_new_streaming_scan: false,
        }
    }

    pub fn id(&self) -> UniqueId {
        self.id
    }

    /// Returns an executor that will read the prefilled cache.
    /// Increments the cache hit count.
    pub fn make_exec(&mut self) -> CacheExec {
        self.hit_count += 1;
        CacheExec { id: self.id }
    }
}

impl Executor for CachePrefill {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        let df = self.input.execute(state)?;
        state.set_df_cache(&self.id, df, self.hit_count);
        Ok(DataFrame::empty())
    }
}

pub struct CacheExec {
    id: UniqueId,
}

impl Executor for CacheExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        Ok(state.get_df_cache(&self.id))
    }
}

pub struct CachePrefiller {
    pub caches: PlIndexMap<UniqueId, CachePrefill>,
    pub phys_plan: Box<dyn Executor>,
}

impl Executor for CachePrefiller {
    fn is_cache_prefiller(&self) -> bool {
        true
    }

    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        if state.verbose() {
            eprintln!("PREFILL CACHES")
        }

        #[cfg(feature = "async")]
        let parallel_scan_exec_limit = {
            // Note, this needs to be less than the size of the tokio blocking threadpool (which
            // defaults to 512).
            let parallel_scan_exec_limit = POOL.current_num_threads().min(128);

            if state.verbose() {
                eprintln!(
                    "CachePrefiller: concurrent streaming scan exec limit: {parallel_scan_exec_limit}"
                )
            }

            Arc::new(tokio::sync::Semaphore::new(parallel_scan_exec_limit))
        };

        #[cfg(feature = "async")]
        let mut scan_handles: Vec<tokio::task::JoinHandle<PolarsResult<()>>> = vec![];

        // Ensure we traverse in discovery order. This will ensure that caches aren't dependent on each
        // other.
        for (_, mut prefill) in self.caches.drain(..) {
            assert_ne!(
                prefill.hit_count,
                0,
                "cache without execs: {}",
                prefill.id()
            );

            let mut state = state.split();
            state.branch_idx += 1;

            #[cfg(feature = "async")]
            if prefill.is_new_streaming_scan {
                let parallel_scan_exec_limit = parallel_scan_exec_limit.clone();

                scan_handles.push(pl_async::get_runtime().spawn(async move {
                    let _permit = parallel_scan_exec_limit.acquire().await.unwrap();

                    tokio::task::spawn_blocking(move || {
                        prefill.execute(&mut state)?;

                        Ok(())
                    })
                    .await
                    .unwrap()
                }));

                continue;
            }

            // This cache node may have dependency on the in-progress scan nodes,
            // ensure all of them complete here.

            #[cfg(feature = "async")]
            if state.verbose() && !scan_handles.is_empty() {
                eprintln!(
                    "CachePrefiller: wait for {} scan executors",
                    scan_handles.len()
                )
            }

            #[cfg(feature = "async")]
            for handle in scan_handles.drain(..) {
                pl_async::get_runtime().block_on(handle).unwrap()?;
            }

            let _df = prefill.execute(&mut state)?;
        }

        #[cfg(feature = "async")]
        if state.verbose() && !scan_handles.is_empty() {
            eprintln!(
                "CachePrefiller: wait for {} scan executors",
                scan_handles.len()
            )
        }

        #[cfg(feature = "async")]
        for handle in scan_handles {
            pl_async::get_runtime().block_on(handle).unwrap()?;
        }

        if state.verbose() {
            eprintln!("EXECUTE PHYS PLAN")
        }

        self.phys_plan.execute(state)
    }
}
