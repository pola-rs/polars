use std::sync::atomic::Ordering;

use polars_io::pl_async;

use super::*;

pub struct CacheExec {
    pub input: Option<Box<dyn Executor>>,
    pub id: usize,
    /// `(cache_hits_before_drop - 1)`
    pub count: u32,
    pub is_new_streaming_scan: bool,
}

impl Executor for CacheExec {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        match &mut self.input {
            // Cache node
            None => {
                if state.verbose() {
                    eprintln!("CACHE HIT: cache id: {:x}", self.id);
                }
                let cache = state.get_df_cache(self.id, self.count);
                let out = cache.1.get().expect("prefilled").clone();
                let previous = cache.0.fetch_sub(1, Ordering::Relaxed);
                if previous == 0 {
                    if state.verbose() {
                        eprintln!("CACHE DROP: cache id: {:x}", self.id);
                    }
                    state.remove_df_cache(self.id);
                }

                Ok(out)
            },
            // Cache Prefill node
            Some(input) => {
                if state.verbose() {
                    eprintln!("CACHE SET: cache id: {:x}", self.id);
                }
                let df = input.execute(state)?;
                let cache = state.get_df_cache(self.id, self.count);
                cache.1.set(df).expect("should be empty");
                Ok(DataFrame::empty())
            },
        }
    }
}

pub struct CachePrefiller {
    pub caches: PlIndexMap<usize, Box<CacheExec>>,
    pub phys_plan: Box<dyn Executor>,
}

impl Executor for CachePrefiller {
    fn execute(&mut self, state: &mut ExecutionState) -> PolarsResult<DataFrame> {
        if state.verbose() {
            eprintln!("PREFILL CACHES")
        }

        #[cfg(feature = "async")]
        let concurrent_scans_limit = {
            let concurrent_scans_limit = POOL.current_num_threads().min(128);

            if state.verbose() {
                eprintln!(
                    "CachePrefiller: concurrent new-streaming scans limit: {}",
                    concurrent_scans_limit
                )
            }

            Arc::new(tokio::sync::Semaphore::new(concurrent_scans_limit))
        };

        #[cfg(feature = "async")]
        let mut scan_handles: Vec<tokio::task::JoinHandle<PolarsResult<()>>> = vec![];

        // Ensure we traverse in discovery order. This will ensure that caches aren't dependent on each
        // other.
        for (_, mut cache_exec) in self.caches.drain(..) {
            let mut state = state.split();
            state.branch_idx += 1;

            #[cfg(feature = "async")]
            if cache_exec.is_new_streaming_scan {
                let concurrent_scans_limit = concurrent_scans_limit.clone();

                scan_handles.push(pl_async::get_runtime().spawn(async move {
                    let _permit = concurrent_scans_limit.acquire().await.unwrap();

                    tokio::task::spawn_blocking(move || {
                        cache_exec.execute(&mut state)?;

                        Ok(())
                    })
                    .await
                    .unwrap()
                }));

                continue;
            }

            for handle in scan_handles.drain(..) {
                pl_async::get_runtime().block_on(handle).unwrap()?;
            }

            let _df = cache_exec.execute(&mut state)?;
        }

        for handle in scan_handles {
            pl_async::get_runtime().block_on(handle).unwrap()?;
        }

        if state.verbose() {
            eprintln!("EXECUTE PHYS PLAN")
        }

        self.phys_plan.execute(state)
    }
}
