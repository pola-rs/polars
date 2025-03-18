pub mod in_memory_linearize;
pub mod late_materialized_df;
pub mod task_handles_ext;

use std::future::Future;
use std::panic::Location;
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use polars_core::prelude::PlHashMap;

static ENABLE: LazyLock<bool> =
    LazyLock::new(|| std::env::var("POLARS_TRACED_AWAIT").as_deref() == Ok("1"));

static LOCAL_TRACERS: LazyLock<Mutex<Vec<Arc<Mutex<TraceStats>>>>> =
    LazyLock::new(|| Mutex::default());

thread_local!(
    static LOCAL_TRACER: Arc<Mutex<TraceStats>> = {
        let arc = Arc::new(Mutex::new(TraceStats::default()));
        LOCAL_TRACERS.lock().push(arc.clone());
        arc
    };
);

#[derive(Default)]
struct TraceStats {
    calls: PlHashMap<&'static Location<'static>, (usize, Duration)>,
}

impl TraceStats {
    fn add(&mut self, loc: &'static Location<'static>, calls: usize, dur: Duration) {
        let c = self.calls.entry(loc).or_default();
        c.0 += calls;
        c.1 += dur;
    }

    fn combine(&mut self, other: &Self) {
        for (loc, (calls, total_dur)) in &other.calls {
            self.add(*loc, *calls, *total_dur);
        }
    }
}

pub fn print_trace_stats() {
    let mut stats = TraceStats::default();
    for local_tracer in LOCAL_TRACERS.lock().iter() {
        stats.combine(&*local_tracer.lock());
    }

    for (loc, (calls, total_dur)) in stats.calls {
        eprintln!(
            "{}:{}: {:>10} {}",
            loc.file(),
            loc.line(),
            calls,
            total_dur.as_millis()
        );
    }
}

pub trait TraceAwait: Sized + Future + Send {
    #[track_caller]
    fn trace_await(self) -> impl Future<Output = Self::Output> + Send {
        let caller_location = Location::caller();

        async move {
            if !*ENABLE {
                return self.await;
            }

            // static ID: AtomicUsize = AtomicUsize::new(0);
            // let local_id = ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // eprintln!(
            //     "begin_await {} {}:{}",
            //     local_id,
            //     caller_location.file(),
            //     caller_location.line()
            // );

            let start = Instant::now();
            let out = self.await;
            let elapsed = start.elapsed();
            LOCAL_TRACER.with(|tracer| {
                tracer.lock().add(caller_location, 1, elapsed);
            });

            // eprintln!(
            //     "finish_await {} {}:{}",
            //     local_id,
            //     caller_location.file(),
            //     caller_location.line()
            // );

            out
        }
    }
}

impl<F: Sized + Future + Send> TraceAwait for F {}
