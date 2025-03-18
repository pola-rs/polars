pub mod in_memory_linearize;
pub mod late_materialized_df;
pub mod task_handles_ext;
use std::future::Future;
use std::panic::Location;
use std::pin::Pin;
use std::sync::{Arc, LazyLock};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use pin_project_lite::pin_project;
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
struct Stats {
    awaits: usize,
    await_time: Duration,
    polls: usize,
    poll_time: Duration,
}

#[derive(Default)]
struct TraceStats {
    stats: PlHashMap<&'static Location<'static>, Stats>,
}

impl TraceStats {
    fn add(&mut self, loc: &'static Location<'static>, stats: &Stats) {
        let c = self.stats.entry(loc).or_default();
        c.awaits += stats.awaits;
        c.await_time += stats.await_time;
        c.polls += stats.polls;
        c.await_time += stats.await_time;
    }

    fn combine(&mut self, other: &Self) {
        for (loc, stats) in &other.stats {
            self.add(loc, stats);
        }
    }
}

pub fn print_trace_stats() {
    let mut trace_stats = TraceStats::default();
    for local_tracer in LOCAL_TRACERS.lock().iter() {
        trace_stats.combine(&*local_tracer.lock());
    }

    for (loc, stats) in trace_stats.stats {
        eprintln!(
            "{}:{}: {:>10} {:>10} {:>10} {:>10}",
            loc.file(),
            loc.line(),
            stats.awaits,
            stats.await_time.as_millis(),
            stats.polls,
            stats.poll_time.as_millis(),
        );
    }

    eprintln!("print finish");
}

pub trait TraceAwait: Sized + Future + Send {
    #[track_caller]
    fn trace_await(self) -> impl Future<Output = Self::Output> + Send {
        let caller_location = Location::caller();
        TracedFuture {
            inner: self,
            caller_location,
            start: None,
        }
    }
}

impl<F: Sized + Future + Send> TraceAwait for F {}

pin_project! {
    struct TracedFuture<F> {
        #[pin]
        inner: F,
        caller_location: &'static Location<'static>,
        start: Option<Instant>
    }
}

impl<F: Future> Future for TracedFuture<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let slf = self.project();
        if !*ENABLE {
            return slf.inner.poll(cx);
        }

        let start = Instant::now();
        let poll_ret = slf.inner.poll(cx);
        let stop = Instant::now();

        if slf.start.is_none() {
            *slf.start = Some(start);
        }

        LOCAL_TRACER.with(|tracer| {
            let mut stats = Stats::default();
            stats.polls = 1;
            stats.poll_time += stop - start;
            if matches!(poll_ret, Poll::Ready(_)) {
                stats.awaits = 1;
                stats.await_time = stop - slf.start.unwrap();
            }
            tracer.lock().add(slf.caller_location, &stats);
        });

        poll_ret
    }
}
