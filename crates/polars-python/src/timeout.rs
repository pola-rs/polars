//! A global process-aborting timeout system, mainly intended for testing.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::time::Duration;

use polars::prelude::{InitHashMaps, PlHashSet};
use polars_utils::priority::Priority;

static TIMEOUT_REQUEST_HANDLER: LazyLock<Sender<TimeoutRequest>> = LazyLock::new(|| {
    let (send, recv) = channel();
    std::thread::Builder::new()
        .name("polars-timeout".to_string())
        .spawn(move || timeout_thread(recv))
        .unwrap();
    send
});

enum TimeoutRequest {
    Start(Duration, u64),
    Cancel(u64),
}

pub fn get_timeout() -> Option<Duration> {
    static TIMEOUT_DISABLED: AtomicBool = AtomicBool::new(false);

    // Fast path so we don't have to keep checking environment variables. Make
    // sure that if you want to use POLARS_TIMEOUT_MS it is set before the first
    // polars call.
    if TIMEOUT_DISABLED.load(Ordering::Relaxed) {
        return None;
    }

    let Ok(timeout) = std::env::var("POLARS_TIMEOUT_MS") else {
        TIMEOUT_DISABLED.store(true, Ordering::Relaxed);
        return None;
    };

    match timeout.parse() {
        Ok(ms) => Some(Duration::from_millis(ms)),
        Err(e) => {
            eprintln!("failed to parse POLARS_TIMEOUT_MS: {e:?}");
            None
        },
    }
}

fn timeout_thread(recv: Receiver<TimeoutRequest>) {
    let mut active_timeouts: PlHashSet<u64> = PlHashSet::new();
    let mut shortest_timeout: BinaryHeap<Priority<Reverse<Duration>, u64>> = BinaryHeap::new();
    loop {
        // Remove cancelled requests.
        while let Some(Priority(_, id)) = shortest_timeout.peek() {
            if active_timeouts.contains(id) {
                break;
            }
            shortest_timeout.pop();
        }

        let request = if let Some(Priority(timeout, _)) = shortest_timeout.peek() {
            match recv.recv_timeout(timeout.0) {
                Err(RecvTimeoutError::Timeout) => {
                    eprintln!("exiting the process, POLARS_TIMEOUT_MS exceeded");
                    std::thread::sleep(Duration::from_secs_f64(1.0));
                    std::process::exit(1);
                },
                r => r.unwrap(),
            }
        } else {
            recv.recv().unwrap()
        };

        match request {
            TimeoutRequest::Start(duration, id) => {
                shortest_timeout.push(Priority(Reverse(duration), id));
                active_timeouts.insert(id);
            },
            TimeoutRequest::Cancel(id) => {
                active_timeouts.remove(&id);
            },
        }
    }
}

pub fn schedule_polars_timeout() -> Option<u64> {
    static TIMEOUT_ID: AtomicU64 = AtomicU64::new(0);

    let timeout = get_timeout()?;
    let id = TIMEOUT_ID.fetch_add(1, Ordering::Relaxed);
    TIMEOUT_REQUEST_HANDLER
        .send(TimeoutRequest::Start(timeout, id))
        .unwrap();
    Some(id)
}

pub fn cancel_polars_timeout(opt_id: Option<u64>) {
    if let Some(id) = opt_id {
        TIMEOUT_REQUEST_HANDLER
            .send(TimeoutRequest::Cancel(id))
            .unwrap();
    }
}
