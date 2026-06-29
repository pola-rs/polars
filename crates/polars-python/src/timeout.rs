//! A global process-aborting timeout system, mainly intended for testing.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::LazyLock;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::time::Duration;

use polars::prelude::{InitHashMaps, PlHashSet};
use polars_utils::priority::Priority;
use polars_utils::relaxed_cell::RelaxedCell;

static TIMEOUT_REQUEST_HANDLER: LazyLock<Sender<TimeoutRequest>> = LazyLock::new(|| {
    let (send, recv) = channel();
    std::thread::Builder::new()
        .name("polars-timeout".to_string())
        .spawn(move || timeout_thread(recv))
        .unwrap();
    send
});

enum TimeoutRequest {
    Start(Duration, u64, Option<String>),
    Cancel(u64),
}

pub fn is_timeout_enabled() -> bool {
    static TIMEOUT_DISABLED: RelaxedCell<bool> = RelaxedCell::new_bool(false);

    // Fast path so we don't have to keep checking environment variables. Make
    // sure that if you want to use POLARS_TIMEOUT_MS it is set before the first
    // polars call.
    if TIMEOUT_DISABLED.load() {
        return false;
    }

    let var = std::env::var("POLARS_TIMEOUT_MS").ok();
    if var.is_none_or(|v| v.is_empty()) {
        TIMEOUT_DISABLED.store(true);
        return false;
    }

    true
}

pub fn get_timeout() -> Option<Duration> {
    if !is_timeout_enabled() {
        return None;
    }

    match std::env::var("POLARS_TIMEOUT_MS").unwrap().parse() {
        Ok(ms) => Some(Duration::from_millis(ms)),
        Err(e) => {
            eprintln!("failed to parse POLARS_TIMEOUT_MS: {e:?}");
            None
        },
    }
}

fn timeout_thread(recv: Receiver<TimeoutRequest>) {
    let mut active_timeouts: PlHashSet<u64> = PlHashSet::new();
    #[allow(clippy::type_complexity)]
    let mut shortest_timeout: BinaryHeap<Priority<Reverse<Duration>, (u64, Option<String>)>> =
        BinaryHeap::new();
    loop {
        // Remove cancelled requests.
        while let Some(Priority(_, (id, _))) = shortest_timeout.peek() {
            if active_timeouts.contains(id) {
                break;
            }
            shortest_timeout.pop();
        }

        let request = if let Some(Priority(timeout, (_, traceback))) = shortest_timeout.peek() {
            match recv.recv_timeout(timeout.0) {
                Err(RecvTimeoutError::Timeout) => {
                    eprint!("exiting the process, POLARS_TIMEOUT_MS exceeded");
                    if let Some(tb) = traceback {
                        eprintln!(", traceback:\n{tb}");
                    } else {
                        eprintln!(", traceback unavailable");
                    }
                    std::thread::sleep(Duration::from_secs_f64(1.0));
                    std::process::exit(1);
                },
                r => r.unwrap(),
            }
        } else {
            recv.recv().unwrap()
        };

        match request {
            TimeoutRequest::Start(duration, id, traceback) => {
                shortest_timeout.push(Priority(Reverse(duration), (id, traceback)));
                active_timeouts.insert(id);
            },
            TimeoutRequest::Cancel(id) => {
                active_timeouts.remove(&id);
            },
        }
    }
}

pub fn schedule_polars_timeout(traceback: Option<String>) -> Option<u64> {
    static TIMEOUT_ID: RelaxedCell<u64> = RelaxedCell::new_u64(0);

    let timeout = get_timeout()?;
    let id = TIMEOUT_ID.fetch_add(1);
    TIMEOUT_REQUEST_HANDLER
        .send(TimeoutRequest::Start(timeout, id, traceback))
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
