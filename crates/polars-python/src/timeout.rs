//! A global process-aborting timeout system, mainly intended for testing.

use std::sync::{Arc, LazyLock, Mutex, Weak};
use std::time::Duration;

use polars_utils::relaxed_cell::RelaxedCell;
use pyo3::sync::PyOnceLock;
use pyo3::types::{IntoPyDict, PyAnyMethods};
use pyo3::{PyResult, Python, intern};

/// Incremented by the watchdog thread every period, starts at 1 so 0 can mean
/// idle.
static TICK: RelaxedCell<u64> = RelaxedCell::new_u64(1);
static SLOTS: Mutex<Vec<Weak<Slot>>> = Mutex::new(Vec::new());

struct Slot {
    python_thread_ident: u64,
    /// Tick at which the current call started, 0 if idle.
    started_at: RelaxedCell<u64>,
}

thread_local! {
    static SLOT: Arc<Slot> = {
        let slot = Arc::new(Slot {
            python_thread_ident: u64::from(unsafe { pyo3::ffi::PyThread_get_thread_ident() }),
            started_at: RelaxedCell::new_u64(0),
        });
        SLOTS.lock().unwrap().push(Arc::downgrade(&slot));
        slot
    };
}

pub fn is_timeout_enabled() -> bool {
    get_timeout().is_some()
}

/// Make sure that if you want to use POLARS_TIMEOUT_MS it is set before the
/// first polars call.
pub fn get_timeout() -> Option<Duration> {
    static TIMEOUT: LazyLock<Option<Duration>> = LazyLock::new(|| {
        let var = std::env::var("POLARS_TIMEOUT_MS").ok()?;
        if var.is_empty() {
            return None;
        }
        match var.parse() {
            Ok(ms) => Some(Duration::from_millis(ms)),
            Err(e) => {
                eprintln!("failed to parse POLARS_TIMEOUT_MS: {e:?}");
                None
            },
        }
    });
    *TIMEOUT
}

fn start_watchdog(py: Python<'_>, timeout: Duration) {
    // Dumps the Python stacks of all threads on abort, even if the GIL is held.
    if let Err(e) = enable_faulthandler(py) {
        eprintln!("failed to enable faulthandler for POLARS_TIMEOUT_MS: {e}");
    }
    std::thread::Builder::new()
        .name("polars-timeout".to_string())
        .spawn(move || watchdog(timeout))
        .unwrap();
}

fn enable_faulthandler(py: Python<'_>) -> PyResult<()> {
    let faulthandler = py.import(intern!(py, "faulthandler"))?;
    if !faulthandler
        .call_method0(intern!(py, "is_enabled"))?
        .extract::<bool>()?
    {
        // The C stack would only show the watchdog thread aborting.
        let kwargs = if py.version_info() >= (3, 14) {
            Some([("c_stack", false)].into_py_dict(py)?)
        } else {
            None
        };
        faulthandler.call_method(intern!(py, "enable"), (2,), kwargs.as_ref())?;
    }
    Ok(())
}

fn watchdog(timeout: Duration) {
    let period = (timeout / 10).clamp(Duration::from_millis(1), Duration::from_secs(1));
    let max_ticks = timeout.as_nanos().div_ceil(period.as_nanos()) as u64;
    loop {
        std::thread::sleep(period);
        let now = TICK.fetch_add(1) + 1;
        let expired = {
            let mut slots = SLOTS.lock().unwrap();
            slots.retain(|slot| slot.strong_count() > 0);
            slots.iter().filter_map(Weak::upgrade).find_map(|slot| {
                let started_at = slot.started_at.load();
                // A call that started at tick t has been running for at least
                // now - t - 1 full periods.
                (started_at != 0 && now - started_at > max_ticks)
                    .then_some(slot.python_thread_ident)
            })
        };
        if let Some(python_thread_ident) = expired {
            eprintln!(
                "exiting the process, POLARS_TIMEOUT_MS exceeded by a call on thread 0x{python_thread_ident:0width$x}, aborting to let faulthandler dump the Python stacks",
                width = 2 * size_of::<std::ffi::c_ulong>()
            );
            // Unlike std::process::abort, this raises SIGABRT on Windows too,
            // which triggers faulthandler.
            unsafe { libc::abort() }
        }
    }
}

/// Stops watching the current call when dropped.
pub struct PolarsTimeoutGuard {
    prev_started_at: u64,
}

impl Drop for PolarsTimeoutGuard {
    fn drop(&mut self) {
        SLOT.with(|slot| slot.started_at.store(self.prev_started_at));
    }
}

#[must_use]
pub fn schedule_polars_timeout(py: Python<'_>) -> Option<PolarsTimeoutGuard> {
    // Unlike Once, PyOnceLock detaches while waiting for another thread's
    // initialization, which may itself need the GIL.
    static WATCHDOG: PyOnceLock<()> = PyOnceLock::new();

    let timeout = get_timeout()?;
    WATCHDOG.get_or_init(py, || start_watchdog(py, timeout));
    let prev_started_at = SLOT.with(|slot| {
        let prev = slot.started_at.load();
        // The outermost call on this thread sets the deadline.
        if prev == 0 {
            slot.started_at.store(TICK.load());
        }
        prev
    });
    Some(PolarsTimeoutGuard { prev_started_at })
}
