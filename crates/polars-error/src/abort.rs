use std::any::Any;
use std::panic::{UnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};

/// Python hooks SIGINT to instead generate a KeyboardInterrupt exception.
/// So we do the same to try and abort long-running computations and return to
/// Python so that the Python exception can be generated.
///
/// We also use this mechanic to abort queries that run out of disk space while
/// spilling, which can happen from anywhere in the out-of-core code, meaning
/// there might not be a suitable PolarsResult return path.
pub enum QueryAborted {
    KeyboardInterrupt,
    OocOutOfDisk,
}

// We use a unique string so we can detect it in backtraces.
static POLARS_ABORT_PREFIX: &str = "__POLARS_ABORT_";
static POLARS_ABORT_KEYBOARD_INTERRUPT_STR: &str = "__POLARS_ABORT_KEYBOARD_INTERRUPT";
static POLARS_ABORT_OOC_OUT_OF_DISK_STR: &str = "__POLARS_ABORT_OOC_OUT_OF_DISK";

// Bottom two bits: abort flags.
// Top 62 bits: number of alive abort catchers.
const ABORT_KEYBOARD_INTERRUPT_BIT: u64 = 1;
const ABORT_OOC_OUT_OF_DISK_BIT: u64 = 2;
const ABORT_CATCHERS_UNIT: u64 = 4;
static ABORT_STATE: AtomicU64 = AtomicU64::new(0);

fn decode_polars_abort(p: &dyn Any) -> Option<QueryAborted> {
    let s = if let Some(s) = p.downcast_ref::<&str>() {
        s
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.as_str()
    } else {
        return None;
    };

    if !s.contains(POLARS_ABORT_PREFIX) {
        return None;
    }

    if s.contains(POLARS_ABORT_KEYBOARD_INTERRUPT_STR) {
        Some(QueryAborted::KeyboardInterrupt)
    } else if s.contains(POLARS_ABORT_OOC_OUT_OF_DISK_STR) {
        Some(QueryAborted::OocOutOfDisk)
    } else {
        unreachable!()
    }
}

pub fn register_polars_abort_mechanism() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |p| {
        // Suppress output if there is an active catcher and the panic message
        // contains the abort string.
        let num_catchers =
            ABORT_STATE.load(Ordering::Relaxed) >> ABORT_CATCHERS_UNIT.trailing_zeros();
        let suppress = num_catchers > 0 && decode_polars_abort(p.payload()).is_some();
        if !suppress {
            default_hook(p);
        }
    }));

    // WASM doesn't support signals, so we just skip installing the hook there.
    #[cfg(not(target_family = "wasm"))]
    unsafe {
        // SAFETY: we only do an atomic op in the signal handler, which is allowed.
        signal_hook::low_level::register(signal_hook::consts::signal::SIGINT, move || {
            // Set the keyboard interrupt flag, but only if there are active catchers.
            ABORT_STATE
                .fetch_update(Ordering::Release, Ordering::Relaxed, |state| {
                    let num_catchers = state >> ABORT_CATCHERS_UNIT.trailing_zeros();
                    if num_catchers > 0 {
                        Some(state | ABORT_KEYBOARD_INTERRUPT_BIT)
                    } else {
                        None
                    }
                })
                .ok();
        })
        .unwrap();
    }
}

pub fn polars_abort_ooc_out_of_disk() -> ! {
    ABORT_STATE
        .fetch_update(Ordering::Release, Ordering::Relaxed, |state| {
            let num_catchers = state >> ABORT_CATCHERS_UNIT.trailing_zeros();
            if num_catchers > 0 {
                Some(state | ABORT_OOC_OUT_OF_DISK_BIT)
            } else {
                None
            }
        })
        .ok();

    std::panic::panic_any(POLARS_ABORT_OOC_OUT_OF_DISK_STR);
}

/// Checks if the abort flag is set, and if yes panics. This function is very cheap.
#[inline(always)]
pub fn try_raise_polars_abort() {
    if ABORT_STATE.load(Ordering::Acquire) & (ABORT_CATCHERS_UNIT - 1) != 0 {
        try_raise_polars_abort_slow()
    }
}

#[inline(never)]
#[cold]
fn try_raise_polars_abort_slow() {
    let state = ABORT_STATE.load(Ordering::Acquire);
    if state & ABORT_KEYBOARD_INTERRUPT_BIT != 0 {
        std::panic::panic_any(POLARS_ABORT_KEYBOARD_INTERRUPT_STR);
    } else if state & ABORT_OOC_OUT_OF_DISK_BIT != 0 {
        std::panic::panic_any(POLARS_ABORT_OOC_OUT_OF_DISK_STR);
    } else {
        unreachable!()
    }
}

/// Runs the passed function, catching any query abortions if they occur
/// while running the function.
pub fn catch_polars_abort<R, F: FnOnce() -> R + UnwindSafe>(try_fn: F) -> Result<R, QueryAborted> {
    // Try to register this catcher (or immediately return if there is an
    // uncaught interrupt).
    try_register_catcher()?;
    let ret = catch_unwind(try_fn);
    unregister_catcher();
    ret.map_err(|p| {
        if let Some(reason) = decode_polars_abort(&*p) {
            reason
        } else {
            std::panic::resume_unwind(p)
        }
    })
}

fn try_register_catcher() -> Result<(), QueryAborted> {
    let old_state = ABORT_STATE.fetch_add(ABORT_CATCHERS_UNIT, Ordering::Relaxed);
    if old_state & (ABORT_CATCHERS_UNIT - 1) != 0 {
        unregister_catcher();

        return if old_state & ABORT_KEYBOARD_INTERRUPT_BIT != 0 {
            Err(QueryAborted::KeyboardInterrupt)
        } else if old_state & ABORT_OOC_OUT_OF_DISK_BIT != 0 {
            Err(QueryAborted::OocOutOfDisk)
        } else {
            unreachable!()
        };
    }
    Ok(())
}

fn unregister_catcher() {
    ABORT_STATE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |state| {
            let num_catchers = state >> ABORT_CATCHERS_UNIT.trailing_zeros();
            if num_catchers > 1 {
                Some(state - ABORT_CATCHERS_UNIT)
            } else {
                // Last catcher, clear abort flags.
                Some(0)
            }
        })
        .ok();
}
