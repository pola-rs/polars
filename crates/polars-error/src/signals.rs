use std::panic::{catch_unwind, UnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};

/// Python hooks SIGINT to instead generate a KeyboardInterrupt exception.
/// So we do the same to try and abort long-running computations and return to
/// Python so that the Python exception can be generated.

pub struct KeyboardInterrupt;

// Bottom bit: interrupt flag.
// Top 63 bits: number of alive interrupt catchers.
static INTERRUPT_STATE: AtomicU64 = AtomicU64::new(0);

pub fn register_polars_keyboard_interrupt_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |p| {
        // Suppress panic output on KeyboardInterrupt.
        if p.payload().downcast_ref::<KeyboardInterrupt>().is_none() {
            default_hook(p);
        }
    }));

    unsafe {
        // SAFETY: we only do an atomic store in the signal handler, which is allowed.
        signal_hook::low_level::register(signal_hook::consts::signal::SIGINT, move || {
            INTERRUPT_STATE.fetch_or(1, Ordering::Relaxed);
        })
        .unwrap();
    }
}

/// Checks if the keyboard interrupt flag is set, and if yes panics with a
/// KeyboardInterrupt. This function is very cheap.
#[inline(always)]
pub fn try_raise_keyboard_interrupt() {
    if INTERRUPT_STATE.load(Ordering::Relaxed) & 1 > 0 {
        try_raise_keyboard_interrupt_slow()
    }
}

#[inline(never)]
#[cold]
fn try_raise_keyboard_interrupt_slow() {
    std::panic::panic_any(KeyboardInterrupt);
}

/// Runs the passed function, catching any KeyboardInterrupts if they occur
/// while running the function.
pub fn catch_keyboard_interrupt<R, F: FnOnce() -> R + UnwindSafe>(
    try_fn: F,
) -> Result<R, KeyboardInterrupt> {
    // Clear any stale interrupts from immediately cancelling the computation,
    // but only if there aren't any other interrupt catchers.
    INTERRUPT_STATE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |state| {
            let num_catchers = state >> 1;
            let interrupt = state & 1;
            if num_catchers > 0 && interrupt != 0 {
                // Interrupt still in progress.
                None
            } else {
                // Clear stale interrupt flag and increment number of catchers.
                Some((state & !1) + 2)
            }
        })
        .map_err(|_| KeyboardInterrupt)?;

    catch_unwind(try_fn).map_err(|p| {
        INTERRUPT_STATE.fetch_sub(2, Ordering::Relaxed);
        match p.downcast::<KeyboardInterrupt>() {
            Ok(_) => KeyboardInterrupt,
            Err(p) => std::panic::resume_unwind(p),
        }
    })
}
