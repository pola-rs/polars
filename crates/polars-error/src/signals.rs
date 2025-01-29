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

    // WASM doesn't support signals, so we just skip installing the hook there.
    #[cfg(not(target_family = "wasm"))]
    unsafe {
        // SAFETY: we only do an atomic op in the signal handler, which is allowed.
        signal_hook::low_level::register(signal_hook::consts::signal::SIGINT, move || {
            // Set the interrupt flag, but only if there are active catchers.
            INTERRUPT_STATE
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |state| {
                    let num_catchers = state >> 1;
                    if num_catchers > 0 {
                        Some(state | 1)
                    } else {
                        None
                    }
                })
                .ok();
        })
        .unwrap();
    }
}

/// Checks if the keyboard interrupt flag is set, and if yes panics with a
/// KeyboardInterrupt. This function is very cheap.
#[inline(always)]
pub fn try_raise_keyboard_interrupt() {
    if INTERRUPT_STATE.load(Ordering::Relaxed) & 1 != 0 {
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
    // Try to register this catcher (or immediately return if there is an
    // uncaught interrupt).
    try_register_catcher()?;
    let ret = catch_unwind(try_fn);
    unregister_catcher();
    ret.map_err(|p| match p.downcast::<KeyboardInterrupt>() {
        Ok(_) => KeyboardInterrupt,
        Err(p) => std::panic::resume_unwind(p),
    })
}

fn try_register_catcher() -> Result<(), KeyboardInterrupt> {
    let old_state = INTERRUPT_STATE.fetch_add(2, Ordering::Relaxed);
    if old_state & 1 != 0 {
        unregister_catcher();
        return Err(KeyboardInterrupt);
    }
    Ok(())
}

fn unregister_catcher() {
    INTERRUPT_STATE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |state| {
            let num_catchers = state >> 1;
            if num_catchers > 1 {
                Some(state - 2)
            } else {
                // Last catcher, clear interrupt flag.
                Some(0)
            }
        })
        .ok();
}
