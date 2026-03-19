use std::cell::RefCell;
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};

use polars::prelude::PolarsError;
use polars_core::error::{to_compute_err, PolarsResult};
/// Gives the caller extra information on how to execute the expression.
pub use polars_ffi::version_0::CallerContext;
pub use pyo3_polars_derive::polars_expr;
use serde::Deserialize;

/// A default opaque kwargs type.
pub type DefaultKwargs = serde_pickle::Value;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

pub fn _parse_kwargs<'a, T>(kwargs: &'a [u8]) -> PolarsResult<T>
where
    T: Deserialize<'a>,
{
    serde_pickle::from_slice(kwargs, Default::default()).map_err(to_compute_err)
}

pub fn _update_last_error(err: PolarsError) {
    let msg = format!("{err}");
    let msg = CString::new(msg).unwrap();
    LAST_ERROR.with(|prev| *prev.borrow_mut() = msg)
}

pub fn _set_panic() {
    let msg = "PANIC";
    let msg = CString::new(msg).unwrap();
    LAST_ERROR.with(|prev| *prev.borrow_mut() = msg)
}

#[no_mangle]
/// # Safety
/// FFI function, so unsafe
pub unsafe extern "C" fn _polars_plugin_get_last_error_message() -> *const std::os::raw::c_char {
    LAST_ERROR.with(|prev| prev.borrow_mut().as_ptr())
}

static INIT: AtomicBool = AtomicBool::new(false);

fn start_up_init() {
    // Set a custom panic hook that only shows output if verbose.
    std::panic::set_hook(Box::new(|info| {
        if polars_config::config().verbose() {
            eprintln!("{info}")
        }
    }));
}

#[no_mangle]
/// # Safety
/// FFI function, so unsafe
pub unsafe extern "C" fn _polars_plugin_get_version() -> u32 {
    if !INIT.swap(true, Ordering::Relaxed) {
        // Plugin version is is always called at least once.
        start_up_init();
    }
    let (major, minor) = polars_ffi::get_version();
    // Stack bits together
    ((major as u32) << 16) + minor as u32
}
