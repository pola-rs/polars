use crate::PolarsResult;

type SignalsFunction = fn() -> PolarsResult<()>;
static mut SIGNALS_FUNCTION: Option<SignalsFunction> = None;

/// Set the function that will be called check_signals.
/// This can be set on startup to enable stopping a query when user input like `ctrl-c` is called.
///
/// # Safety
/// The caller must ensure there is no other thread accessing this function
/// or calling `check_signals`.
pub unsafe fn set_signals_function(function: SignalsFunction) {
    SIGNALS_FUNCTION = Some(function)
}

fn default() -> PolarsResult<()> {
    Ok(())
}

pub fn check_signals() -> PolarsResult<()> {
    let f = unsafe { SIGNALS_FUNCTION.unwrap_or(default) };
    f()
}
