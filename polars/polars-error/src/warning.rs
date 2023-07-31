type WarningFunction = fn(&str);
static mut WARNING_FUNCTION: Option<WarningFunction> = None;

/// Set the function that will be called by the `polars_warn!` macro.
/// You can use this to set logging in polars.
///
/// # Safety
/// The caller must ensure there is no other thread accessing this function
/// or calling `polars_warn!`.
pub unsafe fn set_warning_function(function: WarningFunction) {
    WARNING_FUNCTION = Some(function)
}

fn eprintln(fmt: &str) {
    eprintln!("{}", fmt);
}

pub fn get_warning_function() -> WarningFunction {
    unsafe { WARNING_FUNCTION.unwrap_or(eprintln) }
}
#[macro_export]
macro_rules! polars_warn {
    ($fmt:literal, $($arg:tt)+) => {
        {{
        let func = $crate::get_warning_function();
        func(format!($fmt, $($arg)+).as_ref())
        }}
    };
    ($($arg:tt)+) => {
        polars_warn!("{}", $($arg)+);
    };
}
