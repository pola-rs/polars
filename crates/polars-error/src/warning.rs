use parking_lot::RwLock;

type WarningFunction = fn(&str, PolarsWarning);
static WARNING_FUNCTION: RwLock<WarningFunction> = RwLock::new(eprintln);

fn eprintln(fmt: &str, warning: PolarsWarning) {
    eprintln!("{:?}: {}", warning, fmt);
}

/// Set the function that will be called by the `polars_warn!` macro.
/// You can use this to set logging in polars.
pub fn set_warning_function(function: WarningFunction) {
    *WARNING_FUNCTION.write() = function;
}

pub fn get_warning_function() -> WarningFunction {
    *WARNING_FUNCTION.read()
}

#[derive(Debug)]
pub enum PolarsWarning {
    Deprecation,
    UserWarning,
    CategoricalRemappingWarning,
    MapWithoutReturnDtypeWarning,
}

#[macro_export]
macro_rules! polars_warn {
    ($variant:ident, $fmt:literal $(, $arg:tt)*) => {
        {{
        let func = $crate::get_warning_function();
        let warn = $crate::PolarsWarning::$variant;
        func(format!($fmt, $($arg)*).as_ref(), warn)
        }}
    };
    ($fmt:literal, $($arg:tt)+) => {
        {{
        let func = $crate::get_warning_function();
        func(format!($fmt, $($arg)+).as_ref(), $crate::PolarsWarning::UserWarning)
        }}
    };
    ($($arg:tt)+) => {
        polars_warn!("{}", $($arg)+);
    };
}
