#[macro_export]
macro_rules! format_smartstring {
    ($($arg:tt)*) => {{
        use smartstring::alias::String as SmartString;
        use std::fmt::Write;

        let mut string = SmartString::new();
        write!(string, $($arg)*).unwrap();
        string
    }}
}
