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

#[macro_export]
macro_rules! format_list_container {
    ($e:expr, $start:tt, $end:tt) => {{
        use std::fmt::Write;
        let mut out = String::new();
        out.push($start);
        let mut iter = $e.into_iter();
        let mut next = iter.next();

        loop {
            if let Some(val) = next {
                write!(out, "{val}").unwrap();
            };
            next = iter.next();
            if next.is_some() {
                out.push_str(", ")
            } else {
                break;
            }
        }
        out.push($end);
        out
    };};
}

#[macro_export]
macro_rules! format_list {
    ($e:expr) => {{
        use polars_utils::format_list_container;
        format_list_container!($e, '[', ']')
    };};
}

#[macro_export]
macro_rules! format_tuple {
    ($e:expr) => {{
        use polars_utils::format_list_container;
        format_list_container!($e, '(', ')')
    };};
}
