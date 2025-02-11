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

#[macro_export]
macro_rules! format_list_container_truncated {
    ($e:expr, $start:tt, $end:tt, $max:expr, $quote:expr) => {{
        use std::fmt::Write;
        let mut out = String::new();
        out.push($start);
        let mut iter = $e.into_iter();
        let mut next = iter.next();

        let mut count = 0;
        loop {
            if $max == count {
                write!(out, "...").unwrap();
                break;
            }
            count += 1;

            if let Some(val) = next {
                write!(out, "{}{}{}", $quote, val, $quote).unwrap();
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
macro_rules! format_list_truncated {
    ($e:expr, $max:expr) => {{
        use polars_utils::format_list_container_truncated;
        format_list_container_truncated!($e, '[', ']', $max, "")
    };};
    ($e:expr, $max:expr, $quote:expr) => {{
        use polars_utils::format_list_container_truncated;
        format_list_container_truncated!($e, '[', ']', $max, $quote)
    };};
}
