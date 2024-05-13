#[macro_export]
macro_rules! matches_any_order {
    ($expression1:expr, $expression2:expr,  $( $pattern1:pat_param )|+,  $( $pattern2:pat_param )|+) => {
        (matches!($expression1, $( $pattern1 )|+) && matches!($expression2, $( $pattern2)|+)) ||
        matches!($expression2, $( $pattern1 ) |+) && matches!($expression1, $( $pattern2)|+)
    }
}

#[macro_export]
macro_rules! unreachable_unchecked_release {
    ($($arg:tt)*) => {
        if cfg!(debug_assertions) {
            unreachable!()
        } else {
            unreachable_unchecked()
        }
    };
}

#[macro_export]
macro_rules! format_list {
    ($e:expr) => {{
        use std::fmt::Write;
        let mut out = String::new();
        out.push('[');
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
        out.push_str("]\n");
        out
    };};
}
