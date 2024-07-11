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
macro_rules! no_call_const {
    () => {{
        const { assert!(false, "should not be called") }
        unreachable!()
    }};
}
