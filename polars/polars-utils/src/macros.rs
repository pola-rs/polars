#[macro_export]
macro_rules! matches_any_order {
    ($expression1:expr, $expression2:expr,  $( $pattern1:pat_param )|+,  $( $pattern2:pat_param )|+) => {
        (matches!($expression1, $( $pattern1 )|+) && matches!($expression2, $( $pattern2)|+)) ||
        matches!($expression2, $( $pattern1 ) |+) && matches!($expression1, $( $pattern2)|+)
    }
}
