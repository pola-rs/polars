pub(crate) const DAYS_PER_MONTH: [[i64; 12]; 2] = [
    //J   F   M   A   M   J   J   A   S   O   N   D
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], // non-leap year
    [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], // leap year
];

pub(crate) const fn is_leap_year(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

/// Get the number of days in the given month of the given year
pub(crate) const fn days_in_month(year: i32, month: u8) -> u8 {
    DAYS_PER_MONTH[is_leap_year(year) as usize][(month - 1) as usize] as u8
}

/// nanoseconds per unit
pub const NS_MICROSECOND: i64 = 1_000;
pub const NS_MILLISECOND: i64 = 1_000_000;
pub const NS_SECOND: i64 = 1_000_000_000;
pub const NS_MINUTE: i64 = 60 * NS_SECOND;
pub const NS_HOUR: i64 = 60 * NS_MINUTE;
pub const NS_DAY: i64 = 24 * NS_HOUR;
pub const NS_WEEK: i64 = 7 * NS_DAY;

/// Not-to-exceed (NTE) nanoseconds per unit, accounting for DST.
/// This is an upper bound. Leap seconds do not matter for correctness.
pub const NTE_NS_DAY: i64 = (24 + 1) * NS_HOUR + 1;
pub const NTE_NS_WEEK: i64 = 6 * NS_DAY + NTE_NS_DAY;
