pub(crate) const DAYS_PER_MONTH: [[i64; 12]; 2] = [
    //J   F   M   A   M   J   J   A   S   O   N   D
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], // non-leap year
    [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], // leap year
];

pub(crate) const fn is_leap_year(year: i32) -> bool {
    year % 400 == 0 || (year % 4 == 0 && year % 100 != 0)
}
/// nanoseconds per unit
pub const NS_MICROSECOND: i64 = 1_000;
pub const NS_MILLISECOND: i64 = 1_000_000;
pub const NS_SECOND: i64 = 1_000_000_000;
pub const NS_MINUTE: i64 = 60 * NS_SECOND;
pub const NS_HOUR: i64 = 60 * NS_MINUTE;
pub const NS_DAY: i64 = 24 * NS_HOUR;
pub const NS_WEEK: i64 = 7 * NS_DAY;
