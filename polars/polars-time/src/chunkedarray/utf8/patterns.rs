//! Patterns are grouped together by order of month, day, year. This is to prevent
//! parsing different orders of dates in a single column.

pub(super) static DATE_D_M_Y: &[&str] = &[
    // 8-Jul-2001
    "%v",       // 31-12-2021
    "%d-%m-%Y", // 31-12-21
    "%d-%m-%y", // 31_12_2021
    "%d_%m_%Y", // 31_12_21
    "%d_%m_%y",
];

pub(super) static DATE_Y_M_D: &[&str] = &[
    // 2021-12-31
    "%Y-%m-%d", // 21-12-21
    "%y-%m-%d", // 2021_12_31
    "%Y_%m_%d", // 21_12_21
    "%y_%m_%d",
];

pub(super) static DATETIME_D_M_Y: &[&str] = &[
    // 31/12/21 12:54:98
    "%d/%m/%y %H:%M:%S",
    // 31-12-2021 24:58:01
    "%d-%m-%Y %H:%M:%S",
    // 31-04-2021T02:45:55.555000000
    // microseconds
    "%d-%m-%YT%H:%M:%S.%6f",
    // 31-04-21T02:45:55.555000000
    "%d-%m-%yT%H:%M:%S.%6f",
    // nanoseconds
    "%d-%m-%YT%H:%M:%S.%9f",
    "%d-%m-%yT%H:%M:%S.%9f",
    "%d/%m/%Y 00:00:00",
    "%d-%m-%Y 00:00:00",
    // no times
    "%d-%m-%Y",
    "%d-%m-%y",
];

pub(super) static DATETIME_Y_M_D: &[&str] = &[
    // 21/12/31 12:54:98
    "%y/%m/%d %H:%M:%S",
    // 2021-12-31 24:58:01
    "%Y-%m-%d %H:%M:%S",
    // 21/12/31 24:58:01
    "%y/%m/%d %H:%M:%S",
    //210319 23:58:50
    "%y%m%d %H:%M:%S",
    // 2019-04-18T02:45:55
    // 2021/12/31 12:54:98
    "%Y/%m/%d %H:%M:%S",
    // 2021-12-31 24:58:01
    "%Y-%m-%d %H:%M:%S",
    // 2021/12/31 24:58:01
    "%Y/%m/%d %H:%M:%S",
    // 20210319 23:58:50
    "%Y%m%d %H:%M:%S",
    // 2019-04-18T02:45:55
    "%FT%H:%M:%S",
    // 2019-04-18T02:45:55.555000000
    // microseconds
    "%FT%H:%M:%S.%6f",
    // nanoseconds
    "%FT%H:%M:%S.%9f",
    // no times
    "%F",
    "%Y/%m/%d",
];

#[derive(Eq, Hash, PartialEq, Clone, Copy, Debug)]
pub enum Pattern {
    DateDMY,
    DateYMD,
    DatetimeYMD,
    DatetimeDMY,
}
