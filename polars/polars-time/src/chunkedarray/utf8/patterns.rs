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

/// NOTE: don't use single letter dates like %F
/// polars parsers does not support them, so it will be slower
pub(super) static DATETIME_D_M_Y: &[&str] = &[
    // --
    // supported by polars' parser
    // ---
    // 31/12/21 12:54:48
    "%d/%m/%y %H:%M:%S",
    // 31-12-2021 24:58:01
    "%d-%m-%Y %H:%M:%S",
    // 31-04-2021T02:45:55.555000000
    // milliseconds
    "%d-%m-%YT%H:%M:%S.%3f",
    "%d-%m-%yT%H:%M:%S.%3f",
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
    // 31/12/2021 11:54:48 PM
    "%d/%m/%Y %I:%M:%S %p",
    "%d-%m-%Y %I:%M:%S %p",
    // 31/12/2021 11:54 PM
    "%d/%m/%Y %I:%M %p",
    "%d-%m-%Y %I:%M %p",
];

/// NOTE: don't use single letter dates like %F
/// polars parsers does not support them, so it will be slower
pub(super) static DATETIME_Y_M_D: &[&str] = &[
    // ---
    // ISO8601, generated via the `iso8601_format` test fixture
    // ---
    "%Y/%m/%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y%m%dT%H:%M:%S",
    "%Y/%m/%dT%H%M%S",
    "%Y-%m-%dT%H%M%S",
    "%Y%m%dT%H%M%S",
    "%Y/%m/%dT%H:%M",
    "%Y-%m-%dT%H:%M",
    "%Y%m%dT%H:%M",
    "%Y/%m/%dT%H%M",
    "%Y-%m-%dT%H%M",
    "%Y%m%dT%H%M",
    "%Y/%m/%dT%H:%M:%S.%9f",
    "%Y-%m-%dT%H:%M:%S.%9f",
    "%Y%m%dT%H:%M:%S.%9f",
    "%Y/%m/%dT%H:%M:%S.%6f",
    "%Y-%m-%dT%H:%M:%S.%6f",
    "%Y%m%dT%H:%M:%S.%6f",
    "%Y/%m/%dT%H:%M:%S.%3f",
    "%Y-%m-%dT%H:%M:%S.%3f",
    "%Y%m%dT%H:%M:%S.%3f",
    "%Y/%m/%dT%H%M%S.%9f",
    "%Y-%m-%dT%H%M%S.%9f",
    "%Y%m%dT%H%M%S.%9f",
    "%Y/%m/%dT%H%M%S.%6f",
    "%Y-%m-%dT%H%M%S.%6f",
    "%Y%m%dT%H%M%S.%6f",
    "%Y/%m/%dT%H%M%S.%3f",
    "%Y-%m-%dT%H%M%S.%3f",
    "%Y%m%dT%H%M%S.%3f",
    "%Y/%m/%d",
    "%Y-%m-%d",
    "%Y%m%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y%m%d %H:%M:%S",
    "%Y/%m/%d %H%M%S",
    "%Y-%m-%d %H%M%S",
    "%Y%m%d %H%M%S",
    "%Y/%m/%d %H:%M",
    "%Y-%m-%d %H:%M",
    "%Y%m%d %H:%M",
    "%Y/%m/%d %H%M",
    "%Y-%m-%d %H%M",
    "%Y%m%d %H%M",
    "%Y/%m/%d %H:%M:%S.%9f",
    "%Y-%m-%d %H:%M:%S.%9f",
    "%Y%m%d %H:%M:%S.%9f",
    "%Y/%m/%d %H:%M:%S.%6f",
    "%Y-%m-%d %H:%M:%S.%6f",
    "%Y%m%d %H:%M:%S.%6f",
    "%Y/%m/%d %H:%M:%S.%3f",
    "%Y-%m-%d %H:%M:%S.%3f",
    "%Y%m%d %H:%M:%S.%3f",
    "%Y/%m/%d %H%M%S.%9f",
    "%Y-%m-%d %H%M%S.%9f",
    "%Y%m%d %H%M%S.%9f",
    "%Y/%m/%d %H%M%S.%6f",
    "%Y-%m-%d %H%M%S.%6f",
    "%Y%m%d %H%M%S.%6f",
    "%Y/%m/%d %H%M%S.%3f",
    "%Y-%m-%d %H%M%S.%3f",
    "%Y%m%d %H%M%S.%3f",
    // ---
    // other
    // ---
    // 21/12/31 12:54:48
    "%y/%m/%d %H:%M:%S",
    // 21/12/31 24:58:01
    "%y%m%d %H:%M:%S",
    // 2021/12/31 11:54:48 PM
    "%Y/%m/%d %I:%M:%S %p",
    "%Y-%m-%d %I:%M:%S %p",
    // 2021/12/31 11:54 PM
    "%Y/%m/%d %I:%M %p",
    "%Y-%m-%d %I:%M %p",
    // ---
    // we cannot know this one, because polars needs to know
    // the length of the parsed fmt
    // ---
    "%FT%H:%M:%S%.f",
];

#[derive(Eq, Hash, PartialEq, Clone, Copy, Debug)]
pub enum Pattern {
    DateDMY,
    DateYMD,
    DatetimeYMD,
    DatetimeDMY,
}
