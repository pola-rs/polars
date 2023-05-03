//! Patterns are grouped together by order of month, day, year. This is to prevent
//! parsing different orders of dates in a single column.

pub(super) static DATE_D_M_Y: &[&str] = &[
    "%d-%m-%Y", // 31-12-2021
    "%d/%m/%Y", // 31/12/2021
];

pub(super) static DATE_Y_M_D: &[&str] = &[
    "%Y/%m/%d", // 2021/12/31
    "%Y-%m-%d", // 2021-12-31
];

/// NOTE: don't use single letter dates like %F
/// polars parsers does not support them, so it will be slower
pub(super) static DATETIME_D_M_Y: &[&str] = &[
    // --
    // supported by polars' parser
    // ---
    // 31/12/2021 24:58:01
    "%d/%m/%Y %H:%M:%S",
    // 31-12-2021 24:58
    "%d-%m-%Y %H:%M",
    // 31-12-2021 24:58:01
    "%d-%m-%Y %H:%M:%S",
    // 31-04-2021T02:45:55.555000000
    // milliseconds
    "%d-%m-%YT%H:%M:%S.%3f",
    // microseconds
    "%d-%m-%YT%H:%M:%S.%6f",
    // nanoseconds
    "%d-%m-%YT%H:%M:%S.%9f",
    "%d/%m/%Y 00:00:00",
    "%d-%m-%Y 00:00:00",
    // no times
    "%d-%m-%Y",
];

/// NOTE: don't use single letter dates like %F
/// polars parsers does not support them, so it will be slower
pub(super) static DATETIME_Y_M_D: &[&str] = &[
    // ---
    // ISO8601-like, generated via the `iso8601_format_datetime` test fixture
    // ---
    "%Y/%m/%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y/%m/%dT%H%M%S",
    "%Y-%m-%dT%H%M%S",
    "%Y/%m/%dT%H:%M",
    "%Y-%m-%dT%H:%M",
    "%Y/%m/%dT%H%M",
    "%Y-%m-%dT%H%M",
    "%Y/%m/%dT%H:%M:%S.%9f",
    "%Y-%m-%dT%H:%M:%S.%9f",
    "%Y/%m/%dT%H:%M:%S.%6f",
    "%Y-%m-%dT%H:%M:%S.%6f",
    "%Y/%m/%dT%H:%M:%S.%3f",
    "%Y-%m-%dT%H:%M:%S.%3f",
    "%Y/%m/%dT%H%M%S.%9f",
    "%Y-%m-%dT%H%M%S.%9f",
    "%Y/%m/%dT%H%M%S.%6f",
    "%Y-%m-%dT%H%M%S.%6f",
    "%Y/%m/%dT%H%M%S.%3f",
    "%Y-%m-%dT%H%M%S.%3f",
    "%Y/%m/%d",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H%M%S",
    "%Y-%m-%d %H%M%S",
    "%Y/%m/%d %H:%M",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H%M",
    "%Y-%m-%d %H%M",
    "%Y/%m/%d %H:%M:%S.%9f",
    "%Y-%m-%d %H:%M:%S.%9f",
    "%Y/%m/%d %H:%M:%S.%6f",
    "%Y-%m-%d %H:%M:%S.%6f",
    "%Y/%m/%d %H:%M:%S.%3f",
    "%Y-%m-%d %H:%M:%S.%3f",
    "%Y/%m/%d %H%M%S.%9f",
    "%Y-%m-%d %H%M%S.%9f",
    "%Y/%m/%d %H%M%S.%6f",
    "%Y-%m-%d %H%M%S.%6f",
    "%Y/%m/%d %H%M%S.%3f",
    "%Y-%m-%d %H%M%S.%3f",
    // ---
    // other
    // ---
    // we cannot know this one, because polars needs to know
    // the length of the parsed fmt
    // ---
    "%FT%H:%M:%S%.f",
];

pub(super) static DATETIME_Y_M_D_Z: &[&str] = &[
    // ---
    // ISO8601-like, generated via the `iso8601_tz_aware_format_datetime` test fixture
    // ---
    "%Y/%m/%dT%H:%M:%S%#z",
    "%Y-%m-%dT%H:%M:%S%#z",
    "%Y/%m/%dT%H%M%S%#z",
    "%Y-%m-%dT%H%M%S%#z",
    "%Y/%m/%dT%H:%M%#z",
    "%Y-%m-%dT%H:%M%#z",
    "%Y/%m/%dT%H%M%#z",
    "%Y-%m-%dT%H%M%#z",
    "%Y/%m/%dT%H:%M:%S.%9f%#z",
    "%Y-%m-%dT%H:%M:%S.%9f%#z",
    "%Y/%m/%dT%H:%M:%S.%6f%#z",
    "%Y-%m-%dT%H:%M:%S.%6f%#z",
    "%Y/%m/%dT%H:%M:%S.%3f%#z",
    "%Y-%m-%dT%H:%M:%S.%3f%#z",
    "%Y/%m/%dT%H%M%S.%9f%#z",
    "%Y-%m-%dT%H%M%S.%9f%#z",
    "%Y/%m/%dT%H%M%S.%6f%#z",
    "%Y-%m-%dT%H%M%S.%6f%#z",
    "%Y/%m/%dT%H%M%S.%3f%#z",
    "%Y-%m-%dT%H%M%S.%3f%#z",
    "%Y/%m/%d %H:%M:%S%#z",
    "%Y-%m-%d %H:%M:%S%#z",
    "%Y/%m/%d %H%M%S%#z",
    "%Y-%m-%d %H%M%S%#z",
    "%Y/%m/%d %H:%M%#z",
    "%Y-%m-%d %H:%M%#z",
    "%Y/%m/%d %H%M%#z",
    "%Y-%m-%d %H%M%#z",
    "%Y/%m/%d %H:%M:%S.%9f%#z",
    "%Y-%m-%d %H:%M:%S.%9f%#z",
    "%Y/%m/%d %H:%M:%S.%6f%#z",
    "%Y-%m-%d %H:%M:%S.%6f%#z",
    "%Y/%m/%d %H:%M:%S.%3f%#z",
    "%Y-%m-%d %H:%M:%S.%3f%#z",
    "%Y/%m/%d %H%M%S.%9f%#z",
    "%Y-%m-%d %H%M%S.%9f%#z",
    "%Y/%m/%d %H%M%S.%6f%#z",
    "%Y-%m-%d %H%M%S.%6f%#z",
    "%Y/%m/%d %H%M%S.%3f%#z",
    "%Y-%m-%d %H%M%S.%3f%#z",
    // other
    "%+",
];

#[derive(Eq, Hash, PartialEq, Clone, Copy, Debug)]
pub enum Pattern {
    DateDMY,
    DateYMD,
    DatetimeYMD,
    DatetimeDMY,
    DatetimeYMDZ,
}
