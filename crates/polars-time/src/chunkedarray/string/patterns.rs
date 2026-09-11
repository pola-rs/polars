//! Patterns are grouped together by order of month, day, year. This is to prevent
//! parsing different orders of dates in a single column.

pub(super) static DATE_D_M_Y: &[&str] = &["%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y"];

pub(super) static DATE_Y_M_D: &[&str] = &[
    "%Y-%m-%d", // 2021-12-31
    "%Y/%m/%d", // 2021/12/31
    "%Y.%m.%d", // 2021.12.31
];

/// NOTE: don't use single letter dates like %F
/// polars parsers does not support them, so it will be slower
pub(super) static DATETIME_D_M_Y: &[&str] = &[
    "%d-%m-%YT%H:%M:%S%.f",
    "%d-%m-%YT%H%M%S%.f",
    "%d-%m-%YT%H:%M",
    "%d-%m-%YT%H%M",
    "%d-%m-%Y %H:%M:%S%.f",
    "%d-%m-%Y %H%M%S%.f",
    "%d-%m-%Y %H:%M",
    "%d-%m-%Y %H%M",
    "%d-%m-%Y",
    "%d/%m/%YT%H:%M:%S%.f",
    "%d/%m/%YT%H%M%S%.f",
    "%d/%m/%YT%H:%M",
    "%d/%m/%YT%H%M",
    "%d/%m/%Y %H:%M:%S%.f",
    "%d/%m/%Y %H%M%S%.f",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y %H%M",
    "%d/%m/%Y",
    "%d.%m.%YT%H:%M:%S%.f",
    "%d.%m.%YT%H%M%S%.f",
    "%d.%m.%YT%H:%M",
    "%d.%m.%YT%H%M",
    "%d.%m.%Y %H:%M:%S%.f",
    "%d.%m.%Y %H%M%S%.f",
    "%d.%m.%Y %H:%M",
    "%d.%m.%Y %H%M",
    "%d.%m.%Y",
];

/// NOTE: don't use single letter dates like %F
/// polars parsers does not support them, so it will be slower
pub(super) static DATETIME_Y_M_D: &[&str] = &[
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%dT%H%M%S%.f",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%dT%H%M",
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%d %H%M%S%.f",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H%M",
    "%Y-%m-%d",
    "%Y/%m/%dT%H:%M:%S%.f",
    "%Y/%m/%dT%H%M%S%.f",
    "%Y/%m/%dT%H:%M",
    "%Y/%m/%dT%H%M",
    "%Y/%m/%d %H:%M:%S%.f",
    "%Y/%m/%d %H%M%S%.f",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d %H%M",
    "%Y/%m/%d",
    "%Y.%m.%dT%H:%M:%S%.f",
    "%Y.%m.%dT%H%M%S%.f",
    "%Y.%m.%dT%H:%M",
    "%Y.%m.%dT%H%M",
    "%Y.%m.%d %H:%M:%S%.f",
    "%Y.%m.%d %H%M%S%.f",
    "%Y.%m.%d %H:%M",
    "%Y.%m.%d %H%M",
    "%Y.%m.%d",
    "%Y%m%dT%H%M%S%.f",     // Compact ISO 8601.
    "%Y-%m-%dT%H:%M:%S%.f", // ISO 8601 with dynamic precision and without timezone
];

// Each entry below is listed once per offset colon-style that's actually common
// in practice (`%#z`/no-colon, `%:z`, `%:::z`) to keep inference working for
// them - `%::z` (an offset with a nonzero seconds component) is left out, as
// it's vanishingly rare in real-world data. A literal `Z` variant is also
// listed for each: jiff's strtime engine matches a literal `Z` as plain text
// (not as any `%z`-family directive, none of which accept a bare `Z`), which
// leaves the parsed `BrokenDownTime` without an offset - the same "no
// offset matched" fallback below (`parse_tz_aware_timestamp`) then treats it
// as UTC, same as `strptime`-native RFC 3339 parsers do for a `Z` suffix.
pub(super) static DATETIME_Y_M_D_Z: &[&str] = &[
    "%Y-%m-%dT%H:%M:%S%.f%#z",
    "%Y-%m-%dT%H:%M:%S%.f%:z",
    "%Y-%m-%dT%H:%M:%S%.f%:::z",
    "%Y-%m-%dT%H:%M:%S%.fZ",
    "%Y-%m-%dT%H%M%S%.f%#z",
    "%Y-%m-%dT%H%M%S%.f%:z",
    "%Y-%m-%dT%H%M%S%.f%:::z",
    "%Y-%m-%dT%H%M%S%.fZ",
    "%Y-%m-%dT%H:%M%#z",
    "%Y-%m-%dT%H:%M%:z",
    "%Y-%m-%dT%H:%M%:::z",
    "%Y-%m-%dT%H:%MZ",
    "%Y-%m-%dT%H%M%#z",
    "%Y-%m-%dT%H%M%:z",
    "%Y-%m-%dT%H%M%:::z",
    "%Y-%m-%dT%H%MZ",
    "%Y-%m-%d %H:%M:%S%.f%#z",
    "%Y-%m-%d %H:%M:%S%.f%:z",
    "%Y-%m-%d %H:%M:%S%.f%:::z",
    "%Y-%m-%d %H:%M:%S%.fZ",
    "%Y-%m-%d %H%M%S%.f%#z",
    "%Y-%m-%d %H%M%S%.f%:z",
    "%Y-%m-%d %H%M%S%.f%:::z",
    "%Y-%m-%d %H%M%S%.fZ",
    "%Y-%m-%d %H:%M%#z",
    "%Y-%m-%d %H:%M%:z",
    "%Y-%m-%d %H:%M%:::z",
    "%Y-%m-%d %H:%MZ",
    "%Y-%m-%d %H%M%#z",
    "%Y-%m-%d %H%M%:z",
    "%Y-%m-%d %H%M%:::z",
    "%Y-%m-%d %H%MZ",
    "%Y/%m/%dT%H:%M:%S%.f%#z",
    "%Y/%m/%dT%H:%M:%S%.f%:z",
    "%Y/%m/%dT%H:%M:%S%.f%:::z",
    "%Y/%m/%dT%H:%M:%S%.fZ",
    "%Y/%m/%dT%H%M%S%.f%#z",
    "%Y/%m/%dT%H%M%S%.f%:z",
    "%Y/%m/%dT%H%M%S%.f%:::z",
    "%Y/%m/%dT%H%M%S%.fZ",
    "%Y/%m/%dT%H:%M%#z",
    "%Y/%m/%dT%H:%M%:z",
    "%Y/%m/%dT%H:%M%:::z",
    "%Y/%m/%dT%H:%MZ",
    "%Y/%m/%dT%H%M%#z",
    "%Y/%m/%dT%H%M%:z",
    "%Y/%m/%dT%H%M%:::z",
    "%Y/%m/%dT%H%MZ",
    "%Y/%m/%d %H:%M:%S%.f%#z",
    "%Y/%m/%d %H:%M:%S%.f%:z",
    "%Y/%m/%d %H:%M:%S%.f%:::z",
    "%Y/%m/%d %H:%M:%S%.fZ",
    "%Y/%m/%d %H%M%S%.f%#z",
    "%Y/%m/%d %H%M%S%.f%:z",
    "%Y/%m/%d %H%M%S%.f%:::z",
    "%Y/%m/%d %H%M%S%.fZ",
    "%Y/%m/%d %H:%M%#z",
    "%Y/%m/%d %H:%M%:z",
    "%Y/%m/%d %H:%M%:::z",
    "%Y/%m/%d %H:%MZ",
    "%Y/%m/%d %H%M%#z",
    "%Y/%m/%d %H%M%:z",
    "%Y/%m/%d %H%M%:::z",
    "%Y/%m/%d %H%MZ",
    "%Y.%m.%dT%H:%M:%S%.f%#z",
    "%Y.%m.%dT%H:%M:%S%.f%:z",
    "%Y.%m.%dT%H:%M:%S%.f%:::z",
    "%Y.%m.%dT%H:%M:%S%.fZ",
    "%Y.%m.%dT%H%M%S%.f%#z",
    "%Y.%m.%dT%H%M%S%.f%:z",
    "%Y.%m.%dT%H%M%S%.f%:::z",
    "%Y.%m.%dT%H%M%S%.fZ",
    "%Y.%m.%dT%H:%M%#z",
    "%Y.%m.%dT%H:%M%:z",
    "%Y.%m.%dT%H:%M%:::z",
    "%Y.%m.%dT%H:%MZ",
    "%Y.%m.%dT%H%M%#z",
    "%Y.%m.%dT%H%M%:z",
    "%Y.%m.%dT%H%M%:::z",
    "%Y.%m.%dT%H%MZ",
    "%Y.%m.%d %H:%M:%S%.f%#z",
    "%Y.%m.%d %H:%M:%S%.f%:z",
    "%Y.%m.%d %H:%M:%S%.f%:::z",
    "%Y.%m.%d %H:%M:%S%.fZ",
    "%Y.%m.%d %H%M%S%.f%#z",
    "%Y.%m.%d %H%M%S%.f%:z",
    "%Y.%m.%d %H%M%S%.f%:::z",
    "%Y.%m.%d %H%M%S%.fZ",
    "%Y.%m.%d %H:%M%#z",
    "%Y.%m.%d %H:%M%:z",
    "%Y.%m.%d %H:%M%:::z",
    "%Y.%m.%d %H:%MZ",
    "%Y.%m.%d %H%M%#z",
    "%Y.%m.%d %H%M%:z",
    "%Y.%m.%d %H%M%:::z",
    "%Y.%m.%d %H%MZ",
    "%Y%m%dT%H%M%S%.f%#z",   // Compact ISO 8601.
    "%Y%m%dT%H%M%S%.f%:z",   // Compact ISO 8601.
    "%Y%m%dT%H%M%S%.f%:::z", // Compact ISO 8601.
    "%Y%m%dT%H%M%S%.fZ",     // Compact ISO 8601.
];

pub(super) static TIME_H_M_S: &[&str] = &["%T%.f", "%R"];

#[derive(Eq, Hash, PartialEq, Clone, Copy, Debug)]
pub enum Pattern {
    DateDMY,
    DateYMD,
    DatetimeYMD,
    DatetimeDMY,
    DatetimeYMDZ,
    Time,
}
