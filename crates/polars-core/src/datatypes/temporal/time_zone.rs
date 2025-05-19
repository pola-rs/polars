use polars_error::PolarsResult;
use polars_utils::pl_str::PlSmallStr;

use crate::config;

#[derive(Clone, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimeZone {
    /// Private inner to ensure canonical / parsed time zone repr at construction.
    inner: PlSmallStr,
}

impl TimeZone {
    pub const UTC: TimeZone = unsafe { TimeZone::from_static("UTC") };

    /// Construct from a static string.
    ///
    /// # Safety
    /// This does not perform any validation, the caller is responsible for
    /// ensuring they pass a valid timezone.
    #[inline(always)]
    pub const unsafe fn from_static(tz: &'static str) -> Self {
        Self {
            inner: PlSmallStr::from_static(tz),
        }
    }

    /// # Safety
    /// This does not perform any validation, the caller is responsible for
    /// ensuring they pass a valid timezone.
    pub unsafe fn new_unchecked(zone_str: impl Into<PlSmallStr>) -> Self {
        Self {
            inner: zone_str.into(),
        }
    }

    /// Converts timezones to canonical form.
    ///
    /// If the "timezones" feature is enabled, additionally performs validation and converts to
    /// Etc/GMT form where applicable.
    #[inline]
    pub fn opt_try_new(zone_str: Option<impl Into<PlSmallStr>>) -> PolarsResult<Option<Self>> {
        Self::new_impl(zone_str.map(|x| x.into()))
    }

    fn new_impl(zone_str: Option<PlSmallStr>) -> PolarsResult<Option<Self>> {
        // Needed for selectors https://github.com/pola-rs/polars/pull/9641
        if zone_str.as_deref() == Some("*") {
            return Ok(Some(Self {
                inner: PlSmallStr::from_static("*"),
            }));
        }

        let mut canonical_tz = Self::_canonical_timezone_impl(zone_str);

        #[cfg(feature = "timezones")]
        if let Some(tz) = canonical_tz.as_mut() {
            if Self::validate_time_zone(tz).is_err() {
                match parse_fixed_offset(tz) {
                    Ok(v) => *tz = v,
                    Err(err) => {
                        // This can be used if there are externally created arrow buffers / dtypes
                        // with unknown timezones.
                        if std::env::var("POLARS_IGNORE_TIMEZONE_PARSE_ERROR").as_deref() == Ok("1")
                        {
                            if config::verbose() {
                                eprintln!("WARN: {}", err)
                            }
                        } else {
                            return Err(err.wrap_msg(|s| {
                                format!(
                                    "{}. If you would like to forcibly disable \
                                    timezone validation, set \
                                    POLARS_IGNORE_TIMEZONE_PARSE_ERROR=1.",
                                    s
                                )
                            }));
                        }
                    },
                }
            }
        }

        Ok(canonical_tz.map(|inner| Self { inner }))
    }

    pub fn eq_wildcard_aware(this: &Self, other: &Self) -> bool {
        this == other || this.inner == "*" || other.inner == "*"
    }

    /// Equality where `None` is treated as UTC.
    pub fn eq_none_as_utc(this: Option<&TimeZone>, other: Option<&TimeZone>) -> bool {
        this.unwrap_or(&Self::UTC) == other.unwrap_or(&Self::UTC)
    }

    pub fn _canonical_timezone_impl(tz: Option<PlSmallStr>) -> Option<PlSmallStr> {
        match tz.as_deref() {
            Some("") | None => None,
            #[cfg(feature = "timezones")]
            Some("+00:00") | Some("00:00") | Some("utc") => Some(PlSmallStr::from_static("UTC")),
            Some(_) => tz,
        }
    }

    #[cfg(feature = "timezones")]
    pub fn from_chrono(tz: &chrono_tz::Tz) -> Self {
        use polars_utils::format_pl_smallstr;

        Self {
            inner: format_pl_smallstr!("{}", tz),
        }
    }

    #[cfg(feature = "timezones")]
    pub fn to_chrono(&self) -> PolarsResult<chrono_tz::Tz> {
        parse_time_zone(self)
    }

    #[cfg(feature = "timezones")]
    pub fn validate_time_zone(tz: &str) -> PolarsResult<()> {
        parse_time_zone(tz).map(|_| ())
    }
}

impl std::ops::Deref for TimeZone {
    type Target = PlSmallStr;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::fmt::Debug for TimeZone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.inner, f)
    }
}

impl std::fmt::Display for TimeZone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, f)
    }
}

#[cfg(feature = "timezones")]
static FIXED_OFFSET_PATTERN: &str = r#"(?x)
    ^
    (?P<sign>[-+])?            # optional sign
    (?P<hour>0[0-9]|1[0-4])    # hour (between 0 and 14)
    :?                         # optional separator
    00                         # minute
    $
    "#;

#[cfg(feature = "timezones")]
polars_utils::regex_cache::cached_regex! {
    static FIXED_OFFSET_RE = FIXED_OFFSET_PATTERN;
}

/// Parse a time zone string to [`chrono_tz::Tz`]
#[cfg(feature = "timezones")]
pub fn parse_time_zone(tz: &str) -> PolarsResult<chrono_tz::Tz> {
    match tz.parse::<chrono_tz::Tz>() {
        Ok(tz) => Ok(tz),
        Err(_) => unable_to_parse_err(tz),
    }
}

/// Convert fixed offset to Etc/GMT one from time zone database
///
/// E.g. +01:00 -> Etc/GMT-1
///
/// Note: the sign appears reversed, but is correct, see <https://en.wikipedia.org/wiki/Tz_database#Area>:
/// > In order to conform with the POSIX style, those zone names beginning with
/// > "Etc/GMT" have their sign reversed from the standard ISO 8601 convention.
/// > In the "Etc" area, zones west of GMT have a positive sign and those east
/// > have a negative sign in their name (e.g "Etc/GMT-14" is 14 hours ahead of GMT).
#[cfg(feature = "timezones")]
pub fn parse_fixed_offset(tz: &str) -> PolarsResult<PlSmallStr> {
    use polars_utils::format_pl_smallstr;

    if let Some(caps) = FIXED_OFFSET_RE.captures(tz) {
        let sign = match caps.name("sign").map(|s| s.as_str()) {
            Some("-") => "+",
            _ => "-",
        };
        let hour = caps.name("hour").unwrap().as_str().parse::<i32>().unwrap();
        let etc_tz = format_pl_smallstr!("Etc/GMT{}{}", sign, hour);
        if etc_tz.parse::<chrono_tz::Tz>().is_ok() {
            return Ok(etc_tz);
        }
    }

    unable_to_parse_err(tz)
}

#[cfg(feature = "timezones")]
fn unable_to_parse_err<T>(tz: &str) -> PolarsResult<T> {
    use polars_error::polars_bail;

    polars_bail!(
        ComputeError:
        "unable to parse time zone: '{}'. Please check the \
        Time Zone Database for a list of available time zones.",
        tz
    )
}
