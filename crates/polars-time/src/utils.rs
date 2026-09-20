#[cfg(feature = "timezones")]
pub fn known_timezones() -> Vec<String> {
    jiff::tz::db()
        .available()
        .map(|name| name.as_str().to_string())
        .collect()
}
