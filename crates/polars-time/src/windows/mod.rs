// Ported and adapted from influxdb.
// Credits to their work.
// https://github.com/influxdata/influxdb_iox/blob/main/query/src/func/window/internal.rs
// https://github.com/influxdata/flux/blob/3d6c47d9113fe0d919ddd3d4eef242dfc38ab2fb/interval/window.go
// https://github.com/influxdata/flux/blob/1e9bfd49f21c0e679b42acf6fc515ce05c6dec2b/values/time.go#L40

pub(crate) mod bounds;
pub(crate) mod calendar;
pub(crate) mod duration;
pub(crate) mod group_by;
#[cfg(test)]
mod test;
pub(crate) mod window;
