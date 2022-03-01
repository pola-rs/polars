use crate::prelude::*;
use chrono::prelude::*;
use polars_arrow::export::arrow::temporal_conversions::timestamp_ns_to_datetime;
use polars_core::prelude::*;

#[test]
fn test_date_range() {
    // Test month as interval in date range
    let start = NaiveDate::from_ymd(2022, 1, 1).and_hms(0, 0, 0);
    let end = NaiveDate::from_ymd(2022, 4, 1).and_hms(0, 0, 0);
    let dates = date_range_vec(
        start.timestamp_nanos(),
        end.timestamp_nanos(),
        Duration::parse("1mo"),
        ClosedWindow::Both,
        TimeUnit::Nanoseconds,
    );
    let expected = [
        NaiveDate::from_ymd(2022, 1, 1),
        NaiveDate::from_ymd(2022, 2, 1),
        NaiveDate::from_ymd(2022, 3, 1),
        NaiveDate::from_ymd(2022, 4, 1),
    ]
    .iter()
    .map(|d| d.and_hms(0, 0, 0).timestamp_nanos())
    .collect::<Vec<_>>();
    assert_eq!(dates, expected);
}

#[test]
fn test_feb_date_range() {
    let start = NaiveDate::from_ymd(2022, 2, 1).and_hms(0, 0, 0);
    let end = NaiveDate::from_ymd(2022, 3, 1).and_hms(0, 0, 0);
    let dates = date_range_vec(
        start.timestamp_nanos(),
        end.timestamp_nanos(),
        Duration::parse("1mo"),
        ClosedWindow::Both,
        TimeUnit::Nanoseconds,
    );
    let expected = [
        NaiveDate::from_ymd(2022, 2, 1),
        NaiveDate::from_ymd(2022, 3, 1),
    ]
    .iter()
    .map(|d| d.and_hms(0, 0, 0).timestamp_nanos())
    .collect::<Vec<_>>();
    assert_eq!(dates, expected);
}

fn print_ns(ts: &[i64]) {
    for ts in ts {
        println!("{}", timestamp_ns_to_datetime(*ts));
    }
}

fn take_groups_slice<'a>(groups: &'a GroupsSlice, idx: usize, ts: &'a [i64]) -> &'a [i64] {
    let [first, len] = groups[idx];
    let first = first as usize;
    let len = len as usize;
    &ts[first..first + len]
}

#[test]
fn test_groups_large_interval() {
    let dates = &[
        NaiveDate::from_ymd(2020, 1, 1),
        NaiveDate::from_ymd(2020, 1, 11),
        NaiveDate::from_ymd(2020, 1, 12),
        NaiveDate::from_ymd(2020, 1, 13),
    ];
    let ts = dates
        .iter()
        .map(|d| d.and_hms(0, 0, 0).timestamp_nanos())
        .collect::<Vec<_>>();

    let dur = Duration::parse("2d");
    let w = Window::new(Duration::parse("2d"), dur.clone(), Duration::from_nsecs(0));
    let (groups, _, _) = groupby_windows(w, &ts, false, ClosedWindow::Both, TimeUnit::Nanoseconds);
    assert_eq!(groups.len(), 4);
    assert_eq!(groups[0], [0, 1]);
    assert_eq!(groups[1], [1, 1]);
    assert_eq!(groups[2], [1, 3]);
    assert_eq!(groups[3], [3, 1]);
    let (groups, _, _) = groupby_windows(w, &ts, false, ClosedWindow::Left, TimeUnit::Nanoseconds);
    assert_eq!(groups.len(), 3);
    assert_eq!(groups[2], [3, 1]);
    let (groups, _, _) = groupby_windows(w, &ts, false, ClosedWindow::Right, TimeUnit::Nanoseconds);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[1], [2, 2]);
}

#[test]
fn test_offset() {
    let t = NaiveDate::from_ymd(2020, 1, 2)
        .and_hms(0, 0, 0)
        .timestamp_nanos();
    let w = Window::new(
        Duration::parse("5m"),
        Duration::parse("5m"),
        Duration::parse("-2m"),
    );

    let b = w.get_earliest_bounds_ns(t);
    let start = NaiveDate::from_ymd(2020, 1, 1)
        .and_hms(23, 58, 0)
        .timestamp_nanos();
    assert_eq!(b.start, start);
}

#[test]
fn test_boundaries() {
    let start = NaiveDate::from_ymd(2021, 12, 16).and_hms(0, 0, 0);
    let stop = NaiveDate::from_ymd(2021, 12, 16).and_hms(3, 0, 0);

    let ts = date_range_vec(
        start.timestamp_nanos(),
        stop.timestamp_nanos(),
        Duration::parse("30m"),
        ClosedWindow::Both,
        TimeUnit::Nanoseconds,
    );

    // window:
    // every 2h
    // period 1h
    let w = Window::new(
        Duration::parse("1h"),
        Duration::parse("1h"),
        Duration::parse("0ns"),
    );

    // earliest bound is first datapoint: 2021-12-16 00:00:00
    let b = w.get_earliest_bounds_ns(ts[0]);
    assert_eq!(b.start, start.timestamp_nanos());

    // test closed: "both" (includes both ends of the interval)
    let (groups, lower, higher) =
        groupby_windows(w, &ts, true, ClosedWindow::Both, TimeUnit::Nanoseconds);

    // 1st group
    // expected boundary:
    // 2021-12-16 00:00:00 -> 2021-12-16 01:00:00
    // expected members:
    // 2021-12-16 00:00:00
    // 2021-12-16 00:30:00
    // 2021-12-16 01:00:00
    let g = take_groups_slice(&groups, 0, &ts);
    let t0 = NaiveDate::from_ymd(2021, 12, 16).and_hms(0, 0, 0);
    let t1 = NaiveDate::from_ymd(2021, 12, 16).and_hms(0, 30, 0);
    let t2 = NaiveDate::from_ymd(2021, 12, 16).and_hms(1, 0, 0);
    assert_eq!(
        g,
        &[
            t0.timestamp_nanos(),
            t1.timestamp_nanos(),
            t2.timestamp_nanos()
        ]
    );
    let b_start = NaiveDate::from_ymd(2021, 12, 16).and_hms(0, 0, 0);
    let b_end = NaiveDate::from_ymd(2021, 12, 16).and_hms(1, 0, 0);
    assert_eq!(
        &[lower[0], higher[0]],
        &[b_start.timestamp_nanos(), b_end.timestamp_nanos()]
    );

    // 2nd group
    // expected boundary:
    // 2021-12-16 01:0:00 -> 2021-12-16 02:00:00
    // expected members:
    // 2021-12-16 01:00:00
    // 2021-12-16 01:30:00
    // 2021-12-16 02:00:00
    let g = take_groups_slice(&groups, 1, &ts);
    let t0 = NaiveDate::from_ymd(2021, 12, 16).and_hms(1, 0, 0);
    let t1 = NaiveDate::from_ymd(2021, 12, 16).and_hms(1, 30, 0);
    let t2 = NaiveDate::from_ymd(2021, 12, 16).and_hms(2, 0, 0);
    assert_eq!(
        g,
        &[
            t0.timestamp_nanos(),
            t1.timestamp_nanos(),
            t2.timestamp_nanos()
        ]
    );
    let b_start = NaiveDate::from_ymd(2021, 12, 16).and_hms(1, 0, 0);
    let b_end = NaiveDate::from_ymd(2021, 12, 16).and_hms(2, 0, 0);
    assert_eq!(
        &[lower[1], higher[1]],
        &[b_start.timestamp_nanos(), b_end.timestamp_nanos()]
    );

    assert_eq!(groups[2], [4, 3]);

    // test closed: "left" (should not include right end of interval)
    let (groups, _, _) = groupby_windows(w, &ts, false, ClosedWindow::Left, TimeUnit::Nanoseconds);
    assert_eq!(groups[0], [0, 2]); // 00:00:00 -> 00:30:00
    assert_eq!(groups[1], [2, 2]); // 01:00:00 -> 01:30:00
    assert_eq!(groups[2], [4, 2]); // 02:00:00 -> 02:30:00

    // test closed: "right" (should not include left end of interval)
    let (groups, _, _) = groupby_windows(w, &ts, false, ClosedWindow::Right, TimeUnit::Nanoseconds);
    assert_eq!(groups[0], [1, 2]); // 00:00:00 -> 00:30:00
    assert_eq!(groups[1], [3, 2]); // 01:00:00 -> 01:30:00
    assert_eq!(groups[2], [5, 2]); // 02:00:00 -> 02:30:00

    // test closed: "none" (should not include left or right end of interval)
    let (groups, _, _) = groupby_windows(w, &ts, false, ClosedWindow::None, TimeUnit::Nanoseconds);
    assert_eq!(groups[0], [1, 1]); // 00:00:00 -> 00:30:00
    assert_eq!(groups[1], [3, 1]); // 01:00:00 -> 01:30:00
    assert_eq!(groups[2], [5, 1]); // 02:00:00 -> 02:30:00
}

#[test]
fn test_boundaries_2() {
    let start = NaiveDate::from_ymd(2021, 12, 16).and_hms(0, 0, 0);
    let stop = NaiveDate::from_ymd(2021, 12, 16).and_hms(4, 0, 0);

    let ts = date_range_vec(
        start.timestamp_nanos(),
        stop.timestamp_nanos(),
        Duration::parse("30m"),
        ClosedWindow::Both,
        TimeUnit::Nanoseconds,
    );

    print_ns(&ts);

    // window:
    // every 2h
    // period 1h
    // offset 30m
    let offset = Duration::parse("30m");
    let w = Window::new(Duration::parse("2h"), Duration::parse("1h"), offset);

    // earliest bound is first datapoint: 2021-12-16 00:00:00 + 30m offset: 2021-12-16 00:30:00
    let b = w.get_earliest_bounds_ns(ts[0]);

    assert_eq!(b.start, start.timestamp_nanos() + offset.duration_ns());

    let (groups, lower, higher) =
        groupby_windows(w, &ts, true, ClosedWindow::Left, TimeUnit::Nanoseconds);

    // 1st group
    // expected boundary:
    // 2021-12-16 00:30:00 -> 2021-12-16 01:30:00
    // expected members:
    // 2021-12-16 00:30:00
    // 2021-12-16 01:00:00
    // (note that we don't expect 01:30:00 because we close left (and thus open interval right))
    // see: https://pandas.pydata.org/docs/reference/api/pandas.Interval.html
    let g = take_groups_slice(&groups, 0, &ts);
    let t0 = NaiveDate::from_ymd(2021, 12, 16).and_hms(0, 30, 0);
    let t1 = NaiveDate::from_ymd(2021, 12, 16).and_hms(1, 0, 0);
    assert_eq!(g, &[t0.timestamp_nanos(), t1.timestamp_nanos()]);
    let b_start = NaiveDate::from_ymd(2021, 12, 16).and_hms(0, 30, 0);
    let b_end = NaiveDate::from_ymd(2021, 12, 16).and_hms(1, 30, 0);
    assert_eq!(
        &[lower[0], higher[0]],
        &[b_start.timestamp_nanos(), b_end.timestamp_nanos()]
    );

    // 2nd group
    // expected boundary:
    // 2021-12-16 02:30:00 -> 2021-12-16 03:30:00
    // expected members:
    // 2021-12-16 02:30:00
    // 2021-12-16 03:00:00
    // (note that we don't expect 03:30:00 because we close left)
    let g = take_groups_slice(&groups, 1, &ts);
    let t0 = NaiveDate::from_ymd(2021, 12, 16).and_hms(2, 30, 0);
    let t1 = NaiveDate::from_ymd(2021, 12, 16).and_hms(3, 0, 0);
    assert_eq!(g, &[t0.timestamp_nanos(), t1.timestamp_nanos()]);
    let b_start = NaiveDate::from_ymd(2021, 12, 16).and_hms(2, 30, 0);
    let b_end = NaiveDate::from_ymd(2021, 12, 16).and_hms(3, 30, 0);
    assert_eq!(
        &[lower[1], higher[1]],
        &[b_start.timestamp_nanos(), b_end.timestamp_nanos()]
    );
}

#[test]
fn test_boundaries_ms() {
    let start = NaiveDate::from_ymd(2021, 12, 16).and_hms(0, 0, 0);
    let stop = NaiveDate::from_ymd(2021, 12, 16).and_hms(3, 0, 0);

    let ts = date_range_vec(
        start.timestamp_millis(),
        stop.timestamp_millis(),
        Duration::parse("30m"),
        ClosedWindow::Both,
        TimeUnit::Milliseconds,
    );

    // window:
    // every 2h
    // period 1h
    let w = Window::new(
        Duration::parse("1h"),
        Duration::parse("1h"),
        Duration::parse("0ns"),
    );

    // earliest bound is first datapoint: 2021-12-16 00:00:00
    let b = w.get_earliest_bounds_ms(ts[0]);
    assert_eq!(b.start, start.timestamp_millis());

    // test closed: "both" (includes both ends of the interval)
    let (groups, lower, higher) =
        groupby_windows(w, &ts, true, ClosedWindow::Both, TimeUnit::Milliseconds);

    // 1st group
    // expected boundary:
    // 2021-12-16 00:00:00 -> 2021-12-16 01:00:00
    // expected members:
    // 2021-12-16 00:00:00
    // 2021-12-16 00:30:00
    // 2021-12-16 01:00:00
    let g = take_groups_slice(&groups, 0, &ts);
    let t0 = NaiveDate::from_ymd(2021, 12, 16).and_hms(0, 0, 0);
    let t1 = NaiveDate::from_ymd(2021, 12, 16).and_hms(0, 30, 0);
    let t2 = NaiveDate::from_ymd(2021, 12, 16).and_hms(1, 0, 0);
    assert_eq!(
        g,
        &[
            t0.timestamp_millis(),
            t1.timestamp_millis(),
            t2.timestamp_millis()
        ]
    );
    let b_start = NaiveDate::from_ymd(2021, 12, 16).and_hms(0, 0, 0);
    let b_end = NaiveDate::from_ymd(2021, 12, 16).and_hms(1, 0, 0);
    assert_eq!(
        &[lower[0], higher[0]],
        &[b_start.timestamp_millis(), b_end.timestamp_millis()]
    );

    // 2nd group
    // expected boundary:
    // 2021-12-16 01:0:00 -> 2021-12-16 02:00:00
    // expected members:
    // 2021-12-16 01:00:00
    // 2021-12-16 01:30:00
    // 2021-12-16 02:00:00
    let g = take_groups_slice(&groups, 1, &ts);
    let t0 = NaiveDate::from_ymd(2021, 12, 16).and_hms(1, 0, 0);
    let t1 = NaiveDate::from_ymd(2021, 12, 16).and_hms(1, 30, 0);
    let t2 = NaiveDate::from_ymd(2021, 12, 16).and_hms(2, 0, 0);
    assert_eq!(
        g,
        &[
            t0.timestamp_millis(),
            t1.timestamp_millis(),
            t2.timestamp_millis()
        ]
    );
    let b_start = NaiveDate::from_ymd(2021, 12, 16).and_hms(1, 0, 0);
    let b_end = NaiveDate::from_ymd(2021, 12, 16).and_hms(2, 0, 0);
    assert_eq!(
        &[lower[1], higher[1]],
        &[b_start.timestamp_millis(), b_end.timestamp_millis()]
    );

    assert_eq!(groups[2], [4, 3]);

    // test closed: "left" (should not include right end of interval)
    let (groups, _, _) = groupby_windows(w, &ts, false, ClosedWindow::Left, TimeUnit::Milliseconds);
    assert_eq!(groups[0], [0, 2]); // 00:00:00 -> 00:30:00
    assert_eq!(groups[1], [2, 2]); // 01:00:00 -> 01:30:00
    assert_eq!(groups[2], [4, 2]); // 02:00:00 -> 02:30:00

    // test closed: "right" (should not include left end of interval)
    let (groups, _, _) =
        groupby_windows(w, &ts, false, ClosedWindow::Right, TimeUnit::Milliseconds);
    assert_eq!(groups[0], [1, 2]); // 00:00:00 -> 00:30:00
    assert_eq!(groups[1], [3, 2]); // 01:00:00 -> 01:30:00
    assert_eq!(groups[2], [5, 2]); // 02:00:00 -> 02:30:00

    // test closed: "none" (should not include left or right end of interval)
    let (groups, _, _) = groupby_windows(w, &ts, false, ClosedWindow::None, TimeUnit::Milliseconds);
    assert_eq!(groups[0], [1, 1]); // 00:00:00 -> 00:30:00
    assert_eq!(groups[1], [3, 1]); // 01:00:00 -> 01:30:00
    assert_eq!(groups[2], [5, 1]); // 02:00:00 -> 02:30:00
}

#[test]
fn test_rolling_lookback() {
    // Test month as interval in date range
    let start = NaiveDate::from_ymd(2021, 12, 16).and_hms(0, 0, 0);
    let end = NaiveDate::from_ymd(2021, 12, 16).and_hms(4, 0, 0);
    let dates = date_range_vec(
        start.timestamp_nanos(),
        end.timestamp_nanos(),
        Duration::parse("30m"),
        ClosedWindow::Both,
        TimeUnit::Nanoseconds,
    );

    let groups = groupby_values(
        Duration::parse("2h"),
        Duration::parse("-2h"),
        &dates,
        ClosedWindow::Right,
        TimeUnit::Nanoseconds,
    );
    assert_eq!(dates.len(), groups.len());
    assert_eq!(groups[0], [0, 1]); // bound: 22:00 -> 24:00     time: 24:00
    assert_eq!(groups[1], [0, 2]); // bound: 22:30 -> 00:30     time: 00:30
    assert_eq!(groups[2], [0, 3]); // bound: 23:00 -> 01:00     time: 01:00
    assert_eq!(groups[3], [0, 4]); // bound: 23:30 -> 01:30     time: 01:30
    assert_eq!(groups[4], [1, 4]); // bound: 24:00 -> 02:00     time: 02:00
    assert_eq!(groups[5], [2, 4]); // bound: 00:30 -> 02:30     time: 02:30
    assert_eq!(groups[6], [3, 4]); // bound: 01:00 -> 03:00     time: 03:00
    assert_eq!(groups[7], [4, 4]); // bound: 01:30 -> 03:30     time: 03:30
    assert_eq!(groups[8], [5, 4]); // bound: 02:00 -> 04:00     time: 04:00
}

#[test]
fn test_end_membership() {
    let time = [
        NaiveDate::from_ymd(2021, 2, 1)
            .and_hms(0, 0, 0)
            .timestamp_millis(),
        NaiveDate::from_ymd(2021, 5, 1)
            .and_hms(0, 0, 0)
            .timestamp_millis(),
    ];
    let window = Window::new(
        Duration::parse("1mo"),
        Duration::parse("2mo"),
        Duration::parse("-2mo"),
    );
    // windows
    // 2020-12-01 -> 2021-02-01     members: None
    // 2021-01-01 -> 2021-03-01     members: [0]
    // 2021-02-01 -> 2021-04-01     members: [0]
    // 2021-03-01 -> 2021-05-01     members: None
    // 2021-04-01 -> 2021-06-01     members: [1]
    // 2021-05-01 -> 2021-07-01     members: [1]
    let (groups, _, _) = groupby_windows(
        window,
        &time,
        false,
        ClosedWindow::Left,
        TimeUnit::Milliseconds,
    );
    assert_eq!(groups[0], [0, 1]);
    assert_eq!(groups[1], [0, 1]);
    assert_eq!(groups[2], [1, 1]);
    assert_eq!(groups[3], [1, 1]);
}

#[test]
fn test_groupby_windows_membership_2791() {
    let dates = [0, 0, 2, 2];
    let window = Window::new(
        Duration::parse("1ms"),
        Duration::parse("1ms"),
        Duration::parse("0ns"),
    );
    let (groups, _, _) = groupby_windows(
        window,
        &dates,
        false,
        ClosedWindow::Left,
        TimeUnit::Milliseconds,
    );
    assert_eq!(groups[0], [0, 2]);
    assert_eq!(groups[1], [2, 2]);
}
