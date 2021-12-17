use crate::calendar::{date_range, timestamp_ns_to_datetime};
use crate::duration::Duration;
use crate::groupby::{groupby, ClosedWindow, GroupTuples};
use crate::unit::TimeNanoseconds;
use crate::window::Window;
use chrono::prelude::*;

fn print_ns(ts: &[i64]) {
    for ts in ts {
        println!("{}", timestamp_ns_to_datetime(*ts));
    }
}

fn take_groups(groups: &GroupTuples, idx: usize, ts: &[TimeNanoseconds]) -> Vec<TimeNanoseconds> {
    let group = &groups[idx].1;
    group.iter().map(|idx| ts[*idx as usize]).collect()
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
    let (groups, _, _) = groupby(w, &ts, false, ClosedWindow::Both);
    assert_eq!(groups.len(), 3);
    assert_eq!(groups[0], (0, vec![0]));
    assert_eq!(groups[1], (1, vec![1]));
    assert_eq!(groups[2], (1, vec![1, 2, 3]));
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

    let b = w.get_earliest_bounds(t);
    let start = NaiveDate::from_ymd(2020, 1, 1)
        .and_hms(23, 58, 0)
        .timestamp_nanos();
    assert_eq!(b.start, start);
}

#[test]
fn test_boundaries() {
    let start = NaiveDate::from_ymd(2021, 12, 16).and_hms(0, 0, 0);
    let stop = NaiveDate::from_ymd(2021, 12, 16).and_hms(3, 0, 0);

    let ts = date_range(
        start.timestamp_nanos(),
        stop.timestamp_nanos(),
        Duration::parse("30m"),
        ClosedWindow::Both,
    );

    print_ns(&ts);
    // window:
    // every 2h
    // period 1h
    let w = Window::new(
        Duration::parse("1h"),
        Duration::parse("1h"),
        Duration::parse("0ns"),
    );

    // earliest bound is first datapoint: 2021-12-16 00:00:00
    let b = w.get_earliest_bounds(ts[0]);
    assert_eq!(b.start, start.timestamp_nanos());

    // test closed: "both" (includes both ends of the interval)
    let (groups, lower, higher) = groupby(w, &ts, true, ClosedWindow::Both);

    // 1st group
    // expected boundary:
    // 2021-12-16 00:00:00 -> 2021-12-16 01:00:00
    // expected members:
    // 2021-12-16 00:00:00
    // 2021-12-16 00:30:00
    // 2021-12-16 01:00:00
    let g = take_groups(&groups, 0, &ts);
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
    let g = take_groups(&groups, 1, &ts);
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

    assert_eq!(groups[2].1, &[4, 5, 6]);

    // test closed: "left" (should not include right end of interval)
    let (groups, _, _) = groupby(w, &ts, false, ClosedWindow::Left);
    assert_eq!(groups[0].1, &[0, 1]); // 00:00:00 -> 00:30:00
    assert_eq!(groups[1].1, &[2, 3]); // 01:00:00 -> 01:30:00
    assert_eq!(groups[2].1, &[4, 5]); // 02:00:00 -> 02:30:00

    // test closed: "right" (should not include left end of interval)
    let (groups, _, _) = groupby(w, &ts, false, ClosedWindow::Right);
    assert_eq!(groups[0].1, &[1, 2]); // 00:00:00 -> 00:30:00
    assert_eq!(groups[1].1, &[3, 4]); // 01:00:00 -> 01:30:00
    assert_eq!(groups[2].1, &[5, 6]); // 02:00:00 -> 02:30:00

    // test closed: "none" (should not include left or right end of interval)
    let (groups, _, _) = groupby(w, &ts, false, ClosedWindow::None);
    assert_eq!(groups[0].1, &[1]); // 00:00:00 -> 00:30:00
    assert_eq!(groups[1].1, &[3]); // 01:00:00 -> 01:30:00
    assert_eq!(groups[2].1, &[5]); // 02:00:00 -> 02:30:00
}

#[test]
fn test_boundaries_2() {
    let start = NaiveDate::from_ymd(2021, 12, 16).and_hms(0, 0, 0);
    let stop = NaiveDate::from_ymd(2021, 12, 16).and_hms(4, 0, 0);

    let ts = date_range(
        start.timestamp_nanos(),
        stop.timestamp_nanos(),
        Duration::parse("30m"),
        ClosedWindow::Both,
    );

    print_ns(&ts);

    // window:
    // every 2h
    // period 1h
    // offset 30m
    let offset = Duration::parse("30m");
    let w = Window::new(Duration::parse("2h"), Duration::parse("1h"), offset);

    // earliest bound is first datapoint: 2021-12-16 00:00:00 + 30m offset: 2021-12-16 00:30:00
    let b = w.get_earliest_bounds(ts[0]);

    assert_eq!(b.start, start.timestamp_nanos() + offset.duration());

    let (groups, lower, higher) = groupby(w, &ts, true, ClosedWindow::Left);

    // 1st group
    // expected boundary:
    // 2021-12-16 00:30:00 -> 2021-12-16 01:30:00
    // expected members:
    // 2021-12-16 00:30:00
    // 2021-12-16 01:00:00
    // (note that we don't expect 01:30:00 because we close left (and thus open interval right))
    // see: https://pandas.pydata.org/docs/reference/api/pandas.Interval.html
    let g = take_groups(&groups, 0, &ts);
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
    let g = take_groups(&groups, 1, &ts);
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
