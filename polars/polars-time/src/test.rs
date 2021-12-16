use crate::bounds::Bounds;
use crate::calendar::timestamp_ns_to_datetime;
use crate::duration::Duration;
use crate::groupby::{groupby, ClosedWindow};
use crate::window::Window;
use chrono::prelude::*;

fn date_range(delta: u32) -> Vec<i64> {
    let date = NaiveDate::from_ymd(2001, 1, 1);

    (0..10u32)
        .map(|i| {
            let mut min = delta * i;
            let mut hour = 0;
            if min >= 60 {
                hour = min / 60;
                min = min % 60;
            }

            let time = NaiveTime::from_hms(hour, min, 0);
            NaiveDateTime::new(date, time).timestamp_nanos()
        })
        .collect()
}

fn print_ns(ts: &[i64]) {
    for ts in ts {
        println!("{}", timestamp_ns_to_datetime(*ts));
    }
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
    let (groups, _, _) = groupby(w, &ts, false, ClosedWindow::None);
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
