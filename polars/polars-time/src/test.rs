use crate::bounds::Bounds;
use crate::calendar::timestamp_ns_to_datetime;
use crate::duration::Duration;
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
fn test_window_boundaries() {
    let range_ns = date_range(10);

    let w = Window::new(
        Duration::from_minutes(20),
        Duration::from_minutes(40),
        Duration::from_seconds(0),
    );
    // wrapping_boundary (
    let boundary = Bounds::from(&range_ns);
    let overlapping_bounds = w.get_overlapping_bounds(boundary);

    let hm_start = overlapping_bounds
        .iter()
        .map(|b| {
            let dt = timestamp_ns_to_datetime(b.start);
            (dt.hour(), dt.minute())
        })
        .collect::<Vec<_>>();
    let expected = &[(0, 0), (0, 20), (0, 40), (1, 0), (1, 20), (1, 40)];
    assert_eq!(hm_start, expected);

    let hm_stop = overlapping_bounds
        .iter()
        .map(|b| {
            let dt = timestamp_ns_to_datetime(b.stop);
            (dt.hour(), dt.minute())
        })
        .collect::<Vec<_>>();
    let expected = &[(0, 40), (1, 0), (1, 20), (1, 40), (2, 0), (2, 20)];
    assert_eq!(hm_stop, expected);
}
