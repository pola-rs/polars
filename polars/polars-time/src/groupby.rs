use crate::bounds::Bounds;
use crate::window::Window;

pub type GroupTuples = Vec<(u32, Vec<u32>)>;

pub fn groupby(window: Window, time: &[i64]) -> GroupTuples {
    let boundary = Bounds::from(time);

    let mut group_tuples = Vec::with_capacity(window.estimate_overlapping_bounds(boundary));
    let mut latest_start = 0;

    for bi in window.get_overlapping_bounds_iter(boundary) {
        let mut group = vec![];

        let mut skip_window = false;
        // find starting point of window
        while latest_start < time.len() {
            let t = time[latest_start];
            if bi.is_future(t) {
                skip_window = true;
                break;
            }
            if bi.is_member(t) {
                break;
            }
            latest_start += 1;
        }
        if skip_window {
            latest_start = latest_start.saturating_sub(1);
            continue;
        }

        // subtract 1 because the next window could also start from the same point
        latest_start = latest_start.saturating_sub(1);

        // find members of this window
        let mut i = latest_start;
        if i >= time.len() {
            break;
        }

        while i < time.len() {
            let t = time[i];
            if bi.is_member(t) {
                group.push(i as u32);
            } else if bi.is_future(t) {
                break;
            }
            i += 1
        }

        if !group.is_empty() {
            group_tuples.push((group[0], group))
        }
    }
    group_tuples
}

#[cfg(test)]
mod test {
    use super::*;
    use chrono::{NaiveDate, NaiveDateTime, NaiveTime};

    #[test]
    fn test_group_tuples() {
        let dt = &[
            NaiveDateTime::new(
                NaiveDate::from_ymd(2001, 1, 1),
                NaiveTime::from_hms(1, 0, 0),
            ),
            NaiveDateTime::new(
                NaiveDate::from_ymd(2001, 1, 1),
                NaiveTime::from_hms(1, 0, 15),
            ),
            NaiveDateTime::new(
                NaiveDate::from_ymd(2001, 1, 1),
                NaiveTime::from_hms(1, 0, 30),
            ),
            NaiveDateTime::new(
                NaiveDate::from_ymd(2001, 1, 1),
                NaiveTime::from_hms(1, 0, 45),
            ),
            NaiveDateTime::new(
                NaiveDate::from_ymd(2001, 1, 1),
                NaiveTime::from_hms(1, 1, 0),
            ),
            NaiveDateTime::new(
                NaiveDate::from_ymd(2001, 1, 1),
                NaiveTime::from_hms(1, 1, 15),
            ),
            NaiveDateTime::new(
                NaiveDate::from_ymd(2001, 1, 1),
                NaiveTime::from_hms(1, 1, 30),
            ),
        ];

        let ts = dt.iter().map(|dt| dt.timestamp_nanos()).collect::<Vec<_>>();
        let window = Window::new(
            Duration::from_seconds(30),
            Duration::from_seconds(30),
            Duration::from_seconds(0),
        );
        let gt = groupby(window, &ts)
            .into_iter()
            .map(|g| g.1)
            .collect::<Vec<_>>();

        let expected = &[[0, 1, 2], [2, 3, 4], [4, 5, 6]];
        assert_eq!(gt, expected);
    }
}
