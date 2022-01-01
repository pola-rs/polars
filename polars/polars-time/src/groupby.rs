use crate::bounds::Bounds;
use crate::window::Window;

pub type GroupTuples = Vec<(u32, Vec<u32>)>;

#[derive(Clone, Copy, Debug)]
pub enum ClosedWindow {
    Left,
    Right,
    Both,
    None,
}

pub enum TimeUnit {
    Nanoseconds,
    Milliseconds,
}

pub fn groupby(
    window: Window,
    time: &[i64],
    include_boundaries: bool,
    closed_window: ClosedWindow,
    tu: TimeUnit,
) -> (GroupTuples, Vec<i64>, Vec<i64>) {
    let start = time[0];
    let boundary = if time.len() > 1 {
        // +1 because left or closed boundary could match the next window if it is on the boundary
        let stop = time[time.len() - 1] + 1;
        Bounds::new(start, stop)
    } else {
        let stop = start + 1;
        Bounds::new(start, stop)
    };

    let size = if include_boundaries {
        match tu {
            TimeUnit::Milliseconds => window.estimate_overlapping_bounds_ms(boundary),
            TimeUnit::Nanoseconds => window.estimate_overlapping_bounds_ns(boundary),
        }
    } else {
        0
    };
    let mut lower_bound = Vec::with_capacity(size);
    let mut upper_bound = Vec::with_capacity(size);

    let mut group_tuples = match tu {
        TimeUnit::Nanoseconds => {
            Vec::with_capacity(window.estimate_overlapping_bounds_ns(boundary))
        }
        TimeUnit::Milliseconds => {
            Vec::with_capacity(window.estimate_overlapping_bounds_ms(boundary))
        }
    };
    let mut latest_start = 0;

    for bi in window.get_overlapping_bounds_iter(boundary, tu) {
        let mut group = vec![];

        let mut skip_window = false;
        // find starting point of window
        while latest_start < time.len() {
            let t = time[latest_start];
            if bi.is_future(t) {
                skip_window = true;
                break;
            }
            if bi.is_member(t, closed_window) {
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
            if bi.is_member(t, closed_window) {
                group.push(i as u32);
            } else if bi.is_future(t) {
                break;
            }
            i += 1
        }

        if !group.is_empty() {
            if include_boundaries {
                lower_bound.push(bi.start);
                upper_bound.push(bi.stop);
            }
            group_tuples.push((group[0], group))
        }
    }
    (group_tuples, lower_bound, upper_bound)
}
