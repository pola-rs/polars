use crate::bounds::Bounds;
use crate::unit::TimeNanoseconds;
use crate::window::Window;

pub type GroupTuples = Vec<(u32, Vec<u32>)>;

#[derive(Clone, Copy, Debug)]
pub enum ClosedWindow {
    Left,
    Right,
    Both,
    None,
}

pub fn groupby(
    window: Window,
    time: &[i64],
    include_boundaries: bool,
    closed_window: ClosedWindow,
) -> (GroupTuples, Vec<TimeNanoseconds>, Vec<TimeNanoseconds>) {
    let boundary = Bounds::from(time);
    let size = if include_boundaries {
        window.estimate_overlapping_bounds(boundary)
    } else {
        0
    };
    let mut lower_bound = Vec::with_capacity(size);
    let mut upper_bound = Vec::with_capacity(size);

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
