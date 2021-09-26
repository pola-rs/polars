pub mod no_nulls;
pub mod nulls;
use arrow::bitmap::{Bitmap, MutableBitmap};

type Start = usize;
type End = usize;
type Idx = usize;
type WindowSize = usize;
type Len = usize;

fn det_offsets(i: Idx, window_size: WindowSize, _len: Len) -> (usize, usize) {
    (i.saturating_sub(window_size - 1), i + 1)
}
fn det_offsets_center(i: Idx, window_size: WindowSize, len: Len) -> (usize, usize) {
    (
        i.saturating_sub(window_size / 2),
        std::cmp::min(len, i + window_size / 2),
    )
}

fn create_validity<Fo>(
    min_periods: usize,
    len: usize,
    window_size: usize,
    det_offsets_fn: Fo,
) -> Option<MutableBitmap>
where
    Fo: Fn(Idx, WindowSize, Len) -> (Start, End),
{
    if min_periods > 1 {
        let mut validity = MutableBitmap::with_capacity(len);
        validity.extend_constant(len, true);

        // set the null values at the boundaries

        // head
        for i in 0..len {
            let (start, end) = det_offsets_fn(i, window_size, len);
            if (end - start) < min_periods {
                validity.set(i, false)
            } else {
                break;
            }
        }
        // tail
        for i in (0..len).rev() {
            let (start, end) = det_offsets_fn(i, window_size, len);
            if (end - start) < min_periods {
                validity.set(i, false)
            } else {
                break;
            }
        }

        Some(validity)
    } else {
        None
    }
}
