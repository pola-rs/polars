pub mod buffer;
pub mod storage;

use std::ops;

pub use buffer::Buffer;
pub use storage::SharedStorage;

// Copied from the stdlib.
fn slice_index_fail(start: usize, end: usize, len: usize) -> ! {
    if start > len {
        panic!("range start index {start} out of range for slice of length {len}",)
    }

    if end > len {
        panic!("range end index {end} out of range for slice of length {len}",)
    }

    if start > end {
        panic!("slice index starts at {start} but ends at {end}",)
    }

    // Only reachable if the range was a `RangeInclusive` or a
    // `RangeToInclusive`, with `end == len`.
    panic!("range end index {end} out of range for slice of length {len}",)
}

#[must_use]
pub fn check_range<R>(range: R, bounds: ops::RangeTo<usize>) -> ops::Range<usize>
where
    R: ops::RangeBounds<usize>,
{
    let len = bounds.end;
    let end = match range.end_bound() {
        ops::Bound::Included(&end) if end >= len => slice_index_fail(0, end, len),
        // Cannot overflow because `end < len` implies `end < usize::MAX`.
        ops::Bound::Included(&end) => end + 1,

        ops::Bound::Excluded(&end) if end > len => slice_index_fail(0, end, len),
        ops::Bound::Excluded(&end) => end,
        ops::Bound::Unbounded => len,
    };

    let start = match range.start_bound() {
        ops::Bound::Excluded(&start) if start >= end => slice_index_fail(start, end, len),
        // Cannot overflow because `start < end` implies `start < usize::MAX`.
        ops::Bound::Excluded(&start) => start + 1,

        ops::Bound::Included(&start) if start > end => slice_index_fail(start, end, len),
        ops::Bound::Included(&start) => start,

        ops::Bound::Unbounded => 0,
    };

    ops::Range { start, end }
}

#[must_use]
pub fn decode_range_unchecked<R>(range: R, bounds: ops::RangeTo<usize>) -> ops::Range<usize>
where
    R: ops::RangeBounds<usize>,
{
    let len = bounds.end;
    let end = match range.end_bound() {
        ops::Bound::Included(&end) => end + 1,
        ops::Bound::Excluded(&end) => end,
        ops::Bound::Unbounded => len,
    };

    let start = match range.start_bound() {
        ops::Bound::Excluded(&start) => start + 1,
        ops::Bound::Included(&start) => start,
        ops::Bound::Unbounded => 0,
    };

    ops::Range { start, end }
}
