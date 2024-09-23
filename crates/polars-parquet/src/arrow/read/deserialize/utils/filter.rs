use std::ops::Range;

use arrow::array::Splitable;
use arrow::bitmap::Bitmap;

#[derive(Debug, Clone)]
pub enum Filter {
    Range(Range<usize>),
    Mask(Bitmap),
}

impl Filter {
    pub fn new_limited(x: usize) -> Self {
        Filter::Range(0..x)
    }

    pub fn new_ranged(start: usize, end: usize) -> Self {
        Filter::Range(start..end)
    }

    pub fn new_masked(mask: Bitmap) -> Self {
        Filter::Mask(mask)
    }

    pub fn do_include_at(&self, at: usize) -> bool {
        match self {
            Filter::Range(range) => range.contains(&at),
            Filter::Mask(bitmap) => bitmap.get_bit(at),
        }
    }

    pub(crate) fn num_rows(&self) -> usize {
        match self {
            Filter::Range(range) => range.len(),
            Filter::Mask(bitmap) => bitmap.set_bits(),
        }
    }

    pub(crate) fn split_at(&self, at: usize) -> (Filter, Filter) {
        use Filter as F;
        match self {
            F::Range(range) => {
                let start = range.start;
                let end = range.end;

                if at <= start {
                    (F::Range(0..0), F::Range(start - at..end - at))
                } else if at > end {
                    (F::Range(start..end), F::Range(0..0))
                } else {
                    (F::Range(start..at), F::Range(0..end - at))
                }
            },
            F::Mask(bitmap) => {
                let (lhs, rhs) = bitmap.split_at(at);
                (F::Mask(lhs), F::Mask(rhs))
            },
        }
    }

    pub(crate) fn opt_split_at(
        filter: &Option<Self>,
        at: usize,
    ) -> (Option<Filter>, Option<Filter>) {
        let Some(filter) = filter else {
            return (None, None);
        };

        let (lhs, rhs) = filter.split_at(at);
        (Some(lhs), Some(rhs))
    }

    pub(crate) fn opt_num_rows(filter: &Option<Self>, total_num_rows: usize) -> usize {
        match filter {
            Some(filter) => usize::min(filter.num_rows(), total_num_rows),
            None => total_num_rows,
        }
    }
}
