use std::ops::Range;

use arrow::array::Splitable;
use arrow::bitmap::Bitmap;

use crate::read::expr::ParquetColumnExprRef;

#[derive(Clone)]
pub struct PredicateFilter {
    pub predicate: ParquetColumnExprRef,
    pub include_values: bool,
}

#[derive(Clone)]
pub enum Filter {
    Range(Range<usize>),
    Mask(Bitmap),
    Predicate(PredicateFilter),
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

    pub fn num_rows(&self, total_num_rows: usize) -> usize {
        match self {
            Self::Range(range) => range.len(),
            Self::Mask(bitmap) => bitmap.set_bits(),
            Self::Predicate { .. } => total_num_rows,
        }
    }

    pub fn max_offset(&self, total_num_rows: usize) -> usize {
        match self {
            Self::Range(range) => range.end,
            Self::Mask(bitmap) => bitmap.len(),
            Self::Predicate { .. } => total_num_rows,
        }
    }

    pub(crate) fn split_at(&self, at: usize) -> (Self, Self) {
        match self {
            Self::Range(range) => {
                let start = range.start;
                let end = range.end;

                if at <= start {
                    (Self::Range(0..0), Self::Range(start - at..end - at))
                } else if at > end {
                    (Self::Range(start..end), Self::Range(0..0))
                } else {
                    (Self::Range(start..at), Self::Range(0..end - at))
                }
            },
            Self::Mask(bitmap) => {
                let (lhs, rhs) = bitmap.split_at(at);
                (Self::Mask(lhs), Self::Mask(rhs))
            },
            Self::Predicate(e) => (Self::Predicate(e.clone()), Self::Predicate(e.clone())),
        }
    }

    pub(crate) fn opt_split_at(filter: &Option<Self>, at: usize) -> (Option<Self>, Option<Self>) {
        let Some(filter) = filter else {
            return (None, None);
        };

        let (lhs, rhs) = filter.split_at(at);
        (Some(lhs), Some(rhs))
    }

    pub(crate) fn opt_num_rows(filter: &Option<Self>, total_num_rows: usize) -> usize {
        match filter {
            Some(filter) => usize::min(filter.num_rows(total_num_rows), total_num_rows),
            None => total_num_rows,
        }
    }
}
