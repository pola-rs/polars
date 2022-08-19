use std::iter::FromIterator;
use std::ops::{Index, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use crate::prelude::*;

impl FromIterator<Series> for DataFrame {
    /// # Panics
    ///
    /// Panics if Series have different lengths.
    fn from_iter<T: IntoIterator<Item = Series>>(iter: T) -> Self {
        let v = iter.into_iter().collect();
        DataFrame::new(v).expect("could not create DataFrame from iterator")
    }
}

impl Index<usize> for DataFrame {
    type Output = Series;

    fn index(&self, index: usize) -> &Self::Output {
        &self.columns[index]
    }
}

macro_rules! impl_ranges {
    ($range_type:ty) => {
        impl Index<$range_type> for DataFrame {
            type Output = [Series];

            fn index(&self, index: $range_type) -> &Self::Output {
                &self.columns[index]
            }
        }
    };
}

impl_ranges!(Range<usize>);
impl_ranges!(RangeInclusive<usize>);
impl_ranges!(RangeFrom<usize>);
impl_ranges!(RangeTo<usize>);
impl_ranges!(RangeToInclusive<usize>);
impl_ranges!(RangeFull);

// we don't implement Borrow<str> or AsRef<str> as upstream crates may add impl of trait for usize.
impl Index<&str> for DataFrame {
    type Output = Series;

    fn index(&self, index: &str) -> &Self::Output {
        let idx = self.check_name_to_idx(index).unwrap();
        &self.columns[idx]
    }
}
