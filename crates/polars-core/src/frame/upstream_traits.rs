use std::ops::{Index, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use arrow::record_batch::RecordBatchT;

use crate::prelude::*;

impl TryExtend<RecordBatchT<Box<dyn Array>>> for DataFrame {
    fn try_extend<I: IntoIterator<Item = RecordBatchT<Box<dyn Array>>>>(
        &mut self,
        iter: I,
    ) -> PolarsResult<()> {
        for record_batch in iter {
            self.append_record_batch(record_batch)?;
        }

        Ok(())
    }
}

impl TryExtend<PolarsResult<RecordBatchT<Box<dyn Array>>>> for DataFrame {
    fn try_extend<I: IntoIterator<Item = PolarsResult<RecordBatchT<Box<dyn Array>>>>>(
        &mut self,
        iter: I,
    ) -> PolarsResult<()> {
        for record_batch in iter {
            self.append_record_batch(record_batch?)?;
        }

        Ok(())
    }
}

impl Index<usize> for DataFrame {
    type Output = Column;

    fn index(&self, index: usize) -> &Self::Output {
        &self.columns()[index]
    }
}

macro_rules! impl_ranges {
    ($range_type:ty) => {
        impl Index<$range_type> for DataFrame {
            type Output = [Column];

            fn index(&self, index: $range_type) -> &Self::Output {
                &self.columns()[index]
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
    type Output = Column;

    fn index(&self, index: &str) -> &Self::Output {
        self.column(index).unwrap()
    }
}
