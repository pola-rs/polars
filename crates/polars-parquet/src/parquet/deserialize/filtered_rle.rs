use std::collections::VecDeque;

use super::{HybridDecoderBitmapIter, HybridEncoded};
use crate::parquet::encoding::hybrid_rle::BitmapIter;
use crate::parquet::error::Error;
use crate::parquet::indexes::Interval;

/// Type definition of a [`FilteredHybridBitmapIter`] of [`HybridDecoderBitmapIter`].
pub type FilteredHybridRleDecoderIter<'a> =
    FilteredHybridBitmapIter<'a, HybridDecoderBitmapIter<'a>>;

/// The decoding state of the hybrid-RLE decoder with a maximum definition level of 1
/// that can supports skipped runs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilteredHybridEncoded<'a> {
    /// a bitmap (values, offset, length, skipped_set)
    Bitmap {
        values: &'a [u8],
        offset: usize,
        length: usize,
    },
    Repeated {
        is_set: bool,
        length: usize,
    },
    /// When the run was skipped - contains the number of set values on the skipped run
    Skipped(usize),
}

fn is_set_count(values: &[u8], offset: usize, length: usize) -> usize {
    BitmapIter::new(values, offset, length)
        .filter(|x| *x)
        .count()
}

impl<'a> FilteredHybridEncoded<'a> {
    /// Returns the length of the run in number of items
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            FilteredHybridEncoded::Bitmap { length, .. } => *length,
            FilteredHybridEncoded::Repeated { length, .. } => *length,
            FilteredHybridEncoded::Skipped(_) => 0,
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// An [`Iterator`] adapter over [`HybridEncoded`] that yields [`FilteredHybridEncoded`].
///
/// This iterator adapter is used in combination with
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilteredHybridBitmapIter<'a, I: Iterator<Item = Result<HybridEncoded<'a>, Error>>> {
    iter: I,
    current: Option<(HybridEncoded<'a>, usize)>,
    // a run may end in the middle of an interval, in which case we must
    // split the interval in parts. This tracks the current interval being computed
    current_interval: Option<Interval>,
    selected_rows: VecDeque<Interval>,
    current_items_in_runs: usize,

    total_items: usize,
}

impl<'a, I: Iterator<Item = Result<HybridEncoded<'a>, Error>>> FilteredHybridBitmapIter<'a, I> {
    pub fn new(iter: I, selected_rows: VecDeque<Interval>) -> Self {
        let total_items = selected_rows.iter().map(|x| x.length).sum();
        Self {
            iter,
            current: None,
            current_interval: None,
            selected_rows,
            current_items_in_runs: 0,
            total_items,
        }
    }

    fn advance_current_interval(&mut self, length: usize) {
        if let Some(interval) = &mut self.current_interval {
            interval.start += length;
            interval.length -= length;
            self.total_items -= length;
        }
    }

    /// Returns the number of elements remaining. Note that each run
    /// of the iterator contains more than one element - this is is _not_ equivalent to size_hint.
    pub fn len(&self) -> usize {
        self.total_items
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, I: Iterator<Item = Result<HybridEncoded<'a>, Error>>> Iterator
    for FilteredHybridBitmapIter<'a, I>
{
    type Item = Result<FilteredHybridEncoded<'a>, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        let interval = if let Some(interval) = self.current_interval {
            interval
        } else {
            self.current_interval = self.selected_rows.pop_front();
            self.current_interval?; // case where iteration finishes
            return self.next();
        };

        let (run, offset) = if let Some((run, offset)) = self.current {
            (run, offset)
        } else {
            // a new run
            let run = self.iter.next()?; // no run => something wrong since intervals should only slice items up all runs' length
            match run {
                Ok(run) => {
                    self.current = Some((run, 0));
                },
                Err(e) => return Some(Err(e)),
            }
            return self.next();
        };

        // one of three things can happen:
        // * the start of the interval is not aligned wirh the start of the run => issue a `Skipped` and advance the run / next run
        // * the run contains this interval => consume the interval and keep the run
        // * the run contains part of this interval => consume the run and keep the interval

        match run {
            HybridEncoded::Repeated(is_set, full_run_length) => {
                let run_length = full_run_length - offset;
                // interval.start is from the start of the first run; discount `current_items_in_runs`
                // to get the start from the current run's offset
                let interval_start = interval.start - self.current_items_in_runs;

                if interval_start > 0 {
                    // we need to skip values from the run
                    let to_skip = interval_start;

                    // we only skip up to a run (yield a single skip per multiple runs)
                    let max_skip = full_run_length - offset;
                    let to_skip = to_skip.min(max_skip);

                    let set = if is_set { to_skip } else { 0 };

                    self.current_items_in_runs += to_skip;

                    self.current = if to_skip == max_skip {
                        None
                    } else {
                        Some((run, offset + to_skip))
                    };

                    return Some(Ok(FilteredHybridEncoded::Skipped(set)));
                };

                // slice the bitmap according to current interval
                // note that interval start is from the start of the first run.
                let new_offset = offset + interval_start;

                if interval_start > run_length {
                    let set = if is_set { run_length } else { 0 };

                    self.advance_current_interval(run_length);
                    self.current_items_in_runs += run_length;
                    self.current = None;
                    Some(Ok(FilteredHybridEncoded::Skipped(set)))
                } else {
                    let length = if run_length > interval.length {
                        // interval is fully consumed
                        self.current_items_in_runs += interval.length;

                        // fetch next interval
                        self.total_items -= interval.length;
                        self.current_interval = self.selected_rows.pop_front();

                        self.current = Some((run, offset + interval.length));

                        interval.length
                    } else {
                        // the run is consumed and the interval is shortened accordingly
                        self.current_items_in_runs += run_length;

                        // the interval may cover two runs; shorten the length
                        // to its maximum allowed for this run
                        let length = run_length.min(full_run_length - new_offset);

                        self.advance_current_interval(length);

                        self.current = None;
                        length
                    };
                    Some(Ok(FilteredHybridEncoded::Repeated { is_set, length }))
                }
            },
            HybridEncoded::Bitmap(values, full_run_length) => {
                let run_length = full_run_length - offset;
                // interval.start is from the start of the first run; discount `current_items_in_runs`
                // to get the start from the current run's offset
                let interval_start = interval.start - self.current_items_in_runs;

                if interval_start > 0 {
                    // we need to skip values from the run
                    let to_skip = interval_start;

                    // we only skip up to a run (yield a single skip per multiple runs)
                    let max_skip = full_run_length - offset;
                    let to_skip = to_skip.min(max_skip);

                    let set = is_set_count(values, offset, to_skip);

                    self.current_items_in_runs += to_skip;

                    self.current = if to_skip == max_skip {
                        None
                    } else {
                        Some((run, offset + to_skip))
                    };

                    return Some(Ok(FilteredHybridEncoded::Skipped(set)));
                };

                // slice the bitmap according to current interval
                // note that interval start is from the start of the first run.
                let new_offset = offset + interval_start;

                if interval_start > run_length {
                    let set = is_set_count(values, offset, full_run_length);

                    self.advance_current_interval(run_length);
                    self.current_items_in_runs += run_length;
                    self.current = None;
                    Some(Ok(FilteredHybridEncoded::Skipped(set)))
                } else {
                    let length = if run_length > interval.length {
                        // interval is fully consumed
                        self.current_items_in_runs += interval.length;

                        // fetch next interval
                        self.total_items -= interval.length;
                        self.current_interval = self.selected_rows.pop_front();

                        self.current = Some((run, offset + interval.length));

                        interval.length
                    } else {
                        // the run is consumed and the interval is shortened accordingly
                        self.current_items_in_runs += run_length;

                        // the interval may cover two runs; shorten the length
                        // to its maximum allowed for this run
                        let length = run_length.min(full_run_length - new_offset);

                        self.advance_current_interval(length);

                        self.current = None;
                        length
                    };
                    Some(Ok(FilteredHybridEncoded::Bitmap {
                        values,
                        offset: new_offset,
                        length,
                    }))
                }
            },
        }
    }
}
