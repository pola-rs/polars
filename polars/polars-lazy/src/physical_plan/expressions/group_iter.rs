use std::pin::Pin;

use polars_core::series::unstable::UnstableSeries;

use super::*;

impl<'a> AggregationContext<'a> {
    pub(super) fn iter_groups(
        &mut self,
    ) -> Box<dyn Iterator<Item = Option<UnstableSeries<'_>>> + '_> {
        match self.agg_state() {
            AggState::Literal(_) => {
                self.groups();
                let s = self.series();
                Box::new(LitIter::new(s.array_ref(0).clone(), self.groups.len()))
            }
            AggState::AggregatedFlat(_) => {
                self.groups();
                let s = self.series();
                Box::new(FlatIter::new(s.array_ref(0).clone(), self.groups.len()))
            }
            AggState::AggregatedList(_) => {
                let s = self.series();
                let list = s.list().unwrap();
                Box::new(list.amortized_iter())
            }
            AggState::NotAggregated(_) => {
                // we don't take the owned series as we want a reference
                let _ = self.aggregated();
                let s = self.series();
                let list = s.list().unwrap();
                Box::new(list.amortized_iter())
            }
        }
    }
}

struct LitIter<'a> {
    len: usize,
    offset: usize,
    // UnstableSeries referenced that series
    #[allow(dead_code)]
    series_container: Pin<Box<Series>>,
    item: UnstableSeries<'a>,
}

impl<'a> LitIter<'a> {
    fn new(array: ArrayRef, len: usize) -> Self {
        let mut series_container = Box::pin(Series::try_from(("", array.clone())).unwrap());
        let ref_s = &mut *series_container as *mut Series;
        Self {
            offset: 0,
            len,
            series_container,
            // Safety: we pinned the series so the location is still valid
            item: UnstableSeries::new(unsafe { &mut *ref_s }),
        }
    }
}

impl<'a> Iterator for LitIter<'a> {
    type Item = Option<UnstableSeries<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len == self.offset {
            None
        } else {
            self.offset += 1;
            Some(Some(self.item))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

struct FlatIter<'a> {
    array: ArrayRef,
    offset: usize,
    len: usize,
    // UnstableSeries referenced that series
    #[allow(dead_code)]
    series_container: Pin<Box<Series>>,
    item: UnstableSeries<'a>,
}

impl<'a> FlatIter<'a> {
    fn new(array: ArrayRef, len: usize) -> Self {
        let mut series_container = Box::pin(Series::try_from(("", array.clone())).unwrap());
        let ref_s = &mut *series_container as *mut Series;
        Self {
            array,
            offset: 0,
            len,
            series_container,
            // Safety: we pinned the series so the location is still valid
            item: UnstableSeries::new(unsafe { &mut *ref_s }),
        }
    }
}

impl<'a> Iterator for FlatIter<'a> {
    type Item = Option<UnstableSeries<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len == self.offset {
            None
        } else {
            let mut arr = unsafe { self.array.sliced_unchecked(self.offset, 1) };
            self.offset += 1;
            self.item.swap(&mut arr);
            Some(Some(self.item))
        }
    }
}
