use super::*;
use polars_arrow::export::arrow::array::ArrayRef;
use polars_core::series::unstable::UnstableSeries;
use std::pin::Pin;

impl<'a> AggregationContext<'a> {
    pub(super) fn iter_groups(
        &mut self,
    ) -> Box<dyn Iterator<Item = Option<UnstableSeries<'_>>> + '_> {
        match self.agg_state() {
            AggState::Literal(_) => {
                let s = self.series();
                let s = UnstableSeries::new(s);
                Box::new(LitIter::new(s, self.groups.len()))
            }
            AggState::AggregatedFlat(_) => {
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
    item: UnstableSeries<'a>,
}

impl<'a> LitIter<'a> {
    fn new(s: UnstableSeries<'a>, len: usize) -> Self {
        Self {
            len,
            offset: 0,
            item: s,
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
        let series_container = Box::pin(Series::try_from(("", array.clone())).unwrap());
        let ref_s = &*series_container as *const Series;
        Self {
            array,
            offset: 0,
            len,
            series_container,
            // Safety: we pinned the series so the location is still valid
            item: UnstableSeries::new(unsafe { &*ref_s }),
        }
    }
}

impl<'a> Iterator for FlatIter<'a> {
    type Item = Option<UnstableSeries<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len == self.offset {
            None
        } else {
            let arr = unsafe { Arc::from(self.array.slice_unchecked(self.offset, 1)) };
            self.offset += 1;
            self.item.swap(arr);
            Some(Some(self.item))
        }
    }
}
