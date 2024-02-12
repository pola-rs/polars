use std::pin::Pin;

use polars_core::series::unstable::UnstableSeries;

use super::*;

impl<'a> AggregationContext<'a> {
    /// # Safety
    /// The lifetime of [UnstableSeries] is bound to the iterator. Keeping it alive
    /// longer than the iterator is UB.
    pub(super) unsafe fn iter_groups(
        &mut self,
        keep_names: bool,
    ) -> Box<dyn Iterator<Item = Option<UnstableSeries<'_>>> + '_> {
        match self.agg_state() {
            AggState::Literal(_) => {
                self.groups();
                let s = self.series().rechunk();
                let name = if keep_names { s.name() } else { "" };
                // SAFETY: dtype is correct
                unsafe {
                    Box::new(LitIter::new(
                        s.array_ref(0).clone(),
                        self.groups.len(),
                        s._dtype(),
                        name,
                    ))
                }
            },
            AggState::AggregatedScalar(_) => {
                self.groups();
                let s = self.series();
                let name = if keep_names { s.name() } else { "" };
                // SAFETY: dtype is correct
                unsafe {
                    Box::new(FlatIter::new(
                        s.array_ref(0).clone(),
                        self.groups.len(),
                        s.dtype(),
                        name,
                    ))
                }
            },
            AggState::AggregatedList(_) => {
                let s = self.series();
                let list = s.list().unwrap();
                let name = if keep_names { s.name() } else { "" };
                Box::new(list.amortized_iter_with_name(name))
            },
            AggState::NotAggregated(_) => {
                // we don't take the owned series as we want a reference
                let _ = self.aggregated();
                let s = self.series();
                let list = s.list().unwrap();
                let name = if keep_names { s.name() } else { "" };
                Box::new(list.amortized_iter_with_name(name))
            },
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
    /// # Safety
    /// Caller must ensure the given `logical` dtype belongs to `array`.
    unsafe fn new(array: ArrayRef, len: usize, logical: &DataType, name: &str) -> Self {
        let mut series_container = Box::pin(Series::from_chunks_and_dtype_unchecked(
            name,
            vec![array],
            logical,
        ));

        let ref_s = &mut *series_container as *mut Series;
        Self {
            offset: 0,
            len,
            series_container,
            // SAFETY: we pinned the series so the location is still valid
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
    /// # Safety
    /// Caller must ensure the given `logical` dtype belongs to `array`.
    unsafe fn new(array: ArrayRef, len: usize, logical: &DataType, name: &str) -> Self {
        let mut series_container = Box::pin(Series::from_chunks_and_dtype_unchecked(
            name,
            vec![array.clone()],
            logical,
        ));
        let ref_s = &mut *series_container as *mut Series;
        Self {
            array,
            offset: 0,
            len,
            series_container,
            // SAFETY: we pinned the series so the location is still valid
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
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.offset))
    }
}
