use std::rc::Rc;

use polars_core::series::amortized_iter::AmortSeries;

use super::*;

impl<'a> AggregationContext<'a> {
    pub(super) fn iter_groups(
        &mut self,
        keep_names: bool,
    ) -> Box<dyn Iterator<Item = Option<AmortSeries>> + '_> {
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
                        s.chunks(),
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

struct LitIter {
    len: usize,
    offset: usize,
    // AmortSeries referenced that series
    #[allow(dead_code)]
    series_container: Rc<Series>,
    item: AmortSeries,
}

impl LitIter {
    /// # Safety
    /// Caller must ensure the given `logical` dtype belongs to `array`.
    unsafe fn new(array: ArrayRef, len: usize, logical: &DataType, name: &str) -> Self {
        let series_container = Rc::new(Series::from_chunks_and_dtype_unchecked(
            name,
            vec![array],
            logical,
        ));

        Self {
            offset: 0,
            len,
            series_container: series_container.clone(),
            // SAFETY: we pinned the series so the location is still valid
            item: AmortSeries::new(series_container),
        }
    }
}

impl Iterator for LitIter {
    type Item = Option<AmortSeries>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len == self.offset {
            None
        } else {
            self.offset += 1;
            Some(Some(self.item.clone()))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

struct FlatIter {
    current_array: ArrayRef,
    chunks: Vec<ArrayRef>,
    offset: usize,
    chunk_offset: usize,
    len: usize,
    // AmortSeries referenced that series
    #[allow(dead_code)]
    series_container: Rc<Series>,
    item: AmortSeries,
}

impl FlatIter {
    /// # Safety
    /// Caller must ensure the given `logical` dtype belongs to `array`.
    unsafe fn new(chunks: &[ArrayRef], len: usize, logical: &DataType, name: &str) -> Self {
        let mut stack = Vec::with_capacity(chunks.len());
        for chunk in chunks.iter().rev() {
            stack.push(chunk.clone())
        }
        let current_array = stack.pop().unwrap();
        let series_container = Rc::new(Series::from_chunks_and_dtype_unchecked(
            name,
            vec![current_array.clone()],
            logical,
        ));
        Self {
            current_array,
            chunks: stack,
            offset: 0,
            chunk_offset: 0,
            len,
            series_container: series_container.clone(),
            item: AmortSeries::new(series_container),
        }
    }
}

impl Iterator for FlatIter {
    type Item = Option<AmortSeries>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len == self.offset {
            None
        } else {
            if self.chunk_offset < self.current_array.len() {
                let mut arr = unsafe { self.current_array.sliced_unchecked(self.chunk_offset, 1) };
                unsafe { self.item.swap(&mut arr) };
            } else {
                match self.chunks.pop() {
                    Some(arr) => {
                        self.current_array = arr;
                        self.chunk_offset = 0;
                        return self.next();
                    },
                    None => return None,
                }
            }
            self.offset += 1;
            self.chunk_offset += 1;
            Some(Some(self.item.clone()))
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.offset))
    }
}
