#![allow(unsafe_op_in_unsafe_fn)]
use std::rc::Rc;

use polars_core::series::amortized_iter::AmortSeries;

use super::*;

impl AggregationContext<'_> {
    pub(super) fn iter_groups(
        &mut self,
        keep_names: bool,
    ) -> Box<dyn Iterator<Item = Option<AmortSeries>> + '_> {
        match self.agg_state() {
            AggState::LiteralScalar(_) => {
                self.groups();
                let c = self.get_values().rechunk();
                let name = if keep_names {
                    c.name().clone()
                } else {
                    PlSmallStr::EMPTY
                };
                // SAFETY: dtype is correct
                unsafe {
                    Box::new(LitIter::new(
                        c.as_materialized_series().array_ref(0).clone(),
                        self.groups.len(),
                        c.dtype(),
                        name,
                    ))
                }
            },
            AggState::AggregatedScalar(_) => {
                self.groups();
                let c = self.get_values();
                let name = if keep_names {
                    c.name().clone()
                } else {
                    PlSmallStr::EMPTY
                };
                // SAFETY: dtype is correct
                unsafe {
                    Box::new(FlatIter::new(
                        c.as_materialized_series().chunks(),
                        self.groups.len(),
                        c.dtype(),
                        name,
                    ))
                }
            },
            AggState::AggregatedList(_) => {
                let c = self.get_values();
                let list = c.list().unwrap();
                let name = if keep_names {
                    c.name().clone()
                } else {
                    PlSmallStr::EMPTY
                };
                Box::new(list.amortized_iter_with_name(name))
            },
            AggState::NotAggregated(_) => {
                // we don't take the owned series as we want a reference
                let _ = self.aggregated();
                let c = self.get_values();
                let list = c.list().unwrap();
                let name = if keep_names {
                    c.name().clone()
                } else {
                    PlSmallStr::EMPTY
                };
                Box::new(list.amortized_iter_with_name(name))
            },
        }
    }
}

impl AggregationContext<'_> {
    /// Iterate over groups without greedy aggregation into an AggList.
    pub(super) fn iter_groups_lazy(
        &mut self,
        keep_names: bool,
    ) -> Box<dyn Iterator<Item = Option<AmortSeries>> + '_> {
        match self.agg_state() {
            AggState::NotAggregated(_) => {
                let groups = self.groups();
                let len = groups.len();
                let c = self.get_values().rechunk(); //TODO - do we require rechunk?
                let name = if keep_names {
                    c.name().clone()
                } else {
                    PlSmallStr::EMPTY
                };
                let iter = self.groups().iter();

                // Safety:
                // kdn TODO
                unsafe {
                    Box::new(NotAggLazyIter::new(
                        c.as_materialized_series().array_ref(0).clone(),
                        iter,
                        len,
                        c.dtype(),
                        name,
                    ))
                }
            },
            _ => self.iter_groups(keep_names),
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
    unsafe fn new(array: ArrayRef, len: usize, logical: &DataType, name: PlSmallStr) -> Self {
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
    unsafe fn new(chunks: &[ArrayRef], len: usize, logical: &DataType, name: PlSmallStr) -> Self {
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
        (self.len - self.offset, Some(self.len - self.offset))
    }
}

struct NotAggLazyIter<'a, I: Iterator<Item = GroupsIndicator<'a>>> {
    array: ArrayRef,
    iter: I,
    groups_idx: usize,
    len: usize,
    // AmortSeries referenced that series
    #[allow(dead_code)]
    series_container: Rc<Series>,
    item: AmortSeries,
}

impl<'a, I: Iterator<Item = GroupsIndicator<'a>>> NotAggLazyIter<'a, I> {
    /// # Safety
    /// kdn TODO
    unsafe fn new(
        array: ArrayRef,
        iter: I,
        len: usize,
        logical: &DataType,
        name: PlSmallStr,
    ) -> Self {
        let series_container = Rc::new(Series::from_chunks_and_dtype_unchecked(
            name,
            vec![array.clone()],
            logical,
        ));
        Self {
            array,
            iter,
            groups_idx: 0,
            len,
            series_container: series_container.clone(),
            item: AmortSeries::new(series_container),
        }
    }
}

impl<'a, I: Iterator<Item = GroupsIndicator<'a>>> Iterator for NotAggLazyIter<'a, I> {
    type Item = Option<AmortSeries>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(g) = self.iter.next() {
            self.groups_idx += 1;
            match g {
                GroupsIndicator::Idx(_) => todo!(), //kdn TODO
                GroupsIndicator::Slice(s) => {
                    let mut arr =
                        unsafe { self.array.sliced_unchecked(s[0] as usize, s[1] as usize) };
                    unsafe { self.item.swap(&mut arr) };
                    Some(Some(self.item.clone()))
                },
            }
        } else {
            None
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len - self.groups_idx, Some(self.len - self.groups_idx))
    }
}
