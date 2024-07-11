use super::PageState;
use crate::parquet::error::ParquetResult;
use crate::parquet::indexes::Interval;
use crate::parquet::page::DataPage;

#[derive(Debug)]
pub(crate) struct Filter<'a> {
    selected_rows: &'a [Interval],

    /// Index of the [`Interval`] that is >= `current_index`
    current_interval: usize,
    /// Global offset
    current_index: usize,
}

pub(crate) trait SkipInPlace {
    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()>;
}

impl<'a> Filter<'a> {
    pub fn new(page: &'a DataPage) -> Option<Self> {
        let selected_rows = page.selected_rows()?;

        Some(Self {
            selected_rows,
            current_interval: 0,
            current_index: 0,
        })
    }
}

pub(crate) fn extend_from_state_with_opt_filter<
    'a,
    S: PageState<'a> + SkipInPlace,
    F: FnMut(&mut S, usize) -> ParquetResult<()>,
>(
    state: &mut S,
    filter: &mut Option<Filter>,
    remaining: usize,
    mut collect_fn: F,
) -> ParquetResult<()> {
    match filter {
        None => collect_fn(state, remaining),
        Some(filter) => {
            let mut n = remaining;
            while n > 0 && state.len() > 0 {
                // Skip over all intervals that we have already passed or that are length == 0.
                while filter
                    .selected_rows
                    .get(filter.current_interval)
                    .is_some_and(|iv| {
                        iv.length == 0 || iv.start + iv.length <= filter.current_index
                    })
                {
                    filter.current_interval += 1;
                }

                let Some(iv) = filter.selected_rows.get(filter.current_interval) else {
                    state.skip_in_place(state.len())?;
                    return Ok(());
                };

                // Move to at least the start of the interval
                if filter.current_index < iv.start {
                    state.skip_in_place(iv.start - filter.current_index)?;
                    filter.current_index = iv.start;
                }

                let n_this_round = usize::min(iv.start + iv.length - filter.current_index, n);

                collect_fn(state, n_this_round)?;

                let iv = &filter.selected_rows[filter.current_interval];
                filter.current_index += n_this_round;
                if filter.current_index >= iv.start + iv.length {
                    filter.current_interval += 1;
                }

                n -= n_this_round;
            }

            Ok(())
        },
    }
}
