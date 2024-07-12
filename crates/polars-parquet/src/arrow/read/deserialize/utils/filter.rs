use crate::parquet::indexes::Interval;
use crate::parquet::page::DataPage;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Filter<'a> {
    pub(super) selected_rows: &'a [Interval],

    /// Index of the [`Interval`] that is >= `current_index`
    pub(super) current_interval: usize,
    /// Global offset
    pub(super) current_index: usize,
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
