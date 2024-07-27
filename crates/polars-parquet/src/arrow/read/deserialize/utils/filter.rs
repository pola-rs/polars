use arrow::bitmap::Bitmap;

#[derive(Debug, Clone)]
pub(crate) enum FilterSlice {
    Range(usize, usize),
    Mask(Bitmap),
}

#[derive(Debug, Clone)]
pub struct Filter {
    pub(crate) num_remaining: usize,
    pub(crate) slice: FilterSlice,
}

impl Filter {
    pub fn new_limited(x: usize) -> Self {
        Self {
            num_remaining: x,
            slice: FilterSlice::Range(0, x),
        }
    }

    pub fn new_ranged(start: usize, end: usize) -> Self {
        Self {
            num_remaining: end - start,
            slice: FilterSlice::Range(start, end),
        }
    }

    pub fn new_masked(mask: Bitmap, num_elements: usize) -> Self {
        debug_assert_eq!(mask.set_bits(), num_elements);

        Self {
            num_remaining: num_elements,
            slice: FilterSlice::Mask(mask),
        }
    }

    pub(crate) fn head(&self, num_cells: usize) -> FilterSlice {
        use FilterSlice as FS;
        match &self.slice {
            FS::Range(start, end) => {
                FS::Range(usize::min(*start, num_cells), usize::min(*end, num_cells))
            },
            FS::Mask(bitmap) => FS::Mask(bitmap.clone().sliced(0, num_cells)),
        }
    }

    pub(crate) fn advance_by(&mut self, num_cells: usize, num_values: usize) {
        use FilterSlice as FS;
        match &mut self.slice {
            FS::Range(start, end) => {
                *start = start.saturating_sub(num_cells);
                *end = end.saturating_sub(num_cells);
            },
            FS::Mask(bitmap) => {
                let new_length = bitmap.len().saturating_sub(num_cells);
                bitmap.slice(num_cells, new_length);
            },
        }
        self.num_remaining -= num_values;
    }

    pub(crate) fn opt_head(filter: &Option<Self>, num_cells: usize) -> Option<FilterSlice> {
        Some(filter.as_ref()?.head(num_cells))
    }

    pub(crate) fn opt_advance_by(filter: &mut Option<Filter>, num_cells: usize, num_values: usize) {
        if let Some(filter) = filter.as_mut() {
            filter.advance_by(num_cells, num_values);
        }
    }

    pub(crate) fn opt_num_rows(filter: &Option<Self>, total_num_rows: usize) -> usize {
        match filter {
            Some(filter) => filter.num_remaining,
            None => total_num_rows,
        }
    }
}
