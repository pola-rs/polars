use polars_utils::slice_enum::Slice;

/// Given a `slice` that is relative to the start of a list of files, calculate the slice to apply
/// at a file with a row offset of `current_row_offset`.
pub fn split_slice_at_file(
    current_row_offset_ref: &mut usize,
    n_rows_this_file: usize,
    global_slice_start: usize,
    global_slice_end: usize,
) -> (usize, usize) {
    let current_row_offset = *current_row_offset_ref;
    *current_row_offset_ref += n_rows_this_file;
    match SplitSlicePosition::split_slice_at_file(
        current_row_offset,
        n_rows_this_file,
        Slice::from(global_slice_start..global_slice_end),
    ) {
        SplitSlicePosition::Overlapping(offset, len) => (offset, len),
        SplitSlicePosition::Before | SplitSlicePosition::After => (0, 0),
    }
}

#[derive(Debug)]
pub enum SplitSlicePosition {
    Before,
    Overlapping(usize, usize),
    After,
}

impl SplitSlicePosition {
    /// # Panics
    /// Panics if `global_slice` is [`Slice::Negative`].
    pub fn split_slice_at_file(
        current_row_offset: usize,
        n_rows_this_file: usize,
        global_slice: Slice,
    ) -> Self {
        assert!(global_slice.is_positive());

        // e.g.
        // slice: (offset: 1, len: 1)
        // files:
        //   0: (1 row): current_offset: 0, next_file_offset: 1
        //   1: (1 row): current_offset: 1, next_file_offset: 2
        //   2: (1 row): current_offset: 2, next_file_offset: 3
        // in this example we want to include only file 1.

        let next_row_offset = current_row_offset + n_rows_this_file;

        if next_row_offset <= global_slice.positive_offset() {
            Self::Before
        } else if global_slice
            .end_position()
            .is_some_and(|end| current_row_offset >= end)
        {
            Self::After
        } else {
            let range = global_slice
                .offsetted(current_row_offset)
                .map_or(0..n_rows_this_file, |s| s.resolve(n_rows_this_file));
            Self::Overlapping(range.start, range.len())
        }
    }
}
