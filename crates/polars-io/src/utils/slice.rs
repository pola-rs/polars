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
        global_slice_start..global_slice_end,
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
    pub fn split_slice_at_file(
        current_row_offset: usize,
        n_rows_this_file: usize,
        global_slice: std::ops::Range<usize>,
    ) -> Self {
        // e.g.
        // slice: (start: 1, end: 2)
        // files:
        //   0: (1 row): current_offset: 0, next_file_offset: 1
        //   1: (1 row): current_offset: 1, next_file_offset: 2
        //   2: (1 row): current_offset: 2, next_file_offset: 3
        // in this example we want to include only file 1.

        let next_row_offset = current_row_offset + n_rows_this_file;

        if next_row_offset <= global_slice.start {
            Self::Before
        } else if current_row_offset >= global_slice.end {
            Self::After
        } else {
            let n_rows_to_skip = global_slice.start.saturating_sub(current_row_offset);
            let n_excess_rows = next_row_offset.saturating_sub(global_slice.end);

            Self::Overlapping(
                n_rows_to_skip,
                n_rows_this_file - n_rows_to_skip - n_excess_rows,
            )
        }
    }
}
