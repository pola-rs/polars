/// Given a `slice` that is relative to the start of a list of files, calculate the slice to apply
/// at a file with a row offset of `current_row_offset`.
pub fn split_slice_at_file(
    current_row_offset: &mut usize,
    n_rows_this_file: usize,
    global_slice_start: usize,
    global_slice_end: usize,
) -> (usize, usize) {
    let next_file_offset = *current_row_offset + n_rows_this_file;
    // e.g.
    // slice: (start: 1, end: 2)
    // files:
    //   0: (1 row): current_offset: 0, next_file_offset: 1
    //   1: (1 row): current_offset: 1, next_file_offset: 2
    //   2: (1 row): current_offset: 2, next_file_offset: 3
    // in this example we want to include only file 1.
    let has_overlap_with_slice =
        *current_row_offset < global_slice_end && next_file_offset > global_slice_start;

    let (rel_start, slice_len) = if !has_overlap_with_slice {
        (0, 0)
    } else {
        let n_rows_to_skip = global_slice_start.saturating_sub(*current_row_offset);
        let n_excess_rows = next_file_offset.saturating_sub(global_slice_end);
        (
            n_rows_to_skip,
            n_rows_this_file - n_rows_to_skip - n_excess_rows,
        )
    };

    *current_row_offset = next_file_offset;
    (rel_start, slice_len)
}
