use polars_utils::slice_enum::Slice;

use crate::nodes::io_sources::multi_scan::components::row_counter::RowCounter;
use crate::nodes::io_sources::multi_scan::components::row_deletions::ExternalFilterMask;

/// Represents a [`Slice`] that has been potentially adjusted to account for deleted rows.
#[derive(Debug)]
pub struct PhysicalSlice {
    pub slice: Slice,
    /// Counter that records the number of physical and deleted rows that make up the slice offset,
    /// and `slice_start_position.num_rows() == slice.offset`
    pub slice_start_position: RowCounter,
}

impl PhysicalSlice {
    /// # Panics
    /// Panics if `slice` is [`Slice::Negative`]
    pub fn new(slice: Slice, external_filter_mask: Option<&ExternalFilterMask>) -> Self {
        if let Some(external_filter_mask) = external_filter_mask {
            let requested_offset = slice.positive_offset();

            let physical_slice = external_filter_mask.calc_physical_slice(slice);

            let physical_offset = physical_slice.positive_offset();
            let deleted_in_offset = physical_offset.checked_sub(requested_offset).unwrap();

            let slice_start_position = RowCounter::new(physical_offset, deleted_in_offset);

            PhysicalSlice {
                slice: physical_slice,
                slice_start_position,
            }
        } else {
            let slice_start_position = RowCounter::new(slice.positive_offset(), 0);

            PhysicalSlice {
                slice,
                slice_start_position,
            }
        }
    }
}
