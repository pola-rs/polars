use bitflags::bitflags;

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ReaderCapabilities: u8 {
        /// Supports attaching a row index column.
        ///
        /// Readers may want to implement this if they implement any of:
        /// * NEGATIVE_PRE_SLICE
        /// * SPECIALIZED_FILTER
        ///
        /// If any of the above operations are requested alongside ROW_INDEX, they cannot be pushed
        /// into the reader if ROW_INDEX is not supported in the reader.
        ///
        /// ROW_INDEX will not be needed (or called) for PRE_SLICE, as it gets optimized to be applied
        /// after the PRE_SLICE instead by adjusting the offset.
        const ROW_INDEX = 1 << 0;

        /// Supports slicing with offsets relative to the start of the file (i.e. `offset >= 0 / Slice::Positive`).
        const PRE_SLICE = 1 << 1;

        /// Supports slicing with offsets relative to the end of the file (i.e. `offset < 0 / Slice::Negative`)
        const NEGATIVE_PRE_SLICE = 1 << 2;

        /// Supports specialized filtering (e.g. through the use of metadata) but may not filter
        /// out all rows that don't match the predicate.
        const PARTIAL_FILTER = 1 << 3;

        /// Supports specialized filtering (e.g. through the use of metadata) and will always
        /// filter out all rows that don't match the predicate.
        ///
        /// `PARTIAL_FILTER` should also be enabled if this is enabled.
        const FULL_FILTER = 1 << 4;

        /// The reader supports being passed `Projection::Mapped`.
        const MAPPED_COLUMN_PROJECTION = 1 << 5;

        /// Supports applying an external filter mask.
        const EXTERNAL_FILTER_MASK = 1 << 6;

        /// Signals to the multi-scan pipeline to initialize cloud paths in the file cache before
        /// starting the reader.
        const NEEDS_FILE_CACHE_INIT = 1 << 7;
    }
}
