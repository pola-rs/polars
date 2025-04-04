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
        const ROW_INDEX          = 1 << 0;

        /// Supports slicing with offsets relative to the start of the file (i.e. `offset >= 0 / Slice::Positive`).
        const PRE_SLICE          = 1 << 1;

        /// Supports slicing with offsets relative to the end of the file (i.e. `offset < 0 / Slice::Negative`)
        const NEGATIVE_PRE_SLICE = 1 << 2;

        /// Supports specialized filtering (e.g. through the use of metadata).
        const SPECIALIZED_FILTER = 1 << 3;
    }
}
