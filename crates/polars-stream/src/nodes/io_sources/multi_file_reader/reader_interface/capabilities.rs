use bitflags::bitflags;

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ReaderCapabilities: u8 {
        /// ROW_INDEX: Readers may want to implement this if they implement any of:
        /// * NEGATIVE_PRE_SLICE
        /// * FILTER
        ///
        /// If any of the above operations are requested alongside ROW_INDEX, they cannot be pushed
        /// into the reader if ROW_INDEX is not supported in the reader.
        ///
        /// ROW_INDEX will not be needed (or called) for PRE_SLICE, as it gets optimized to be applied
        /// after the PRE_SLICE instead by adjusting the offset.
        const ROW_INDEX          = 1 << 0;
        const PRE_SLICE          = 1 << 1;
        const NEGATIVE_PRE_SLICE = 1 << 2;
        const FILTER             = 1 << 3;
    }
}
