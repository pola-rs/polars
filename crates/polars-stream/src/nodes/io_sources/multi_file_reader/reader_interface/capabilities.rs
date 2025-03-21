use bitflags::bitflags;

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ReaderCapabilities: u8 {
        const ROW_INDEX          = 1 << 0;
        const PRE_SLICE          = 1 << 1;
        const NEGATIVE_PRE_SLICE = 1 << 2;
        const FILTER             = 1 << 3;
    }
}
