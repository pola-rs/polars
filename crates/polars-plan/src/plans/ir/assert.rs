bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, std::hash::Hash, PartialEq, Eq)]
    #[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct AssertFlags: u8 {
        const WARN_ON_FAIL       = 0x01;
        const ALLOW_PREDICATE_PD = 0x02;
    }
}
