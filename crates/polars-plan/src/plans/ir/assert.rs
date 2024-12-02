#[derive(Clone, Copy, Debug, std::hash::Hash)]
#[repr(u8)]
#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OnAssertionFail {
    Warn,
    Error,
}

impl std::fmt::Display for OnAssertionFail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OnAssertionFail::Warn => f.write_str("warn"),
            OnAssertionFail::Error => f.write_str("error"),
        }
    }
}
