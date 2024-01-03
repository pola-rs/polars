pub struct View {
    /// The length of the string/bytes.
    pub length: u32,
    /// First 4 bytes of string/bytes data.
    pub prefix: u32,
    /// The buffer index.
    pub buffer_idx: u32,
    /// The offset into the buffer.
    pub offset: u32,
}

impl From<u128> for View {
    #[inline]
    fn from(value: u128) -> Self {
        Self {
            length: value as u32,
            prefix: (value >> 64) as u32,
            buffer_idx: (value >> 64) as u32,
            offset: (value >> 96) as u32,
        }
    }
}

impl From<View> for u128 {
    #[inline]
    fn from(value: View) -> Self {
        value.length as u128
            | ((value.prefix as u128) << 32)
            | ((value.buffer_idx as u128) << 64)
            | ((value.offset as u128) << 96)
    }
}
