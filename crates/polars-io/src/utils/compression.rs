// magic numbers
pub mod magic {
    pub const GZIP: [u8; 2] = [31, 139];
    pub const ZLIB0: [u8; 2] = [0x78, 0x01];
    pub const ZLIB1: [u8; 2] = [0x78, 0x9C];
    pub const ZLIB2: [u8; 2] = [0x78, 0xDA];
    pub const ZSTD: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];
}

/// check if csv file is compressed
pub fn is_compressed(bytes: &[u8]) -> bool {
    use magic::*;

    bytes.starts_with(&ZLIB0)
        || bytes.starts_with(&ZLIB1)
        || bytes.starts_with(&ZLIB2)
        || bytes.starts_with(&GZIP)
        || bytes.starts_with(&ZSTD)
}
