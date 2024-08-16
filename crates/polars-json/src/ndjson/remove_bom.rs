pub fn remove_bom(bytes: &[u8]) -> &[u8] {
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        // UTF-8 BOM
        &bytes[3..]
    } else if bytes.starts_with(&[0xFE, 0xFF]) || bytes.starts_with(&[0xFF, 0xFE]) {
        // UTF-16 BOM
        &bytes[2..]
    } else {
        bytes
    }
}