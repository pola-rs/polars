/// Parse a UUID using the common Python/RFC spellings and PostgreSQL's
/// additional allowance for hyphens between hexadecimal groups.
///
/// The canonical, simple, braced, and URN forms stay on `uuid`'s allocation-free
/// fast path. The fallback only allocates for non-standard hyphen placement.
pub fn parse_uuid_str(value: &str) -> Option<u128> {
    if let Ok(value) = uuid::Uuid::parse_str(value) {
        return Some(value.as_u128());
    }

    let value = value
        .strip_prefix("urn:uuid:")
        .or_else(|| value.strip_prefix("URN:UUID:"))
        .unwrap_or(value);
    let value = value
        .strip_prefix('{')
        .and_then(|value| value.strip_suffix('}'))
        .unwrap_or(value);

    if !value.contains('-') {
        return None;
    }

    let mut normalized = String::with_capacity(32);
    let mut digits_since_hyphen = 0usize;
    for byte in value.bytes() {
        if byte == b'-' {
            if digits_since_hyphen == 0 || digits_since_hyphen % 4 != 0 {
                return None;
            }
            digits_since_hyphen = 0;
        } else if byte.is_ascii_hexdigit() {
            normalized.push(char::from(byte));
            digits_since_hyphen += 1;
        } else {
            return None;
        }
    }
    if normalized.len() != 32 || digits_since_hyphen == 0 {
        return None;
    }

    uuid::Uuid::parse_str(&normalized)
        .ok()
        .map(|value| value.as_u128())
}

#[cfg(test)]
mod tests {
    use super::parse_uuid_str;

    const VALUE: u128 = 0xa0eebc999c0b4ef8bb6d6bb9bd380a11;

    #[test]
    fn parses_interoperable_text_forms() {
        for value in [
            "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
            "A0EEBC999C0B4EF8BB6D6BB9BD380A11",
            "a0eebc999c0b4ef8bb6d6bb9bd380a11",
            "{a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11}",
            "urn:uuid:a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
            "a0ee-bc99-9c0b-4ef8-bb6d-6bb9-bd38-0a11",
        ] {
            assert_eq!(parse_uuid_str(value), Some(VALUE), "{value}");
        }
    }

    #[test]
    fn rejects_invalid_text_forms() {
        for value in [
            "",
            "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a1",
            "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a1z",
            "a0e-ebc999c0b4ef8bb6d6bb9bd380a11",
            "-a0eebc999c0b4ef8bb6d6bb9bd380a11",
        ] {
            assert_eq!(parse_uuid_str(value), None, "{value}");
        }
    }
}
