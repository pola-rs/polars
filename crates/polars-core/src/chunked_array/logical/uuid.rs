use super::*;
use crate::prelude::arity::unary_elementwise;
use crate::prelude::*;

pub type UuidChunked = Logical<UuidType, UInt128Type>;

/// Parse a UUID using the common Python/RFC spellings and PostgreSQL's
/// additional allowance for hyphens between hexadecimal groups.
///
/// The canonical, simple, braced, and URN forms stay on `uuid`'s allocation-free
/// fast path. The fallback only allocates for non-standard hyphen placement.
pub fn parse_uuid_str(value: &str) -> Option<u128> {
    if let Ok(value) = ::uuid::Uuid::parse_str(value) {
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

    ::uuid::Uuid::parse_str(&normalized)
        .ok()
        .map(|value| value.as_u128())
}

impl UInt128Chunked {
    pub fn into_uuid(self) -> UuidChunked {
        // SAFETY: every 128-bit bit pattern is a valid UUID value.
        unsafe { UuidChunked::new_logical(self, DataType::Uuid) }
    }
}

impl UuidChunked {
    /// Extract the four-bit UUID version field.
    pub fn version(&self) -> UInt8Chunked {
        unary_elementwise(&self.phys, |value| {
            value.map(|value| ((value >> 76) & 0x0f) as u8)
        })
    }

    /// Extract the Unix epoch millisecond timestamp embedded in UUIDv7 values.
    ///
    /// Non-v7 values are rejected in strict mode and become null otherwise.
    pub fn timestamp_ms(&self, strict: bool) -> PolarsResult<Int64Chunked> {
        let mut out = PrimitiveChunkedBuilder::<Int64Type>::new(self.name().clone(), self.len());
        for value in self.phys.iter() {
            match value {
                None => out.append_null(),
                Some(value) if ((value >> 76) & 0x0f) == 7 => {
                    out.append_value((value >> 80) as i64)
                },
                Some(value) if strict => {
                    polars_bail!(
                        ComputeError:
                        "cannot extract a UUIDv7 timestamp from UUID version {}",
                        (value >> 76) & 0x0f,
                    )
                },
                Some(_) => out.append_null(),
            }
        }
        Ok(out.finish())
    }
}

impl LogicalType for UuidChunked {
    fn dtype(&self) -> &DataType {
        &DataType::Uuid
    }

    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        self.phys.get_any_value(i).map(|av| av.as_uuid())
    }

    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        self.phys.get_any_value_unchecked(i).as_uuid()
    }

    fn cast_with_options(
        &self,
        dtype: &DataType,
        _cast_options: CastOptions,
    ) -> PolarsResult<Series> {
        match dtype {
            DataType::Uuid => Ok(self.clone().into_series()),
            DataType::UInt128 => Ok(self.phys.clone().into_series()),
            DataType::String => {
                let mut out = StringChunkedBuilder::new(self.name().clone(), self.len());
                let mut buffer = ::uuid::Uuid::encode_buffer();
                for value in self.phys.iter() {
                    match value {
                        Some(value) => out.append_value(
                            ::uuid::Uuid::from_u128(value)
                                .as_hyphenated()
                                .encode_lower(&mut buffer),
                        ),
                        None => out.append_null(),
                    }
                }
                Ok(out.finish().into_series())
            },
            DataType::Binary => {
                let out = self
                    .phys
                    .iter()
                    .map(|value| value.map(u128::to_be_bytes))
                    .collect::<BinaryChunked>();
                Ok(out.with_name(self.name().clone()).into_series())
            },
            dtype => polars_bail!(InvalidOperation: "cannot cast UUID to {dtype}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_uuid_str;

    const VALUE: u128 = 0xa0eebc999c0b4ef8bb6d6bb9bd380a11;

    #[test]
    fn parses_interoperable_text_forms() {
        for value in [
            "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
            "A0EEBC99-9C0B-4EF8-BB6D-6BB9BD380A11",
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

    #[test]
    fn extracts_version_and_v7_timestamp() {
        use crate::prelude::*;

        let values = UInt128Chunked::from_iter_options(
            "id".into(),
            [
                parse_uuid_str("019482e4-1441-7aad-8127-eec99573b0a0"),
                parse_uuid_str("a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"),
                None,
            ]
            .into_iter(),
        )
        .into_uuid();

        assert_eq!(
            values.version().iter().collect::<Vec<_>>(),
            [Some(7), Some(4), None]
        );
        assert!(values.timestamp_ms(true).is_err());
        assert_eq!(
            values
                .timestamp_ms(false)
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            [Some(1737362773057), None, None]
        );
    }

    #[test]
    fn uuid_series_iteration_and_cast_policy() {
        use crate::prelude::*;

        let series =
            UInt128Chunked::from_iter_options("id".into(), [Some(VALUE), None].into_iter())
                .into_uuid()
                .into_series();

        assert_eq!(series.iter().next(), Some(AnyValue::Uuid(VALUE)));
        assert_eq!(
            series
                .cast(&DataType::Binary)
                .unwrap()
                .binary()
                .unwrap()
                .get(0),
            Some(VALUE.to_be_bytes().as_slice())
        );
        assert!(series.cast(&DataType::Float64).is_err());
        assert!(series.cast(&DataType::Boolean).is_err());
    }
}
