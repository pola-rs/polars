/// Converts numeric type to a `String`
#[inline]
pub fn lexical_to_bytes<N: lexical_core::ToLexical>(n: N) -> Vec<u8> {
    let mut buf = Vec::<u8>::with_capacity(N::FORMATTED_SIZE_DECIMAL);
    lexical_to_bytes_mut(n, &mut buf);
    buf
}

/// Converts numeric type to a `String`
#[inline]
pub fn lexical_to_bytes_mut<N: lexical_core::ToLexical>(n: N, buf: &mut Vec<u8>) {
    buf.clear();
    buf.reserve(N::FORMATTED_SIZE_DECIMAL);
    unsafe {
        // JUSTIFICATION
        //  Benefit
        //      Allows using the faster serializer lexical core and convert to string
        //  Soundness
        //      Length of buf is set as written length afterwards. lexical_core
        //      creates a valid string, so doesn't need to be checked.
        let slice = std::slice::from_raw_parts_mut(buf.as_mut_ptr(), buf.capacity());

        //  Safety:
        //  Omits an unneeded bound check as we just ensured that we reserved `N::FORMATTED_SIZE_DECIMAL`
        #[cfg(debug_assertions)]
        {
            let len = lexical_core::write(n, slice).len();
            buf.set_len(len);
        }
        #[cfg(not(debug_assertions))]
        {
            let len = lexical_core::write_unchecked(n, slice).len();
            buf.set_len(len);
        }
    }
}

/// Converts numeric type to a `String`
#[inline]
pub fn lexical_to_string<N: lexical_core::ToLexical>(n: N) -> String {
    unsafe { String::from_utf8_unchecked(lexical_to_bytes(n)) }
}
