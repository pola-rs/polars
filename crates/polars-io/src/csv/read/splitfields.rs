#[cfg(not(feature = "simd"))]
mod inner {
    /// An adapted version of std::iter::Split.
    /// This exists solely because we cannot split the lines naively as
    pub(crate) struct SplitFields<'a> {
        v: &'a [u8],
        separator: u8,
        finished: bool,
        quote_char: u8,
        quoting: bool,
        eol_char: u8,
    }

    impl<'a> SplitFields<'a> {
        pub(crate) fn new(
            slice: &'a [u8],
            separator: u8,
            quote_char: Option<u8>,
            eol_char: u8,
        ) -> Self {
            Self {
                v: slice,
                separator,
                finished: false,
                quote_char: quote_char.unwrap_or(b'"'),
                quoting: quote_char.is_some(),
                eol_char,
            }
        }

        unsafe fn finish_eol(
            &mut self,
            need_escaping: bool,
            idx: usize,
        ) -> Option<(&'a [u8], bool)> {
            self.finished = true;
            debug_assert!(idx <= self.v.len());
            Some((self.v.get_unchecked(..idx), need_escaping))
        }

        fn finish(&mut self, need_escaping: bool) -> Option<(&'a [u8], bool)> {
            self.finished = true;
            Some((self.v, need_escaping))
        }

        fn eof_oel(&self, current_ch: u8) -> bool {
            current_ch == self.separator || current_ch == self.eol_char
        }
    }

    impl<'a> Iterator for SplitFields<'a> {
        // the bool is used to indicate that it requires escaping
        type Item = (&'a [u8], bool);

        #[inline]
        fn next(&mut self) -> Option<(&'a [u8], bool)> {
            if self.finished {
                return None;
            } else if self.v.is_empty() {
                return self.finish(false);
            }

            let mut needs_escaping = false;
            // There can be strings with separators:
            // "Street, City",

            // SAFETY:
            // we have checked bounds
            let pos = if self.quoting && unsafe { *self.v.get_unchecked(0) } == self.quote_char {
                needs_escaping = true;
                // There can be pair of double-quotes within string.
                // Each of the embedded double-quote characters must be represented
                // by a pair of double-quote characters:
                // e.g. 1997,Ford,E350,"Super, ""luxurious"" truck",20020

                // denotes if we are in a string field, started with a quote
                let mut in_field = false;

                let mut idx = 0u32;
                let mut current_idx = 0u32;
                // micro optimizations
                #[allow(clippy::explicit_counter_loop)]
                for &c in self.v.iter() {
                    if c == self.quote_char {
                        // toggle between string field enclosure
                        //      if we encounter a starting '"' -> in_field = true;
                        //      if we encounter a closing '"' -> in_field = false;
                        in_field = !in_field;
                    }

                    if !in_field && self.eof_oel(c) {
                        if c == self.eol_char {
                            // SAFETY:
                            // we are in bounds
                            return unsafe {
                                self.finish_eol(needs_escaping, current_idx as usize)
                            };
                        }
                        idx = current_idx;
                        break;
                    }
                    current_idx += 1;
                }

                if idx == 0 {
                    return self.finish(needs_escaping);
                }

                idx as usize
            } else {
                match self.v.iter().position(|&c| self.eof_oel(c)) {
                    None => return self.finish(needs_escaping),
                    Some(idx) => unsafe {
                        // SAFETY:
                        // idx was just found
                        if *self.v.get_unchecked(idx) == self.eol_char {
                            return self.finish_eol(needs_escaping, idx);
                        } else {
                            idx
                        }
                    },
                }
            };

            unsafe {
                debug_assert!(pos <= self.v.len());
                // SAFETY:
                // we are in bounds
                let ret = Some((self.v.get_unchecked(..pos), needs_escaping));
                self.v = self.v.get_unchecked(pos + 1..);
                ret
            }
        }
    }
}

#[cfg(feature = "simd")]
mod inner {
    use std::simd::prelude::*;

    use polars_utils::clmul::prefix_xorsum_inclusive;
    use polars_utils::slice::GetSaferUnchecked;
    use polars_utils::unwrap::UnwrapUncheckedRelease;

    const SIMD_SIZE: usize = 64;
    type SimdVec = u8x64;

    /// An adapted version of std::iter::Split.
    /// This exists solely because we cannot split the lines naively as
    pub(crate) struct SplitFields<'a> {
        pub v: &'a [u8],
        separator: u8,
        pub finished: bool,
        quote_char: u8,
        quoting: bool,
        eol_char: u8,
        simd_separator: SimdVec,
        simd_eol_char: SimdVec,
        simd_quote_char: SimdVec,
        previous_valid_ends: u64,
    }

    impl<'a> SplitFields<'a> {
        pub(crate) fn new(
            slice: &'a [u8],
            separator: u8,
            quote_char: Option<u8>,
            eol_char: u8,
        ) -> Self {
            let simd_separator = SimdVec::splat(separator);
            let simd_eol_char = SimdVec::splat(eol_char);
            let quoting = quote_char.is_some();
            let quote_char = quote_char.unwrap_or(b'"');
            let simd_quote_char = SimdVec::splat(quote_char);

            Self {
                v: slice,
                separator,
                finished: false,
                quote_char,
                quoting,
                eol_char,
                simd_separator,
                simd_eol_char,
                simd_quote_char,
                previous_valid_ends: 0,
            }
        }

        unsafe fn finish_eol(
            &mut self,
            need_escaping: bool,
            idx: usize,
        ) -> Option<(&'a [u8], bool)> {
            self.finished = true;
            debug_assert!(idx <= self.v.len());
            Some((self.v.get_unchecked(..idx), need_escaping))
        }

        #[inline]
        fn finish(&mut self, need_escaping: bool) -> Option<(&'a [u8], bool)> {
            self.finished = true;
            Some((self.v, need_escaping))
        }

        fn eof_oel(&self, current_ch: u8) -> bool {
            current_ch == self.separator || current_ch == self.eol_char
        }
    }

    impl<'a> Iterator for SplitFields<'a> {
        // the bool is used to indicate that it requires escaping
        type Item = (&'a [u8], bool);

        #[inline]
        fn next(&mut self) -> Option<(&'a [u8], bool)> {
            // First check cached value as this is hot.
            if self.previous_valid_ends != 0 {
                let pos = self.previous_valid_ends.trailing_zeros() as usize;
                self.previous_valid_ends >>= (pos + 1) as u64;

                unsafe {
                    debug_assert!(pos < self.v.len());
                    // SAFETY:
                    // we are in bounds
                    let bytes = self.v.get_unchecked_release(..pos);
                    self.v = self.v.get_unchecked_release(pos + 1..);
                    let ret = Some((
                        bytes,
                        bytes
                            .first()
                            .map(|c| *c == self.quote_char && self.quoting)
                            .unwrap_or(false),
                    ));

                    return ret;
                }
            }
            if self.finished {
                return None;
            }
            if self.v.is_empty() {
                return self.finish(false);
            }

            let mut needs_escaping = false;
            // There can be strings with separators:
            // "Street, City",

            // SAFETY:
            // we have checked bounds
            let pos = if self.quoting && unsafe { *self.v.get_unchecked(0) } == self.quote_char {
                let mut total_idx = 0;
                needs_escaping = true;
                let mut not_in_field_previous_iter = true;

                loop {
                    let bytes = unsafe { self.v.get_unchecked_release(total_idx..) };

                    if bytes.len() > SIMD_SIZE {
                        let lane: [u8; SIMD_SIZE] = unsafe {
                            bytes
                                .get_unchecked(0..SIMD_SIZE)
                                .try_into()
                                .unwrap_unchecked_release()
                        };
                        let simd_bytes = SimdVec::from(lane);
                        let has_eol = simd_bytes.simd_eq(self.simd_eol_char);
                        let has_sep = simd_bytes.simd_eq(self.simd_separator);
                        let quote_mask = simd_bytes.simd_eq(self.simd_quote_char).to_bitmask();
                        let mut end_mask = (has_sep | has_eol).to_bitmask();

                        let mut not_in_quote_field = prefix_xorsum_inclusive(quote_mask);

                        if not_in_field_previous_iter {
                            not_in_quote_field = !not_in_quote_field;
                        }
                        not_in_field_previous_iter =
                            (not_in_quote_field & (1 << (SIMD_SIZE - 1))) > 0;
                        end_mask &= not_in_quote_field;

                        if end_mask != 0 {
                            let pos = end_mask.trailing_zeros() as usize;
                            total_idx += pos;
                            debug_assert!(
                                self.v[total_idx] == self.eol_char
                                    || self.v[total_idx] == self.separator
                            );

                            if pos == SIMD_SIZE - 1 {
                                self.previous_valid_ends = 0;
                            } else {
                                self.previous_valid_ends = end_mask >> (pos + 1) as u64;
                            }

                            break;
                        } else {
                            total_idx += SIMD_SIZE;
                        }
                    } else {
                        // There can be a pair of double-quotes within a string.
                        // Each of the embedded double-quote characters must be represented
                        // by a pair of double-quote characters:
                        // e.g. 1997,Ford,E350,"Super, ""luxurious"" truck",20020

                        // denotes if we are in a string field, started with a quote
                        let mut in_field = !not_in_field_previous_iter;

                        // usize::MAX is unset.
                        let mut idx = usize::MAX;
                        let mut current_idx = 0;
                        // micro optimizations
                        #[allow(clippy::explicit_counter_loop)]
                        for &c in bytes.iter() {
                            if c == self.quote_char {
                                // toggle between string field enclosure
                                //      if we encounter a starting '"' -> in_field = true;
                                //      if we encounter a closing '"' -> in_field = false;
                                in_field = !in_field;
                            }

                            if !in_field && self.eof_oel(c) {
                                if c == self.eol_char {
                                    // SAFETY:
                                    // we are in bounds
                                    return unsafe {
                                        self.finish_eol(needs_escaping, current_idx + total_idx)
                                    };
                                }
                                idx = current_idx;
                                break;
                            }
                            current_idx += 1;
                        }

                        if idx == usize::MAX {
                            return self.finish(needs_escaping);
                        }

                        total_idx += idx;
                        debug_assert!(
                            self.v[total_idx] == self.eol_char
                                || self.v[total_idx] == self.separator
                        );
                        break;
                    }
                }
                total_idx
            } else {
                let mut total_idx = 0;

                loop {
                    let bytes = unsafe { self.v.get_unchecked_release(total_idx..) };

                    if bytes.len() > SIMD_SIZE {
                        let lane: [u8; SIMD_SIZE] = unsafe {
                            bytes
                                .get_unchecked(0..SIMD_SIZE)
                                .try_into()
                                .unwrap_unchecked_release()
                        };
                        let simd_bytes = SimdVec::from(lane);
                        let has_eol_char = simd_bytes.simd_eq(self.simd_eol_char);
                        let has_separator = simd_bytes.simd_eq(self.simd_separator);
                        let has_any_mask = (has_separator | has_eol_char).to_bitmask();

                        if has_any_mask != 0 {
                            total_idx += has_any_mask.trailing_zeros() as usize;
                            break;
                        } else {
                            total_idx += SIMD_SIZE;
                        }
                    } else {
                        match bytes.iter().position(|&c| self.eof_oel(c)) {
                            None => return self.finish(needs_escaping),
                            Some(idx) => {
                                total_idx += idx;
                                break;
                            },
                        }
                    }
                }
                unsafe {
                    if *self.v.get_unchecked_release(total_idx) == self.eol_char {
                        return self.finish_eol(needs_escaping, total_idx);
                    } else {
                        total_idx
                    }
                }
            };

            unsafe {
                debug_assert!(pos < self.v.len());
                // SAFETY:
                // we are in bounds
                let ret = Some((self.v.get_unchecked(..pos), needs_escaping));
                self.v = self.v.get_unchecked(pos + 1..);
                ret
            }
        }
    }
}

pub(crate) use inner::SplitFields;

#[cfg(test)]
mod test {
    use super::SplitFields;

    #[test]
    fn test_splitfields() {
        let input = "\"foo\",\"bar\"";
        let mut fields = SplitFields::new(input.as_bytes(), b',', Some(b'"'), b'\n');

        assert_eq!(fields.next(), Some(("\"foo\"".as_bytes(), true)));
        assert_eq!(fields.next(), Some(("\"bar\"".as_bytes(), true)));
        assert_eq!(fields.next(), None);

        let input2 = "\"foo\n bar\";\"baz\";12345";
        let mut fields2 = SplitFields::new(input2.as_bytes(), b';', Some(b'"'), b'\n');

        assert_eq!(fields2.next(), Some(("\"foo\n bar\"".as_bytes(), true)));
        assert_eq!(fields2.next(), Some(("\"baz\"".as_bytes(), true)));
        assert_eq!(fields2.next(), Some(("12345".as_bytes(), false)));
        assert_eq!(fields2.next(), None);
    }
}
