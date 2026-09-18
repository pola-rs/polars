use polars_core::prelude::StringChunked;

// Inlined from std.
fn convert_while_ascii(b: &[u8], convert: fn(&u8) -> u8, out: &mut Vec<u8>) {
    out.clear();
    out.reserve(b.len());

    const USIZE_SIZE: usize = size_of::<usize>();
    const MAGIC_UNROLL: usize = 2;
    const N: usize = USIZE_SIZE * MAGIC_UNROLL;
    const NONASCII_MASK: usize = usize::from_ne_bytes([0x80; USIZE_SIZE]);

    let mut i = 0;
    unsafe {
        while i + N <= b.len() {
            // SAFETY: we have checks the sizes `b` and `out`.
            let in_chunk = b.get_unchecked(i..i + N);
            let out_chunk = out.spare_capacity_mut().get_unchecked_mut(i..i + N);

            let mut bits = 0;
            for j in 0..MAGIC_UNROLL {
                // Read the bytes 1 usize at a time (unaligned since we haven't checked the alignment).
                // SAFETY: in_chunk is valid bytes in the range.
                bits |= in_chunk.as_ptr().cast::<usize>().add(j).read_unaligned();
            }
            // If our chunks aren't ascii, then return only the prior bytes as init.
            if bits & NONASCII_MASK != 0 {
                break;
            }

            // Perform the case conversions on N bytes (gets heavily autovec'd).
            for j in 0..N {
                // SAFETY: in_chunk and out_chunk are valid bytes in the range.
                let out = out_chunk.get_unchecked_mut(j);
                out.write(convert(in_chunk.get_unchecked(j)));
            }

            // Mark these bytes as initialised.
            i += N;
        }

        while i < b.len() {
            // SAFETY: `i` is in bounds of `b`, as just checked.
            let byte = *b.get_unchecked(i);
            if !byte.is_ascii() {
                break;
            }

            // SAFETY: the reservation above left room for every byte of `b`, and the conversion
            // of an ASCII byte is a byte like any other: the bytes written stay valid UTF-8.
            out.spare_capacity_mut()
                .get_unchecked_mut(i)
                .write(convert(&byte));
            i += 1;
        }

        out.set_len(i);
    }
}

fn to_lowercase_helper(source: &str, buf: &mut Vec<u8>) {
    convert_while_ascii(source.as_bytes(), u8::to_ascii_lowercase, buf);

    // SAFETY: we know this is a valid char boundary since
    // out.len() is only progressed if ASCII bytes are found.
    let converted = buf.len();
    let rest = unsafe { source.get_unchecked(converted..) };

    // SAFETY: We have written only valid ASCII to our vec.
    let mut s = unsafe { String::from_utf8_unchecked(std::mem::take(buf)) };

    for (i, c) in rest[..].char_indices() {
        if c == 'Σ' {
            // Σ maps to σ, except at the end of a word where it maps to ς.
            // This is the only conditional (contextual) but language-independent mapping
            // in `SpecialCasing.txt`,
            // so hard-code it rather than have a generic "condition" mechanism.
            // See https://github.com/rust-lang/rust/issues/26035
            //
            // The context is read out of the whole string rather than out of what is left of it:
            // the bytes the ASCII pass above took are part of the word this sigma may end, and
            // dropping them made `"a".repeat(16) + "Σ"` lowercase to σ where `str::to_lowercase`
            // gives ς.
            map_uppercase_sigma(source, converted + i, &mut s)
        } else {
            for lower in c.to_lowercase() {
                s.push(lower);
            }
        }
    }

    fn map_uppercase_sigma(from: &str, i: usize, to: &mut String) {
        // See https://www.unicode.org/versions/Unicode7.0.0/ch03.pdf#G33992
        // for the definition of `Final_Sigma`.
        debug_assert!('Σ'.len_utf8() == 2);
        let is_word_final = case_ignorable_then_cased(from[..i].chars().rev())
            && !case_ignorable_then_cased(from[i + 2..].chars());
        to.push_str(if is_word_final { "ς" } else { "σ" });
    }

    fn case_ignorable_then_cased<I: Iterator<Item = char>>(iter: I) -> bool {
        use super::unicode_internals::{Case_Ignorable, Cased};
        #[allow(clippy::skip_while_next)]
        match iter.skip_while(|&c| Case_Ignorable(c)).next() {
            Some(c) => Cased(c),
            None => false,
        }
    }

    // Put buf back for next iteration.
    *buf = s.into_bytes();
}

pub(super) fn to_lowercase<'a>(ca: &'a StringChunked) -> StringChunked {
    // Amortize allocation.
    let mut buf = Vec::new();
    let f = |s: &'a str| -> &'a str {
        to_lowercase_helper(s, &mut buf);
        // SAFETY: apply_mut will copy value from buf before next iteration.
        let slice = unsafe { std::str::from_utf8_unchecked(&buf) };
        unsafe { std::mem::transmute::<&str, &'a str>(slice) }
    };
    ca.apply_mut(f)
}

// Inlined from std.
pub(super) fn to_uppercase<'a>(ca: &'a StringChunked) -> StringChunked {
    // Amortize allocation.
    let mut buf = Vec::new();
    let f = |s: &'a str| -> &'a str {
        convert_while_ascii(s.as_bytes(), u8::to_ascii_uppercase, &mut buf);

        // SAFETY: we know this is a valid char boundary since
        // out.len() is only progressed if ascii bytes are found.
        let rest = unsafe { s.get_unchecked(buf.len()..) };

        // SAFETY: We have written only valid ASCII to our vec.
        let mut s = unsafe { String::from_utf8_unchecked(std::mem::take(&mut buf)) };

        for c in rest.chars() {
            for upper in c.to_uppercase() {
                s.push(upper);
            }
        }

        // Put buf back for next iteration.
        buf = s.into_bytes();

        // SAFETY: apply_mut will copy value from buf before next iteration.
        let slice = unsafe { std::str::from_utf8_unchecked(&buf) };
        unsafe { std::mem::transmute::<&str, &'a str>(slice) }
    };
    ca.apply_mut(f)
}

#[cfg(feature = "nightly")]
pub(super) fn to_titlecase<'a>(ca: &'a StringChunked) -> StringChunked {
    // Amortize allocation.
    let mut buf = Vec::new();

    // Temporary scratch space.
    // We have a double copy as we first convert to lowercase and then copy to `buf`.
    let mut scratch = Vec::new();
    let f = |s: &'a str| -> &'a str {
        to_lowercase_helper(s, &mut scratch);
        let lowercased = unsafe { std::str::from_utf8_unchecked(&scratch) };

        // SAFETY: the buffer is clear, empty string is valid UTF-8.
        buf.clear();
        let mut s = unsafe { String::from_utf8_unchecked(std::mem::take(&mut buf)) };

        let mut next_is_upper = true;
        for c in lowercased.chars() {
            if next_is_upper {
                for upper in c.to_uppercase() {
                    s.push(upper);
                }
            } else {
                s.push(c);
            }
            next_is_upper = !c.is_alphabetic();
        }

        // Put buf back for next iteration.
        buf = s.into_bytes();

        // SAFETY: apply_mut will copy value from buf before next iteration.
        let slice = unsafe { std::str::from_utf8_unchecked(&buf) };
        unsafe { std::mem::transmute::<&str, &'a str>(slice) }
    };
    ca.apply_mut(f)
}
