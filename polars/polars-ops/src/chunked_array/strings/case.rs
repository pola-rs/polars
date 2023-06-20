#[cfg(feature = "nightly")]
use core::unicode::conversions;

use polars_core::prelude::Utf8Chunked;

// inlined from std
fn convert_while_ascii(b: &[u8], convert: fn(&u8) -> u8, out: &mut Vec<u8>) {
    unsafe {
        out.set_len(0);
        out.reserve(b.len());
    }

    const USIZE_SIZE: usize = std::mem::size_of::<usize>();
    const MAGIC_UNROLL: usize = 2;
    const N: usize = USIZE_SIZE * MAGIC_UNROLL;
    const NONASCII_MASK: usize = usize::from_ne_bytes([0x80; USIZE_SIZE]);

    let mut i = 0;
    unsafe {
        while i + N <= b.len() {
            // Safety: we have checks the sizes `b` and `out` to know that our
            let in_chunk = b.get_unchecked(i..i + N);
            let out_chunk = out.spare_capacity_mut().get_unchecked_mut(i..i + N);

            let mut bits = 0;
            for j in 0..MAGIC_UNROLL {
                // read the bytes 1 usize at a time (unaligned since we haven't checked the alignment)
                // safety: in_chunk is valid bytes in the range
                bits |= in_chunk.as_ptr().cast::<usize>().add(j).read_unaligned();
            }
            // if our chunks aren't ascii, then return only the prior bytes as init
            if bits & NONASCII_MASK != 0 {
                break;
            }

            // perform the case conversions on N bytes (gets heavily autovec'd)
            for j in 0..N {
                // safety: in_chunk and out_chunk is valid bytes in the range
                let out = out_chunk.get_unchecked_mut(j);
                out.write(convert(in_chunk.get_unchecked(j)));
            }

            // mark these bytes as initialised
            i += N;
        }
        out.set_len(i);
    }
}

#[cfg(not(feature = "nightly"))]
pub(super) fn to_lowercase<'a>(ca: &'a Utf8Chunked) -> Utf8Chunked {
    // this amortizes allocations and will not be freed
    // so will have size of max(len)
    let mut buf = Vec::new();

    // this is one that will be set if we cannot convert ascii
    // this length will change every iteration we must use this
    let mut buf2 = Vec::new();
    let f = |s: &'a str| {
        convert_while_ascii(s.as_bytes(), u8::to_ascii_lowercase, &mut buf);
        let slice = if buf.len() < s.len() {
            buf2 = s.to_lowercase().into_bytes();
            buf2.as_ref()
        } else {
            buf.as_ref()
        };
        // extend lifetime
        // lifetime is bound to 'a
        let slice = unsafe { std::str::from_utf8_unchecked(slice) };
        unsafe { std::mem::transmute::<&str, &'a str>(slice) }
    };
    ca.apply_mut(f)
}

#[cfg(feature = "nightly")]
fn to_lowercase_helper(s: &str, buf: &mut Vec<u8>) {
    convert_while_ascii(s.as_bytes(), u8::to_ascii_lowercase, buf);

    // Safety: we know this is a valid char boundary since
    // out.len() is only progressed if ascii bytes are found
    let rest = unsafe { s.get_unchecked(buf.len()..) };

    // Safety: We have written only valid ASCII to our vec
    let mut s = unsafe { String::from_utf8_unchecked(std::mem::take(buf)) };

    for (i, c) in rest[..].char_indices() {
        if c == 'Σ' {
            // Σ maps to σ, except at the end of a word where it maps to ς.
            // This is the only conditional (contextual) but language-independent mapping
            // in `SpecialCasing.txt`,
            // so hard-code it rather than have a generic "condition" mechanism.
            // See https://github.com/rust-lang/rust/issues/26035
            map_uppercase_sigma(rest, i, &mut s)
        } else {
            match conversions::to_lower(c) {
                [a, '\0', _] => s.push(a),
                [a, b, '\0'] => {
                    s.push(a);
                    s.push(b);
                }
                [a, b, c] => {
                    s.push(a);
                    s.push(b);
                    s.push(c);
                }
            }
        }
    }

    fn map_uppercase_sigma(from: &str, i: usize, to: &mut String) {
        // See https://www.unicode.org/versions/Unicode7.0.0/ch03.pdf#G33992
        // for the definition of `Final_Sigma`.
        debug_assert!('Σ'.len_utf8() == 2);
        let is_word_final = case_ignoreable_then_cased(from[..i].chars().rev())
            && !case_ignoreable_then_cased(from[i + 2..].chars());
        to.push_str(if is_word_final { "ς" } else { "σ" });
    }

    fn case_ignoreable_then_cased<I: Iterator<Item = char>>(iter: I) -> bool {
        use core::unicode::{Case_Ignorable, Cased};
        #[allow(clippy::skip_while_next)]
        match iter.skip_while(|&c| Case_Ignorable(c)).next() {
            Some(c) => Cased(c),
            None => false,
        }
    }
    // put buf back for next iteration
    *buf = s.into_bytes();
}

// inlined from std
#[cfg(feature = "nightly")]
pub(super) fn to_lowercase<'a>(ca: &'a Utf8Chunked) -> Utf8Chunked {
    // amortize allocation
    let mut buf = Vec::new();
    let f = |s: &'a str| {
        to_lowercase_helper(s, &mut buf);

        // extend lifetime
        // lifetime is bound to 'a
        let slice = unsafe { std::str::from_utf8_unchecked(&buf) };
        unsafe { std::mem::transmute::<&str, &'a str>(slice) }
    };
    ca.apply_mut(f)
}

#[cfg(not(feature = "nightly"))]
pub(super) fn to_uppercase<'a>(ca: &'a Utf8Chunked) -> Utf8Chunked {
    // this amortizes allocations and will not be freed
    // so will have size of max(len)
    let mut buf = Vec::new();

    // this is one that will be set if we cannot convert ascii
    // this length will change every iteration we must use this
    let mut buf2 = Vec::new();
    let f = |s: &'a str| {
        convert_while_ascii(s.as_bytes(), u8::to_ascii_uppercase, &mut buf);
        let slice = if buf.len() < s.len() {
            buf2 = s.to_uppercase().into_bytes();
            buf2.as_ref()
        } else {
            buf.as_ref()
        };
        // extend lifetime
        // lifetime is bound to 'a
        let slice = unsafe { std::str::from_utf8_unchecked(slice) };
        unsafe { std::mem::transmute::<&str, &'a str>(slice) }
    };
    ca.apply_mut(f)
}

#[inline]
#[cfg(feature = "nightly")]
fn push_char_to_upper(c: char, s: &mut String) {
    match conversions::to_upper(c) {
        [a, '\0', _] => s.push(a),
        [a, b, '\0'] => {
            s.push(a);
            s.push(b);
        }
        [a, b, c] => {
            s.push(a);
            s.push(b);
            s.push(c);
        }
    }
}

// inlined from std
#[cfg(feature = "nightly")]
pub(super) fn to_uppercase<'a>(ca: &'a Utf8Chunked) -> Utf8Chunked {
    // amortize allocation
    let mut buf = Vec::new();
    let f = |s: &'a str| {
        convert_while_ascii(s.as_bytes(), u8::to_ascii_uppercase, &mut buf);

        // Safety: we know this is a valid char boundary since
        // out.len() is only progressed if ascii bytes are found
        let rest = unsafe { s.get_unchecked(buf.len()..) };

        // Safety: We have written only valid ASCII to our vec
        let mut s = unsafe { String::from_utf8_unchecked(std::mem::take(&mut buf)) };

        for c in rest.chars() {
            push_char_to_upper(c, &mut s);
        }

        // put buf back for next iteration
        buf = s.into_bytes();

        // extend lifetime
        // lifetime is bound to 'a
        let slice = unsafe { std::str::from_utf8_unchecked(&buf) };
        unsafe { std::mem::transmute::<&str, &'a str>(slice) }
    };
    ca.apply_mut(f)
}

#[cfg(feature = "nightly")]
pub(super) fn to_titlecase<'a>(ca: &'a Utf8Chunked) -> Utf8Chunked {
    // amortize allocation
    let mut buf = Vec::new();

    // temporary scratch
    // we have a double copy as we first convert to lowercase
    // and then copy to `buf`
    let mut scratch = Vec::new();
    let f = |s: &'a str| {
        unsafe {
            buf.set_len(0);
        }
        // this helper sets scratch len to 0
        to_lowercase_helper(s, &mut scratch);

        let mut next_is_upper = true;

        let lowercased = unsafe { std::str::from_utf8_unchecked(&scratch) };

        let mut s = unsafe { String::from_utf8_unchecked(std::mem::take(&mut buf)) };

        for c in lowercased.chars() {
            let is_whitespace = c.is_whitespace();
            if is_whitespace || !next_is_upper {
                next_is_upper = is_whitespace;
                s.push(c);
            } else {
                next_is_upper = false;
                push_char_to_upper(c, &mut s);
            }
        }

        // put buf back for next iteration
        buf = s.into_bytes();

        // extend lifetime
        // lifetime is bound to 'a
        let slice = unsafe { std::str::from_utf8_unchecked(&buf) };
        unsafe { std::mem::transmute::<&str, &'a str>(slice) }
    };
    ca.apply_mut(f)
}
