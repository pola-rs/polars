use polars_core::prelude::StringChunked;
use polars_defs::expr::UnicodeForm;
use unicode_normalization::UnicodeNormalization;

pub fn normalize_with<'a, F: Fn(&str, &mut String)>(
    ca: &'a StringChunked,
    normalizer: F,
) -> StringChunked {
    let mut buffer = String::new();

    let f = |s: &'a str| -> &'a str {
        buffer.clear();
        normalizer(s, &mut buffer);

        // SAFETY: `apply_mut` copies the value out before it calls back, so the buffer is free to
        // be written over for the next element.
        unsafe { std::mem::transmute::<&str, &'a str>(buffer.as_str()) }
    };

    ca.apply_mut(f)
}

pub fn normalize(ca: &StringChunked, form: UnicodeForm) -> StringChunked {
    match form {
        UnicodeForm::NFC => normalize_with(ca, |s, b| b.extend(s.nfc())),
        UnicodeForm::NFKC => normalize_with(ca, |s, b| b.extend(s.nfkc())),
        UnicodeForm::NFD => normalize_with(ca, |s, b| b.extend(s.nfd())),
        UnicodeForm::NFKD => normalize_with(ca, |s, b| b.extend(s.nfkd())),
    }
}
