use polars_core::prelude::{StringChunked, StringChunkedBuilder};
use unicode_normalization::UnicodeNormalization;

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UnicodeForm {
    NFC,
    NFKC,
    NFD,
    NFKD,
}

pub fn normalize_with<F: Fn(&str, &mut String)>(
    ca: &StringChunked,
    normalizer: F,
) -> StringChunked {
    let mut buffer = String::new();
    let mut builder = StringChunkedBuilder::new(ca.name().clone(), ca.len());
    for opt_s in ca.iter() {
        if let Some(s) = opt_s {
            buffer.clear();
            normalizer(s, &mut buffer);
            builder.append_value(&buffer);
        } else {
            builder.append_null();
        }
    }
    builder.finish()
}

pub fn normalize(ca: &StringChunked, form: UnicodeForm) -> StringChunked {
    match form {
        UnicodeForm::NFC => normalize_with(ca, |s, b| b.extend(s.nfc())),
        UnicodeForm::NFKC => normalize_with(ca, |s, b| b.extend(s.nfkc())),
        UnicodeForm::NFD => normalize_with(ca, |s, b| b.extend(s.nfd())),
        UnicodeForm::NFKD => normalize_with(ca, |s, b| b.extend(s.nfkd())),
    }
}
