use polars_core::prelude::arity::unary_elementwise;
use polars_core::prelude::StringChunked;
use unicode_normalization::UnicodeNormalization;

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UnicodeForm {
    NFC,
    NFKC,
    NFD,
    NFKD,
}

pub fn normalize(ca: &StringChunked, form: UnicodeForm) -> StringChunked {
    unary_elementwise(ca, |val| {
        val.map(|x| match form {
            UnicodeForm::NFC => x.nfc().collect::<String>(),
            UnicodeForm::NFKC => x.nfkc().collect::<String>(),
            UnicodeForm::NFD => x.nfd().collect::<String>(),
            UnicodeForm::NFKD => x.nfkd().collect::<String>(),
        })
    })
}
