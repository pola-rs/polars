use crate::{
    encoding::{hybrid_rle, plain_byte_array::BinaryIter},
    error::Error,
    page::{split_buffer, DataPage},
    parquet_bridge::{Encoding, Repetition},
};

use super::utils;

#[derive(Debug)]
pub struct Dictionary<'a, P> {
    pub indexes: hybrid_rle::HybridRleDecoder<'a>,
    pub dict: P,
}

impl<'a, P> Dictionary<'a, P> {
    pub fn try_new(page: &'a DataPage, dict: P) -> Result<Self, Error> {
        let indexes = utils::dict_indices_decoder(page)?;

        Ok(Self { indexes, dict })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.indexes.size_hint().0
    }
}

#[allow(clippy::large_enum_variant)]
pub enum BinaryPageState<'a, P> {
    Optional(utils::DefLevelsDecoder<'a>, BinaryIter<'a>),
    Required(BinaryIter<'a>),
    RequiredDictionary(Dictionary<'a, P>),
    OptionalDictionary(utils::DefLevelsDecoder<'a>, Dictionary<'a, P>),
}

impl<'a, P> BinaryPageState<'a, P> {
    pub fn try_new(page: &'a DataPage, dict: Option<P>) -> Result<Self, Error> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;

        match (page.encoding(), dict, is_optional) {
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), false) => {
                Dictionary::try_new(page, dict).map(Self::RequiredDictionary)
            }
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), true) => {
                Ok(Self::OptionalDictionary(
                    utils::DefLevelsDecoder::try_new(page)?,
                    Dictionary::try_new(page, dict)?,
                ))
            }
            (Encoding::Plain, _, true) => {
                let (_, _, values) = split_buffer(page)?;

                let validity = utils::DefLevelsDecoder::try_new(page)?;
                let values = BinaryIter::new(values, None);

                Ok(Self::Optional(validity, values))
            }
            (Encoding::Plain, _, false) => {
                let (_, _, values) = split_buffer(page)?;
                let values = BinaryIter::new(values, Some(page.num_values()));

                Ok(Self::Required(values))
            }
            _ => Err(Error::FeatureNotSupported(format!(
                "Viewing page for encoding {:?} for binary type",
                page.encoding(),
            ))),
        }
    }
}
