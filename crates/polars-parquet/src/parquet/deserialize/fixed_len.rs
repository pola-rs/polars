use crate::{
    encoding::hybrid_rle,
    error::Error,
    page::{split_buffer, DataPage},
    parquet_bridge::{Encoding, Repetition},
    schema::types::PhysicalType,
};

use super::utils;

#[derive(Debug)]
pub struct FixexBinaryIter<'a> {
    values: std::slice::ChunksExact<'a, u8>,
}

impl<'a> FixexBinaryIter<'a> {
    pub fn new(values: &'a [u8], size: usize) -> Self {
        let values = values.chunks_exact(size);
        Self { values }
    }
}

impl<'a> Iterator for FixexBinaryIter<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.values.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.values.size_hint()
    }
}

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
pub enum FixedLenBinaryPageState<'a, P> {
    Optional(utils::DefLevelsDecoder<'a>, FixexBinaryIter<'a>),
    Required(FixexBinaryIter<'a>),
    RequiredDictionary(Dictionary<'a, P>),
    OptionalDictionary(utils::DefLevelsDecoder<'a>, Dictionary<'a, P>),
}

impl<'a, P> FixedLenBinaryPageState<'a, P> {
    pub fn try_new(page: &'a DataPage, dict: Option<P>) -> Result<Self, Error> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;

        let size: usize = if let PhysicalType::FixedLenByteArray(size) =
            page.descriptor.primitive_type.physical_type
        {
            size
        } else {
            return Err(Error::InvalidParameter(
                "FixedLenBinaryPageState must be initialized by pages of FixedLenByteArray"
                    .to_string(),
            ));
        };

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
                let values = FixexBinaryIter::new(values, size);

                Ok(Self::Optional(validity, values))
            }
            (Encoding::Plain, _, false) => {
                let (_, _, values) = split_buffer(page)?;
                let values = FixexBinaryIter::new(values, size);

                Ok(Self::Required(values))
            }
            _ => Err(Error::FeatureNotSupported(format!(
                "Viewing page for encoding {:?} for binary type",
                page.encoding(),
            ))),
        }
    }
}
