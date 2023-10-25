use crate::{
    encoding::hybrid_rle,
    error::Error,
    page::{split_buffer, DataPage},
    parquet_bridge::{Encoding, Repetition},
    types::{decode, NativeType},
};

use super::utils;

/// Typedef of an iterator over PLAIN page values
pub type Casted<'a, T> = std::iter::Map<std::slice::ChunksExact<'a, u8>, fn(&'a [u8]) -> T>;

/// Views the values of the data page as [`Casted`] to [`NativeType`].
pub fn native_cast<T: NativeType>(page: &DataPage) -> Result<Casted<T>, Error> {
    let (_, _, values) = split_buffer(page)?;
    if values.len() % std::mem::size_of::<T>() != 0 {
        return Err(Error::oos(
            "A primitive page data's len must be a multiple of the type",
        ));
    }

    Ok(values
        .chunks_exact(std::mem::size_of::<T>())
        .map(decode::<T>))
}

#[derive(Debug)]
pub struct Dictionary<'a, P> {
    pub indexes: hybrid_rle::HybridRleDecoder<'a>,
    pub dict: P,
}

impl<'a, P> Dictionary<'a, P> {
    pub fn try_new(page: &'a DataPage, dict: P) -> Result<Self, Error> {
        let indexes = utils::dict_indices_decoder(page)?;

        Ok(Self { dict, indexes })
    }

    pub fn len(&self) -> usize {
        self.indexes.size_hint().0
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The deserialization state of a `DataPage` of `Primitive` parquet primitive type
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum NativePageState<'a, T, P>
where
    T: NativeType,
{
    /// A page of optional values
    Optional(utils::DefLevelsDecoder<'a>, Casted<'a, T>),
    /// A page of required values
    Required(Casted<'a, T>),
    /// A page of required, dictionary-encoded values
    RequiredDictionary(Dictionary<'a, P>),
    /// A page of optional, dictionary-encoded values
    OptionalDictionary(utils::DefLevelsDecoder<'a>, Dictionary<'a, P>),
}

impl<'a, T: NativeType, P> NativePageState<'a, T, P> {
    /// Tries to create [`NativePageState`]
    /// # Error
    /// Errors iff the page is not a `NativePageState`
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
                let validity = utils::DefLevelsDecoder::try_new(page)?;
                let values = native_cast(page)?;

                Ok(Self::Optional(validity, values))
            }
            (Encoding::Plain, _, false) => native_cast(page).map(Self::Required),
            _ => Err(Error::FeatureNotSupported(format!(
                "Viewing page for encoding {:?} for native type {}",
                page.encoding(),
                std::any::type_name::<T>()
            ))),
        }
    }
}
