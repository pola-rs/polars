use crate::{
    encoding::hybrid_rle::BitmapIter,
    error::Error,
    page::{split_buffer, DataPage},
    parquet_bridge::{Encoding, Repetition},
};

use super::utils;

// The state of a `DataPage` of `Boolean` parquet boolean type
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum BooleanPageState<'a> {
    Optional(utils::DefLevelsDecoder<'a>, BitmapIter<'a>),
    Required(&'a [u8], usize),
}

impl<'a> BooleanPageState<'a> {
    pub fn try_new(page: &'a DataPage) -> Result<Self, Error> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;

        match (page.encoding(), is_optional) {
            (Encoding::Plain, true) => {
                let validity = utils::DefLevelsDecoder::try_new(page)?;

                let (_, _, values) = split_buffer(page)?;
                let values = BitmapIter::new(values, 0, values.len() * 8);

                Ok(Self::Optional(validity, values))
            }
            (Encoding::Plain, false) => {
                let (_, _, values) = split_buffer(page)?;
                Ok(Self::Required(values, page.num_values()))
            }
            _ => Err(Error::InvalidParameter(format!(
                "Viewing page for encoding {:?} for boolean type not supported",
                page.encoding(),
            ))),
        }
    }
}
