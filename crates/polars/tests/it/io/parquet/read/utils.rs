use polars_parquet::parquet::encoding::hybrid_rle::{self, BitmapIter, HybridRleDecoder};
use polars_parquet::parquet::error::{ParquetError, ParquetResult};
use polars_parquet::parquet::page::{split_buffer, DataPage, EncodedSplitBuffer};
use polars_parquet::parquet::read::levels::get_bit_width;
use polars_parquet::parquet::schema::Repetition;
use polars_parquet::parquet::types::{decode, NativeType};
use polars_parquet::read::PhysicalType;
use polars_parquet::write::Encoding;

pub(super) fn dict_indices_decoder(page: &DataPage) -> ParquetResult<HybridRleDecoder> {
    let EncodedSplitBuffer {
        rep: _,
        def: _,
        values: indices_buffer,
    } = split_buffer(page)?;

    // SPEC: Data page format: the bit width used to encode the entry ids stored as 1 byte (max bit width = 32),
    // SPEC: followed by the values encoded using RLE/Bit packed described above (with the given bit width).
    let bit_width = indices_buffer[0];
    if bit_width > 32 {
        panic!("Bit width of dictionary pages cannot be larger than 32",);
    }
    let indices_buffer = &indices_buffer[1..];

    Ok(hybrid_rle::HybridRleDecoder::new(
        indices_buffer,
        bit_width as u32,
        page.num_values(),
    ))
}

/// Decoder of definition levels.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum DefLevelsDecoder<'a> {
    /// When the maximum definition level is larger than 1
    Levels(HybridRleDecoder<'a>, u32),
}

impl<'a> DefLevelsDecoder<'a> {
    pub fn try_new(page: &'a DataPage) -> ParquetResult<Self> {
        let EncodedSplitBuffer {
            rep: _,
            def: def_levels,
            values: _,
        } = split_buffer(page)?;

        let max_def_level = page.descriptor.max_def_level;
        Ok({
            let iter =
                HybridRleDecoder::new(def_levels, get_bit_width(max_def_level), page.num_values());
            Self::Levels(iter, max_def_level as u32)
        })
    }
}

pub fn deserialize_optional<C: Clone, I: Iterator<Item = ParquetResult<C>>>(
    validity: DefLevelsDecoder,
    values: I,
) -> ParquetResult<Vec<Option<C>>> {
    match validity {
        DefLevelsDecoder::Levels(levels, max_level) => {
            deserialize_levels(levels, max_level, values)
        },
    }
}

fn deserialize_levels<C: Clone, I: Iterator<Item = Result<C, ParquetError>>>(
    levels: HybridRleDecoder,
    max: u32,
    mut values: I,
) -> Result<Vec<Option<C>>, ParquetError> {
    levels
        .collect()?
        .into_iter()
        .map(|x| {
            if x == max {
                values.next().transpose()
            } else {
                Ok(None)
            }
        })
        .collect()
}

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
    pub fn try_new(page: &'a DataPage, dict: P) -> ParquetResult<Self> {
        let indexes = dict_indices_decoder(page)?;

        Ok(Self { indexes, dict })
    }
}

#[allow(clippy::large_enum_variant)]
pub enum FixedLenBinaryPageState<'a, P> {
    Optional(DefLevelsDecoder<'a>, FixexBinaryIter<'a>),
    Required(FixexBinaryIter<'a>),
    RequiredDictionary(Dictionary<'a, P>),
    OptionalDictionary(DefLevelsDecoder<'a>, Dictionary<'a, P>),
}

impl<'a, P> FixedLenBinaryPageState<'a, P> {
    pub fn try_new(page: &'a DataPage, dict: Option<P>) -> ParquetResult<Self> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;

        let size: usize = if let PhysicalType::FixedLenByteArray(size) =
            page.descriptor.primitive_type.physical_type
        {
            size
        } else {
            return Err(ParquetError::InvalidParameter(
                "FixedLenBinaryPageState must be initialized by pages of FixedLenByteArray"
                    .to_string(),
            ));
        };

        match (page.encoding(), dict, is_optional) {
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), false) => {
                Dictionary::try_new(page, dict).map(Self::RequiredDictionary)
            },
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), true) => {
                Ok(Self::OptionalDictionary(
                    DefLevelsDecoder::try_new(page)?,
                    Dictionary::try_new(page, dict)?,
                ))
            },
            (Encoding::Plain, _, true) => {
                let EncodedSplitBuffer {
                    rep: _,
                    def: _,
                    values,
                } = split_buffer(page)?;

                let validity = DefLevelsDecoder::try_new(page)?;
                let values = FixexBinaryIter::new(values, size);

                Ok(Self::Optional(validity, values))
            },
            (Encoding::Plain, _, false) => {
                let EncodedSplitBuffer {
                    rep: _,
                    def: _,
                    values,
                } = split_buffer(page)?;
                let values = FixexBinaryIter::new(values, size);

                Ok(Self::Required(values))
            },
            _ => Err(ParquetError::FeatureNotSupported(format!(
                "Viewing page for encoding {:?} for binary type",
                page.encoding(),
            ))),
        }
    }
}

/// Typedef of an iterator over PLAIN page values
pub type Casted<'a, T> = std::iter::Map<std::slice::ChunksExact<'a, u8>, fn(&'a [u8]) -> T>;

/// Views the values of the data page as [`Casted`] to [`NativeType`].
pub fn native_cast<T: NativeType>(page: &DataPage) -> ParquetResult<Casted<T>> {
    let EncodedSplitBuffer {
        rep: _,
        def: _,
        values,
    } = split_buffer(page)?;
    if values.len() % std::mem::size_of::<T>() != 0 {
        panic!("A primitive page data's len must be a multiple of the type");
    }

    Ok(values
        .chunks_exact(std::mem::size_of::<T>())
        .map(decode::<T>))
}

/// The deserialization state of a `DataPage` of `Primitive` parquet primitive type
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum NativePageState<'a, T, P>
where
    T: NativeType,
{
    /// A page of optional values
    Optional(DefLevelsDecoder<'a>, Casted<'a, T>),
    /// A page of required values
    Required(Casted<'a, T>),
    /// A page of required, dictionary-encoded values
    RequiredDictionary(Dictionary<'a, P>),
    /// A page of optional, dictionary-encoded values
    OptionalDictionary(DefLevelsDecoder<'a>, Dictionary<'a, P>),
}

impl<'a, T: NativeType, P> NativePageState<'a, T, P> {
    /// Tries to create [`NativePageState`]
    /// # Error
    /// Errors iff the page is not a `NativePageState`
    pub fn try_new(page: &'a DataPage, dict: Option<P>) -> ParquetResult<Self> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;

        match (page.encoding(), dict, is_optional) {
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), false) => {
                Dictionary::try_new(page, dict).map(Self::RequiredDictionary)
            },
            (Encoding::PlainDictionary | Encoding::RleDictionary, Some(dict), true) => {
                Ok(Self::OptionalDictionary(
                    DefLevelsDecoder::try_new(page)?,
                    Dictionary::try_new(page, dict)?,
                ))
            },
            (Encoding::Plain, _, true) => {
                let validity = DefLevelsDecoder::try_new(page)?;
                let values = native_cast(page)?;

                Ok(Self::Optional(validity, values))
            },
            (Encoding::Plain, _, false) => native_cast(page).map(Self::Required),
            _ => Err(ParquetError::FeatureNotSupported(format!(
                "Viewing page for encoding {:?} for native type {}",
                page.encoding(),
                std::any::type_name::<T>()
            ))),
        }
    }
}

// The state of a `DataPage` of `Boolean` parquet boolean type
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum BooleanPageState<'a> {
    Optional(DefLevelsDecoder<'a>, BitmapIter<'a>),
    Required(&'a [u8], usize),
}

impl<'a> BooleanPageState<'a> {
    pub fn try_new(page: &'a DataPage) -> ParquetResult<Self> {
        let is_optional =
            page.descriptor.primitive_type.field_info.repetition == Repetition::Optional;

        match (page.encoding(), is_optional) {
            (Encoding::Plain, true) => {
                let validity = DefLevelsDecoder::try_new(page)?;

                let values = split_buffer(page)?.values;
                let values = BitmapIter::new(values, 0, values.len() * 8);

                Ok(Self::Optional(validity, values))
            },
            (Encoding::Plain, false) => {
                let values = split_buffer(page)?.values;
                Ok(Self::Required(values, page.num_values()))
            },
            _ => Err(ParquetError::InvalidParameter(format!(
                "Viewing page for encoding {:?} for boolean type not supported",
                page.encoding(),
            ))),
        }
    }
}
