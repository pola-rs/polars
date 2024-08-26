use polars_parquet::parquet::encoding::hybrid_rle::FnTranslator;
use polars_parquet::parquet::error::ParquetResult;
use polars_parquet::parquet::page::DataPage;
use polars_parquet::parquet::types::NativeType;
use polars_parquet::read::ParquetError;

use super::dictionary::PrimitivePageDict;
use super::hybrid_rle_iter;
use super::utils::{deserialize_optional, NativePageState};

/// The deserialization state of a `DataPage` of `Primitive` parquet primitive type
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum PageState<'a, T>
where
    T: NativeType,
{
    Nominal(NativePageState<'a, T, &'a PrimitivePageDict<T>>),
}

impl<'a, T: NativeType> PageState<'a, T> {
    /// Tries to create [`NativePageState`]
    /// # Error
    /// Errors iff the page is not a `NativePageState`
    pub fn try_new(
        page: &'a DataPage,
        dict: Option<&'a PrimitivePageDict<T>>,
    ) -> Result<Self, ParquetError> {
        NativePageState::try_new(page, dict).map(Self::Nominal)
    }
}

pub fn page_to_vec<T: NativeType>(
    page: &DataPage,
    dict: Option<&PrimitivePageDict<T>>,
) -> ParquetResult<Vec<Option<T>>> {
    assert_eq!(page.descriptor.max_rep_level, 0);
    let state = PageState::<T>::try_new(page, dict)?;

    match state {
        PageState::Nominal(state) => match state {
            NativePageState::Optional(validity, mut values) => {
                deserialize_optional(validity, values.by_ref().map(Ok))
            },
            NativePageState::Required(values) => Ok(values.map(Some).collect()),
            NativePageState::RequiredDictionary(dict) => {
                let dictionary = FnTranslator(|x| dict.dict.value(x as usize).copied().map(Some));
                dict.indexes.translate_and_collect(&dictionary)
            },
            NativePageState::OptionalDictionary(validity, dict) => {
                let values =
                    hybrid_rle_iter(dict.indexes)?.map(|x| dict.dict.value(x as usize).copied());
                deserialize_optional(validity, values)
            },
        },
    }
}
