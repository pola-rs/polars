use polars_parquet::parquet::error::ParquetResult;
use polars_parquet::parquet::page::DataPage;

use super::dictionary::FixedLenByteArrayPageDict;
use super::utils::{deserialize_optional, FixedLenBinaryPageState};

pub fn page_to_vec(
    page: &DataPage,
    dict: Option<&FixedLenByteArrayPageDict>,
) -> ParquetResult<Vec<Option<Vec<u8>>>> {
    assert_eq!(page.descriptor.max_rep_level, 0);

    let state = FixedLenBinaryPageState::try_new(page, dict)?;

    match state {
        FixedLenBinaryPageState::Optional(validity, values) => {
            deserialize_optional(validity, values.map(|x| Ok(x.to_vec())))
        },
        FixedLenBinaryPageState::Required(values) => {
            Ok(values.map(|x| x.to_vec()).map(Some).collect())
        },
        FixedLenBinaryPageState::RequiredDictionary(dict) => dict
            .indexes
            .map(|x| dict.dict.value(x as usize).map(|x| x.to_vec()).map(Some))
            .collect(),
        FixedLenBinaryPageState::OptionalDictionary(validity, dict) => {
            let values = dict
                .indexes
                .map(|x| dict.dict.value(x as usize).map(|x| x.to_vec()));
            deserialize_optional(validity, values)
        },
    }
}
