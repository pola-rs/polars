use polars_parquet::parquet::deserialize::FixedLenBinaryPageState;
use polars_parquet::parquet::error::Result;
use polars_parquet::parquet::page::DataPage;

use super::dictionary::BinaryPageDict;
use super::utils::deserialize_optional;

pub fn page_to_vec(page: &DataPage, dict: Option<&BinaryPageDict>) -> Result<Vec<Option<Vec<u8>>>> {
    assert_eq!(page.descriptor.max_rep_level, 0);

    let state = FixedLenBinaryPageState::try_new(page, dict)?;

    match state {
        FixedLenBinaryPageState::Optional(validity, values) => {
            deserialize_optional(validity, values.map(|x| Ok(x.to_vec())))
        },
        FixedLenBinaryPageState::Required(values) => values
            .map(|x| Ok(x.to_vec()))
            .map(Some)
            .map(|x| x.transpose())
            .collect(),
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
