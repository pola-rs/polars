use polars_parquet::parquet::encoding::hybrid_rle::FnTranslator;
use polars_parquet::parquet::error::ParquetResult;
use polars_parquet::parquet::page::DataPage;

use super::dictionary::BinaryPageDict;
use super::utils::deserialize_optional;
use crate::io::parquet::read::hybrid_rle_iter;
use crate::io::parquet::read::utils::FixedLenBinaryPageState;

pub fn page_to_vec(
    page: &DataPage,
    dict: Option<&BinaryPageDict>,
) -> ParquetResult<Vec<Option<Vec<u8>>>> {
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
        FixedLenBinaryPageState::RequiredDictionary(dict) => {
            let dictionary =
                FnTranslator(|v| dict.dict.value(v as usize).map(|v| Some(v.to_vec())));
            dict.indexes.translate_and_collect(&dictionary)
        },
        FixedLenBinaryPageState::OptionalDictionary(validity, dict) => {
            let values = hybrid_rle_iter(dict.indexes)?
                .map(|x| dict.dict.value(x as usize).map(|x| x.to_vec()));
            deserialize_optional(validity, values)
        },
    }
}
