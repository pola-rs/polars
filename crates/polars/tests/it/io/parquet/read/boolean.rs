use polars_parquet::parquet::encoding::hybrid_rle::BitmapIter;
use polars_parquet::parquet::error::ParquetResult;
use polars_parquet::parquet::page::DataPage;

use super::utils::deserialize_optional;
use crate::io::parquet::read::utils::BooleanPageState;

pub fn page_to_vec(page: &DataPage) -> ParquetResult<Vec<Option<bool>>> {
    assert_eq!(page.descriptor.max_rep_level, 0);

    let state = BooleanPageState::try_new(page)?;

    match state {
        BooleanPageState::Optional(validity, mut values) => {
            deserialize_optional(validity, values.by_ref().map(Ok))
        },
        BooleanPageState::Required(bitmap, length) => {
            Ok(BitmapIter::new(bitmap, 0, length).map(Some).collect())
        },
    }
}
