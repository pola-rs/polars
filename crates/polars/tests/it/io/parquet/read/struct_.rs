use polars_parquet::parquet::encoding::hybrid_rle::HybridRleDecoder;
use polars_parquet::parquet::error::ParquetResult;
use polars_parquet::parquet::page::{split_buffer, DataPage, EncodedSplitBuffer};
use polars_parquet::parquet::read::levels::get_bit_width;

use super::hybrid_rle_iter;

pub fn extend_validity(val: &mut Vec<bool>, page: &DataPage) -> ParquetResult<()> {
    let EncodedSplitBuffer {
        rep: _,
        def: def_levels,
        values: _,
    } = split_buffer(page)?;
    let length = page.num_values();

    if page.descriptor.max_def_level == 0 {
        return Ok(());
    }

    let def_level_encoding = (
        &page.definition_level_encoding(),
        page.descriptor.max_def_level,
    );

    let def_levels = HybridRleDecoder::new(def_levels, get_bit_width(def_level_encoding.1), length);

    val.reserve(length);
    hybrid_rle_iter(def_levels)?.for_each(|x| {
        val.push(x != 0);
    });
    Ok(())
}
