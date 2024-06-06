use polars_parquet::parquet::deserialize::{
    DefLevelsDecoder, HybridDecoderBitmapIter, HybridEncoded,
};
use polars_parquet::parquet::encoding::hybrid_rle::{BitmapIter, HybridRleDecoder};
use polars_parquet::parquet::error::Error;

pub fn deserialize_optional<C: Clone, I: Iterator<Item = Result<C, Error>>>(
    validity: DefLevelsDecoder,
    values: I,
) -> Result<Vec<Option<C>>, Error> {
    match validity {
        DefLevelsDecoder::Bitmap(bitmap) => deserialize_bitmap(bitmap, values),
        DefLevelsDecoder::Levels(levels, max_level) => {
            deserialize_levels(levels, max_level, values)
        },
    }
}

fn deserialize_bitmap<C: Clone, I: Iterator<Item = Result<C, Error>>>(
    mut validity: HybridDecoderBitmapIter,
    mut values: I,
) -> Result<Vec<Option<C>>, Error> {
    let mut deserialized = Vec::with_capacity(validity.len());

    validity.try_for_each(|run| match run {
        HybridEncoded::Bitmap(bitmap, length) => {
            BitmapIter::new(bitmap, 0, length).try_for_each(|x| {
                if x {
                    deserialized.push(values.next().transpose()?);
                } else {
                    deserialized.push(None);
                }
                Result::<_, Error>::Ok(())
            })
        },
        HybridEncoded::Repeated(is_set, length) => {
            if is_set {
                deserialized.reserve(length);
                for x in values.by_ref().take(length) {
                    deserialized.push(Some(x?))
                }
            } else {
                deserialized.extend(std::iter::repeat(None).take(length))
            }
            Ok(())
        },
    })?;
    Ok(deserialized)
}

fn deserialize_levels<C: Clone, I: Iterator<Item = Result<C, Error>>>(
    levels: HybridRleDecoder,
    max: u32,
    mut values: I,
) -> Result<Vec<Option<C>>, Error> {
    levels
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
