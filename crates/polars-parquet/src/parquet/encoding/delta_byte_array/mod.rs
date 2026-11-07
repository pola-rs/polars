mod decoder;
mod encoder;

pub use decoder::Decoder;
pub use encoder::encode;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::error::ParquetError;

    #[test]
    fn basic() -> Result<(), ParquetError> {
        let data = vec![b"Hello".as_ref(), b"Helicopter"];
        let mut buffer = vec![];
        encode(data.clone().into_iter(), &mut buffer);

        // "Hello", then "Helicopter" as the first 3 bytes of "Hello" and "icopter".
        let decoder = Decoder::try_new(&buffer)?;
        assert_eq!(decoder.values(), b"Helloicopter");
        assert_eq!(decoder.prefix_lengths.collect::<Vec<i64>>()?, [0, 3]);
        assert_eq!(decoder.suffix_lengths.collect::<Vec<i64>>()?, [5, 7]);
        Ok(())
    }
}
