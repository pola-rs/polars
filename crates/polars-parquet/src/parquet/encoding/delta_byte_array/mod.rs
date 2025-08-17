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

        let mut decoder = Decoder::try_new(&buffer)?;
        let prefixes = decoder.by_ref().collect::<Result<Vec<_>, _>>()?;
        assert_eq!(prefixes, vec![b"Hello".to_vec(), b"Helicopter".to_vec()]);

        // move to the values
        let values = decoder.values();
        assert_eq!(values, b"Helloicopter");
        Ok(())
    }
}
