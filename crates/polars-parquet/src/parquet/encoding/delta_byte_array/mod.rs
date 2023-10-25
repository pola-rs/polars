mod decoder;
mod encoder;

pub use decoder::Decoder;
pub use encoder::encode;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    #[test]
    fn basic() -> Result<(), Error> {
        let data = vec![b"Hello".as_ref(), b"Helicopter"];
        let mut buffer = vec![];
        encode(data.clone().into_iter(), &mut buffer);

        let mut decoder = Decoder::try_new(&buffer)?;
        let prefixes = decoder.by_ref().collect::<Result<Vec<_>, _>>()?;
        assert_eq!(prefixes, vec![0, 3]);

        // move to the lengths
        let mut decoder = decoder.into_lengths()?;

        let lengths = decoder.by_ref().collect::<Result<Vec<_>, _>>()?;
        assert_eq!(lengths, vec![5, 7]);

        // move to the values
        let values = decoder.values();
        assert_eq!(values, b"Helloicopter");
        Ok(())
    }
}
