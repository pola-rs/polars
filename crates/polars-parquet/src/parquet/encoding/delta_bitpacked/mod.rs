mod decoder;
mod encoder;

pub use decoder::Decoder;
pub use encoder::encode;

#[cfg(test)]
mod tests {
    use crate::error::Error;

    use super::*;

    #[test]
    fn basic() -> Result<(), Error> {
        let data = vec![1, 3, 1, 2, 3];

        let mut buffer = vec![];
        encode(data.clone().into_iter(), &mut buffer);
        let iter = Decoder::try_new(&buffer)?;

        let result = iter.collect::<Result<Vec<_>, _>>()?;
        assert_eq!(result, data);
        Ok(())
    }

    #[test]
    fn negative_value() -> Result<(), Error> {
        let data = vec![1, 3, -1, 2, 3];

        let mut buffer = vec![];
        encode(data.clone().into_iter(), &mut buffer);
        let iter = Decoder::try_new(&buffer)?;

        let result = iter.collect::<Result<Vec<_>, _>>()?;
        assert_eq!(result, data);
        Ok(())
    }

    #[test]
    fn some() -> Result<(), Error> {
        let data = vec![
            -2147483648,
            -1777158217,
            -984917788,
            -1533539476,
            -731221386,
            -1322398478,
            906736096,
        ];

        let mut buffer = vec![];
        encode(data.clone().into_iter(), &mut buffer);
        let iter = Decoder::try_new(&buffer)?;

        let result = iter.collect::<Result<Vec<_>, Error>>()?;
        assert_eq!(result, data);
        Ok(())
    }

    #[test]
    fn more_than_one_block() -> Result<(), Error> {
        let mut data = vec![1, 3, -1, 2, 3, 10, 1];
        for x in 0..128 {
            data.push(x - 10)
        }

        let mut buffer = vec![];
        encode(data.clone().into_iter(), &mut buffer);
        let iter = Decoder::try_new(&buffer)?;

        let result = iter.collect::<Result<Vec<_>, _>>()?;
        assert_eq!(result, data);
        Ok(())
    }

    #[test]
    fn test_another() -> Result<(), Error> {
        let data = vec![2, 3, 1, 2, 1];

        let mut buffer = vec![];
        encode(data.clone().into_iter(), &mut buffer);
        let len = buffer.len();
        let mut iter = Decoder::try_new(&buffer)?;

        let result = iter.by_ref().collect::<Result<Vec<_>, _>>()?;
        assert_eq!(result, data);

        assert_eq!(iter.consumed_bytes(), len);
        Ok(())
    }
}
