mod decoder;
mod encoder;
mod fuzz;

pub(crate) use decoder::{Decoder, DeltaGatherer, SumGatherer};
pub(crate) use encoder::encode;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::error::{ParquetError, ParquetResult};

    #[test]
    fn basic() -> Result<(), ParquetError> {
        let data = vec![1, 3, 1, 2, 3];

        let mut buffer = vec![];
        encode(data.clone().into_iter(), &mut buffer, 1);
        let (iter, _) = Decoder::try_new(&buffer)?;

        let result = iter.collect::<Vec<_>>()?;
        assert_eq!(result, data);
        Ok(())
    }

    #[test]
    fn negative_value() -> Result<(), ParquetError> {
        let data = vec![1, 3, -1, 2, 3];

        let mut buffer = vec![];
        encode(data.clone().into_iter(), &mut buffer, 1);
        let (iter, _) = Decoder::try_new(&buffer)?;

        let result = iter.collect::<Vec<_>>()?;
        assert_eq!(result, data);
        Ok(())
    }

    #[test]
    fn some() -> Result<(), ParquetError> {
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
        encode(data.clone().into_iter(), &mut buffer, 1);
        let (iter, _) = Decoder::try_new(&buffer)?;

        let result = iter.collect::<Vec<_>>()?;
        assert_eq!(result, data);
        Ok(())
    }

    #[test]
    fn more_than_one_block() -> Result<(), ParquetError> {
        let mut data = vec![1, 3, -1, 2, 3, 10, 1];
        for x in 0..128 {
            data.push(x - 10)
        }

        let mut buffer = vec![];
        encode(data.clone().into_iter(), &mut buffer, 1);
        let (iter, _) = Decoder::try_new(&buffer)?;

        let result = iter.collect::<Vec<_>>()?;
        assert_eq!(result, data);
        Ok(())
    }

    #[test]
    fn test_another() -> Result<(), ParquetError> {
        let data = vec![2, 3, 1, 2, 1];

        let mut buffer = vec![];
        encode(data.clone().into_iter(), &mut buffer, 1);
        let (iter, _) = Decoder::try_new(&buffer)?;

        let result = iter.collect::<Vec<_>>()?;
        assert_eq!(result, data);

        Ok(())
    }

    #[test]
    fn overflow_constant() -> ParquetResult<()> {
        let data = vec![i64::MIN, i64::MAX, i64::MIN, i64::MAX];

        let mut buffer = vec![];
        encode(data.clone().into_iter(), &mut buffer, 1);
        let (iter, _) = Decoder::try_new(&buffer)?;

        let result = iter.collect::<Vec<_>>()?;
        assert_eq!(result, data);

        Ok(())
    }

    #[test]
    fn overflow_vary() -> ParquetResult<()> {
        let data = vec![
            0,
            i64::MAX,
            i64::MAX - 1,
            i64::MIN + 1,
            i64::MAX,
            i64::MIN + 2,
        ];

        let mut buffer = vec![];
        encode(data.clone().into_iter(), &mut buffer, 1);
        let (iter, _) = Decoder::try_new(&buffer)?;

        let result = iter.collect::<Vec<_>>()?;
        assert_eq!(result, data);

        Ok(())
    }
}
