mod decoder;
mod encoder;

pub(crate) use decoder::Decoder;
pub(crate) use encoder::encode;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::error::ParquetError;

    #[test]
    fn basic() -> Result<(), ParquetError> {
        let data = vec!["aa", "bbb", "a", "aa", "b"];

        let mut buffer = vec![];
        encode(data.into_iter().map(|x| x.as_bytes()), &mut buffer);

        let mut iter = Decoder::try_new(&buffer)?;

        let result = iter.by_ref().collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            result,
            vec![
                b"aa".as_ref(),
                b"bbb".as_ref(),
                b"a".as_ref(),
                b"aa".as_ref(),
                b"b".as_ref()
            ]
        );

        let result = iter.values;
        assert_eq!(result, b"aabbbaaab".as_ref());
        Ok(())
    }

    #[test]
    fn many_numbers() -> Result<(), ParquetError> {
        let mut data = vec![];
        for i in 0..136 {
            data.push(format!("a{}", i))
        }

        let expected = data
            .iter()
            .map(|v| v.as_bytes().to_vec())
            .collect::<Vec<_>>();

        let mut buffer = vec![];
        encode(data.into_iter(), &mut buffer);

        let mut iter = Decoder::try_new(&buffer)?;

        let result = iter.by_ref().collect::<Result<Vec<_>, _>>()?;
        assert_eq!(result, expected);

        Ok(())
    }
}
