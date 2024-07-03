mod decoder;

pub use decoder::Decoder;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::error::ParquetError;
    use crate::parquet::types::NativeType;

    #[test]
    fn round_trip_f32() -> Result<(), ParquetError> {
        let data = vec![1.0e-2_f32, 2.5_f32, 3.0e2_f32];
        let mut buffer = vec![];
        encode(&data, &mut buffer);

        let mut decoder = Decoder::try_new(&buffer, std::mem::size_of::<f32>())?;
        let values = decoder
            .iter_converted(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();

        assert_eq!(data, values);

        Ok(())
    }

    #[test]
    fn round_trip_f64() -> Result<(), ParquetError> {
        let data = vec![1.0e-2_f64, 2.5_f64, 3.0e2_f64];
        let mut buffer = vec![];
        encode(&data, &mut buffer);

        let mut decoder = Decoder::try_new(&buffer, std::mem::size_of::<f64>())?;
        let values = decoder
            .iter_converted(|bytes| f64::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();

        assert_eq!(data, values);

        Ok(())
    }

    #[test]
    fn fails_for_invalid_values_size() -> Result<(), ParquetError> {
        let buffer = vec![0; 12];

        let result = Decoder::try_new(&buffer, 8);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn fails_for_invalid_element_size() -> Result<(), ParquetError> {
        let buffer = vec![0; 16];

        let result = Decoder::try_new(&buffer, 16);
        assert!(result.is_err());

        Ok(())
    }

    fn encode<T: NativeType>(data: &[T], buffer: &mut Vec<u8>) {
        let element_size = std::mem::size_of::<T>();
        let num_elements = data.len();
        let total_length = std::mem::size_of_val(data);
        buffer.resize(total_length, 0);

        for (i, v) in data.iter().enumerate() {
            let value_bytes = v.to_le_bytes();
            let value_bytes_ref = value_bytes.as_ref();
            for n in 0..element_size {
                buffer[(num_elements * n) + i] = value_bytes_ref[n];
            }
        }
    }
}
