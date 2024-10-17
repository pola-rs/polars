mod decode;
#[cfg(feature = "async")]
mod decode_async;
mod encode;
#[cfg(feature = "async")]
mod encode_async;

pub use decode::{VarInt, VarIntReader};

#[cfg(feature = "async")]
pub use decode_async::VarIntAsyncReader;
pub use encode::VarIntWriter;
#[cfg(feature = "async")]
pub use encode_async::VarIntAsyncWriter;

#[cfg(test)]
mod tests {
    #[cfg(feature = "async")]
    use super::VarIntAsyncReader;
    #[cfg(feature = "async")]
    use super::VarIntAsyncWriter;

    use super::VarInt;
    use super::VarIntReader;
    use super::VarIntWriter;

    #[test]
    fn test_required_space() {
        assert_eq!((0u32).required_space(), 1);
        assert_eq!((1u32).required_space(), 1);
        assert_eq!((128u32).required_space(), 2);
        assert_eq!((16384u32).required_space(), 3);
        assert_eq!((2097151u32).required_space(), 3);
        assert_eq!((2097152u32).required_space(), 4);
    }

    #[test]
    fn test_encode_u64() {
        assert_eq!((0u32).encode_var_vec(), vec![0b00000000]);
        assert_eq!((300u32).encode_var_vec(), vec![0b10101100, 0b00000010]);
    }

    #[test]
    fn test_identity_u64() {
        for i in 1u64..100 {
            assert_eq!(
                u64::decode_var(i.encode_var_vec().as_slice()).unwrap(),
                (i, 1)
            );
        }
        for i in 16400u64..16500 {
            assert_eq!(
                u64::decode_var(i.encode_var_vec().as_slice()).unwrap(),
                (i, 3)
            );
        }
    }

    #[test]
    fn test_decode_max_u64() {
        let max_vec_encoded = vec![0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01];
        assert_eq!(
            u64::decode_var(max_vec_encoded.as_slice()).unwrap().0,
            u64::max_value()
        );
    }

    #[test]
    fn test_encode_i64() {
        assert_eq!((0i64).encode_var_vec(), (0u32).encode_var_vec());
        assert_eq!((150i64).encode_var_vec(), (300u32).encode_var_vec());
        assert_eq!((-150i64).encode_var_vec(), (299u32).encode_var_vec());
        assert_eq!(
            (-2147483648i64).encode_var_vec(),
            (4294967295u64).encode_var_vec()
        );
        assert_eq!(
            (i64::max_value() as i64).encode_var_vec(),
            &[0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]
        );
        assert_eq!(
            (i64::min_value() as i64).encode_var_vec(),
            &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]
        );
    }

    #[test]
    fn test_decode_min_i64() {
        let min_vec_encoded = vec![0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01];
        assert_eq!(
            i64::decode_var(min_vec_encoded.as_slice()).unwrap().0,
            i64::min_value()
        );
    }

    #[test]
    fn test_decode_max_i64() {
        let max_vec_encoded = vec![0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01];
        assert_eq!(
            i64::decode_var(max_vec_encoded.as_slice()).unwrap().0,
            i64::max_value()
        );
    }

    #[test]
    fn test_encode_i16() {
        assert_eq!((150i16).encode_var_vec(), (300u32).encode_var_vec());
        assert_eq!((-150i16).encode_var_vec(), (299u32).encode_var_vec());
    }

    #[test]
    fn test_reader_writer() {
        let mut buf = Vec::with_capacity(128);

        let i1: u32 = 1;
        let i2: u32 = 65532;
        let i3: u32 = 4200123456;
        let i4: i64 = i3 as i64 * 1000;
        let i5: i32 = -32456;

        assert!(buf.write_varint(i1).is_ok());
        assert!(buf.write_varint(i2).is_ok());
        assert!(buf.write_varint(i3).is_ok());
        assert!(buf.write_varint(i4).is_ok());
        assert!(buf.write_varint(i5).is_ok());

        let mut reader: &[u8] = buf.as_ref();

        assert_eq!(i1, reader.read_varint().unwrap());
        assert_eq!(i2, reader.read_varint().unwrap());
        assert_eq!(i3, reader.read_varint().unwrap());
        assert_eq!(i4, reader.read_varint().unwrap());
        assert_eq!(i5, reader.read_varint().unwrap());

        assert!(reader.read_varint::<u32>().is_err());
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_async_reader() {
        let mut buf = Vec::with_capacity(128);

        let i1: u32 = 1;
        let i2: u32 = 65532;
        let i3: u32 = 4200123456;
        let i4: i64 = i3 as i64 * 1000;
        let i5: i32 = -32456;

        buf.write_varint_async(i1).await.unwrap();
        buf.write_varint_async(i2).await.unwrap();
        buf.write_varint_async(i3).await.unwrap();
        buf.write_varint_async(i4).await.unwrap();
        buf.write_varint_async(i5).await.unwrap();

        let mut reader: &[u8] = buf.as_ref();

        assert_eq!(i1, reader.read_varint_async().await.unwrap());
        assert_eq!(i2, reader.read_varint_async().await.unwrap());
        assert_eq!(i3, reader.read_varint_async().await.unwrap());
        assert_eq!(i4, reader.read_varint_async().await.unwrap());
        assert_eq!(i5, reader.read_varint_async().await.unwrap());
        assert!(reader.read_varint_async::<u32>().await.is_err());
    }

    #[test]
    fn test_unterminated_varint() {
        let buf = vec![0xffu8; 12];
        let mut read = buf.as_slice();
        assert!(read.read_varint::<u64>().is_err());
    }

    #[test]
    fn test_unterminated_varint_2() {
        let buf = [0xff, 0xff];
        let mut read = &buf[..];
        assert!(read.read_varint::<u64>().is_err());
    }

    #[test]
    fn test_decode_extra_bytes_u64() {
        let mut encoded = 0x12345u64.encode_var_vec();
        assert_eq!(u64::decode_var(&encoded[..]), Some((0x12345, 3)));

        encoded.push(0x99);
        assert_eq!(u64::decode_var(&encoded[..]), Some((0x12345, 3)));

        let encoded = [0xFF, 0xFF, 0xFF];
        assert_eq!(u64::decode_var(&encoded[..]), None);

        // Overflow
        let mut encoded = vec![0xFF; 64];
        encoded.push(0x00);
        assert_eq!(u64::decode_var(&encoded[..]), None);
    }

    #[test]
    fn test_decode_extra_bytes_i64() {
        let mut encoded = (-0x12345i64).encode_var_vec();
        assert_eq!(i64::decode_var(&encoded[..]), Some((-0x12345, 3)));

        encoded.push(0x99);
        assert_eq!(i64::decode_var(&encoded[..]), Some((-0x12345, 3)));

        let encoded = [0xFF, 0xFF, 0xFF];
        assert_eq!(i64::decode_var(&encoded[..]), None);

        // Overflow
        let mut encoded = vec![0xFF; 64];
        encoded.push(0x00);
        assert_eq!(i64::decode_var(&encoded[..]), None);
    }

    #[test]
    fn test_regression_22() {
        let encoded: Vec<u8> = (0x112233u64).encode_var_vec();
        assert_eq!(
            encoded.as_slice().read_varint::<i8>().unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
    }
}
