use polars_error::{to_compute_err, PolarsResult};

fn serialize_impl<W, T>(writer: W, value: &T) -> PolarsResult<()>
where
    W: std::io::Write,
    T: serde::ser::Serialize,
{
    bincode::serialize_into(writer, value).map_err(to_compute_err)
}

pub fn deserialize_impl<T, R>(reader: R) -> PolarsResult<T>
where
    T: serde::de::DeserializeOwned,
    R: std::io::Read,
{
    bincode::deserialize_from(reader).map_err(to_compute_err)
}

/// Mainly used to enable compression when serializing the final outer value.
/// For intermediate serialization steps, the function in the module should
/// be used instead.
pub struct SerializeOptions {
    compression: bool,
}

impl SerializeOptions {
    pub fn new(compression: bool) -> Self {
        Self { compression }
    }

    pub fn serialize_into_writer<W, T>(&self, writer: W, value: &T) -> PolarsResult<()>
    where
        W: std::io::Write,
        T: serde::ser::Serialize,
    {
        if self.compression {
            let writer = flate2::write::ZlibEncoder::new(writer, flate2::Compression::fast());
            serialize_impl(writer, value)
        } else {
            serialize_impl(writer, value)
        }
    }

    pub fn deserialize_from_reader<T, R>(&self, reader: R) -> PolarsResult<T>
    where
        T: serde::de::DeserializeOwned,
        R: std::io::Read,
    {
        if self.compression {
            deserialize_impl(flate2::read::ZlibDecoder::new(reader))
        } else {
            deserialize_impl(reader)
        }
    }

    pub fn serialize_to_bytes<T>(&self, value: &T) -> PolarsResult<Vec<u8>>
    where
        T: serde::ser::Serialize,
    {
        let mut v = vec![];

        self.serialize_into_writer(&mut v, value)?;

        Ok(v)
    }
}

pub fn serialize_into_writer<W, T>(writer: W, value: &T) -> PolarsResult<()>
where
    W: std::io::Write,
    T: serde::ser::Serialize,
{
    serialize_impl(writer, value)
}

pub fn deserialize_from_reader<T, R>(reader: R) -> PolarsResult<T>
where
    T: serde::de::DeserializeOwned,
    R: std::io::Read,
{
    deserialize_impl(reader)
}

pub fn serialize_to_bytes<T>(value: &T) -> PolarsResult<Vec<u8>>
where
    T: serde::ser::Serialize,
{
    let mut v = vec![];

    serialize_into_writer(&mut v, value)?;

    Ok(v)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_serde_skip_enum() {
        #[derive(Default, Debug, PartialEq)]
        struct MyType(Option<usize>);

        // Note: serde(skip) must be at the end of enums
        #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
        enum Enum {
            A,
            #[serde(skip)]
            B(MyType),
        }

        impl Default for Enum {
            fn default() -> Self {
                Self::B(MyType(None))
            }
        }

        let v = Enum::A;
        let b = super::serialize_to_bytes(&v).unwrap();
        let r: Enum = super::deserialize_from_reader(b.as_slice()).unwrap();

        assert_eq!(r, v);

        let v = Enum::A;
        let b = super::SerializeOptions::new(true)
            .serialize_to_bytes(&v)
            .unwrap();
        let r: Enum = super::SerializeOptions::new(true)
            .deserialize_from_reader(b.as_slice())
            .unwrap();

        assert_eq!(r, v);
    }
}
