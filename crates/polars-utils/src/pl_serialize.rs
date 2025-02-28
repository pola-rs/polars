//! Centralized Polars serialization entry.
//!
//! Currently provides two serialization scheme's.
//! - Self-describing (and thus more forward compatible) activated with `FC: true`
//! - Compact activated with `FC: false`
use polars_error::{to_compute_err, PolarsResult};

fn serialize_impl<W, T, const FC: bool>(writer: W, value: &T) -> PolarsResult<()>
where
    W: std::io::Write,
    T: serde::ser::Serialize,
{
    if FC {
        let mut s = rmp_serde::Serializer::new(writer).with_struct_map();
        value.serialize(&mut s).map_err(to_compute_err)
    } else {
        bincode::serialize_into(writer, value).map_err(to_compute_err)
    }
}

pub fn deserialize_impl<T, R, const FC: bool>(reader: R) -> PolarsResult<T>
where
    T: serde::de::DeserializeOwned,
    R: std::io::Read,
{
    if FC {
        rmp_serde::from_read(reader).map_err(to_compute_err)
    } else {
        bincode::deserialize_from(reader).map_err(to_compute_err)
    }
}

/// Mainly used to enable compression when serializing the final outer value.
/// For intermediate serialization steps, the function in the module should
/// be used instead.
pub struct SerializeOptions {
    compression: bool,
}

impl SerializeOptions {
    pub fn with_compression(mut self, compression: bool) -> Self {
        self.compression = compression;
        self
    }

    pub fn serialize_into_writer<W, T, const FC: bool>(
        &self,
        writer: W,
        value: &T,
    ) -> PolarsResult<()>
    where
        W: std::io::Write,
        T: serde::ser::Serialize,
    {
        if self.compression {
            let writer = flate2::write::ZlibEncoder::new(writer, flate2::Compression::fast());
            serialize_impl::<_, _, FC>(writer, value)
        } else {
            serialize_impl::<_, _, FC>(writer, value)
        }
    }

    pub fn deserialize_from_reader<T, R, const FC: bool>(&self, reader: R) -> PolarsResult<T>
    where
        T: serde::de::DeserializeOwned,
        R: std::io::Read,
    {
        if self.compression {
            deserialize_impl::<_, _, FC>(flate2::read::ZlibDecoder::new(reader))
        } else {
            deserialize_impl::<_, _, FC>(reader)
        }
    }

    pub fn serialize_to_bytes<T, const FC: bool>(&self, value: &T) -> PolarsResult<Vec<u8>>
    where
        T: serde::ser::Serialize,
    {
        let mut v = vec![];

        self.serialize_into_writer::<_, _, FC>(&mut v, value)?;

        Ok(v)
    }
}

#[allow(clippy::derivable_impls)]
impl Default for SerializeOptions {
    fn default() -> Self {
        Self { compression: false }
    }
}

pub fn serialize_into_writer<W, T, const FC: bool>(writer: W, value: &T) -> PolarsResult<()>
where
    W: std::io::Write,
    T: serde::ser::Serialize,
{
    serialize_impl::<_, _, FC>(writer, value)
}

pub fn deserialize_from_reader<T, R, const FC: bool>(reader: R) -> PolarsResult<T>
where
    T: serde::de::DeserializeOwned,
    R: std::io::Read,
{
    deserialize_impl::<_, _, FC>(reader)
}

pub fn serialize_to_bytes<T, const FC: bool>(value: &T) -> PolarsResult<Vec<u8>>
where
    T: serde::ser::Serialize,
{
    let mut v = vec![];

    serialize_into_writer::<_, _, FC>(&mut v, value)?;

    Ok(v)
}

/// Potentially avoids copying memory compared to a naive `Vec::<u8>::deserialize`.
///
/// This is essentially boilerplate for visiting bytes without copying where possible.
pub fn deserialize_map_bytes<'de, D, O>(
    deserializer: D,
    mut func: impl for<'b> FnMut(std::borrow::Cow<'b, [u8]>) -> O,
) -> Result<O, D::Error>
where
    D: serde::de::Deserializer<'de>,
{
    // Lets us avoid monomorphizing the visitor
    let mut out: Option<O> = None;
    struct V<'f>(&'f mut (dyn for<'b> FnMut(std::borrow::Cow<'b, [u8]>)));

    deserializer.deserialize_bytes(V(&mut |v| drop(out.replace(func(v)))))?;

    return Ok(out.unwrap());

    impl<'de> serde::de::Visitor<'de> for V<'_> {
        type Value = ();

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("deserialize_map_bytes")
        }

        fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            self.0(std::borrow::Cow::Borrowed(v));
            Ok(())
        }

        fn visit_byte_buf<E>(self, v: Vec<u8>) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            self.0(std::borrow::Cow::Owned(v));
            Ok(())
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            // This is not ideal, but we hit here if the serialization format is JSON.
            let bytes = std::iter::from_fn(|| seq.next_element::<u8>().transpose())
                .collect::<Result<Vec<_>, A::Error>>()?;

            self.0(std::borrow::Cow::Owned(bytes));
            Ok(())
        }
    }
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
        let b = super::serialize_to_bytes::<_, false>(&v).unwrap();
        let r: Enum = super::deserialize_from_reader::<_, _, false>(b.as_slice()).unwrap();

        assert_eq!(r, v);

        let v = Enum::A;
        let b = super::SerializeOptions::default()
            .serialize_to_bytes::<_, false>(&v)
            .unwrap();
        let r: Enum = super::SerializeOptions::default()
            .deserialize_from_reader::<_, _, false>(b.as_slice())
            .unwrap();

        assert_eq!(r, v);
    }
}
