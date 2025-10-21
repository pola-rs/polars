//! Centralized Polars serialization entry.
//!
//! Currently provides two serialization scheme's.
//! - Self-describing (and thus more forward compatible) activated with `FC: true`
//! - Compact activated with `FC: false`
use polars_error::{PolarsResult, to_compute_err};

fn config() -> bincode::config::Configuration {
    bincode::config::standard()
        .with_no_limit()
        .with_variable_int_encoding()
}

fn serialize_impl<W, T, const FC: bool>(mut writer: W, value: &T) -> PolarsResult<()>
where
    W: std::io::Write,
    T: serde::ser::Serialize,
{
    if FC {
        let mut s = rmp_serde::Serializer::new(writer).with_struct_map();
        value.serialize(&mut s).map_err(to_compute_err)
    } else {
        bincode::serde::encode_into_std_write(value, &mut writer, config())
            .map_err(to_compute_err)
            .map(|_| ())
    }
}

pub fn deserialize_impl<T, R, const FC: bool>(mut reader: R) -> PolarsResult<T>
where
    T: serde::de::DeserializeOwned,
    R: std::io::Read,
{
    if FC {
        rmp_serde::from_read(reader).map_err(to_compute_err)
    } else {
        bincode::serde::decode_from_std_read(&mut reader, config()).map_err(to_compute_err)
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

/// Serialize function customized for `DslPlan`, with stack overflow protection.
pub fn serialize_dsl<W, T>(writer: W, value: &T) -> PolarsResult<()>
where
    W: std::io::Write,
    T: serde::ser::Serialize,
{
    let mut s = rmp_serde::Serializer::new(writer).with_struct_map();
    let s = serde_stacker::Serializer::new(&mut s);
    value.serialize(s).map_err(to_compute_err)
}

/// Deserialize function customized for `DslPlan`, with stack overflow protection.
pub fn deserialize_dsl<T, R>(reader: R) -> PolarsResult<T>
where
    T: serde::de::DeserializeOwned,
    R: std::io::Read,
{
    let mut de = rmp_serde::Deserializer::new(reader);
    de.set_max_depth(usize::MAX);
    let de = serde_stacker::Deserializer::new(&mut de);
    T::deserialize(de).map_err(to_compute_err)
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
    struct V<'f>(&'f mut dyn for<'b> FnMut(std::borrow::Cow<'b, [u8]>));

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

thread_local! {
    pub static USE_CLOUDPICKLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(feature = "python")]
pub fn python_object_serialize(
    pyobj: &pyo3::Py<pyo3::PyAny>,
    buf: &mut Vec<u8>,
) -> PolarsResult<()> {
    use pyo3::Python;
    use pyo3::pybacked::PyBackedBytes;
    use pyo3::types::{PyAnyMethods, PyModule};

    use crate::python_function::PYTHON3_VERSION;

    let mut use_cloudpickle = USE_CLOUDPICKLE.get();
    let dumped = Python::attach(|py| {
        // Pickle with whatever pickling method was selected.
        if use_cloudpickle {
            let cloudpickle = PyModule::import(py, "cloudpickle")?.getattr("dumps")?;
            cloudpickle.call1((pyobj.clone_ref(py),))?
        } else {
            let pickle = PyModule::import(py, "pickle")?.getattr("dumps")?;
            match pickle.call1((pyobj.clone_ref(py),)) {
                Ok(dumped) => dumped,
                Err(_) => {
                    use_cloudpickle = true;
                    let cloudpickle = PyModule::import(py, "cloudpickle")?.getattr("dumps")?;
                    cloudpickle.call1((pyobj.clone_ref(py),))?
                },
            }
        }
        .extract::<PyBackedBytes>()
    })?;

    // Write pickle metadata
    buf.push(use_cloudpickle as u8);
    buf.extend_from_slice(&*PYTHON3_VERSION);

    // Write UDF
    buf.extend_from_slice(&dumped);
    Ok(())
}

#[cfg(feature = "python")]
pub fn python_object_deserialize(buf: &[u8]) -> PolarsResult<pyo3::Py<pyo3::PyAny>> {
    use polars_error::polars_ensure;
    use pyo3::Python;
    use pyo3::types::{PyAnyMethods, PyBytes, PyModule};

    use crate::python_function::PYTHON3_VERSION;

    // Handle pickle metadata
    let use_cloudpickle = buf[0] != 0;
    if use_cloudpickle {
        let ser_py_version = &buf[1..3];
        let cur_py_version = *PYTHON3_VERSION;
        polars_ensure!(
            ser_py_version == cur_py_version,
            InvalidOperation:
            "current Python version {:?} does not match the Python version used to serialize the UDF {:?}",
            (3, cur_py_version[0], cur_py_version[1]),
            (3, ser_py_version[0], ser_py_version[1] )
        );
    }
    let buf = &buf[3..];

    Python::attach(|py| {
        let loads = PyModule::import(py, "pickle")?.getattr("loads")?;
        let arg = (PyBytes::new(py, buf),);
        let python_function = loads.call1(arg)?;
        Ok(python_function.into())
    })
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
