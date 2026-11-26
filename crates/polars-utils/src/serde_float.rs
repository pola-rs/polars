//! Serde helpers that keep NaN and infinity in human readable formats such as
//! JSON, which cannot represent them as numbers. Those values are written as the
//! strings `"NaN"`, `"inf"` and `"-inf"`. Binary formats are not affected.
use std::fmt;

use num_traits::AsPrimitive;
use serde::de::{self, DeserializeOwned, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub fn serialize<T, S>(v: &T, serializer: S) -> Result<S::Ok, S::Error>
where
    T: AsPrimitive<f64> + Serialize,
    S: Serializer,
{
    let f: f64 = v.as_();
    if serializer.is_human_readable() && !f.is_finite() {
        let s = if f.is_nan() {
            "NaN"
        } else if f > 0.0 {
            "inf"
        } else {
            "-inf"
        };
        serializer.serialize_str(s)
    } else {
        v.serialize(serializer)
    }
}

pub fn deserialize<'de, T, D>(d: D) -> Result<T, D::Error>
where
    T: DeserializeOwned + Copy + 'static,
    f64: AsPrimitive<T>,
    D: Deserializer<'de>,
{
    if d.is_human_readable() {
        d.deserialize_any(FloatVisitor).map(|v| v.as_())
    } else {
        T::deserialize(d)
    }
}

struct FloatVisitor;

impl Visitor<'_> for FloatVisitor {
    type Value = f64;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a number or one of \"NaN\", \"inf\", \"-inf\"")
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<f64, E> {
        Ok(v)
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> Result<f64, E> {
        Ok(v as f64)
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<f64, E> {
        Ok(v as f64)
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<f64, E> {
        match v {
            "NaN" => Ok(f64::NAN),
            "inf" => Ok(f64::INFINITY),
            "-inf" => Ok(f64::NEG_INFINITY),
            _ => Err(E::invalid_value(de::Unexpected::Str(v), &self)),
        }
    }
}

pub mod option_slice {
    use super::*;

    struct Wrap(f64);

    impl Serialize for Wrap {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            super::serialize(&self.0, serializer)
        }
    }

    impl<'de> Deserialize<'de> for Wrap {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            super::deserialize(d).map(Wrap)
        }
    }

    pub fn serialize<S: Serializer>(v: &[Option<f64>], serializer: S) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() {
            serializer.collect_seq(v.iter().map(|x| x.map(Wrap)))
        } else {
            v.serialize(serializer)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Box<[Option<f64>]>, D::Error> {
        if d.is_human_readable() {
            let v = Vec::<Option<Wrap>>::deserialize(d)?;
            Ok(v.into_iter().map(|x| x.map(|x| x.0)).collect())
        } else {
            Box::deserialize(d)
        }
    }
}
