use crate::prelude::*;
use crate::serde::DeDataType;
use serde::de::{MapAccess, Visitor};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::Cow;
use std::fmt::Formatter;

impl Serialize for Series {
    fn serialize<S>(
        &self,
        serializer: S,
    ) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        if let Ok(ca) = self.i32() {
            ca.serialize(serializer)
        } else if let Ok(ca) = self.u32() {
            ca.serialize(serializer)
        } else if let Ok(ca) = self.i64() {
            ca.serialize(serializer)
        } else if let Ok(ca) = self.u64() {
            ca.serialize(serializer)
        } else if let Ok(ca) = self.f32() {
            ca.serialize(serializer)
        } else if let Ok(ca) = self.f64() {
            ca.serialize(serializer)
        } else if let Ok(ca) = self.date() {
            ca.serialize(serializer)
        } else if let Ok(ca) = self.datetime() {
            ca.serialize(serializer)
        } else if let Ok(ca) = self.utf8() {
            ca.serialize(serializer)
        } else if let Ok(ca) = self.bool() {
            ca.serialize(serializer)
        } else if let Ok(ca) = self.categorical() {
            #[cfg(feature = "dtype-categorical")]
            {
                ca.serialize(serializer)
            }
            #[cfg(not(feature = "dtype-categorical"))]
            {
                panic!("activate dtype-categorical");
            }
        } else if let Ok(ca) = self.list() {
            ca.serialize(serializer)
        } else {
            // cast small integers to i32
            self.cast(&DataType::Int32).unwrap().serialize(serializer)
        }
    }
}

impl<'de> Deserialize<'de> for Series {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, <D as Deserializer<'de>>::Error>
    where
        D: Deserializer<'de>,
    {
        const FIELDS: &[&str] = &["name", "datatype", "values"];

        struct SeriesVisitor;

        impl<'de> Visitor<'de> for SeriesVisitor {
            type Value = Series;

            fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
                formatter
                    .write_str("struct {name: <name>, datatype: <dtype>, values: <values array>}")
            }

            fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut name: Option<Cow<'de, str>> = None;
                let mut dtype = None;
                let mut values_set = false;
                let mut count = 0;
                while let Some(key) = map.next_key::<Cow<str>>().unwrap() {
                    count += 1;
                    match key.as_ref() {
                        "name" => {
                            name = match map.next_value::<Cow<str>>() {
                                Ok(s) => Some(s),
                                Err(_) => Some(Cow::Owned(map.next_value::<String>()?)),
                            };
                        }
                        "datatype" => {
                            dtype = Some(map.next_value()?);
                        }
                        "values" => {
                            // we delay calling next_value until we know the dtype
                            values_set = true;
                            if count != 3 {
                                return Err(de::Error::custom(
                                    "field values should be behind name and datatype",
                                ));
                            }
                            break;
                        }
                        fld => return Err(de::Error::unknown_field(fld, FIELDS)),
                    }
                }
                if !values_set {
                    return Err(de::Error::missing_field("values"));
                }
                let name = name.ok_or_else(|| de::Error::missing_field("name"))?;
                let dtype = dtype.ok_or_else(|| de::Error::missing_field("datatype"))?;

                match dtype {
                    #[cfg(feature = "dtype-i8")]
                    DeDataType::Int8 => {
                        let values: Vec<Option<i8>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    }
                    #[cfg(feature = "dtype-u8")]
                    DeDataType::UInt8 => {
                        let values: Vec<Option<u8>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    }
                    DeDataType::Int32 => {
                        let values: Vec<Option<i32>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    }
                    DeDataType::UInt32 => {
                        let values: Vec<Option<u32>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    }
                    DeDataType::Int64 => {
                        let values: Vec<Option<i64>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    }
                    DeDataType::UInt64 => {
                        let values: Vec<Option<u64>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    }
                    #[cfg(feature = "dtype-date")]
                    DeDataType::Date => {
                        let values: Vec<Option<i32>> = map.next_value()?;
                        Ok(Series::new(&name, values).cast(&DataType::Date).unwrap())
                    }
                    #[cfg(feature = "dtype-datetime")]
                    DeDataType::Datetime => {
                        let values: Vec<Option<i64>> = map.next_value()?;
                        Ok(Series::new(&name, values)
                            .cast(&DataType::Datetime)
                            .unwrap())
                    }
                    DeDataType::Boolean => {
                        let values: Vec<Option<bool>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    }
                    DeDataType::Float32 => {
                        let values: Vec<Option<f32>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    }
                    DeDataType::Float64 => {
                        let values: Vec<Option<f64>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    }
                    DeDataType::Utf8 => {
                        let values: Vec<Option<Cow<str>>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    }
                    DeDataType::List => {
                        let values: Vec<Option<Series>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    }
                    dt => {
                        panic!("{:?} dtype deserialization not yet implemented", dt)
                    }
                }
            }
        }

        deserializer.deserialize_map(SeriesVisitor)
    }
}
