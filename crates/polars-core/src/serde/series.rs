use std::borrow::Cow;
use std::fmt::Formatter;

use serde::de::{Error as DeError, MapAccess, Visitor};
#[cfg(feature = "object")]
use serde::ser::Error as SerError;
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

#[cfg(feature = "dtype-array")]
use crate::chunked_array::builder::get_fixed_size_list_builder;
use crate::chunked_array::builder::AnonymousListBuilder;
use crate::chunked_array::metadata::MetadataFlags;
use crate::prelude::*;

impl Serialize for Series {
    fn serialize<S>(
        &self,
        serializer: S,
    ) -> std::result::Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        match self.dtype() {
            DataType::Binary => {
                let ca = self.binary().unwrap();
                ca.serialize(serializer)
            },
            DataType::List(_) => {
                let ca = self.list().unwrap();
                ca.serialize(serializer)
            },
            #[cfg(feature = "dtype-array")]
            DataType::Array(_, _) => {
                let ca = self.array().unwrap();
                ca.serialize(serializer)
            },
            DataType::Boolean => {
                let ca = self.bool().unwrap();
                ca.serialize(serializer)
            },
            DataType::String => {
                let ca = self.str().unwrap();
                ca.serialize(serializer)
            },
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => {
                let ca = self.struct_().unwrap();
                ca.serialize(serializer)
            },
            #[cfg(feature = "dtype-date")]
            DataType::Date => {
                let ca = self.date().unwrap();
                ca.serialize(serializer)
            },
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => {
                let ca = self.datetime().unwrap();
                ca.serialize(serializer)
            },
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                let ca = self.categorical().unwrap();
                ca.serialize(serializer)
            },
            #[cfg(feature = "dtype-duration")]
            DataType::Duration(_) => {
                let ca = self.duration().unwrap();
                ca.serialize(serializer)
            },
            #[cfg(feature = "dtype-time")]
            DataType::Time => {
                let ca = self.time().unwrap();
                ca.serialize(serializer)
            },
            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(_, _) => {
                let ca = self.decimal().unwrap();
                ca.serialize(serializer)
            },
            DataType::Null => {
                let ca = self.null().unwrap();
                ca.serialize(serializer)
            },
            #[cfg(feature = "object")]
            DataType::Object(_, _) => Err(S::Error::custom(
                "serializing data of type Object is not supported",
            )),
            dt => {
                with_match_physical_numeric_polars_type!(dt, |$T| {
                let ca: &ChunkedArray<$T> = self.as_ref().as_ref().as_ref();
                ca.serialize(serializer)
                })
            },
        }
    }
}

impl<'de> Deserialize<'de> for Series {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, <D as Deserializer<'de>>::Error>
    where
        D: Deserializer<'de>,
    {
        const FIELDS: &[&str] = &["name", "datatype", "bit_settings", "values"];

        struct SeriesVisitor;

        impl<'de> Visitor<'de> for SeriesVisitor {
            type Value = Series;

            fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
                formatter
                    .write_str("struct {name: <name>, datatype: <dtype>, bit_settings?: <settings>, values: <values array>}")
            }

            fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut name: Option<Cow<'de, str>> = None;
                let mut dtype = None;
                let mut bit_settings: Option<MetadataFlags> = None;
                let mut values_set = false;
                while let Some(key) = map.next_key::<Cow<str>>().unwrap() {
                    match key.as_ref() {
                        "name" => {
                            name = match map.next_value::<Cow<str>>() {
                                Ok(s) => Some(s),
                                Err(_) => Some(Cow::Owned(map.next_value::<String>()?)),
                            };
                        },
                        "datatype" => {
                            dtype = Some(map.next_value()?);
                        },
                        "bit_settings" => {
                            bit_settings = Some(map.next_value()?);
                        },
                        "values" => {
                            // we delay calling next_value until we know the dtype
                            values_set = true;
                            break;
                        },
                        fld => return Err(de::Error::unknown_field(fld, FIELDS)),
                    }
                }
                if !values_set {
                    return Err(de::Error::missing_field("values"));
                }
                let name = name.ok_or_else(|| de::Error::missing_field("name"))?;
                let dtype = dtype.ok_or_else(|| de::Error::missing_field("datatype"))?;

                let mut s = match dtype {
                    #[cfg(feature = "dtype-i8")]
                    DataType::Int8 => {
                        let values: Vec<Option<i8>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    },
                    #[cfg(feature = "dtype-u8")]
                    DataType::UInt8 => {
                        let values: Vec<Option<u8>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    },
                    #[cfg(feature = "dtype-i16")]
                    DataType::Int16 => {
                        let values: Vec<Option<i16>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    },
                    #[cfg(feature = "dtype-u16")]
                    DataType::UInt16 => {
                        let values: Vec<Option<u16>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    },
                    DataType::Int32 => {
                        let values: Vec<Option<i32>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    },
                    DataType::UInt32 => {
                        let values: Vec<Option<u32>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    },
                    DataType::Int64 => {
                        let values: Vec<Option<i64>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    },
                    DataType::UInt64 => {
                        let values: Vec<Option<u64>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    },
                    #[cfg(feature = "dtype-date")]
                    DataType::Date => {
                        let values: Vec<Option<i32>> = map.next_value()?;
                        Ok(Series::new(&name, values).cast(&DataType::Date).unwrap())
                    },
                    #[cfg(feature = "dtype-datetime")]
                    DataType::Datetime(tu, tz) => {
                        let values: Vec<Option<i64>> = map.next_value()?;
                        Ok(Series::new(&name, values)
                            .cast(&DataType::Datetime(tu, tz))
                            .unwrap())
                    },
                    #[cfg(feature = "dtype-duration")]
                    DataType::Duration(tu) => {
                        let values: Vec<Option<i64>> = map.next_value()?;
                        Ok(Series::new(&name, values)
                            .cast(&DataType::Duration(tu))
                            .unwrap())
                    },
                    #[cfg(feature = "dtype-time")]
                    DataType::Time => {
                        let values: Vec<Option<i64>> = map.next_value()?;
                        Ok(Series::new(&name, values).cast(&DataType::Time).unwrap())
                    },
                    #[cfg(feature = "dtype-decimal")]
                    DataType::Decimal(precision, Some(scale)) => {
                        let values: Vec<Option<i128>> = map.next_value()?;
                        Ok(ChunkedArray::from_slice_options(&name, &values)
                            .into_decimal_unchecked(precision, scale)
                            .into_series())
                    },
                    DataType::Boolean => {
                        let values: Vec<Option<bool>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    },
                    DataType::Float32 => {
                        let values: Vec<Option<f32>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    },
                    DataType::Float64 => {
                        let values: Vec<Option<f64>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    },
                    DataType::String => {
                        let values: Vec<Option<Cow<str>>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    },
                    DataType::List(inner) => {
                        let values: Vec<Option<Series>> = map.next_value()?;
                        let mut lb = AnonymousListBuilder::new(&name, values.len(), Some(*inner));
                        for value in &values {
                            lb.append_opt_series(value.as_ref()).map_err(|e| {
                                de::Error::custom(format!("could not append series to list: {e}"))
                            })?;
                        }
                        Ok(lb.finish().into_series())
                    },
                    #[cfg(feature = "dtype-array")]
                    DataType::Array(inner, width) => {
                        let values: Vec<Option<Series>> = map.next_value()?;
                        let mut builder =
                            get_fixed_size_list_builder(&inner, values.len(), width, &name)
                                .map_err(|e| {
                                    de::Error::custom(format!(
                                        "could not get supported list builder: {e}"
                                    ))
                                })?;
                        for value in &values {
                            if let Some(s) = value {
                                // we only have one chunk per series as we serialize it in this way.
                                let arr = &s.chunks()[0];
                                // SAFETY, we are within bounds
                                unsafe {
                                    builder.push_unchecked(arr.as_ref(), 0);
                                }
                            } else {
                                // SAFETY, we are within bounds
                                unsafe {
                                    builder.push_null();
                                }
                            }
                        }
                        Ok(builder.finish().into_series())
                    },
                    DataType::Binary => {
                        let values: Vec<Option<Cow<[u8]>>> = map.next_value()?;
                        Ok(Series::new(&name, values))
                    },
                    #[cfg(feature = "dtype-struct")]
                    DataType::Struct(_) => {
                        let values: Vec<Series> = map.next_value()?;
                        let ca = StructChunked::from_series(&name, &values).unwrap();
                        let mut s = ca.into_series();
                        s.rename(&name);
                        Ok(s)
                    },
                    #[cfg(feature = "dtype-categorical")]
                    dt @ (DataType::Categorical(_, _) | DataType::Enum(_, _)) => {
                        let values: Vec<Option<Cow<str>>> = map.next_value()?;
                        Ok(Series::new(&name, values).cast(&dt).unwrap())
                    },
                    DataType::Null => {
                        let values: Vec<usize> = map.next_value()?;
                        let len = values.first().unwrap();
                        Ok(Series::new_null(&name, *len))
                    },
                    dt => Err(A::Error::custom(format!(
                        "deserializing data of type {dt} is not supported"
                    ))),
                }?;

                if let Some(f) = bit_settings {
                    s.set_flags(f)
                }
                Ok(s)
            }
        }

        deserializer.deserialize_map(SeriesVisitor)
    }
}
