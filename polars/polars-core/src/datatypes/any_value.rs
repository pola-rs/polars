use arrow::types::PrimitiveType;
#[cfg(feature = "dtype-categorical")]
use polars_utils::sync::SyncPtr;
use polars_utils::unwrap::UnwrapUncheckedRelease;

use super::*;

#[cfg(feature = "object")]
#[derive(Debug)]
pub struct OwnedObject(pub Box<dyn PolarsObjectSafe>);

#[cfg(feature = "object")]
impl Clone for OwnedObject {
    fn clone(&self) -> Self {
        Self(self.0.to_boxed())
    }
}

#[derive(Debug, Clone, Default)]
pub enum AnyValue<'a> {
    #[default]
    Null,
    /// A binary true or false.
    Boolean(bool),
    /// A UTF8 encoded string type.
    Utf8(&'a str),
    /// An unsigned 8-bit integer number.
    UInt8(u8),
    /// An unsigned 16-bit integer number.
    UInt16(u16),
    /// An unsigned 32-bit integer number.
    UInt32(u32),
    /// An unsigned 64-bit integer number.
    UInt64(u64),
    /// An 8-bit integer number.
    Int8(i8),
    /// A 16-bit integer number.
    Int16(i16),
    /// A 32-bit integer number.
    Int32(i32),
    /// A 64-bit integer number.
    Int64(i64),
    /// A 32-bit floating point number.
    Float32(f32),
    /// A 64-bit floating point number.
    Float64(f64),
    /// A 32-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in days (32 bits).
    #[cfg(feature = "dtype-date")]
    Date(i32),
    /// A 64-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in nanoseconds (64 bits).
    #[cfg(feature = "dtype-datetime")]
    Datetime(i64, TimeUnit, &'a Option<TimeZone>),
    // A 64-bit integer representing difference between date-times in [`TimeUnit`]
    #[cfg(feature = "dtype-duration")]
    Duration(i64, TimeUnit),
    /// A 64-bit time representing the elapsed time since midnight in nanoseconds
    #[cfg(feature = "dtype-time")]
    Time(i64),
    #[cfg(feature = "dtype-categorical")]
    // If syncptr is_null the data is in the rev-map
    // otherwise it is in the array pointer
    Categorical(u32, &'a RevMapping, SyncPtr<Utf8Array<i64>>),
    /// Nested type, contains arrays that are filled with one of the datetypes.
    List(Series),
    #[cfg(feature = "object")]
    /// Can be used to fmt and implements Any, so can be downcasted to the proper value type.
    #[cfg(feature = "object")]
    Object(&'a dyn PolarsObjectSafe),
    #[cfg(feature = "object")]
    ObjectOwned(OwnedObject),
    #[cfg(feature = "dtype-struct")]
    // 3 pointers and thus not larger than string/vec
    Struct(usize, &'a StructArray, &'a [Field]),
    #[cfg(feature = "dtype-struct")]
    StructOwned(Box<(Vec<AnyValue<'a>>, Vec<Field>)>),
    /// An UTF8 encoded string type.
    Utf8Owned(smartstring::alias::String),
    #[cfg(feature = "dtype-binary")]
    Binary(&'a [u8]),
    #[cfg(feature = "dtype-binary")]
    BinaryOwned(Vec<u8>),
}

#[cfg(feature = "serde")]
impl Serialize for AnyValue<'_> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let name = "AnyValue";
        match self {
            AnyValue::Null => serializer.serialize_unit_variant(name, 0, "Null"),
            AnyValue::Int8(v) => serializer.serialize_newtype_variant(name, 1, "Int8", v),
            AnyValue::Int16(v) => serializer.serialize_newtype_variant(name, 2, "Int16", v),
            AnyValue::Int32(v) => serializer.serialize_newtype_variant(name, 3, "Int32", v),
            AnyValue::Int64(v) => serializer.serialize_newtype_variant(name, 4, "Int64", v),
            AnyValue::UInt8(v) => serializer.serialize_newtype_variant(name, 5, "UInt8", v),
            AnyValue::UInt16(v) => serializer.serialize_newtype_variant(name, 6, "UInt16", v),
            AnyValue::UInt32(v) => serializer.serialize_newtype_variant(name, 7, "UInt32", v),
            AnyValue::UInt64(v) => serializer.serialize_newtype_variant(name, 8, "UInt64", v),
            AnyValue::Float32(v) => serializer.serialize_newtype_variant(name, 9, "Float32", v),
            AnyValue::Float64(v) => serializer.serialize_newtype_variant(name, 10, "Float64", v),
            AnyValue::List(v) => serializer.serialize_newtype_variant(name, 11, "List", v),
            AnyValue::Boolean(v) => serializer.serialize_newtype_variant(name, 12, "Bool", v),
            // both utf8 variants same number
            AnyValue::Utf8(v) => serializer.serialize_newtype_variant(name, 13, "Utf8Owned", v),
            AnyValue::Utf8Owned(v) => {
                serializer.serialize_newtype_variant(name, 13, "Utf8Owned", v.as_str())
            }
            #[cfg(feature = "dtype-binary")]
            AnyValue::Binary(v) => serializer.serialize_newtype_variant(name, 14, "BinaryOwned", v),
            #[cfg(feature = "dtype-binary")]
            AnyValue::BinaryOwned(v) => {
                serializer.serialize_newtype_variant(name, 14, "BinaryOwned", v)
            }
            _ => todo!(),
        }
    }
}

#[cfg(feature = "serde")]
impl<'a> Deserialize<'a> for AnyValue<'static> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'a>,
    {
        #[repr(u8)]
        enum AvField {
            Null,
            Int8,
            Int16,
            Int32,
            Int64,
            UInt8,
            UInt16,
            UInt32,
            UInt64,
            Float32,
            Float64,
            List,
            Bool,
            Utf8Owned,
            #[cfg(feature = "dtype-binary")]
            BinaryOwned,
        }
        const VARIANTS: &[&str] = &[
            "Null",
            "UInt8",
            "UInt16",
            "UInt32",
            "UInt64",
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "Float32",
            "Float64",
            "List",
            "Boolean",
            "Utf8Owned",
            "BinaryOwned",
        ];
        #[cfg(feature = "dtype-binary")]
        const LAST: u8 = unsafe { std::mem::transmute::<_, u8>(AvField::BinaryOwned) };
        #[cfg(not(feature = "dtype-binary"))]
        const LAST: u8 = unsafe { std::mem::transmute::<_, u8>(AvField::Utf8Owned) };

        struct FieldVisitor;

        impl Visitor<'_> for FieldVisitor {
            type Value = AvField;

            fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
                write!(formatter, "an integer between 0-{LAST}")
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
            where
                E: Error,
            {
                let field: u8 = NumCast::from(v).ok_or_else(|| {
                    serde::de::Error::invalid_value(
                        Unexpected::Signed(v),
                        &"expected value that fits into u8",
                    )
                })?;

                // safety:
                // we are repr: u8 and check last value that we are in bounds
                let field = unsafe {
                    if field <= LAST {
                        std::mem::transmute::<u8, AvField>(field)
                    } else {
                        return Err(serde::de::Error::invalid_value(
                            Unexpected::Signed(v),
                            &"expected value that fits into AnyValue's number of fields",
                        ));
                    }
                };
                Ok(field)
            }

            fn visit_str<E>(self, v: &str) -> std::result::Result<Self::Value, E>
            where
                E: Error,
            {
                self.visit_bytes(v.as_bytes())
            }

            fn visit_bytes<E>(self, v: &[u8]) -> std::result::Result<Self::Value, E>
            where
                E: Error,
            {
                let field = match v {
                    b"Null" => AvField::Null,
                    b"Int8" => AvField::Int8,
                    b"Int16" => AvField::Int16,
                    b"Int32" => AvField::Int32,
                    b"Int64" => AvField::Int64,
                    b"UInt8" => AvField::UInt8,
                    b"UInt16" => AvField::UInt16,
                    b"UInt32" => AvField::UInt32,
                    b"UInt64" => AvField::UInt64,
                    b"Float32" => AvField::Float32,
                    b"Float64" => AvField::Float64,
                    b"List" => AvField::List,
                    b"Bool" => AvField::Bool,
                    b"Utf8Owned" | b"Utf8" => AvField::Utf8Owned,
                    #[cfg(feature = "dtype-binary")]
                    b"BinaryOwned" | b"Binary" => AvField::BinaryOwned,
                    _ => {
                        return Err(serde::de::Error::unknown_variant(
                            &String::from_utf8_lossy(v),
                            VARIANTS,
                        ))
                    }
                };
                Ok(field)
            }
        }

        impl<'a> Deserialize<'a> for AvField {
            fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
            where
                D: Deserializer<'a>,
            {
                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct OuterVisitor;

        impl<'b> Visitor<'b> for OuterVisitor {
            type Value = AnyValue<'static>;

            fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
                write!(formatter, "enum AnyValue")
            }

            fn visit_enum<A>(self, data: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: EnumAccess<'b>,
            {
                let out = match data.variant()? {
                    (AvField::Null, _variant) => AnyValue::Null,
                    (AvField::Int8, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Int8(value)
                    }
                    (AvField::Int16, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Int16(value)
                    }
                    (AvField::Int32, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Int32(value)
                    }
                    (AvField::Int64, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Int64(value)
                    }
                    (AvField::UInt8, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::UInt8(value)
                    }
                    (AvField::UInt16, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::UInt16(value)
                    }
                    (AvField::UInt32, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::UInt32(value)
                    }
                    (AvField::UInt64, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::UInt64(value)
                    }
                    (AvField::Float32, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Float32(value)
                    }
                    (AvField::Float64, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Float64(value)
                    }
                    (AvField::Bool, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::Boolean(value)
                    }
                    (AvField::List, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::List(value)
                    }
                    (AvField::Utf8Owned, variant) => {
                        let value: String = variant.newtype_variant()?;
                        AnyValue::Utf8Owned(value.into())
                    }
                    #[cfg(feature = "dtype-binary")]
                    (AvField::BinaryOwned, variant) => {
                        let value = variant.newtype_variant()?;
                        AnyValue::BinaryOwned(value)
                    }
                };
                Ok(out)
            }
        }
        deserializer.deserialize_enum("AnyValue", VARIANTS, OuterVisitor)
    }
}

impl<'a> AnyValue<'a> {
    pub fn dtype(&self) -> DataType {
        use AnyValue::*;
        match self.as_borrowed() {
            Null => DataType::Unknown,
            Int8(_) => DataType::Int8,
            Int16(_) => DataType::Int16,
            Int32(_) => DataType::Int32,
            Int64(_) => DataType::Int64,
            UInt8(_) => DataType::UInt8,
            UInt16(_) => DataType::UInt16,
            UInt32(_) => DataType::UInt32,
            UInt64(_) => DataType::UInt64,
            Float32(_) => DataType::Float32,
            Float64(_) => DataType::Float64,
            #[cfg(feature = "dtype-date")]
            Date(_) => DataType::Date,
            #[cfg(feature = "dtype-datetime")]
            Datetime(_, tu, tz) => DataType::Datetime(tu, tz.clone()),
            #[cfg(feature = "dtype-time")]
            Time(_) => DataType::Time,
            #[cfg(feature = "dtype-duration")]
            Duration(_, tu) => DataType::Duration(tu),
            Boolean(_) => DataType::Boolean,
            Utf8(_) => DataType::Utf8,
            #[cfg(feature = "dtype-categorical")]
            Categorical(_, _, _) => DataType::Categorical(None),
            List(s) => DataType::List(Box::new(s.dtype().clone())),
            #[cfg(feature = "dtype-struct")]
            Struct(_, _, fields) => DataType::Struct(fields.to_vec()),
            #[cfg(feature = "dtype-struct")]
            StructOwned(payload) => DataType::Struct(payload.1.clone()),
            #[cfg(feature = "dtype-binary")]
            Binary(_) => DataType::Binary,
            _ => unimplemented!(),
        }
    }
    /// Extract a numerical value from the AnyValue
    #[doc(hidden)]
    #[cfg(feature = "private")]
    #[inline]
    pub fn extract<T: NumCast>(&self) -> Option<T> {
        use AnyValue::*;
        match self {
            Int8(v) => NumCast::from(*v),
            Int16(v) => NumCast::from(*v),
            Int32(v) => NumCast::from(*v),
            Int64(v) => NumCast::from(*v),
            UInt8(v) => NumCast::from(*v),
            UInt16(v) => NumCast::from(*v),
            UInt32(v) => NumCast::from(*v),
            UInt64(v) => NumCast::from(*v),
            Float32(v) => NumCast::from(*v),
            Float64(v) => NumCast::from(*v),
            #[cfg(feature = "dtype-date")]
            Date(v) => NumCast::from(*v),
            #[cfg(feature = "dtype-datetime")]
            Datetime(v, _, _) => NumCast::from(*v),
            #[cfg(feature = "dtype-time")]
            Time(v) => NumCast::from(*v),
            #[cfg(feature = "dtype-duration")]
            Duration(v, _) => NumCast::from(*v),
            Boolean(v) => {
                if *v {
                    NumCast::from(1)
                } else {
                    NumCast::from(0)
                }
            }
            _ => None,
        }
    }

    #[inline]
    pub fn try_extract<T: NumCast>(&self) -> PolarsResult<T> {
        self.extract().ok_or_else(|| {
            PolarsError::ComputeError(
                format!(
                    "could not extract number from AnyValue of dtype: '{:?}'",
                    self.dtype()
                )
                .into(),
            )
        })
    }
}

impl From<AnyValue<'_>> for DataType {
    fn from(value: AnyValue<'_>) -> Self {
        value.dtype()
    }
}

impl<'a> Hash for AnyValue<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        use AnyValue::*;
        match self {
            Null => state.write_u64(u64::MAX / 2 + 135123),
            Int8(v) => state.write_i8(*v),
            Int16(v) => state.write_i16(*v),
            Int32(v) => state.write_i32(*v),
            Int64(v) => state.write_i64(*v),
            UInt8(v) => state.write_u8(*v),
            UInt16(v) => state.write_u16(*v),
            UInt32(v) => state.write_u32(*v),
            UInt64(v) => state.write_u64(*v),
            Utf8(v) => state.write(v.as_bytes()),
            Utf8Owned(v) => state.write(v.as_bytes()),
            Float32(v) => state.write_u32(v.to_bits()),
            Float64(v) => state.write_u64(v.to_bits()),
            #[cfg(feature = "dtype-binary")]
            Binary(v) => state.write(v),
            #[cfg(feature = "dtype-binary")]
            BinaryOwned(v) => state.write(v),
            Boolean(v) => state.write_u8(*v as u8),
            List(v) => Hash::hash(&Wrap(v.clone()), state),
            _ => unimplemented!(),
        }
    }
}

impl<'a> Eq for AnyValue<'a> {}

impl<'a, T> From<Option<T>> for AnyValue<'a>
where
    T: Into<AnyValue<'a>>,
{
    #[inline]
    fn from(a: Option<T>) -> Self {
        match a {
            None => AnyValue::Null,
            Some(v) => v.into(),
        }
    }
}

impl<'a> AnyValue<'a> {
    #[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
    pub(crate) fn into_date(self) -> Self {
        match self {
            #[cfg(feature = "dtype-date")]
            AnyValue::Int32(v) => AnyValue::Date(v),
            AnyValue::Null => AnyValue::Null,
            dt => panic!("cannot create date from other type. dtype: {dt}"),
        }
    }
    #[cfg(feature = "dtype-datetime")]
    pub(crate) fn into_datetime(self, tu: TimeUnit, tz: &'a Option<TimeZone>) -> Self {
        match self {
            AnyValue::Int64(v) => AnyValue::Datetime(v, tu, tz),
            AnyValue::Null => AnyValue::Null,
            dt => panic!("cannot create date from other type. dtype: {dt}"),
        }
    }

    #[cfg(feature = "dtype-duration")]
    pub(crate) fn into_duration(self, tu: TimeUnit) -> Self {
        match self {
            AnyValue::Int64(v) => AnyValue::Duration(v, tu),
            AnyValue::Null => AnyValue::Null,
            dt => panic!("cannot create date from other type. dtype: {dt}"),
        }
    }

    #[cfg(feature = "dtype-time")]
    pub(crate) fn into_time(self) -> Self {
        match self {
            AnyValue::Int64(v) => AnyValue::Time(v),
            AnyValue::Null => AnyValue::Null,
            dt => panic!("cannot create date from other type. dtype: {dt}"),
        }
    }

    #[must_use]
    pub fn add(&self, rhs: &AnyValue) -> Self {
        use AnyValue::*;
        match (self, rhs) {
            (Null, _) => Null,
            (_, Null) => Null,
            (Int32(l), Int32(r)) => Int32(l + r),
            (Int64(l), Int64(r)) => Int64(l + r),
            (UInt32(l), UInt32(r)) => UInt32(l + r),
            (UInt64(l), UInt64(r)) => UInt64(l + r),
            (Float32(l), Float32(r)) => Float32(l + r),
            (Float64(l), Float64(r)) => Float64(l + r),
            _ => todo!(),
        }
    }

    #[inline]
    pub fn as_borrowed(&self) -> AnyValue<'_> {
        match self {
            #[cfg(feature = "dtype-binary")]
            AnyValue::BinaryOwned(data) => AnyValue::Binary(data),
            AnyValue::Utf8Owned(data) => AnyValue::Utf8(data),
            av => av.clone(),
        }
    }

    /// Try to coerce to an AnyValue with static lifetime.
    /// This can be done if it does not borrow any values.
    #[inline]
    pub fn into_static(self) -> PolarsResult<AnyValue<'static>> {
        use AnyValue::*;
        let av = match self {
            Null => Null,
            Int8(v) => Int8(v),
            Int16(v) => Int16(v),
            Int32(v) => Int32(v),
            Int64(v) => Int64(v),
            UInt8(v) => UInt8(v),
            UInt16(v) => UInt16(v),
            UInt32(v) => UInt32(v),
            UInt64(v) => UInt64(v),
            Boolean(v) => Boolean(v),
            Float32(v) => Float32(v),
            Float64(v) => Float64(v),
            #[cfg(feature = "dtype-date")]
            Date(v) => Date(v),
            #[cfg(feature = "dtype-time")]
            Time(v) => Time(v),
            List(v) => List(v),
            Utf8(v) => Utf8Owned(v.into()),
            Utf8Owned(v) => Utf8Owned(v),
            #[cfg(feature = "dtype-binary")]
            Binary(v) => BinaryOwned(v.to_vec()),
            #[cfg(feature = "dtype-binary")]
            BinaryOwned(v) => BinaryOwned(v),
            #[cfg(feature = "object")]
            Object(v) => ObjectOwned(OwnedObject(v.to_boxed())),
            dt => {
                return Err(PolarsError::ComputeError(
                    format!("cannot get static AnyValue from {dt}").into(),
                ))
            }
        };
        Ok(av)
    }

    /// Get a reference to the `&str` contained within [`AnyValue`].
    pub fn get_str(&self) -> Option<&str> {
        match self {
            AnyValue::Utf8(s) => Some(s),
            AnyValue::Utf8Owned(s) => Some(s),
            #[cfg(feature = "dtype-categorical")]
            AnyValue::Categorical(idx, rev, arr) => {
                let s = if arr.is_null() {
                    rev.get(*idx)
                } else {
                    unsafe { arr.deref_unchecked().value(*idx as usize) }
                };
                Some(s)
            }
            _ => None,
        }
    }
}

impl<'a> From<AnyValue<'a>> for Option<i64> {
    fn from(val: AnyValue<'a>) -> Self {
        use AnyValue::*;
        match val {
            Null => None,
            Int32(v) => Some(v as i64),
            Int64(v) => Some(v),
            UInt32(v) => Some(v as i64),
            _ => todo!(),
        }
    }
}

impl PartialEq for AnyValue<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        use AnyValue::*;
        match (self.as_borrowed(), other.as_borrowed()) {
            (UInt8(l), UInt8(r)) => l == r,
            (UInt16(l), UInt16(r)) => l == r,
            (UInt32(l), UInt32(r)) => l == r,
            (UInt64(l), UInt64(r)) => l == r,
            (Int8(l), Int8(r)) => l == r,
            (Int16(l), Int16(r)) => l == r,
            (Int32(l), Int32(r)) => l == r,
            (Int64(l), Int64(r)) => l == r,
            (Float32(l), Float32(r)) => l == r,
            (Float64(l), Float64(r)) => l == r,
            #[cfg(feature = "dtype-time")]
            (Time(l), Time(r)) => l == r,
            #[cfg(all(feature = "dtype-datetime", feature = "dtype-date"))]
            (Date(l), Date(r)) => l == r,
            #[cfg(all(feature = "dtype-datetime", feature = "dtype-date"))]
            (Datetime(l, tul, tzl), Datetime(r, tur, tzr)) => l == r && tul == tur && tzl == tzr,
            (Boolean(l), Boolean(r)) => l == r,
            (List(l), List(r)) => l == r,
            #[cfg(feature = "dtype-binary")]
            (Binary(l), Binary(r)) => l == r,
            (Utf8(l), Utf8(r)) => l == r,
            #[cfg(feature = "object")]
            (Object(_), Object(_)) => panic!("eq between object not supported"),
            // should it?
            (Null, Null) => true,
            #[cfg(feature = "dtype-categorical")]
            (Categorical(idx_l, rev_l, _), Categorical(idx_r, rev_r, _)) => match (rev_l, rev_r) {
                (RevMapping::Global(_, _, id_l), RevMapping::Global(_, _, id_r)) => {
                    id_l == id_r && idx_l == idx_r
                }
                (RevMapping::Local(arr_l), RevMapping::Local(arr_r)) => {
                    std::ptr::eq(arr_l, arr_r) && idx_l == idx_r
                }
                _ => false,
            },
            #[cfg(feature = "dtype-duration")]
            (Duration(l, tu_l), Duration(r, tu_r)) => l == r && tu_l == tu_r,
            _ => false,
        }
    }
}

impl PartialOrd for AnyValue<'_> {
    /// Only implemented for the same types and physical types!
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use AnyValue::*;
        match (self.as_borrowed(), &other.as_borrowed()) {
            (UInt8(l), UInt8(r)) => l.partial_cmp(r),
            (UInt16(l), UInt16(r)) => l.partial_cmp(r),
            (UInt32(l), UInt32(r)) => l.partial_cmp(r),
            (UInt64(l), UInt64(r)) => l.partial_cmp(r),
            (Int8(l), Int8(r)) => l.partial_cmp(r),
            (Int16(l), Int16(r)) => l.partial_cmp(r),
            (Int32(l), Int32(r)) => l.partial_cmp(r),
            (Int64(l), Int64(r)) => l.partial_cmp(r),
            (Float32(l), Float32(r)) => l.partial_cmp(r),
            (Float64(l), Float64(r)) => l.partial_cmp(r),
            (Utf8(l), Utf8(r)) => l.partial_cmp(*r),
            #[cfg(feature = "dtype-binary")]
            (Binary(l), Binary(r)) => l.partial_cmp(*r),
            _ => None,
        }
    }
}

#[cfg(test)]
mod test {
    #[cfg(feature = "dtype-categorical")]
    use super::*;

    #[test]
    #[cfg(feature = "dtype-categorical")]
    fn test_arrow_dtypes_to_polars() {
        let dtypes = [
            (
                ArrowDataType::Duration(ArrowTimeUnit::Nanosecond),
                DataType::Duration(TimeUnit::Nanoseconds),
            ),
            (
                ArrowDataType::Duration(ArrowTimeUnit::Millisecond),
                DataType::Duration(TimeUnit::Milliseconds),
            ),
            (
                ArrowDataType::Date64,
                DataType::Datetime(TimeUnit::Milliseconds, None),
            ),
            (
                ArrowDataType::Timestamp(ArrowTimeUnit::Nanosecond, None),
                DataType::Datetime(TimeUnit::Nanoseconds, None),
            ),
            (
                ArrowDataType::Timestamp(ArrowTimeUnit::Microsecond, None),
                DataType::Datetime(TimeUnit::Microseconds, None),
            ),
            (
                ArrowDataType::Timestamp(ArrowTimeUnit::Millisecond, None),
                DataType::Datetime(TimeUnit::Milliseconds, None),
            ),
            (
                ArrowDataType::Timestamp(ArrowTimeUnit::Second, None),
                DataType::Datetime(TimeUnit::Milliseconds, None),
            ),
            (
                ArrowDataType::Timestamp(ArrowTimeUnit::Second, Some("".to_string())),
                DataType::Datetime(TimeUnit::Milliseconds, Some("".to_string())),
            ),
            (ArrowDataType::LargeUtf8, DataType::Utf8),
            (ArrowDataType::Utf8, DataType::Utf8),
            #[cfg(feature = "dtype-binary")]
            (ArrowDataType::LargeBinary, DataType::Binary),
            #[cfg(feature = "dtype-binary")]
            (ArrowDataType::Binary, DataType::Binary),
            (
                ArrowDataType::Time64(ArrowTimeUnit::Nanosecond),
                DataType::Time,
            ),
            (
                ArrowDataType::Time64(ArrowTimeUnit::Millisecond),
                DataType::Time,
            ),
            (
                ArrowDataType::Time64(ArrowTimeUnit::Microsecond),
                DataType::Time,
            ),
            (ArrowDataType::Time64(ArrowTimeUnit::Second), DataType::Time),
            (
                ArrowDataType::Time32(ArrowTimeUnit::Nanosecond),
                DataType::Time,
            ),
            (
                ArrowDataType::Time32(ArrowTimeUnit::Millisecond),
                DataType::Time,
            ),
            (
                ArrowDataType::Time32(ArrowTimeUnit::Microsecond),
                DataType::Time,
            ),
            (ArrowDataType::Time32(ArrowTimeUnit::Second), DataType::Time),
            (
                ArrowDataType::List(Box::new(ArrowField::new(
                    "item",
                    ArrowDataType::Float64,
                    true,
                ))),
                DataType::List(DataType::Float64.into()),
            ),
            (
                ArrowDataType::LargeList(Box::new(ArrowField::new(
                    "item",
                    ArrowDataType::Float64,
                    true,
                ))),
                DataType::List(DataType::Float64.into()),
            ),
            (
                ArrowDataType::Dictionary(IntegerType::UInt32, ArrowDataType::Utf8.into(), false),
                DataType::Categorical(None),
            ),
            (
                ArrowDataType::Dictionary(
                    IntegerType::UInt32,
                    ArrowDataType::LargeUtf8.into(),
                    false,
                ),
                DataType::Categorical(None),
            ),
            (
                ArrowDataType::Dictionary(
                    IntegerType::UInt64,
                    ArrowDataType::LargeUtf8.into(),
                    false,
                ),
                DataType::Categorical(None),
            ),
        ];

        for (dt_a, dt_p) in dtypes {
            let dt: DataType = (&dt_a).into();

            assert_eq!(dt_p, dt);
        }
    }
}
pub trait GetAnyValue {
    /// # Safety
    ///
    /// Get an value without doing bound checks.
    unsafe fn get_unchecked(&self, index: usize) -> AnyValue;
}

impl GetAnyValue for ArrayRef {
    // Should only be called with physical types
    unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
        match self.data_type() {
            ArrowDataType::Int8 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i8>>()
                    .unwrap_unchecked_release();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Int8(v),
                }
            }
            ArrowDataType::Int16 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i16>>()
                    .unwrap_unchecked_release();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Int16(v),
                }
            }
            ArrowDataType::Int32 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i32>>()
                    .unwrap_unchecked_release();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Int32(v),
                }
            }
            ArrowDataType::Int64 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i64>>()
                    .unwrap_unchecked_release();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Int64(v),
                }
            }
            ArrowDataType::UInt8 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<u8>>()
                    .unwrap_unchecked_release();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::UInt8(v),
                }
            }
            ArrowDataType::UInt16 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<u16>>()
                    .unwrap_unchecked_release();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::UInt16(v),
                }
            }
            ArrowDataType::UInt32 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<u32>>()
                    .unwrap_unchecked_release();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::UInt32(v),
                }
            }
            ArrowDataType::UInt64 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<u64>>()
                    .unwrap_unchecked_release();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::UInt64(v),
                }
            }
            ArrowDataType::Float32 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<f32>>()
                    .unwrap_unchecked_release();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Float32(v),
                }
            }
            ArrowDataType::Float64 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<PrimitiveArray<f64>>()
                    .unwrap_unchecked_release();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Float64(v),
                }
            }
            ArrowDataType::Boolean => {
                let arr = self
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .unwrap_unchecked_release();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Boolean(v),
                }
            }
            ArrowDataType::LargeUtf8 => {
                let arr = self
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .unwrap_unchecked_release();
                match arr.get_unchecked(index) {
                    None => AnyValue::Null,
                    Some(v) => AnyValue::Utf8(v),
                }
            }
            _ => unimplemented!(),
        }
    }
}

impl<K: NumericNative> From<K> for AnyValue<'_> {
    fn from(value: K) -> Self {
        unsafe {
            match K::PRIMITIVE {
                PrimitiveType::Int8 => {
                    AnyValue::Int8(NumCast::from(value).unwrap_unchecked_release())
                }
                PrimitiveType::Int16 => {
                    AnyValue::Int16(NumCast::from(value).unwrap_unchecked_release())
                }
                PrimitiveType::Int32 => {
                    AnyValue::Int32(NumCast::from(value).unwrap_unchecked_release())
                }
                PrimitiveType::Int64 => {
                    AnyValue::Int64(NumCast::from(value).unwrap_unchecked_release())
                }
                PrimitiveType::UInt8 => {
                    AnyValue::UInt8(NumCast::from(value).unwrap_unchecked_release())
                }
                PrimitiveType::UInt16 => {
                    AnyValue::UInt16(NumCast::from(value).unwrap_unchecked_release())
                }
                PrimitiveType::UInt32 => {
                    AnyValue::UInt32(NumCast::from(value).unwrap_unchecked_release())
                }
                PrimitiveType::UInt64 => {
                    AnyValue::UInt64(NumCast::from(value).unwrap_unchecked_release())
                }
                PrimitiveType::Float32 => {
                    AnyValue::Float32(NumCast::from(value).unwrap_unchecked_release())
                }
                PrimitiveType::Float64 => {
                    AnyValue::Float64(NumCast::from(value).unwrap_unchecked_release())
                }
                // not supported by polars
                _ => unreachable!(),
            }
        }
    }
}
