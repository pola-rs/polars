use super::*;

pub type TimeZone = String;

#[derive(Clone, Debug, Default)]
pub enum DataType {
    Boolean,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
    #[cfg(feature = "dtype-i128")]
    /// Fixed point decimal type with precision and scale.
    /// This is backed by 128 bits which allows for 38 significant digits.
    Decimal128(Option<(usize, usize)>),
    /// String data
    Utf8,
    #[cfg(feature = "dtype-binary")]
    Binary,
    /// A 32-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in days (32 bits).
    Date,
    /// A 64-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in the given timeunit (64 bits).
    Datetime(TimeUnit, Option<TimeZone>),
    // 64-bit integer representing difference between times in milliseconds or nanoseconds
    Duration(TimeUnit),
    /// A 64-bit time representing the elapsed time since midnight in nanoseconds
    Time,
    List(Box<DataType>),
    #[cfg(feature = "object")]
    /// A generic type that can be used in a `Series`
    /// &'static str can be used to determine/set inner type
    Object(&'static str),
    Null,
    #[cfg(feature = "dtype-categorical")]
    // The RevMapping has the internal state.
    // This is ignored with casts, comparisons, hashing etc.
    Categorical(Option<Arc<RevMapping>>),
    #[cfg(feature = "dtype-struct")]
    Struct(Vec<Field>),
    // some logical types we cannot know statically, e.g. Datetime
    #[default]
    Unknown,
}

impl Hash for DataType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state)
    }
}

impl PartialEq for DataType {
    fn eq(&self, other: &Self) -> bool {
        use DataType::*;
        {
            match (self, other) {
                // Don't include rev maps in comparisons
                #[cfg(feature = "dtype-categorical")]
                (Categorical(_), Categorical(_)) => true,
                (Datetime(tu_l, tz_l), Datetime(tu_r, tz_r)) => tu_l == tu_r && tz_l == tz_r,
                (List(left_inner), List(right_inner)) => left_inner == right_inner,
                #[cfg(feature = "dtype-duration")]
                (Duration(tu_l), Duration(tu_r)) => tu_l == tu_r,
                #[cfg(feature = "object")]
                (Object(lhs), Object(rhs)) => lhs == rhs,
                #[cfg(feature = "dtype-struct")]
                (Struct(lhs), Struct(rhs)) => lhs == rhs,
                _ => std::mem::discriminant(self) == std::mem::discriminant(other),
            }
        }
    }
}

impl Eq for DataType {}

impl DataType {
    pub fn value_within_range(&self, other: AnyValue) -> bool {
        use DataType::*;
        match self {
            UInt8 => other.extract::<u8>().is_some(),
            #[cfg(feature = "dtype-u16")]
            UInt16 => other.extract::<u16>().is_some(),
            UInt32 => other.extract::<u32>().is_some(),
            UInt64 => other.extract::<u64>().is_some(),
            #[cfg(feature = "dtype-i8")]
            Int8 => other.extract::<i8>().is_some(),
            #[cfg(feature = "dtype-i16")]
            Int16 => other.extract::<i16>().is_some(),
            Int32 => other.extract::<i32>().is_some(),
            Int64 => other.extract::<i64>().is_some(),
            _ => false,
        }
    }

    pub fn inner_dtype(&self) -> Option<&DataType> {
        if let DataType::List(inner) = self {
            Some(inner)
        } else {
            None
        }
    }

    /// Convert to the physical data type
    #[must_use]
    pub fn to_physical(&self) -> DataType {
        use DataType::*;
        match self {
            Date => Int32,
            Datetime(_, _) => Int64,
            Duration(_) => Int64,
            Time => Int64,
            #[cfg(feature = "dtype-categorical")]
            Categorical(_) => UInt32,
            _ => self.clone(),
        }
    }

    /// Check if this [`DataType`] is a logical type
    pub fn is_logical(&self) -> bool {
        self != &self.to_physical()
    }

    /// Check if this [`DataType`] is a temporal type
    pub fn is_temporal(&self) -> bool {
        use DataType::*;
        matches!(self, Date | Datetime(_, _) | Duration(_) | Time)
    }

    /// Check if datatype is a primitive type. By that we mean that
    /// it is not a container type.
    pub fn is_primitive(&self) -> bool {
        #[cfg(feature = "dtype-binary")]
        {
            self.is_numeric()
                | matches!(self, DataType::Boolean | DataType::Utf8 | DataType::Binary)
        }

        #[cfg(not(feature = "dtype-binary"))]
        {
            self.is_numeric() | matches!(self, DataType::Boolean | DataType::Utf8)
        }
    }

    /// Check if this [`DataType`] is a numeric type
    pub fn is_numeric(&self) -> bool {
        // allow because it cannot be replaced when object feature is activated
        #[allow(clippy::match_like_matches_macro)]
        match self {
            DataType::Utf8
            | DataType::List(_)
            | DataType::Date
            | DataType::Datetime(_, _)
            | DataType::Duration(_)
            | DataType::Boolean
            | DataType::Unknown
            | DataType::Null => false,
            #[cfg(feature = "dtype-binary")]
            DataType::Binary => false,
            #[cfg(feature = "object")]
            DataType::Object(_) => false,
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => false,
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(_) => false,
            _ => true,
        }
    }

    pub fn is_float(&self) -> bool {
        matches!(self, DataType::Float32 | DataType::Float64)
    }

    pub fn is_integer(&self) -> bool {
        self.is_numeric() && !matches!(self, DataType::Float32 | DataType::Float64)
    }

    pub fn is_signed(&self) -> bool {
        // allow because it cannot be replaced when object feature is activated
        #[allow(clippy::match_like_matches_macro)]
        match self {
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => true,
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => true,
            DataType::Int32 | DataType::Int64 => true,
            _ => false,
        }
    }
    pub fn is_unsigned(&self) -> bool {
        self.is_numeric() && !self.is_signed()
    }

    /// Convert to an Arrow data type.
    #[inline]
    pub fn to_arrow(&self) -> ArrowDataType {
        use DataType::*;
        match self {
            Boolean => ArrowDataType::Boolean,
            UInt8 => ArrowDataType::UInt8,
            UInt16 => ArrowDataType::UInt16,
            UInt32 => ArrowDataType::UInt32,
            UInt64 => ArrowDataType::UInt64,
            Int8 => ArrowDataType::Int8,
            Int16 => ArrowDataType::Int16,
            Int32 => ArrowDataType::Int32,
            Int64 => ArrowDataType::Int64,
            Float32 => ArrowDataType::Float32,
            Float64 => ArrowDataType::Float64,
            #[cfg(feature = "dtype-i128")]
            Decimal128(_) => todo!(),
            Utf8 => ArrowDataType::LargeUtf8,
            #[cfg(feature = "dtype-binary")]
            Binary => ArrowDataType::LargeBinary,
            Date => ArrowDataType::Date32,
            Datetime(unit, tz) => ArrowDataType::Timestamp(unit.to_arrow(), tz.clone()),
            Duration(unit) => ArrowDataType::Duration(unit.to_arrow()),
            Time => ArrowDataType::Time64(ArrowTimeUnit::Nanosecond),
            List(dt) => ArrowDataType::LargeList(Box::new(arrow::datatypes::Field::new(
                "item",
                dt.to_arrow(),
                true,
            ))),
            Null => ArrowDataType::Null,
            #[cfg(feature = "object")]
            Object(_) => panic!("cannot convert object to arrow"),
            #[cfg(feature = "dtype-categorical")]
            Categorical(_) => ArrowDataType::Dictionary(
                IntegerType::UInt32,
                Box::new(ArrowDataType::LargeUtf8),
                false,
            ),
            #[cfg(feature = "dtype-struct")]
            Struct(fields) => {
                let fields = fields.iter().map(|fld| fld.to_arrow()).collect();
                ArrowDataType::Struct(fields)
            }
            Unknown => unreachable!(),
        }
    }
}

impl PartialEq<ArrowDataType> for DataType {
    fn eq(&self, other: &ArrowDataType) -> bool {
        let dt: DataType = other.into();
        self == &dt
    }
}

impl Display for DataType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            DataType::Null => "null",
            DataType::Boolean => "bool",
            DataType::UInt8 => "u8",
            DataType::UInt16 => "u16",
            DataType::UInt32 => "u32",
            DataType::UInt64 => "u64",
            DataType::Int8 => "i8",
            DataType::Int16 => "i16",
            DataType::Int32 => "i32",
            DataType::Int64 => "i64",
            DataType::Float32 => "f32",
            DataType::Float64 => "f64",
            #[cfg(feature = "dtype-i128")]
            DataType::Decimal128(_) => "i128",
            DataType::Utf8 => "str",
            #[cfg(feature = "dtype-binary")]
            DataType::Binary => "binary",
            DataType::Date => "date",
            DataType::Datetime(tu, tz) => {
                let s = match tz {
                    None => format!("datetime[{tu}]"),
                    Some(tz) => format!("datetime[{tu}, {tz}]"),
                };
                return f.write_str(&s);
            }
            DataType::Duration(tu) => return write!(f, "duration[{tu}]"),
            DataType::Time => "time",
            DataType::List(tp) => return write!(f, "list[{tp}]"),
            #[cfg(feature = "object")]
            DataType::Object(s) => s,
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => "cat",
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => return write!(f, "struct[{}]", fields.len()),
            DataType::Unknown => unreachable!(),
        };
        f.write_str(s)
    }
}

pub fn merge_dtypes(left: &DataType, right: &DataType) -> PolarsResult<DataType> {
    // TODO! add struct
    use DataType::*;
    match (left, right) {
        #[cfg(feature = "dtype-categorical")]
        (Categorical(Some(rev_map_l)), Categorical(Some(rev_map_r))) => {
            let rev_map = merge_categorical_map(rev_map_l, rev_map_r)?;
            Ok(DataType::Categorical(Some(rev_map)))
        }
        (List(inner_l), List(inner_r)) => {
            let merged = merge_dtypes(inner_l, inner_r)?;
            Ok(DataType::List(Box::new(merged)))
        }
        (left, right) if left == right => Ok(left.clone()),
        _ => Err(PolarsError::ComputeError(
            "Coult not merge datatypes".into(),
        )),
    }
}
