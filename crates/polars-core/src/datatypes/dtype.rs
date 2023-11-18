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
    #[cfg(feature = "dtype-decimal")]
    /// Fixed point decimal type optional precision and non-negative scale.
    /// This is backed by a signed 128-bit integer which allows for up to 38 significant digits.
    Decimal(Option<usize>, Option<usize>), // precision/scale; scale being None means "infer"
    /// String data
    Utf8,
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
    /// A nested list with a fixed size in each row
    #[cfg(feature = "dtype-array")]
    Array(Box<DataType>, usize),
    /// A nested list with a variable size in each row
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

pub trait AsRefDataType {
    fn as_ref_dtype(&self) -> &DataType;
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
                (Struct(lhs), Struct(rhs)) => Vec::as_ptr(lhs) == Vec::as_ptr(rhs) || lhs == rhs,
                #[cfg(feature = "dtype-array")]
                (Array(left_inner, left_width), Array(right_inner, right_width)) => {
                    left_width == right_width && left_inner == right_inner
                },
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

    /// Get the inner data type of a nested type.
    pub fn inner_dtype(&self) -> Option<&DataType> {
        match self {
            DataType::List(inner) => Some(inner),
            #[cfg(feature = "dtype-array")]
            DataType::Array(inner, _) => Some(inner),
            _ => None,
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
            List(dt) => List(Box::new(dt.to_physical())),
            #[cfg(feature = "dtype-struct")]
            Struct(fields) => {
                let new_fields = fields
                    .iter()
                    .map(|s| Field::new(s.name(), s.data_type().to_physical()))
                    .collect();
                Struct(new_fields)
            },
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
        self.is_numeric() | matches!(self, DataType::Boolean | DataType::Utf8 | DataType::Binary)
    }

    /// Check if this [`DataType`] is a basic numeric type (excludes Decimal).
    pub fn is_numeric(&self) -> bool {
        self.is_float() || self.is_integer()
    }

    /// Check if this [`DataType`] is a Decimal type (of any scale/precision).
    pub fn is_decimal(&self) -> bool {
        match self {
            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(_, _) => true,
            _ => false,
        }
    }

    /// Check if this [`DataType`] is a basic floating point type (excludes Decimal).
    pub fn is_float(&self) -> bool {
        matches!(self, DataType::Float32 | DataType::Float64)
    }

    /// Check if this [`DataType`] is an integer.
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
        )
    }

    pub fn is_signed_integer(&self) -> bool {
        // allow because it cannot be replaced when object feature is activated
        match self {
            DataType::Int64 | DataType::Int32 => true,
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => true,
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => true,
            _ => false,
        }
    }

    pub fn is_unsigned_integer(&self) -> bool {
        match self {
            DataType::UInt64 | DataType::UInt32 => true,
            #[cfg(feature = "dtype-u8")]
            DataType::UInt8 => true,
            #[cfg(feature = "dtype-u16")]
            DataType::UInt16 => true,
            _ => false,
        }
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
            #[cfg(feature = "dtype-decimal")]
            // note: what else can we do here other than setting precision to 38?..
            Decimal(precision, scale) => ArrowDataType::Decimal(
                (*precision).unwrap_or(38),
                scale.unwrap_or(0), // and what else can we do here?
            ),
            Utf8 => ArrowDataType::LargeUtf8,
            Binary => ArrowDataType::LargeBinary,
            Date => ArrowDataType::Date32,
            Datetime(unit, tz) => ArrowDataType::Timestamp(unit.to_arrow(), tz.clone()),
            Duration(unit) => ArrowDataType::Duration(unit.to_arrow()),
            Time => ArrowDataType::Time64(ArrowTimeUnit::Nanosecond),
            #[cfg(feature = "dtype-array")]
            Array(dt, size) => ArrowDataType::FixedSizeList(
                Box::new(arrow::datatypes::Field::new("item", dt.to_arrow(), true)),
                *size,
            ),
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
            },
            Unknown => unreachable!(),
        }
    }

    pub fn is_nested_null(&self) -> bool {
        use DataType::*;
        match self {
            Null => true,
            List(field) => field.is_nested_null(),
            #[cfg(feature = "dtype-struct")]
            Struct(fields) => fields.iter().all(|fld| fld.dtype.is_nested_null()),
            _ => false,
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
            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(precision, scale) => {
                return match (precision, scale) {
                    (Some(precision), Some(scale)) => {
                        f.write_str(&format!("decimal[{precision},{scale}]"))
                    },
                    (None, Some(scale)) => f.write_str(&format!("decimal[*,{scale}]")),
                    _ => f.write_str("decimal[?]"), // shouldn't happen
                };
            },
            DataType::Utf8 => "str",
            DataType::Binary => "binary",
            DataType::Date => "date",
            DataType::Datetime(tu, tz) => {
                let s = match tz {
                    None => format!("datetime[{tu}]"),
                    Some(tz) => format!("datetime[{tu}, {tz}]"),
                };
                return f.write_str(&s);
            },
            DataType::Duration(tu) => return write!(f, "duration[{tu}]"),
            DataType::Time => "time",
            #[cfg(feature = "dtype-array")]
            DataType::Array(tp, size) => return write!(f, "array[{tp}, {size}]"),
            DataType::List(tp) => return write!(f, "list[{tp}]"),
            #[cfg(feature = "object")]
            DataType::Object(s) => s,
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => "cat",
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => return write!(f, "struct[{}]", fields.len()),
            DataType::Unknown => "unknown",
        };
        f.write_str(s)
    }
}

pub fn merge_dtypes(left: &DataType, right: &DataType) -> PolarsResult<DataType> {
    // TODO! add struct
    use DataType::*;
    Ok(match (left, right) {
        #[cfg(feature = "dtype-categorical")]
        (Categorical(Some(rev_map_l)), Categorical(Some(rev_map_r))) => {
            match (&**rev_map_l, &**rev_map_r) {
                (RevMapping::Global(_, _, idl), RevMapping::Global(_, _, idr)) if idl == idr => {
                    let mut merger = GlobalRevMapMerger::new(rev_map_l.clone());
                    merger.merge_map(rev_map_r)?;
                    Categorical(Some(merger.finish()))
                },
                (RevMapping::Local(_, idl), RevMapping::Local(_, idr)) if idl == idr => {
                    left.clone()
                },
                _ => polars_bail!(string_cache_mismatch),
            }
        },
        (List(inner_l), List(inner_r)) => {
            let merged = merge_dtypes(inner_l, inner_r)?;
            List(Box::new(merged))
        },
        #[cfg(feature = "dtype-array")]
        (Array(inner_l, width_l), Array(inner_r, width_r)) => {
            polars_ensure!(width_l == width_r, ComputeError: "widths of FixedSizeWidth Series are not equal");
            let merged = merge_dtypes(inner_l, inner_r)?;
            Array(Box::new(merged), *width_l)
        },
        (left, right) if left == right => left.clone(),
        _ => polars_bail!(ComputeError: "unable to merge datatypes"),
    })
}
