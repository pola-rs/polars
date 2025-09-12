use std::collections::BTreeMap;

use arrow::datatypes::{
    DTYPE_CATEGORICAL_NEW, DTYPE_ENUM_VALUES_LEGACY, DTYPE_ENUM_VALUES_NEW, Metadata,
};
#[cfg(feature = "dtype-array")]
use polars_utils::format_tuple;
use polars_utils::itertools::Itertools;
#[cfg(any(feature = "serde-lazy", feature = "serde"))]
use serde::{Deserialize, Serialize};
pub use temporal::time_zone::TimeZone;

use super::*;
#[cfg(feature = "object")]
use crate::chunked_array::object::registry::get_object_physical_type;
use crate::utils::materialize_dyn_int;

static MAINTAIN_PL_TYPE: &str = "maintain_type";
static PL_KEY: &str = "pl";

pub trait MetaDataExt: IntoMetadata {
    fn pl_enum_metadata(&self) -> Option<&str> {
        let md = self.into_metadata_ref();
        let values = md
            .get(DTYPE_ENUM_VALUES_NEW)
            .or_else(|| md.get(DTYPE_ENUM_VALUES_LEGACY));
        Some(values?.as_str())
    }

    fn pl_categorical_metadata(&self) -> Option<&str> {
        // We ignore DTYPE_CATEGORICAL_LEGACY here, as we already map all
        // string-typed arrow dictionaries to the global Categories, and the
        // legacy metadata format only specifies the now-removed physical
        // ordering parameter.
        Some(
            self.into_metadata_ref()
                .get(DTYPE_CATEGORICAL_NEW)?
                .as_str(),
        )
    }

    fn maintain_type(&self) -> bool {
        let metadata = self.into_metadata_ref();
        metadata.get(PL_KEY).map(|s| s.as_str()) == Some(MAINTAIN_PL_TYPE)
    }
}

impl MetaDataExt for Metadata {}
pub trait IntoMetadata {
    #[allow(clippy::wrong_self_convention)]
    fn into_metadata_ref(&self) -> &Metadata;
}

impl IntoMetadata for Metadata {
    fn into_metadata_ref(&self) -> &Metadata {
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(
    any(feature = "serde", feature = "serde-lazy"),
    derive(Serialize, Deserialize)
)]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum UnknownKind {
    Ufunc,
    // Hold the value to determine the concrete size.
    Int(i128),
    Float,
    // Can be Categorical or String
    Str,
    #[default]
    Any,
}

impl UnknownKind {
    pub fn materialize(&self) -> Option<DataType> {
        let dtype = match self {
            UnknownKind::Int(v) => materialize_dyn_int(*v).dtype(),
            UnknownKind::Float => DataType::Float64,
            UnknownKind::Str => DataType::String,
            UnknownKind::Any | UnknownKind::Ufunc => return None,
        };
        Some(dtype)
    }
}

#[derive(Clone)]
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
    Int128,
    Float32,
    Float64,
    /// Fixed point decimal type optional precision and non-negative scale.
    /// This is backed by a signed 128-bit integer which allows for up to 38 significant digits.
    /// Meaning max precision is 38.
    #[cfg(feature = "dtype-decimal")]
    Decimal(Option<usize>, Option<usize>), // precision/scale; scale being None means "infer"
    /// String data
    String,
    Binary,
    BinaryOffset,
    /// A 32-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in days (32 bits).
    Date,
    /// A 64-bit date representing the elapsed time since UNIX epoch (1970-01-01)
    /// in the given timeunit (64 bits).
    Datetime(TimeUnit, Option<TimeZone>),
    /// 64-bit integer representing difference between times in milliseconds or nanoseconds
    Duration(TimeUnit),
    /// A 64-bit time representing the elapsed time since midnight in nanoseconds
    Time,
    /// A nested list with a fixed size in each row
    #[cfg(feature = "dtype-array")]
    Array(Box<DataType>, usize),
    /// A nested list with a variable size in each row
    List(Box<DataType>),
    /// A generic type that can be used in a `Series`
    /// &'static str can be used to determine/set inner type
    #[cfg(feature = "object")]
    Object(&'static str),
    Null,
    #[cfg(feature = "dtype-categorical")]
    Categorical(Arc<Categories>, Arc<CategoricalMapping>),
    // It is an Option, so that matching Enum/Categoricals can take the same guards.
    #[cfg(feature = "dtype-categorical")]
    Enum(Arc<FrozenCategories>, Arc<CategoricalMapping>),
    #[cfg(feature = "dtype-struct")]
    Struct(Vec<Field>),
    // some logical types we cannot know statically, e.g. Datetime
    Unknown(UnknownKind),
}

impl Default for DataType {
    fn default() -> Self {
        DataType::Unknown(UnknownKind::Any)
    }
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
                #[cfg(feature = "dtype-categorical")]
                (Categorical(cats_l, _), Categorical(cats_r, _)) => Arc::ptr_eq(cats_l, cats_r),
                #[cfg(feature = "dtype-categorical")]
                (Enum(fcats_l, _), Enum(fcats_r, _)) => Arc::ptr_eq(fcats_l, fcats_r),
                (Datetime(tu_l, tz_l), Datetime(tu_r, tz_r)) => tu_l == tu_r && tz_l == tz_r,
                (List(left_inner), List(right_inner)) => left_inner == right_inner,
                #[cfg(feature = "dtype-duration")]
                (Duration(tu_l), Duration(tu_r)) => tu_l == tu_r,
                #[cfg(feature = "dtype-decimal")]
                (Decimal(l_prec, l_scale), Decimal(r_prec, r_scale)) => {
                    let is_prec_eq = l_prec.is_none() || r_prec.is_none() || l_prec == r_prec;
                    let is_scale_eq = l_scale.is_none() || r_scale.is_none() || l_scale == r_scale;

                    is_prec_eq && is_scale_eq
                },
                #[cfg(feature = "object")]
                (Object(lhs), Object(rhs)) => lhs == rhs,
                #[cfg(feature = "dtype-struct")]
                (Struct(lhs), Struct(rhs)) => {
                    std::ptr::eq(Vec::as_ptr(lhs), Vec::as_ptr(rhs)) || lhs == rhs
                },
                #[cfg(feature = "dtype-array")]
                (Array(left_inner, left_width), Array(right_inner, right_width)) => {
                    left_width == right_width && left_inner == right_inner
                },
                (Unknown(l), Unknown(r)) => match (l, r) {
                    (UnknownKind::Int(_), UnknownKind::Int(_)) => true,
                    _ => l == r,
                },
                _ => std::mem::discriminant(self) == std::mem::discriminant(other),
            }
        }
    }
}

impl Eq for DataType {}

impl DataType {
    pub const IDX_DTYPE: Self = {
        #[cfg(not(feature = "bigidx"))]
        {
            DataType::UInt32
        }
        #[cfg(feature = "bigidx")]
        {
            DataType::UInt64
        }
    };

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

    /// Check if the whole dtype is known.
    pub fn is_known(&self) -> bool {
        match self {
            DataType::List(inner) => inner.is_known(),
            #[cfg(feature = "dtype-array")]
            DataType::Array(inner, _) => inner.is_known(),
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => fields.iter().all(|fld| fld.dtype.is_known()),
            DataType::Unknown(_) => false,
            _ => true,
        }
    }

    /// Materialize this datatype if it is unknown. All other datatypes
    /// are left unchanged.
    pub fn materialize_unknown(self, allow_unknown: bool) -> PolarsResult<DataType> {
        match self {
            DataType::Unknown(u) => match u.materialize() {
                Some(known) => Ok(known),
                None => {
                    if allow_unknown {
                        Ok(DataType::Unknown(u))
                    } else {
                        polars_bail!(SchemaMismatch: "failed to materialize unknown type")
                    }
                },
            },
            DataType::List(inner) => Ok(DataType::List(Box::new(
                inner.materialize_unknown(allow_unknown)?,
            ))),
            #[cfg(feature = "dtype-array")]
            DataType::Array(inner, size) => Ok(DataType::Array(
                Box::new(inner.materialize_unknown(allow_unknown)?),
                size,
            )),
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => Ok(DataType::Struct(
                fields
                    .into_iter()
                    .map(|f| {
                        PolarsResult::Ok(Field::new(
                            f.name,
                            f.dtype.materialize_unknown(allow_unknown)?,
                        ))
                    })
                    .try_collect_vec()?,
            )),
            _ => Ok(self),
        }
    }

    #[cfg(feature = "dtype-array")]
    /// Get the full shape of a multidimensional array.
    pub fn get_shape(&self) -> Option<Vec<usize>> {
        fn get_shape_impl(dt: &DataType, shape: &mut Vec<usize>) {
            if let DataType::Array(inner, size) = dt {
                shape.push(*size);
                get_shape_impl(inner, shape);
            }
        }

        if let DataType::Array(inner, size) = self {
            let mut shape = vec![*size];
            get_shape_impl(inner, &mut shape);
            Some(shape)
        } else {
            None
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

    /// Get the inner data type of a nested type.
    pub fn into_inner_dtype(self) -> Option<DataType> {
        match self {
            DataType::List(inner) => Some(*inner),
            #[cfg(feature = "dtype-array")]
            DataType::Array(inner, _) => Some(*inner),
            _ => None,
        }
    }

    /// Get the inner data type of a nested type.
    pub fn try_into_inner_dtype(self) -> PolarsResult<DataType> {
        match self {
            DataType::List(inner) => Ok(*inner),
            #[cfg(feature = "dtype-array")]
            DataType::Array(inner, _) => Ok(*inner),
            dt => polars_bail!(InvalidOperation: "cannot get inner datatype of `{dt}`"),
        }
    }

    /// Get the absolute inner data type of a nested type.
    pub fn leaf_dtype(&self) -> &DataType {
        let mut prev = self;
        while let Some(dtype) = prev.inner_dtype() {
            prev = dtype
        }
        prev
    }

    #[cfg(feature = "dtype-array")]
    /// Get the inner data type of a multidimensional array.
    pub fn array_leaf_dtype(&self) -> Option<&DataType> {
        let mut prev = self;
        match prev {
            DataType::Array(_, _) => {
                while let DataType::Array(inner, _) = &prev {
                    prev = inner;
                }
                Some(prev)
            },
            _ => None,
        }
    }

    /// Cast the leaf types of Lists/Arrays and keep the nesting.
    pub fn cast_leaf(&self, to: DataType) -> DataType {
        use DataType::*;
        match self {
            List(inner) => List(Box::new(inner.cast_leaf(to))),
            #[cfg(feature = "dtype-array")]
            Array(inner, size) => Array(Box::new(inner.cast_leaf(to)), *size),
            _ => to,
        }
    }

    /// Return whether the cast to `to` makes sense.
    ///
    /// If it `None`, we are not sure.
    pub fn can_cast_to(&self, to: &DataType) -> Option<bool> {
        if self == to {
            return Some(true);
        }
        if self.is_primitive_numeric() && to.is_primitive_numeric() {
            return Some(true);
        }

        if self.is_null() {
            return Some(true);
        }

        use DataType as D;
        Some(match (self, to) {
            #[cfg(feature = "dtype-categorical")]
            (D::Categorical(_, _) | D::Enum(_, _), D::Binary)
            | (D::Binary, D::Categorical(_, _) | D::Enum(_, _)) => false, // TODO @ cat-rework: why can we not cast to Binary?

            #[cfg(feature = "object")]
            (D::Object(_), D::Object(_)) => true,
            #[cfg(feature = "object")]
            (D::Object(_), _) | (_, D::Object(_)) => false,

            (D::Boolean, dt) | (dt, D::Boolean) => match dt {
                dt if dt.is_primitive_numeric() => true,
                #[cfg(feature = "dtype-decimal")]
                D::Decimal(_, _) => true,
                D::String | D::Binary => true,
                _ => false,
            },

            (D::List(from), D::List(to)) => from.can_cast_to(to)?,
            #[cfg(feature = "dtype-array")]
            (D::Array(from, l_width), D::Array(to, r_width)) => {
                l_width == r_width && from.can_cast_to(to)?
            },
            #[cfg(feature = "dtype-struct")]
            (D::Struct(l_fields), D::Struct(r_fields)) => {
                if l_fields.is_empty() {
                    return Some(true);
                }

                if l_fields.len() != r_fields.len() {
                    return Some(false);
                }

                for (l, r) in l_fields.iter().zip(r_fields) {
                    if !l.dtype().can_cast_to(r.dtype())? {
                        return Some(false);
                    }
                }

                true
            },

            // @NOTE: we are being conversative
            _ => return None,
        })
    }

    pub fn implode(self) -> DataType {
        DataType::List(Box::new(self))
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
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => Int128,
            #[cfg(feature = "dtype-categorical")]
            Categorical(cats, _) => cats.physical().dtype(),
            #[cfg(feature = "dtype-categorical")]
            Enum(fcats, _) => fcats.physical().dtype(),
            #[cfg(feature = "dtype-array")]
            Array(dt, width) => Array(Box::new(dt.to_physical()), *width),
            List(dt) => List(Box::new(dt.to_physical())),
            #[cfg(feature = "dtype-struct")]
            Struct(fields) => {
                let new_fields = fields
                    .iter()
                    .map(|s| Field::new(s.name().clone(), s.dtype().to_physical()))
                    .collect();
                Struct(new_fields)
            },
            _ => self.clone(),
        }
    }

    pub fn is_supported_list_arithmetic_input(&self) -> bool {
        self.is_primitive_numeric() || self.is_bool() || self.is_null()
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
    /// it is not a nested or logical type.
    pub fn is_primitive(&self) -> bool {
        self.is_primitive_numeric()
            | matches!(
                self,
                DataType::Boolean | DataType::String | DataType::Binary
            )
    }

    /// Check if this [`DataType`] is a primitive numeric type (excludes Decimal).
    pub fn is_primitive_numeric(&self) -> bool {
        self.is_float() || self.is_integer()
    }

    /// Check if this [`DataType`] is a boolean.
    pub fn is_bool(&self) -> bool {
        matches!(self, DataType::Boolean)
    }

    /// Check if this [`DataType`] is a list.
    pub fn is_list(&self) -> bool {
        matches!(self, DataType::List(_))
    }

    /// Check if this [`DataType`] is an array.
    pub fn is_array(&self) -> bool {
        #[cfg(feature = "dtype-array")]
        {
            matches!(self, DataType::Array(_, _))
        }
        #[cfg(not(feature = "dtype-array"))]
        {
            false
        }
    }

    pub fn is_nested(&self) -> bool {
        self.is_list() || self.is_struct() || self.is_array()
    }

    /// Check if this [`DataType`] is a struct
    pub fn is_struct(&self) -> bool {
        #[cfg(feature = "dtype-struct")]
        {
            matches!(self, DataType::Struct(_))
        }
        #[cfg(not(feature = "dtype-struct"))]
        {
            false
        }
    }

    pub fn is_binary(&self) -> bool {
        matches!(self, DataType::Binary)
    }

    pub fn is_date(&self) -> bool {
        matches!(self, DataType::Date)
    }
    pub fn is_datetime(&self) -> bool {
        matches!(self, DataType::Datetime(..))
    }

    pub fn is_duration(&self) -> bool {
        matches!(self, DataType::Duration(..))
    }

    pub fn is_object(&self) -> bool {
        #[cfg(feature = "object")]
        {
            matches!(self, DataType::Object(_))
        }
        #[cfg(not(feature = "object"))]
        {
            false
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, DataType::Null)
    }

    pub fn contains_views(&self) -> bool {
        use DataType::*;
        match self {
            Binary | String => true,
            List(inner) => inner.contains_views(),
            #[cfg(feature = "dtype-array")]
            Array(inner, _) => inner.contains_views(),
            #[cfg(feature = "dtype-struct")]
            Struct(fields) => fields.iter().any(|field| field.dtype.contains_views()),
            _ => false,
        }
    }

    pub fn contains_categoricals(&self) -> bool {
        use DataType::*;
        match self {
            #[cfg(feature = "dtype-categorical")]
            Categorical(_, _) | Enum(_, _) => true,
            List(inner) => inner.contains_categoricals(),
            #[cfg(feature = "dtype-array")]
            Array(inner, _) => inner.contains_categoricals(),
            #[cfg(feature = "dtype-struct")]
            Struct(fields) => fields
                .iter()
                .any(|field| field.dtype.contains_categoricals()),
            _ => false,
        }
    }

    pub fn contains_objects(&self) -> bool {
        use DataType::*;
        match self {
            #[cfg(feature = "object")]
            Object(_) => true,
            List(inner) => inner.contains_objects(),
            #[cfg(feature = "dtype-array")]
            Array(inner, _) => inner.contains_objects(),
            #[cfg(feature = "dtype-struct")]
            Struct(fields) => fields.iter().any(|field| field.dtype.contains_objects()),
            _ => false,
        }
    }

    pub fn contains_list_recursive(&self) -> bool {
        use DataType as D;
        match self {
            D::List(_) => true,
            #[cfg(feature = "dtype-array")]
            D::Array(inner, _) => inner.contains_list_recursive(),
            #[cfg(feature = "dtype-struct")]
            D::Struct(fields) => fields
                .iter()
                .any(|field| field.dtype.contains_list_recursive()),
            _ => false,
        }
    }

    pub fn contains_unknown(&self) -> bool {
        use DataType as D;
        match self {
            D::Unknown(_) => true,
            D::List(inner) => inner.contains_unknown(),
            #[cfg(feature = "dtype-array")]
            D::Array(inner, _) => inner.contains_unknown(),
            #[cfg(feature = "dtype-struct")]
            D::Struct(fields) => fields.iter().any(|field| field.dtype.contains_unknown()),
            _ => false,
        }
    }

    /// Check if type is sortable
    pub fn is_ord(&self) -> bool {
        let phys = self.to_physical();
        phys.is_primitive_numeric()
            || self.is_decimal()
            || matches!(
                phys,
                DataType::Binary | DataType::String | DataType::Boolean
            )
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
    /// Note, this also includes `Unknown(UnknownKind::Float)`.
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            DataType::Float32 | DataType::Float64 | DataType::Unknown(UnknownKind::Float)
        )
    }

    /// Check if this [`DataType`] is an integer. Note, this also includes `Unknown(UnknownKind::Int)`.
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::Int128
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Unknown(UnknownKind::Int(_))
        )
    }

    pub fn is_signed_integer(&self) -> bool {
        // allow because it cannot be replaced when object feature is activated
        matches!(
            self,
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 | DataType::Int128
        )
    }

    pub fn is_unsigned_integer(&self) -> bool {
        matches!(
            self,
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64,
        )
    }

    pub fn is_string(&self) -> bool {
        matches!(self, DataType::String | DataType::Unknown(UnknownKind::Str))
    }

    pub fn is_categorical(&self) -> bool {
        #[cfg(feature = "dtype-categorical")]
        {
            matches!(self, DataType::Categorical(_, _))
        }
        #[cfg(not(feature = "dtype-categorical"))]
        {
            false
        }
    }

    pub fn is_enum(&self) -> bool {
        #[cfg(feature = "dtype-categorical")]
        {
            matches!(self, DataType::Enum(_, _))
        }
        #[cfg(not(feature = "dtype-categorical"))]
        {
            false
        }
    }

    /// Convert to an Arrow Field.
    pub fn to_arrow_field(&self, name: PlSmallStr, compat_level: CompatLevel) -> ArrowField {
        let metadata = match self {
            #[cfg(feature = "dtype-categorical")]
            DataType::Enum(fcats, _map) => {
                let cats = fcats.categories();
                let strings_size: usize = cats
                    .values_iter()
                    .map(|s| (s.len() + 1).ilog10() as usize + 1 + s.len())
                    .sum();
                let mut encoded = String::with_capacity(strings_size);
                for cat in cats.values_iter() {
                    encoded.push_str(itoa::Buffer::new().format(cat.len()));
                    encoded.push(';');
                    encoded.push_str(cat);
                }
                Some(BTreeMap::from([(
                    PlSmallStr::from_static(DTYPE_ENUM_VALUES_NEW),
                    PlSmallStr::from_string(encoded),
                )]))
            },
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(cats, _) => {
                let mut encoded = String::new();
                encoded.push_str(itoa::Buffer::new().format(cats.name().len()));
                encoded.push(';');
                encoded.push_str(cats.name());
                encoded.push_str(itoa::Buffer::new().format(cats.namespace().len()));
                encoded.push(';');
                encoded.push_str(cats.namespace());
                encoded.push_str(cats.physical().as_str());
                encoded.push(';');

                Some(BTreeMap::from([(
                    PlSmallStr::from_static(DTYPE_CATEGORICAL_NEW),
                    PlSmallStr::from_string(encoded),
                )]))
            },
            DataType::BinaryOffset => Some(BTreeMap::from([(
                PlSmallStr::from_static(PL_KEY),
                PlSmallStr::from_static(MAINTAIN_PL_TYPE),
            )])),
            _ => None,
        };

        let field = ArrowField::new(name, self.to_arrow(compat_level), true);

        if let Some(metadata) = metadata {
            field.with_metadata(metadata)
        } else {
            field
        }
    }

    /// Try to get the maximum value for this datatype.
    pub fn max(&self) -> PolarsResult<Scalar> {
        use DataType::*;
        let v = match self {
            Int8 => Scalar::from(i8::MAX),
            Int16 => Scalar::from(i16::MAX),
            Int32 => Scalar::from(i32::MAX),
            Int64 => Scalar::from(i64::MAX),
            Int128 => Scalar::from(i128::MAX),
            UInt8 => Scalar::from(u8::MAX),
            UInt16 => Scalar::from(u16::MAX),
            UInt32 => Scalar::from(u32::MAX),
            UInt64 => Scalar::from(u64::MAX),
            Float32 => Scalar::from(f32::INFINITY),
            Float64 => Scalar::from(f64::INFINITY),
            #[cfg(feature = "dtype-time")]
            Time => Scalar::new(Time, AnyValue::Time(NS_IN_DAY - 1)),
            dt => polars_bail!(ComputeError: "cannot determine upper bound for dtype `{}`", dt),
        };
        Ok(v)
    }

    /// Try to get the minimum value for this datatype.
    pub fn min(&self) -> PolarsResult<Scalar> {
        use DataType::*;
        let v = match self {
            Int8 => Scalar::from(i8::MIN),
            Int16 => Scalar::from(i16::MIN),
            Int32 => Scalar::from(i32::MIN),
            Int64 => Scalar::from(i64::MIN),
            Int128 => Scalar::from(i128::MIN),
            UInt8 => Scalar::from(u8::MIN),
            UInt16 => Scalar::from(u16::MIN),
            UInt32 => Scalar::from(u32::MIN),
            UInt64 => Scalar::from(u64::MIN),
            Float32 => Scalar::from(f32::NEG_INFINITY),
            Float64 => Scalar::from(f64::NEG_INFINITY),
            #[cfg(feature = "dtype-time")]
            Time => Scalar::new(Time, AnyValue::Time(0)),
            dt => polars_bail!(ComputeError: "cannot determine lower bound for dtype `{}`", dt),
        };
        Ok(v)
    }

    /// Convert to an Arrow data type.
    #[inline]
    pub fn to_arrow(&self, compat_level: CompatLevel) -> ArrowDataType {
        self.try_to_arrow(compat_level).unwrap()
    }

    #[inline]
    pub fn try_to_arrow(&self, compat_level: CompatLevel) -> PolarsResult<ArrowDataType> {
        use DataType::*;
        match self {
            Boolean => Ok(ArrowDataType::Boolean),
            UInt8 => Ok(ArrowDataType::UInt8),
            UInt16 => Ok(ArrowDataType::UInt16),
            UInt32 => Ok(ArrowDataType::UInt32),
            UInt64 => Ok(ArrowDataType::UInt64),
            Int8 => Ok(ArrowDataType::Int8),
            Int16 => Ok(ArrowDataType::Int16),
            Int32 => Ok(ArrowDataType::Int32),
            Int64 => Ok(ArrowDataType::Int64),
            Int128 => Ok(ArrowDataType::Int128),
            Float32 => Ok(ArrowDataType::Float32),
            Float64 => Ok(ArrowDataType::Float64),
            #[cfg(feature = "dtype-decimal")]
            Decimal(precision, scale) => {
                let precision = (*precision).unwrap_or(38);
                polars_ensure!(precision <= 38 && precision > 0, InvalidOperation: "decimal precision should be <= 38 & >= 1");

                Ok(ArrowDataType::Decimal(
                    precision,
                    scale.unwrap_or(0), // and what else can we do here?
                ))
            },
            String => {
                let dt = if compat_level.0 >= 1 {
                    ArrowDataType::Utf8View
                } else {
                    ArrowDataType::LargeUtf8
                };
                Ok(dt)
            },
            Binary => {
                let dt = if compat_level.0 >= 1 {
                    ArrowDataType::BinaryView
                } else {
                    ArrowDataType::LargeBinary
                };
                Ok(dt)
            },
            Date => Ok(ArrowDataType::Date32),
            Datetime(unit, tz) => Ok(ArrowDataType::Timestamp(
                unit.to_arrow(),
                tz.as_deref().cloned(),
            )),
            Duration(unit) => Ok(ArrowDataType::Duration(unit.to_arrow())),
            Time => Ok(ArrowDataType::Time64(ArrowTimeUnit::Nanosecond)),
            #[cfg(feature = "dtype-array")]
            Array(dt, size) => Ok(dt
                .try_to_arrow(compat_level)?
                .to_fixed_size_list(*size, true)),
            List(dt) => Ok(ArrowDataType::LargeList(Box::new(
                dt.to_arrow_field(LIST_VALUES_NAME, compat_level),
            ))),
            Null => Ok(ArrowDataType::Null),
            #[cfg(feature = "object")]
            Object(_) => Ok(get_object_physical_type()),
            #[cfg(feature = "dtype-categorical")]
            Categorical(_, _) | Enum(_, _) => {
                let arrow_phys = match self.cat_physical().unwrap() {
                    CategoricalPhysical::U8 => IntegerType::UInt8,
                    CategoricalPhysical::U16 => IntegerType::UInt16,
                    CategoricalPhysical::U32 => IntegerType::UInt32,
                };

                let values = if compat_level.0 >= 1 {
                    ArrowDataType::Utf8View
                } else {
                    ArrowDataType::LargeUtf8
                };

                Ok(ArrowDataType::Dictionary(
                    arrow_phys,
                    Box::new(values),
                    false,
                ))
            },
            #[cfg(feature = "dtype-struct")]
            Struct(fields) => {
                let fields = fields
                    .iter()
                    .map(|fld| fld.to_arrow(compat_level))
                    .collect();
                Ok(ArrowDataType::Struct(fields))
            },
            BinaryOffset => Ok(ArrowDataType::LargeBinary),
            Unknown(kind) => {
                let dt = match kind {
                    UnknownKind::Any | UnknownKind::Ufunc => ArrowDataType::Unknown,
                    UnknownKind::Float => ArrowDataType::Float64,
                    UnknownKind::Str => ArrowDataType::Utf8View,
                    UnknownKind::Int(v) => {
                        return materialize_dyn_int(*v).dtype().try_to_arrow(compat_level);
                    },
                };
                Ok(dt)
            },
        }
    }

    pub fn is_nested_null(&self) -> bool {
        use DataType::*;
        match self {
            Null => true,
            List(field) => field.is_nested_null(),
            #[cfg(feature = "dtype-array")]
            Array(field, _) => field.is_nested_null(),
            #[cfg(feature = "dtype-struct")]
            Struct(fields) => fields.iter().all(|fld| fld.dtype.is_nested_null()),
            _ => false,
        }
    }

    /// Answers if this type matches the given type of a schema.
    ///
    /// Allows (nested) Null types in this type to match any type in the schema,
    /// but not vice versa. In such a case Ok(true) is returned, because a cast
    /// is necessary. If no cast is necessary Ok(false) is returned, and an
    /// error is returned if the types are incompatible.
    pub fn matches_schema_type(&self, schema_type: &DataType) -> PolarsResult<bool> {
        match (self, schema_type) {
            (DataType::List(l), DataType::List(r)) => l.matches_schema_type(r),
            #[cfg(feature = "dtype-array")]
            (DataType::Array(l, sl), DataType::Array(r, sr)) => {
                Ok(l.matches_schema_type(r)? && sl == sr)
            },
            #[cfg(feature = "dtype-struct")]
            (DataType::Struct(l), DataType::Struct(r)) => {
                if l.len() != r.len() {
                    polars_bail!(SchemaMismatch: "structs have different number of fields: {} vs {}", l.len(), r.len());
                }
                let mut must_cast = false;
                for (l, r) in l.iter().zip(r.iter()) {
                    must_cast |= l.dtype.matches_schema_type(&r.dtype)?;
                }
                Ok(must_cast)
            },
            (DataType::Null, DataType::Null) => Ok(false),
            #[cfg(feature = "dtype-decimal")]
            (DataType::Decimal(_, s1), DataType::Decimal(_, s2)) => Ok(s1 != s2),
            // We don't allow the other way around, only if our current type is
            // null and the schema isn't we allow it.
            (DataType::Null, _) => Ok(true),
            #[cfg(feature = "dtype-categorical")]
            (DataType::Categorical(l, _), DataType::Categorical(r, _)) => {
                ensure_same_categories(l, r)?;
                Ok(false)
            },
            #[cfg(feature = "dtype-categorical")]
            (DataType::Enum(l, _), DataType::Enum(r, _)) => {
                ensure_same_frozen_categories(l, r)?;
                Ok(false)
            },

            (l, r) if l == r => Ok(false),
            (l, r) => {
                polars_bail!(SchemaMismatch: "type {:?} is incompatible with expected type {:?}", l, r)
            },
        }
    }

    #[inline]
    pub fn is_unknown(&self) -> bool {
        matches!(self, DataType::Unknown(_))
    }

    pub fn nesting_level(&self) -> usize {
        let mut level = 0;
        let mut slf = self;
        while let Some(inner_dtype) = slf.inner_dtype() {
            level += 1;
            slf = inner_dtype;
        }
        level
    }

    /// If this dtype is a Categorical or Enum, returns the physical backing type.
    #[cfg(feature = "dtype-categorical")]
    pub fn cat_physical(&self) -> PolarsResult<CategoricalPhysical> {
        match self {
            DataType::Categorical(cats, _) => Ok(cats.physical()),
            DataType::Enum(fcats, _) => Ok(fcats.physical()),
            _ => {
                polars_bail!(SchemaMismatch: "invalid dtype: expected an Enum or Categorical type, received '{:?}'", self)
            },
        }
    }

    /// If this dtype is a Categorical or Enum, returns the underlying mapping.
    #[cfg(feature = "dtype-categorical")]
    pub fn cat_mapping(&self) -> PolarsResult<&Arc<CategoricalMapping>> {
        match self {
            DataType::Categorical(_, mapping) | DataType::Enum(_, mapping) => Ok(mapping),
            _ => {
                polars_bail!(SchemaMismatch: "invalid dtype: expected an Enum or Categorical type, received '{:?}'", self)
            },
        }
    }

    #[cfg(feature = "dtype-categorical")]
    pub fn from_categories(cats: Arc<Categories>) -> Self {
        let mapping = cats.mapping();
        Self::Categorical(cats, mapping)
    }

    #[cfg(feature = "dtype-categorical")]
    pub fn from_frozen_categories(fcats: Arc<FrozenCategories>) -> Self {
        let mapping = fcats.mapping().clone();
        Self::Enum(fcats, mapping)
    }

    pub fn is_numeric(&self) -> bool {
        self.is_integer() || self.is_float() || self.is_decimal()
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
            DataType::Int128 => "i128",
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
            DataType::String => "str",
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
            DataType::Array(_, _) => {
                let tp = self.array_leaf_dtype().unwrap();

                let dims = self.get_shape().unwrap();
                let shape = if dims.len() == 1 {
                    format!("{}", dims[0])
                } else {
                    format_tuple!(dims)
                };
                return write!(f, "array[{tp}, {shape}]");
            },
            DataType::List(tp) => return write!(f, "list[{tp}]"),
            #[cfg(feature = "object")]
            DataType::Object(s) => s,
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) => "cat",
            #[cfg(feature = "dtype-categorical")]
            DataType::Enum(_, _) => "enum",
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => return write!(f, "struct[{}]", fields.len()),
            DataType::Unknown(kind) => match kind {
                UnknownKind::Ufunc => "unknown ufunc",
                UnknownKind::Any => "unknown",
                UnknownKind::Int(_) => "dyn int",
                UnknownKind::Float => "dyn float",
                UnknownKind::Str => "dyn str",
            },
            DataType::BinaryOffset => "binary[offset]",
        };
        f.write_str(s)
    }
}

impl std::fmt::Debug for DataType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use DataType::*;
        match self {
            Boolean => write!(f, "Boolean"),
            UInt8 => write!(f, "UInt8"),
            UInt16 => write!(f, "UInt16"),
            UInt32 => write!(f, "UInt32"),
            UInt64 => write!(f, "UInt64"),
            Int8 => write!(f, "Int8"),
            Int16 => write!(f, "Int16"),
            Int32 => write!(f, "Int32"),
            Int64 => write!(f, "Int64"),
            Int128 => write!(f, "Int128"),
            Float32 => write!(f, "Float32"),
            Float64 => write!(f, "Float64"),
            String => write!(f, "String"),
            Binary => write!(f, "Binary"),
            BinaryOffset => write!(f, "BinaryOffset"),
            Date => write!(f, "Date"),
            Time => write!(f, "Time"),
            Duration(unit) => write!(f, "Duration('{unit}')"),
            Datetime(unit, opt_tz) => {
                if let Some(tz) = opt_tz {
                    write!(f, "Datetime('{unit}', '{tz}')")
                } else {
                    write!(f, "Datetime('{unit}')")
                }
            },
            #[cfg(feature = "dtype-decimal")]
            Decimal(opt_p, opt_s) => match (opt_p, opt_s) {
                (None, None) => write!(f, "Decimal(None, None)"),
                (None, Some(s)) => write!(f, "Decimal(None, {s})"),
                (Some(p), None) => write!(f, "Decimal({p}, None)"),
                (Some(p), Some(s)) => write!(f, "Decimal({p}, {s})"),
            },
            #[cfg(feature = "dtype-array")]
            Array(inner, size) => write!(f, "Array({inner:?}, {size})"),
            List(inner) => write!(f, "List({inner:?})"),
            #[cfg(feature = "dtype-struct")]
            Struct(fields) => {
                let mut first = true;
                write!(f, "Struct({{")?;
                for field in fields {
                    if !first {
                        write!(f, ", ")?;
                    }
                    write!(f, "'{}': {:?}", field.name(), field.dtype())?;
                    first = false;
                }
                write!(f, "}})")
            },
            #[cfg(feature = "dtype-categorical")]
            Categorical(cats, _) => {
                if cats.is_global() {
                    write!(f, "Categorical")
                } else if cats.namespace().is_empty() && cats.physical() == CategoricalPhysical::U32
                {
                    write!(f, "Categorical('{}')", cats.name())
                } else {
                    write!(
                        f,
                        "Categorical('{}', '{}', {:?})",
                        cats.name(),
                        cats.namespace(),
                        cats.physical()
                    )
                }
            },
            #[cfg(feature = "dtype-categorical")]
            Enum(_, _) => write!(f, "Enum([...])"),
            #[cfg(feature = "object")]
            Object(_) => write!(f, "Object"),
            Null => write!(f, "Null"),
            Unknown(kind) => write!(f, "Unknown({kind:?})"),
        }
    }
}

pub fn merge_dtypes(left: &DataType, right: &DataType) -> PolarsResult<DataType> {
    use DataType::*;
    Ok(match (left, right) {
        #[cfg(feature = "dtype-categorical")]
        (Categorical(cats_l, map), Categorical(cats_r, _)) => {
            ensure_same_categories(cats_l, cats_r)?;
            Categorical(cats_l.clone(), map.clone())
        },
        #[cfg(feature = "dtype-categorical")]
        (Enum(fcats_l, map), Enum(fcats_r, _)) => {
            ensure_same_frozen_categories(fcats_l, fcats_r)?;
            Enum(fcats_l.clone(), map.clone())
        },
        (List(inner_l), List(inner_r)) => {
            let merged = merge_dtypes(inner_l, inner_r)?;
            List(Box::new(merged))
        },
        #[cfg(feature = "dtype-struct")]
        (Struct(inner_l), Struct(inner_r)) => {
            polars_ensure!(inner_l.len() == inner_r.len(), ComputeError: "cannot combine structs with differing amounts of fields ({} != {})", inner_l.len(), inner_r.len());
            let fields = inner_l.iter().zip(inner_r.iter()).map(|(l, r)| {
                polars_ensure!(l.name() == r.name(), ComputeError: "cannot combine structs with different fields ({} != {})", l.name(), r.name());
                let merged = merge_dtypes(l.dtype(), r.dtype())?;
                Ok(Field::new(l.name().clone(), merged))
            }).collect::<PolarsResult<Vec<_>>>()?;
            Struct(fields)
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

fn collect_nested_types(
    dtype: &DataType,
    result: &mut PlHashSet<DataType>,
    include_compound_types: bool,
) {
    match dtype {
        DataType::List(inner) => {
            if include_compound_types {
                result.insert(dtype.clone());
            }
            collect_nested_types(inner, result, include_compound_types);
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(inner, _) => {
            if include_compound_types {
                result.insert(dtype.clone());
            }
            collect_nested_types(inner, result, include_compound_types);
        },
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(fields) => {
            if include_compound_types {
                result.insert(dtype.clone());
            }
            for field in fields {
                collect_nested_types(field.dtype(), result, include_compound_types);
            }
        },
        _ => {
            result.insert(dtype.clone());
        },
    }
}

pub fn unpack_dtypes(dtype: &DataType, include_compound_types: bool) -> PlHashSet<DataType> {
    let mut result = PlHashSet::new();
    collect_nested_types(dtype, &mut result, include_compound_types);
    result
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct CompatLevel(pub(crate) u16);

impl CompatLevel {
    pub const fn newest() -> CompatLevel {
        CompatLevel(1)
    }

    pub const fn oldest() -> CompatLevel {
        CompatLevel(0)
    }

    // The following methods are only used internally

    #[doc(hidden)]
    pub fn with_level(level: u16) -> PolarsResult<CompatLevel> {
        if level > CompatLevel::newest().0 {
            polars_bail!(InvalidOperation: "invalid compat level");
        }
        Ok(CompatLevel(level))
    }

    #[doc(hidden)]
    pub fn get_level(&self) -> u16 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "dtype-array")]
    #[test]
    fn test_unpack_primitive_dtypes() {
        let inner_type = DataType::Float64;
        let array_type = DataType::Array(Box::new(inner_type), 10);
        let list_type = DataType::List(Box::new(array_type));

        let result = unpack_dtypes(&list_type, false);

        let mut expected = PlHashSet::new();
        expected.insert(DataType::Float64);

        assert_eq!(result, expected)
    }

    #[cfg(feature = "dtype-array")]
    #[test]
    fn test_unpack_compound_dtypes() {
        let inner_type = DataType::Float64;
        let array_type = DataType::Array(Box::new(inner_type), 10);
        let list_type = DataType::List(Box::new(array_type.clone()));

        let result = unpack_dtypes(&list_type, true);

        let mut expected = PlHashSet::new();
        expected.insert(list_type);
        expected.insert(array_type);
        expected.insert(DataType::Float64);

        assert_eq!(result, expected)
    }
}
