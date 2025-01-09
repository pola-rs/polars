use std::collections::BTreeMap;

use arrow::datatypes::{Metadata, DTYPE_CATEGORICAL, DTYPE_ENUM_VALUES};
#[cfg(feature = "dtype-array")]
use polars_utils::format_tuple;
use polars_utils::itertools::Itertools;
#[cfg(any(feature = "serde-lazy", feature = "serde"))]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

use super::*;
#[cfg(feature = "object")]
use crate::chunked_array::object::registry::ObjectRegistry;
use crate::utils::materialize_dyn_int;

pub type TimeZone = PlSmallStr;

static MAINTAIN_PL_TYPE: &str = "maintain_type";
static PL_KEY: &str = "pl";

pub trait MetaDataExt: IntoMetadata {
    fn is_enum(&self) -> bool {
        let metadata = self.into_metadata_ref();
        metadata.get(DTYPE_ENUM_VALUES).is_some()
    }

    fn categorical(&self) -> Option<CategoricalOrdering> {
        let metadata = self.into_metadata_ref();
        match metadata.get(DTYPE_CATEGORICAL)?.as_str() {
            "lexical" => Some(CategoricalOrdering::Lexical),
            // Default is Physical
            _ => Some(CategoricalOrdering::Physical),
        }
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
pub enum UnknownKind {
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
            UnknownKind::Any => return None,
        };
        Some(dtype)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Default, IntoStaticStr)]
#[cfg_attr(
    any(feature = "serde-lazy", feature = "serde"),
    derive(Serialize, Deserialize)
)]
#[strum(serialize_all = "snake_case")]
pub enum CategoricalOrdering {
    #[default]
    Physical,
    Lexical,
}

#[derive(Clone, Debug)]
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
    Object(&'static str, Option<Arc<ObjectRegistry>>),
    Null,
    // The RevMapping has the internal state.
    // This is ignored with comparisons, hashing etc.
    #[cfg(feature = "dtype-categorical")]
    Categorical(Option<Arc<RevMapping>>, CategoricalOrdering),
    // It is an Option, so that matching Enum/Categoricals can take the same guards.
    #[cfg(feature = "dtype-categorical")]
    Enum(Option<Arc<RevMapping>>, CategoricalOrdering),
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
                // Don't include rev maps in comparisons
                // TODO: include ordering in comparison
                (Categorical(_, _ordering_l), Categorical(_, _ordering_r)) => true,
                #[cfg(feature = "dtype-categorical")]
                // None means select all Enum dtypes. This is for operation `pl.col(pl.Enum)`
                (Enum(None, _), Enum(_, _)) | (Enum(_, _), Enum(None, _)) => true,
                #[cfg(feature = "dtype-categorical")]
                (Enum(Some(cat_lhs), _), Enum(Some(cat_rhs), _)) => {
                    cat_lhs.get_categories() == cat_rhs.get_categories()
                },
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
                (Object(lhs, _), Object(rhs, _)) => lhs == rhs,
                #[cfg(feature = "dtype-struct")]
                (Struct(lhs), Struct(rhs)) => Vec::as_ptr(lhs) == Vec::as_ptr(rhs) || lhs == rhs,
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
    /// Standardize timezones to consistent values.
    pub(crate) fn canonical_timezone(tz: &Option<PlSmallStr>) -> Option<TimeZone> {
        match tz.as_deref() {
            Some("") | None => None,
            #[cfg(feature = "timezones")]
            Some("+00:00") | Some("00:00") | Some("utc") => Some(PlSmallStr::from_static("UTC")),
            Some(v) => Some(PlSmallStr::from_str(v)),
        }
    }

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
            | (D::Binary, D::Categorical(_, _) | D::Enum(_, _)) => false,

            #[cfg(feature = "object")]
            (D::Object(_, _), D::Object(_, _)) => true,
            #[cfg(feature = "object")]
            (D::Object(_, _), _) | (_, D::Object(_, _)) => false,

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
            Categorical(_, _) | Enum(_, _) => UInt32,
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

    pub fn is_object(&self) -> bool {
        #[cfg(feature = "object")]
        {
            matches!(self, DataType::Object(_, _))
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
            #[cfg(feature = "dtype-categorical")]
            Categorical(_, _) | Enum(_, _) => true,
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
            Object(_, _) => true,
            List(inner) => inner.contains_objects(),
            #[cfg(feature = "dtype-array")]
            Array(inner, _) => inner.contains_objects(),
            #[cfg(feature = "dtype-struct")]
            Struct(fields) => fields.iter().any(|field| field.dtype.contains_objects()),
            _ => false,
        }
    }

    /// Check if type is sortable
    pub fn is_ord(&self) -> bool {
        #[cfg(feature = "dtype-categorical")]
        let is_cat = matches!(self, DataType::Categorical(_, _) | DataType::Enum(_, _));
        #[cfg(not(feature = "dtype-categorical"))]
        let is_cat = false;

        let phys = self.to_physical();
        (phys.is_primitive_numeric()
            || self.is_decimal()
            || matches!(
                phys,
                DataType::Binary | DataType::String | DataType::Boolean
            ))
            && !is_cat
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

    /// Convert to an Arrow Field
    pub fn to_arrow_field(&self, name: PlSmallStr, compat_level: CompatLevel) -> ArrowField {
        let metadata = match self {
            #[cfg(feature = "dtype-categorical")]
            DataType::Enum(Some(revmap), _) => {
                let cats = revmap.get_categories();
                let mut encoded = String::with_capacity(cats.len() * 10);
                for cat in cats.values_iter() {
                    encoded.push_str(itoa::Buffer::new().format(cat.len()));
                    encoded.push(';');
                    encoded.push_str(cat);
                }
                Some(BTreeMap::from([(
                    PlSmallStr::from_static(DTYPE_ENUM_VALUES),
                    PlSmallStr::from_string(encoded),
                )]))
            },
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, ordering) => Some(BTreeMap::from([(
                PlSmallStr::from_static(DTYPE_CATEGORICAL),
                PlSmallStr::from_static(ordering.into()),
            )])),
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
            Datetime(unit, tz) => Ok(ArrowDataType::Timestamp(unit.to_arrow(), tz.clone())),
            Duration(unit) => Ok(ArrowDataType::Duration(unit.to_arrow())),
            Time => Ok(ArrowDataType::Time64(ArrowTimeUnit::Nanosecond)),
            #[cfg(feature = "dtype-array")]
            Array(dt, size) => Ok(dt
                .try_to_arrow(compat_level)?
                .to_fixed_size_list(*size, true)),
            List(dt) => Ok(ArrowDataType::LargeList(Box::new(
                dt.to_arrow_field(PlSmallStr::from_static("item"), compat_level),
            ))),
            Null => Ok(ArrowDataType::Null),
            #[cfg(feature = "object")]
            Object(_, Some(reg)) => Ok(reg.physical_dtype.clone()),
            #[cfg(feature = "object")]
            Object(_, None) => {
                // FIXME: find out why we have Objects floating around without a
                // known dtype.
                // polars_bail!(InvalidOperation: "cannot convert Object dtype without registry to Arrow")
                Ok(ArrowDataType::Unknown)
            },
            #[cfg(feature = "dtype-categorical")]
            Categorical(_, _) | Enum(_, _) => {
                let values = if compat_level.0 >= 1 {
                    ArrowDataType::Utf8View
                } else {
                    ArrowDataType::LargeUtf8
                };
                Ok(ArrowDataType::Dictionary(
                    IntegerType::UInt32,
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
                    UnknownKind::Any => ArrowDataType::Unknown,
                    UnknownKind::Float => ArrowDataType::Float64,
                    UnknownKind::Str => ArrowDataType::Utf8View,
                    UnknownKind::Int(v) => {
                        return materialize_dyn_int(*v).dtype().try_to_arrow(compat_level)
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

    // Answers if this type matches the given type of a schema.
    //
    // Allows (nested) Null types in this type to match any type in the schema,
    // but not vice versa. In such a case Ok(true) is returned, because a cast
    // is necessary. If no cast is necessary Ok(false) is returned, and an
    // error is returned if the types are incompatible.
    pub fn matches_schema_type(&self, schema_type: &DataType) -> PolarsResult<bool> {
        match (self, schema_type) {
            (DataType::List(l), DataType::List(r)) => l.matches_schema_type(r),
            #[cfg(feature = "dtype-array")]
            (DataType::Array(l, sl), DataType::Array(r, sr)) => {
                Ok(l.matches_schema_type(r)? && sl == sr)
            },
            #[cfg(feature = "dtype-struct")]
            (DataType::Struct(l), DataType::Struct(r)) => {
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
            (l, r) if l == r => Ok(false),
            (l, r) => {
                polars_bail!(SchemaMismatch: "type {:?} is incompatible with expected type {:?}", l, r)
            },
        }
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
                return write!(f, "array[{tp}, {}]", shape);
            },
            DataType::List(tp) => return write!(f, "list[{tp}]"),
            #[cfg(feature = "object")]
            DataType::Object(s, _) => s,
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, _) => "cat",
            #[cfg(feature = "dtype-categorical")]
            DataType::Enum(_, _) => "enum",
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => return write!(f, "struct[{}]", fields.len()),
            DataType::Unknown(kind) => match kind {
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

pub fn merge_dtypes(left: &DataType, right: &DataType) -> PolarsResult<DataType> {
    use DataType::*;
    Ok(match (left, right) {
        #[cfg(feature = "dtype-categorical")]
        (Categorical(Some(rev_map_l), ordering), Categorical(Some(rev_map_r), _)) => {
            match (&**rev_map_l, &**rev_map_r) {
                (RevMapping::Global(_, _, idl), RevMapping::Global(_, _, idr)) if idl == idr => {
                    let mut merger = GlobalRevMapMerger::new(rev_map_l.clone());
                    merger.merge_map(rev_map_r)?;
                    Categorical(Some(merger.finish()), *ordering)
                },
                (RevMapping::Local(_, idl), RevMapping::Local(_, idr)) if idl == idr => {
                    left.clone()
                },
                _ => polars_bail!(string_cache_mismatch),
            }
        },
        #[cfg(feature = "dtype-categorical")]
        (Enum(Some(rev_map_l), _), Enum(Some(rev_map_r), _)) => {
            match (&**rev_map_l, &**rev_map_r) {
                (RevMapping::Local(_, idl), RevMapping::Local(_, idr)) if idl == idr => {
                    left.clone()
                },
                _ => polars_bail!(ComputeError: "can not combine with different categories"),
            }
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

#[cfg(feature = "dtype-categorical")]
pub fn create_enum_dtype(categories: Utf8ViewArray) -> DataType {
    let rev_map = RevMapping::build_local(categories);
    DataType::Enum(Some(Arc::new(rev_map)), Default::default())
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
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
