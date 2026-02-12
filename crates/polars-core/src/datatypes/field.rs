use arrow::datatypes::{IntervalUnit, Metadata};
use polars_dtype::categorical::CategoricalPhysical;
use polars_error::feature_gated;
use polars_utils::pl_str::PlSmallStr;

use super::*;
use crate::config::check_allow_importing_interval_as_struct;
pub static POLARS_OBJECT_EXTENSION_NAME: &str = "_POLARS_PYTHON_OBJECT";

/// Characterizes the name and the [`DataType`] of a column.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    any(feature = "serde", feature = "serde-lazy"),
    derive(Serialize, Deserialize)
)]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct Field {
    pub name: PlSmallStr,
    pub dtype: DataType,
}

impl From<Field> for (PlSmallStr, DataType) {
    fn from(value: Field) -> Self {
        (value.name, value.dtype)
    }
}

pub type FieldRef = Arc<Field>;

impl Field {
    /// Creates a new `Field`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let f1 = Field::new("Fruit name".into(), DataType::String);
    /// let f2 = Field::new("Lawful".into(), DataType::Boolean);
    /// let f2 = Field::new("Departure".into(), DataType::Time);
    /// ```
    #[inline]
    pub fn new(name: PlSmallStr, dtype: DataType) -> Self {
        Field { name, dtype }
    }

    /// Returns a reference to the `Field` name.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let f = Field::new("Year".into(), DataType::Int32);
    ///
    /// assert_eq!(f.name(), "Year");
    /// ```
    #[inline]
    pub fn name(&self) -> &PlSmallStr {
        &self.name
    }

    /// Returns a reference to the `Field` datatype.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let f = Field::new("Birthday".into(), DataType::Date);
    ///
    /// assert_eq!(f.dtype(), &DataType::Date);
    /// ```
    #[inline]
    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }

    /// Sets the `Field` datatype.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let mut f = Field::new("Temperature".into(), DataType::Int32);
    /// f.coerce(DataType::Float32);
    ///
    /// assert_eq!(f, Field::new("Temperature".into(), DataType::Float32));
    /// ```
    pub fn coerce(&mut self, dtype: DataType) {
        self.dtype = dtype;
    }

    /// Sets the `Field` name.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let mut f = Field::new("Atomic number".into(), DataType::UInt32);
    /// f.set_name("Proton".into());
    ///
    /// assert_eq!(f, Field::new("Proton".into(), DataType::UInt32));
    /// ```
    pub fn set_name(&mut self, name: PlSmallStr) {
        self.name = name;
    }

    /// Returns this `Field`, renamed.
    pub fn with_name(mut self, name: PlSmallStr) -> Self {
        self.name = name;
        self
    }

    // Returns this `Field`, with a different datatype.
    pub fn with_dtype(mut self, dtype: DataType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Converts the `Field` to an `arrow::datatypes::Field`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let f = Field::new("Value".into(), DataType::Int64);
    /// let af = arrow::datatypes::Field::new("Value".into(), arrow::datatypes::ArrowDataType::Int64, true);
    ///
    /// assert_eq!(f.to_arrow(CompatLevel::newest()), af);
    /// ```
    pub fn to_arrow(&self, compat_level: CompatLevel) -> ArrowField {
        self.dtype.to_arrow_field(self.name.clone(), compat_level)
    }

    pub fn to_physical(&self) -> Field {
        Self {
            name: self.name.clone(),
            dtype: self.dtype().to_physical(),
        }
    }
}

impl AsRef<DataType> for Field {
    fn as_ref(&self) -> &DataType {
        &self.dtype
    }
}

impl AsRef<DataType> for DataType {
    fn as_ref(&self) -> &DataType {
        self
    }
}

impl DataType {
    pub fn boxed(self) -> Box<DataType> {
        Box::new(self)
    }

    pub fn from_arrow_field(field: &ArrowField) -> DataType {
        Self::from_arrow(&field.dtype, field.metadata.as_deref())
    }

    pub fn from_arrow_dtype(dt: &ArrowDataType) -> DataType {
        Self::from_arrow(dt, None)
    }

    pub fn from_arrow(dt: &ArrowDataType, md: Option<&Metadata>) -> DataType {
        match dt {
            ArrowDataType::Null => DataType::Null,
            ArrowDataType::UInt8 => DataType::UInt8,
            ArrowDataType::UInt16 => DataType::UInt16,
            ArrowDataType::UInt32 => DataType::UInt32,
            ArrowDataType::UInt64 => DataType::UInt64,
            #[cfg(feature = "dtype-u128")]
            ArrowDataType::UInt128 => DataType::UInt128,
            ArrowDataType::Int8 => DataType::Int8,
            ArrowDataType::Int16 => DataType::Int16,
            ArrowDataType::Int32 => DataType::Int32,
            ArrowDataType::Int64 => DataType::Int64,
            #[cfg(feature = "dtype-i128")]
            ArrowDataType::Int128 => DataType::Int128,
            ArrowDataType::Boolean => DataType::Boolean,
            #[cfg(feature = "dtype-f16")]
            ArrowDataType::Float16 => DataType::Float16,
            ArrowDataType::Float32 => DataType::Float32,
            ArrowDataType::Float64 => DataType::Float64,
            #[cfg(feature = "dtype-array")]
            ArrowDataType::FixedSizeList(f, size) => {
                DataType::Array(DataType::from_arrow_field(f).boxed(), *size)
            },
            ArrowDataType::LargeList(f) | ArrowDataType::List(f) => {
                DataType::List(DataType::from_arrow_field(f).boxed())
            },
            ArrowDataType::Date32 => DataType::Date,
            ArrowDataType::Timestamp(tu, tz) => {
                DataType::Datetime(tu.into(), TimeZone::opt_try_new(tz.clone()).unwrap())
            },
            ArrowDataType::Duration(tu) => DataType::Duration(tu.into()),
            ArrowDataType::Date64 => DataType::Datetime(TimeUnit::Milliseconds, None),
            ArrowDataType::Time64(_) | ArrowDataType::Time32(_) => DataType::Time,

            #[cfg(feature = "dtype-categorical")]
            ArrowDataType::Dictionary(_, value_type, _) => {
                // The metadata encoding here must match DataType::to_arrow_field.
                if let Some(mut enum_md) = md.and_then(|md| md.pl_enum_metadata()) {
                    let cats = move || {
                        if enum_md.is_empty() {
                            return None;
                        }

                        let len;
                        (len, enum_md) = enum_md.split_once(';').unwrap();
                        let len = len.parse::<usize>().unwrap();
                        let cat;
                        (cat, enum_md) = enum_md.split_at(len);
                        Some(cat)
                    };

                    let fcats = FrozenCategories::new(std::iter::from_fn(cats)).unwrap();
                    DataType::from_frozen_categories(fcats)
                } else if let Some(mut cat_md) = md.and_then(|md| md.pl_categorical_metadata()) {
                    let name_len;
                    (name_len, cat_md) = cat_md.split_once(';').unwrap();
                    let name_len = name_len.parse::<usize>().unwrap();
                    let name;
                    (name, cat_md) = cat_md.split_at(name_len);

                    let namespace_len;
                    (namespace_len, cat_md) = cat_md.split_once(';').unwrap();
                    let namespace_len = namespace_len.parse::<usize>().unwrap();
                    let namespace;
                    (namespace, cat_md) = cat_md.split_at(namespace_len);

                    let (physical, _rest) = cat_md.split_once(';').unwrap();

                    let physical: CategoricalPhysical = physical.parse().ok().unwrap();
                    let cats = Categories::new(
                        PlSmallStr::from_str(name),
                        PlSmallStr::from_str(namespace),
                        physical,
                    );
                    DataType::from_categories(cats)
                } else if matches!(
                    value_type.as_ref(),
                    ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View
                ) {
                    DataType::from_categories(Categories::global())
                } else {
                    Self::from_arrow(value_type, None)
                }
            },

            #[cfg(feature = "dtype-struct")]
            ArrowDataType::Struct(fields) => {
                DataType::Struct(fields.iter().map(|fld| fld.into()).collect())
            },
            #[cfg(not(feature = "dtype-struct"))]
            ArrowDataType::Struct(_) => {
                panic!("activate the 'dtype-struct' feature to handle struct data types")
            },
            ArrowDataType::Extension(ext) if ext.name.as_str() == POLARS_OBJECT_EXTENSION_NAME => {
                #[cfg(feature = "object")]
                {
                    DataType::Object("object")
                }
                #[cfg(not(feature = "object"))]
                {
                    panic!("activate the 'object' feature to be able to load POLARS_EXTENSION_TYPE")
                }
            },
            #[cfg(feature = "dtype-extension")]
            ArrowDataType::Extension(ext) => {
                use crate::prelude::extension::get_extension_type_or_storage;
                let storage = DataType::from_arrow(&ext.inner, md);
                match get_extension_type_or_storage(&ext.name, &storage, ext.metadata.as_deref()) {
                    Some(typ) => DataType::Extension(typ, Box::new(storage)),
                    None => storage,
                }
            },
            #[cfg(feature = "dtype-decimal")]
            ArrowDataType::Decimal(precision, scale) => DataType::Decimal(*precision, *scale),
            ArrowDataType::Utf8View | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8 => {
                DataType::String
            },
            ArrowDataType::BinaryView => DataType::Binary,
            ArrowDataType::LargeBinary if md.is_some() => {
                let md = md.unwrap();
                if md.maintain_type() {
                    DataType::BinaryOffset
                } else {
                    DataType::Binary
                }
            },
            ArrowDataType::LargeBinary | ArrowDataType::Binary => DataType::Binary,
            ArrowDataType::FixedSizeBinary(_) => DataType::Binary,
            ArrowDataType::Map(inner, _is_sorted) => {
                DataType::List(Self::from_arrow_field(inner).boxed())
            },
            ArrowDataType::Interval(IntervalUnit::MonthDayNano) => {
                check_allow_importing_interval_as_struct("month_day_nano_interval").unwrap();
                feature_gated!("dtype-struct", DataType::_month_days_ns_struct_type())
            },
            ArrowDataType::Interval(IntervalUnit::MonthDayMillis) => {
                check_allow_importing_interval_as_struct("month_day_millisecond_interval").unwrap();
                feature_gated!("dtype-struct", DataType::_month_days_ns_struct_type())
            },
            dt => panic!(
                "Arrow datatype {dt:?} not supported by Polars. \
                You probably need to activate that data-type feature."
            ),
        }
    }
}

impl From<&ArrowField> for Field {
    fn from(f: &ArrowField) -> Self {
        Field::new(f.name.clone(), DataType::from_arrow_field(f))
    }
}
