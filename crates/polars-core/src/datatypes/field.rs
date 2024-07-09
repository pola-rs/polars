use smartstring::alias::String as SmartString;

use super::*;

/// Characterizes the name and the [`DataType`] of a column.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    any(feature = "serde", feature = "serde-lazy"),
    derive(Serialize, Deserialize)
)]
pub struct Field {
    pub name: SmartString,
    pub dtype: DataType,
}

pub type FieldRef = Arc<Field>;

impl Field {
    /// Creates a new `Field`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let f1 = Field::new("Fruit name", DataType::String);
    /// let f2 = Field::new("Lawful", DataType::Boolean);
    /// let f2 = Field::new("Departure", DataType::Time);
    /// ```
    #[inline]
    pub fn new(name: &str, dtype: DataType) -> Self {
        Field {
            name: name.into(),
            dtype,
        }
    }

    pub fn from_owned(name: SmartString, dtype: DataType) -> Self {
        Field { name, dtype }
    }

    /// Returns a reference to the `Field` name.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let f = Field::new("Year", DataType::Int32);
    ///
    /// assert_eq!(f.name(), "Year");
    /// ```
    #[inline]
    pub fn name(&self) -> &SmartString {
        &self.name
    }

    /// Returns a reference to the `Field` datatype.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let f = Field::new("Birthday", DataType::Date);
    ///
    /// assert_eq!(f.data_type(), &DataType::Date);
    /// ```
    #[inline]
    pub fn data_type(&self) -> &DataType {
        &self.dtype
    }

    /// Sets the `Field` datatype.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let mut f = Field::new("Temperature", DataType::Int32);
    /// f.coerce(DataType::Float32);
    ///
    /// assert_eq!(f, Field::new("Temperature", DataType::Float32));
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
    /// let mut f = Field::new("Atomic number", DataType::UInt32);
    /// f.set_name("Proton".into());
    ///
    /// assert_eq!(f, Field::new("Proton", DataType::UInt32));
    /// ```
    pub fn set_name(&mut self, name: SmartString) {
        self.name = name;
    }

    /// Converts the `Field` to an `arrow::datatypes::Field`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let f = Field::new("Value", DataType::Int64);
    /// let af = arrow::datatypes::Field::new("Value", arrow::datatypes::ArrowDataType::Int64, true);
    ///
    /// assert_eq!(f.to_arrow(CompatLevel::newest()), af);
    /// ```
    pub fn to_arrow(&self, compat_level: CompatLevel) -> ArrowField {
        self.dtype.to_arrow_field(self.name.as_str(), compat_level)
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

    pub fn from_arrow(dt: &ArrowDataType, bin_to_view: bool) -> DataType {
        match dt {
            ArrowDataType::Null => DataType::Null,
            ArrowDataType::UInt8 => DataType::UInt8,
            ArrowDataType::UInt16 => DataType::UInt16,
            ArrowDataType::UInt32 => DataType::UInt32,
            ArrowDataType::UInt64 => DataType::UInt64,
            ArrowDataType::Int8 => DataType::Int8,
            ArrowDataType::Int16 => DataType::Int16,
            ArrowDataType::Int32 => DataType::Int32,
            ArrowDataType::Int64 => DataType::Int64,
            ArrowDataType::Boolean => DataType::Boolean,
            ArrowDataType::Float32 => DataType::Float32,
            ArrowDataType::Float64 => DataType::Float64,
            #[cfg(feature = "dtype-array")]
            ArrowDataType::FixedSizeList(f, size) => DataType::Array(DataType::from_arrow(f.data_type(), bin_to_view).boxed(), *size),
            ArrowDataType::LargeList(f) | ArrowDataType::List(f) => DataType::List(DataType::from_arrow(f.data_type(), bin_to_view).boxed()),
            ArrowDataType::Date32 => DataType::Date,
            ArrowDataType::Timestamp(tu, tz) => DataType::Datetime(tu.into(), DataType::canonical_timezone(tz)),
            ArrowDataType::Duration(tu) => DataType::Duration(tu.into()),
            ArrowDataType::Date64 => DataType::Datetime(TimeUnit::Milliseconds, None),
            ArrowDataType::Time64(_) | ArrowDataType::Time32(_) => DataType::Time,
            #[cfg(feature = "dtype-categorical")]
            ArrowDataType::Dictionary(_, _, _) => DataType::Categorical(None,Default::default()),
            #[cfg(feature = "dtype-struct")]
            ArrowDataType::Struct(fields) => {
                DataType::Struct(fields.iter().map(|fld| fld.into()).collect())
            }
            #[cfg(not(feature = "dtype-struct"))]
            ArrowDataType::Struct(_) => {
                panic!("activate the 'dtype-struct' feature to handle struct data types")
            }
            ArrowDataType::Extension(name, _, _) if name == "POLARS_EXTENSION_TYPE" => {
                #[cfg(feature = "object")]
                {
                    DataType::Object("extension", None)
                }
                #[cfg(not(feature = "object"))]
                {
                    panic!("activate the 'object' feature to be able to load POLARS_EXTENSION_TYPE")
                }
            }
            #[cfg(feature = "dtype-decimal")]
            ArrowDataType::Decimal(precision, scale) => DataType::Decimal(Some(*precision), Some(*scale)),
            ArrowDataType::Utf8View |ArrowDataType::LargeUtf8 | ArrowDataType::Utf8 => DataType::String,
            ArrowDataType::BinaryView => DataType::Binary,
            ArrowDataType::LargeBinary | ArrowDataType::Binary => {
                if bin_to_view {
                    DataType::Binary
                } else {

                    DataType::BinaryOffset
                }
            },
            ArrowDataType::FixedSizeBinary(_) => DataType::Binary,
            dt => panic!("Arrow datatype {dt:?} not supported by Polars. You probably need to activate that data-type feature."),
        }
    }
}

impl From<&ArrowDataType> for DataType {
    fn from(dt: &ArrowDataType) -> Self {
        Self::from_arrow(dt, true)
    }
}

impl From<&ArrowField> for Field {
    fn from(f: &ArrowField) -> Self {
        Field::new(&f.name, f.data_type().into())
    }
}
