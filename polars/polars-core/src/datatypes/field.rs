use super::*;

/// Characterizes the name and the [`DataType`] of a column.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    any(feature = "serde", feature = "serde-lazy"),
    derive(Serialize, Deserialize)
)]
pub struct Field {
    pub name: String,
    pub dtype: DataType,
}

impl Field {
    /// Creates a new `Field`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let f1 = Field::new("Fruit name", DataType::Utf8);
    /// let f2 = Field::new("Lawful", DataType::Boolean);
    /// let f2 = Field::new("Departure", DataType::Time);
    /// ```
    #[inline]
    pub fn new(name: &str, dtype: DataType) -> Self {
        Field {
            name: name.to_string(),
            dtype,
        }
    }

    pub fn from_owned(name: String, dtype: DataType) -> Self {
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
    pub fn name(&self) -> &String {
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
    /// f.set_name("Proton".to_owned());
    ///
    /// assert_eq!(f, Field::new("Proton", DataType::UInt32));
    /// ```
    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    /// Converts the `Field` to an `arrow::datatypes::Field`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars_core::prelude::*;
    /// let f = Field::new("Value", DataType::Int64);
    /// let af = arrow::datatypes::Field::new("Value", arrow::datatypes::DataType::Int64, true);
    ///
    /// assert_eq!(f.to_arrow(), af);
    /// ```
    pub fn to_arrow(&self) -> ArrowField {
        ArrowField::new(&self.name, self.dtype.to_arrow(), true)
    }
}

impl From<&ArrowDataType> for DataType {
    fn from(dt: &ArrowDataType) -> Self {
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
            ArrowDataType::LargeList(f) => DataType::List(Box::new(f.data_type().into())),
            ArrowDataType::List(f) => DataType::List(Box::new(f.data_type().into())),
            ArrowDataType::Date32 => DataType::Date,
            ArrowDataType::Timestamp(tu, tz) => DataType::Datetime(tu.into(), tz.clone()),
            ArrowDataType::Duration(tu) => DataType::Duration(tu.into()),
            ArrowDataType::Date64 => DataType::Datetime(TimeUnit::Milliseconds, None),
            ArrowDataType::LargeUtf8 | ArrowDataType::Utf8 => DataType::Utf8,
            #[cfg(feature = "dtype-binary")]
            ArrowDataType::LargeBinary | ArrowDataType::Binary => DataType::Binary,
            ArrowDataType::Time64(_) | ArrowDataType::Time32(_) => DataType::Time,
            #[cfg(feature = "dtype-categorical")]
            ArrowDataType::Dictionary(_, _, _) => DataType::Categorical(None),
            #[cfg(feature = "dtype-struct")]
            ArrowDataType::Struct(fields) => {
                let fields: Vec<Field> = fields.iter().map(|fld| fld.into()).collect();
                DataType::Struct(fields)
            }
            ArrowDataType::Extension(name, _, _) if name == "POLARS_EXTENSION_TYPE" => {
                #[cfg(feature = "object")]
                {
                    DataType::Object("extension")
                }
                #[cfg(not(feature = "object"))]
                {
                    panic!("activate the 'object' feature to be able to load POLARS_EXTENSION_TYPE")
                }
            }
            dt => panic!("Arrow datatype {dt:?} not supported by Polars. You probably need to activate that data-type feature."),
        }
    }
}

impl From<&ArrowField> for Field {
    fn from(f: &ArrowField) -> Self {
        Field::new(&f.name, f.data_type().into())
    }
}
