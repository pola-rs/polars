use arrow::datatypes::{Metadata, DTYPE_ENUM_VALUES};
use polars_utils::pl_str::PlSmallStr;

use super::*;
pub static EXTENSION_NAME: &str = "POLARS_EXTENSION_TYPE";

/// Characterizes the name and the [`DataType`] of a column.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    any(feature = "serde", feature = "serde-lazy"),
    derive(Serialize, Deserialize)
)]
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
        Self::from_arrow(&field.dtype, true, field.metadata.as_deref())
    }

    pub fn from_arrow_dtype(dt: &ArrowDataType) -> DataType {
        Self::from_arrow(dt, true, None)
    }

    pub fn from_arrow(dt: &ArrowDataType, bin_to_view: bool, md: Option<&Metadata>) -> DataType {
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
            #[cfg(feature = "dtype-i128")]
            ArrowDataType::Int128 => DataType::Int128,
            ArrowDataType::Boolean => DataType::Boolean,
            ArrowDataType::Float16 => DataType::Float32,
            ArrowDataType::Float32 => DataType::Float32,
            ArrowDataType::Float64 => DataType::Float64,
            #[cfg(feature = "dtype-array")]
            ArrowDataType::FixedSizeList(f, size) => DataType::Array(DataType::from_arrow_field(f).boxed(), *size),
            ArrowDataType::LargeList(f) | ArrowDataType::List(f) => DataType::List(DataType::from_arrow_field(f).boxed()),
            ArrowDataType::Date32 => DataType::Date,
            ArrowDataType::Timestamp(tu, tz) => DataType::Datetime(tu.into(), DataType::canonical_timezone(tz)),
            ArrowDataType::Duration(tu) => DataType::Duration(tu.into()),
            ArrowDataType::Date64 => DataType::Datetime(TimeUnit::Milliseconds, None),
            ArrowDataType::Time64(_) | ArrowDataType::Time32(_) => DataType::Time,
            #[cfg(feature = "dtype-categorical")]
            ArrowDataType::Dictionary(_, value_type, _) => {
                if md.map(|md| md.is_enum()).unwrap_or(false) {
                    let md = md.unwrap();
                    let encoded = md.get(DTYPE_ENUM_VALUES).unwrap();
                    let mut encoded = encoded.as_str();
                    let mut cats = MutableBinaryViewArray::<str>::new();

                    // Data is encoded as <len in ascii><sep ';'><payload>
                    // We know thus that len is only [0-9] and the first ';' doesn't belong to the
                    // payload.
                    while let Some(pos) = encoded.find(';') {
                            let (len, remainder) =  encoded.split_at(pos);
                            // Split off ';'
                            encoded = &remainder[1..];
                            let len = len.parse::<usize>().unwrap();

                            let (value, remainder) = encoded.split_at(len);
                            cats.push_value(value);
                            encoded = remainder;
                    }
                    DataType::Enum(Some(Arc::new(RevMapping::build_local(cats.into()))), Default::default())
                } else if let Some(ordering) = md.and_then(|md| md.categorical()) {
                    DataType::Categorical(None, ordering)
                } else if matches!(value_type.as_ref(), ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View) {
                    DataType::Categorical(None, Default::default())
                } else {
                    Self::from_arrow(value_type, bin_to_view, None)
                }
            },
            #[cfg(feature = "dtype-struct")]
            ArrowDataType::Struct(fields) => {
                DataType::Struct(fields.iter().map(|fld| fld.into()).collect())
            }
            #[cfg(not(feature = "dtype-struct"))]
            ArrowDataType::Struct(_) => {
                panic!("activate the 'dtype-struct' feature to handle struct data types")
            }
            ArrowDataType::Extension(ext) if ext.name.as_str() == EXTENSION_NAME => {
                #[cfg(feature = "object")]
                {
                    DataType::Object("object", None)
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

impl From<&ArrowField> for Field {
    fn from(f: &ArrowField) -> Self {
        Field::new(f.name.clone(), DataType::from_arrow_field(f))
    }
}
