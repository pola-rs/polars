use std::ops::Deref;

use polars_error::{PolarsResult, polars_err};

use super::{DataType, Field};
use crate::frame::validation::find_duplicate_name;
use crate::schema::{Schema, SchemaExt};

/// The fields of a `Struct` dtype, with unique names.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(
    any(feature = "serde", feature = "serde-lazy"),
    derive(serde::Serialize)
)]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "dsl-schema", schemars(transparent))]
#[cfg_attr(any(feature = "serde", feature = "serde-lazy"), serde(transparent))]
pub struct StructFields(Vec<Field>);

impl StructFields {
    pub fn try_new(fields: Vec<Field>) -> PolarsResult<Self> {
        if let Some(name) = find_duplicate_name(&fields, |f| f.name.as_str()) {
            return Err(polars_err!(duplicate_field = name));
        }
        Ok(Self(fields))
    }

    pub fn try_from_iter(fields: impl IntoIterator<Item = Field>) -> PolarsResult<Self> {
        Self::try_new(fields.into_iter().collect())
    }

    pub fn as_slice(&self) -> &[Field] {
        &self.0
    }

    pub fn into_vec(self) -> Vec<Field> {
        self.0
    }

    /// The same names with the dtypes given by `f`.
    pub fn map_fields(&self, mut f: impl FnMut(&Field) -> DataType) -> Self {
        Self(
            self.0
                .iter()
                .map(|fld| Field::new(fld.name.clone(), f(fld)))
                .collect(),
        )
    }

    /// The fields for which `f` returns a dtype, with their names unchanged.
    pub fn filter_map_fields(&self, mut f: impl FnMut(&Field) -> Option<DataType>) -> Self {
        Self(
            self.0
                .iter()
                .filter_map(|fld| Some(Field::new(fld.name.clone(), f(fld)?)))
                .collect(),
        )
    }

    /// The same names with the dtypes given by `f`.
    pub fn try_map_fields(
        &self,
        mut f: impl FnMut(&Field) -> PolarsResult<DataType>,
    ) -> PolarsResult<Self> {
        let fields = self
            .0
            .iter()
            .map(|fld| Ok(Field::new(fld.name.clone(), f(fld)?)))
            .collect::<PolarsResult<_>>()?;
        Ok(Self(fields))
    }

    /// Mutable access to the field dtypes; names cannot change, so they stay unique.
    pub fn dtypes_mut(&mut self) -> impl Iterator<Item = &mut DataType> {
        self.0.iter_mut().map(|f| &mut f.dtype)
    }
}

impl Deref for StructFields {
    type Target = [Field];

    fn deref(&self) -> &[Field] {
        &self.0
    }
}

impl From<Field> for StructFields {
    fn from(field: Field) -> Self {
        Self(vec![field])
    }
}

impl From<&Schema> for StructFields {
    fn from(schema: &Schema) -> Self {
        Self(schema.iter_fields().collect())
    }
}

impl IntoIterator for StructFields {
    type Item = Field;
    type IntoIter = std::vec::IntoIter<Field>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a StructFields {
    type Item = &'a Field;
    type IntoIter = std::slice::Iter<'a, Field>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

#[cfg(any(feature = "serde", feature = "serde-lazy"))]
impl<'de> serde::Deserialize<'de> for StructFields {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let fields = Vec::<Field>::deserialize(deserializer)?;
        Self::try_new(fields).map_err(serde::de::Error::custom)
    }
}
