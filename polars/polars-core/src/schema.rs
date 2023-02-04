use std::fmt::{Debug, Formatter};

use indexmap::IndexMap;
#[cfg(feature = "serde-lazy")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;

#[derive(Eq, Clone, Default)]
#[cfg_attr(feature = "serde-lazy", derive(Serialize, Deserialize))]
pub struct Schema {
    inner: PlIndexMap<String, DataType>,
}

// IndexMap does not care about order.
impl PartialEq for Schema {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl Debug for Schema {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Schema:")?;
        for (name, dtype) in self.inner.iter() {
            writeln!(f, "name: {name}, data type: {dtype:?}")?;
        }
        Ok(())
    }
}

impl<I, J> From<I> for Schema
where
    I: Iterator<Item = J>,
    J: Into<Field>,
{
    fn from(iter: I) -> Self {
        let mut map: PlIndexMap<_, _> =
            IndexMap::with_capacity_and_hasher(iter.size_hint().0, ahash::RandomState::default());
        for fld in iter {
            let fld = fld.into();
            map.insert(fld.name().clone(), fld.data_type().clone());
        }
        Self { inner: map }
    }
}

impl<J> FromIterator<J> for Schema
where
    J: Into<Field>,
{
    fn from_iter<I: IntoIterator<Item = J>>(iter: I) -> Self {
        Schema::from(iter.into_iter())
    }
}

impl Schema {
    // could not implement TryFrom
    pub fn try_from_fallible<I>(flds: I) -> PolarsResult<Self>
    where
        I: IntoIterator<Item = PolarsResult<Field>>,
    {
        let iter = flds.into_iter();
        let mut map: PlIndexMap<_, _> =
            IndexMap::with_capacity_and_hasher(iter.size_hint().0, ahash::RandomState::default());
        for fld in iter {
            let fld = fld?;
            map.insert(fld.name().clone(), fld.data_type().clone());
        }
        Ok(Self { inner: map })
    }

    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let map: PlIndexMap<_, _> =
            IndexMap::with_capacity_and_hasher(capacity, ahash::RandomState::default());
        Self { inner: map }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn rename(&mut self, old: &str, new: String) -> Option<()> {
        // we first append the new name
        // and then remove the old name
        // this works because the removed slot is swapped with the last value in the indexmap
        let dtype = self.inner.get(old)?.clone();
        self.inner.insert(new, dtype);
        self.inner.swap_remove(old);
        Some(())
    }

    pub fn insert_index(&self, index: usize, name: String, dtype: DataType) -> Option<Self> {
        // 0 and self.len() 0 is allowed
        if index > self.len() {
            return None;
        }
        let mut new = Self::default();
        let mut iter = self
            .inner
            .iter()
            .map(|(name, dtype)| (name.clone(), dtype.clone()));
        new.inner.extend((&mut iter).take(index));
        new.inner.insert(name, dtype);
        new.inner.extend(iter);
        Some(new)
    }

    pub fn get(&self, name: &str) -> Option<&DataType> {
        self.inner.get(name)
    }

    pub fn try_get(&self, name: &str) -> PolarsResult<&DataType> {
        self.get(name)
            .ok_or_else(|| PolarsError::SchemaFieldNotFound(name.to_string().into()))
    }

    pub fn try_get_full(&self, name: &str) -> PolarsResult<(usize, &String, &DataType)> {
        self.inner
            .get_full(name)
            .ok_or_else(|| PolarsError::SchemaFieldNotFound(name.to_string().into()))
    }

    pub fn remove(&mut self, name: &str) -> Option<DataType> {
        self.inner.remove(name)
    }

    pub fn get_full(&self, name: &str) -> Option<(usize, &String, &DataType)> {
        self.inner.get_full(name)
    }

    pub fn get_field(&self, name: &str) -> Option<Field> {
        self.inner
            .get(name)
            .map(|dtype| Field::new(name, dtype.clone()))
    }

    pub fn try_get_field(&self, name: &str) -> PolarsResult<Field> {
        self.inner
            .get(name)
            .ok_or_else(|| PolarsError::SchemaFieldNotFound(name.to_string().into()))
            .map(|dtype| Field::new(name, dtype.clone()))
    }

    pub fn get_index(&self, index: usize) -> Option<(&String, &DataType)> {
        self.inner.get_index(index)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    pub fn get_index_mut(&mut self, index: usize) -> Option<(&mut String, &mut DataType)> {
        self.inner.get_index_mut(index)
    }

    pub fn coerce_by_name(&mut self, name: &str, dtype: DataType) -> Option<()> {
        *self.inner.get_mut(name)? = dtype;
        Some(())
    }

    pub fn coerce_by_index(&mut self, index: usize, dtype: DataType) -> Option<()> {
        *self.inner.get_index_mut(index)?.1 = dtype;
        Some(())
    }

    /// Insert a new column in the [`Schema`]
    ///
    /// If an equivalent name already exists in the schema: the name remains and
    /// retains in its place in the order, its corresponding value is updated
    /// with [`DataType`] and the older dtype is returned inside `Some(_)`.
    ///
    /// If no equivalent key existed in the map: the new name-dtype pair is
    /// inserted, last in order, and `None` is returned.
    ///
    /// Computes in **O(1)** time (amortized average).
    pub fn with_column(&mut self, name: String, dtype: DataType) -> Option<DataType> {
        self.inner.insert(name, dtype)
    }

    pub fn merge(&mut self, other: Self) {
        self.inner.extend(other.inner.into_iter())
    }

    pub fn to_arrow(&self) -> ArrowSchema {
        let fields: Vec<_> = self
            .inner
            .iter()
            .map(|(name, dtype)| ArrowField::new(name, dtype.to_arrow(), true))
            .collect();
        ArrowSchema::from(fields)
    }

    pub fn iter_fields(&self) -> impl Iterator<Item = Field> + ExactSizeIterator + '_ {
        self.inner
            .iter()
            .map(|(name, dtype)| Field::new(name, dtype.clone()))
    }

    pub fn iter_dtypes(&self) -> impl Iterator<Item = &DataType> + ExactSizeIterator + '_ {
        self.inner.iter().map(|(_name, dtype)| dtype)
    }

    pub fn iter_names(&self) -> impl Iterator<Item = &String> + '_ + ExactSizeIterator {
        self.inner.iter().map(|(name, _dtype)| name)
    }
    pub fn iter(&self) -> impl Iterator<Item = (&String, &DataType)> + '_ {
        self.inner.iter()
    }
}

pub type SchemaRef = Arc<Schema>;

impl IntoIterator for Schema {
    type Item = (String, DataType);
    type IntoIter = <PlIndexMap<String, DataType> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

/// This trait exists to be unify the API of polars Schema and arrows Schema
#[cfg(feature = "private")]
pub trait IndexOfSchema: Debug {
    /// Get the index of column by name.
    fn index_of(&self, name: &str) -> Option<usize>;

    fn try_index_of(&self, name: &str) -> PolarsResult<usize> {
        self.index_of(name).ok_or_else(|| {
            PolarsError::SchemaMisMatch(
                format!("Unable to get field named \"{name}\" from schema: {self:?}",).into(),
            )
        })
    }
}

impl IndexOfSchema for Schema {
    fn index_of(&self, name: &str) -> Option<usize> {
        self.inner.get_index_of(name)
    }
}

impl IndexOfSchema for ArrowSchema {
    fn index_of(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }
}
