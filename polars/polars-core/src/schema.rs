use crate::prelude::*;
use indexmap::IndexMap;
use std::fmt::{Debug, Formatter};

#[derive(PartialEq, Eq, Clone, Default)]
pub struct Schema {
    inner: PlIndexMap<String, DataType>,
}

impl Debug for Schema {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Schema:")?;
        for (name, dtype) in self.inner.iter() {
            writeln!(f, "name: {}, data type: {:?}", name, dtype)?;
        }
        Ok(())
    }
}

impl<I, J> From<I> for Schema
where
    I: IntoIterator<Item = J>,
    J: Into<Field>,
{
    fn from(flds: I) -> Self {
        let iter = flds.into_iter();
        let mut map: PlIndexMap<_, _> =
            IndexMap::with_capacity_and_hasher(iter.size_hint().0, ahash::RandomState::default());
        for fld in iter {
            let fld = fld.into();
            map.insert(fld.name().clone(), fld.data_type().clone());
        }
        Self { inner: map }
    }
}

impl Schema {
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

    pub fn get_full(&self, name: &str) -> Option<(usize, &String, &DataType)> {
        self.inner.get_full(name)
    }

    pub fn get_field(&self, name: &str) -> Option<Field> {
        self.inner
            .get(name)
            .map(|dtype| Field::new(name, dtype.clone()))
    }

    pub fn get_index(&self, index: usize) -> Option<(&String, &DataType)> {
        self.inner.get_index(index)
    }

    pub fn coerce_by_name(&mut self, name: &str, dtype: DataType) -> Option<()> {
        *self.inner.get_mut(name)? = dtype;
        Some(())
    }

    pub fn coerce_by_index(&mut self, index: usize, dtype: DataType) -> Option<()> {
        *self.inner.get_index_mut(index)?.1 = dtype;
        Some(())
    }

    pub fn with_column(&mut self, name: String, dtype: DataType) {
        self.inner.insert(name, dtype);
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

    pub fn iter_fields(&self) -> impl Iterator<Item = Field> + '_ {
        self.inner
            .iter()
            .map(|(name, dtype)| Field::new(name, dtype.clone()))
    }

    pub fn iter_dtypes(&self) -> impl Iterator<Item = &DataType> + '_ {
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

/// This trait exists to be unify the API of polars Schema and arrows Schema
#[cfg(feature = "private")]
pub trait IndexOfSchema: Debug {
    /// Get the index of column by name.
    fn index_of(&self, name: &str) -> Option<usize>;

    fn try_index_of(&self, name: &str) -> Result<usize> {
        self.index_of(name).ok_or_else(|| {
            PolarsError::SchemaMisMatch(
                format!(
                    "Unable to get field named \"{}\" from schema: {:?}",
                    name, self
                )
                .into(),
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
