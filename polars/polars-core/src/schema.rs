use std::fmt::{Debug, Formatter};

use indexmap::IndexMap;
#[cfg(feature = "serde-lazy")]
use serde::{Deserialize, Serialize};
use smartstring::alias::String as SmartString;

use crate::prelude::*;

/// A map from field/column name (`String`) to the type of that field/column (`DataType`)
#[derive(Eq, Clone, Default)]
#[cfg_attr(feature = "serde-lazy", derive(Serialize, Deserialize))]
pub struct Schema {
    inner: PlIndexMap<SmartString, DataType>,
}

// IndexMap keeps track of the underlying order of its entries, so Schemas will only compare equal if they have the same
// fields in the same order
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

impl<'a> FromIterator<&'a arrow::datatypes::Field> for Schema {
    fn from_iter<T: IntoIterator<Item = &'a arrow::datatypes::Field>>(fields: T) -> Self {
        fields
            .into_iter()
            .map(|fld| Field::new(&fld.name, (&fld.data_type).into()))
            .collect()
    }
}

impl<F> FromIterator<F> for Schema
where
    F: Into<Field>,
{
    fn from_iter<T: IntoIterator<Item = F>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let mut map: PlIndexMap<_, _> =
            IndexMap::with_capacity_and_hasher(iter.size_hint().0, ahash::RandomState::default());
        for fld in iter {
            let fld = fld.into();

            #[cfg(feature = "dtype-decimal")]
            let fld = match fld.dtype {
                DataType::Decimal(_, _) => {
                    if crate::config::decimal_is_active() {
                        fld
                    } else {
                        let mut fld = fld.clone();
                        fld.coerce(DataType::Float64);
                        fld
                    }
                }
                _ => fld,
            };

            map.insert(fld.name().clone(), fld.data_type().clone());
        }
        Self { inner: map }
    }
}

impl Schema {
    /// Create a new, empty schema
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Create a new, empty schema with capacity
    ///
    /// If you know the number of fields you have ahead of time, using this is more efficient that using [`new`].
    pub fn with_capacity(capacity: usize) -> Self {
        let map: PlIndexMap<_, _> =
            IndexMap::with_capacity_and_hasher(capacity, ahash::RandomState::default());
        Self { inner: map }
    }

    /// The number of fields in the schema
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Rename field `old` to `new`, and return the (owned) old name
    ///
    /// If `old` is not present in the schema, the schema is not modified and `None` is returned. Otherwise the schema
    /// is updated and `Some(old_name)` is returned.
    pub fn rename(&mut self, old: &str, new: SmartString) -> Option<SmartString> {
        // Remove `old`, get the corresponding index and dtype, and move the last item in the map to that position
        let (old_index, old_name, dtype) = self.inner.swap_remove_full(old)?;
        // Insert the same dtype under the new name at the end of the map and store that index
        let (new_index, _) = self.inner.insert_full(new, dtype);
        // Swap the two indices to move the originally last element back to the end and to move the new element back to
        // its original position
        self.inner.swap_indices(old_index, new_index);

        Some(old_name)
    }

    /// Insert a field, with `name` and `dtype`, at the given `index`
    ///
    /// If a field named `name` already exists, it is updated with the new dtype. Regardless, the field is always moved
    /// to the given index.
    ///
    /// Valid indices range from `0` (front of the schema) to `self.len()` (after the end of the schema).
    ///
    /// Returns: if `index <= self.len()`, then `Ok(Some(old_dtype))` if a field with this name already existed, else
    /// `Ok(None)`. If `index > self.len()`, returns `Err(PolarsError)`.
    pub fn insert_index(
        &mut self,
        index: usize,
        name: SmartString,
        dtype: DataType,
    ) -> PolarsResult<Option<DataType>> {
        // 0 and self.len() are allowed
        polars_ensure!(
            index <= self.len(),
            ComputeError:
                "index {} is out of bounds for schema with length {} (the max index allowed is self.len())",
                    index,
                    self.len()
        );

        let old_dtype = self.inner.insert(name, dtype);
        self.inner.move_index(self.len() - 1, index);
        Ok(old_dtype)
    }

    /// Get the dtype of the field named `name`, or `None` if the field doesn't exist
    pub fn get(&self, name: &str) -> Option<&DataType> {
        self.inner.get(name)
    }

    /// Return all data about the field named `name`: its index in the schema, its name, and its dtype
    ///
    /// Returns `Some((index, &name, &dtype))` if the field exists, `None` if it doesn't.
    pub fn get_full(&self, name: &str) -> Option<(usize, &SmartString, &DataType)> {
        self.inner.get_full(name)
    }

    /// Swap-remove a field by name and, if the field existed, return its dtype
    ///
    /// If the field does not exist, the schema is not modified and `None` is returned.
    ///
    /// This method does a `swap_remove`, which is O(1) but **changes the order of the schema**: the field named `name`
    /// is replaced by the last field, which takes its position. For a slower, but order-preserving, method, use
    /// [`shift_remove`].
    pub fn remove(&mut self, name: &str) -> Option<DataType> {
        self.inner.swap_remove(name)
    }

    /// Remove a field by name, preserving order, and, if the field existed, return its dtype
    ///
    /// If the field does not exist, the schema is not modified and `None` is returned.
    ///
    /// This method does a `shift_remove`, which preserves the order of the fields in the schema but **is O(n)**. For a
    /// faster, but not order-preserving, method, use [`remove`].
    pub fn swap_remove(&mut self, name: &str) -> Option<DataType> {
        self.inner.shift_remove(name)
    }

    /// Look up the name in the schema and return an owned [`Field`] by cloning the data
    ///
    /// Returns `None` if the field does not exist.
    ///
    /// This method constructs the `Field` by cloning the name and dtype. For a version that returns references, see
    /// [`get`] or [`get_full`].
    pub fn get_field(&self, name: &str) -> Option<Field> {
        self.inner
            .get(name)
            .map(|dtype| Field::new(name, dtype.clone()))
    }

    /// Get references to the name and dtype of the field at `index`
    ///
    /// If `index` is inbounds, returns `Some((&name, &dtype))`, else `None`. See [`get_at_index_mut`] for a mutable
    /// version.
    pub fn get_at_index(&self, index: usize) -> Option<(&SmartString, &DataType)> {
        self.inner.get_index(index)
    }

    /// Get mutable references to the name and dtype of the field at `index`
    ///
    /// If `index` is inbounds, returns `Some((&mut name, &mut dtype))`, else `None`. See [`get_at_index`] for an
    /// immutable version.
    pub fn get_at_index_mut(&mut self, index: usize) -> Option<(&mut SmartString, &mut DataType)> {
        self.inner.get_index_mut(index)
    }

    /// Whether the schema contains a field named `name`
    pub fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// Change the field named `name` to the given `dtype` and return the previous dtype
    ///
    /// If the name exists in the schema, returns `Some(old_dtype)`, else `None`.
    pub fn set_dtype(&mut self, name: &str, dtype: DataType) -> Option<DataType> {
        let old_dtype = self.inner.get_mut(name)?;
        Some(std::mem::replace(old_dtype, dtype))
    }

    /// Change the field at the given index to the given `dtype` and return the previous dtype
    ///
    /// If the index is in bounds, returns `Some(old_dtype)`, else `None`.
    pub fn set_dtype_at_index(&mut self, index: usize, dtype: DataType) -> Option<DataType> {
        let (_, old_dtype) = self.inner.get_index_mut(index)?;
        Some(std::mem::replace(old_dtype, dtype))
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
    pub fn with_column(&mut self, name: SmartString, dtype: DataType) -> Option<DataType> {
        self.inner.insert(name, dtype)
    }

    /// Merge `other` into `self`
    ///
    /// Merging logic:
    /// - Fields that occur in `self` but not `other` are unmodified
    /// - Fields that occur in `other` but not `self` are appended, in order, to the end of `self`
    /// - Fields that occur in both `self` and `other` are updated with the dtype from `other`, but keep their original
    ///   index
    pub fn merge(&mut self, other: Self) {
        self.inner.extend(other.inner)
    }

    /// Convert self to `ArrowSchema` by cloning the fields
    pub fn to_arrow(&self) -> ArrowSchema {
        let fields: Vec<_> = self
            .inner
            .iter()
            .map(|(name, dtype)| ArrowField::new(name.as_str(), dtype.to_arrow(), true))
            .collect();
        ArrowSchema::from(fields)
    }

    /// Iterates the `Field`s in this schema, constructing them anew by cloning each `(&name, &dtype)` pair
    ///
    /// Note that this clones each name and dtype in order to form an owned `Field`. For a clone-free version, use
    /// [`iter`], which returns `(&name, &dtype)`.
    pub fn iter_fields(&self) -> impl Iterator<Item = Field> + ExactSizeIterator + '_ {
        self.inner
            .iter()
            .map(|(name, dtype)| Field::new(name, dtype.clone()))
    }

    /// Iterates over references to the dtypes in this schema
    pub fn iter_dtypes(&self) -> impl Iterator<Item = &DataType> + ExactSizeIterator + '_ {
        self.inner.iter().map(|(_name, dtype)| dtype)
    }

    /// Iterates over references to the names in this schema
    pub fn iter_names(&self) -> impl Iterator<Item = &SmartString> + '_ + ExactSizeIterator {
        self.inner.iter().map(|(name, _dtype)| name)
    }

    /// Iterates over the `(&name, &dtype)` pairs in this schema
    ///
    /// For an owned version, use [`iter_fields`], which clones the data to iterate owned `Field`s
    pub fn iter(&self) -> impl Iterator<Item = (&SmartString, &DataType)> + '_ {
        self.inner.iter()
    }
}

pub type SchemaRef = Arc<Schema>;

impl IntoIterator for Schema {
    type Item = (SmartString, DataType);
    type IntoIter = <PlIndexMap<SmartString, DataType> as IntoIterator>::IntoIter;

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
            polars_err!(SchemaMismatch: "unable to get field '{}' from schema: {:?}", name, self)
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
