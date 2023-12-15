use std::fmt::{Debug, Formatter};

use arrow::datatypes::ArrowSchemaRef;
use indexmap::map::MutableKeys;
use indexmap::IndexMap;
#[cfg(feature = "serde-lazy")]
use serde::{Deserialize, Serialize};
use smartstring::alias::String as SmartString;

use crate::prelude::*;
use crate::utils::try_get_supertype;

/// A map from field/column name ([`String`](smartstring::alias::String)) to the type of that field/column ([`DataType`])
#[derive(Eq, Clone, Default)]
#[cfg_attr(feature = "serde-lazy", derive(Serialize, Deserialize))]
pub struct Schema {
    inner: PlIndexMap<SmartString, DataType>,
}

// Schemas will only compare equal if they have the same fields in the same order. We can't use `self.inner ==
// other.inner` because [`IndexMap`] ignores order when checking equality, but we don't want to ignore it.
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

impl From<&[Series]> for Schema {
    fn from(value: &[Series]) -> Self {
        value.iter().map(|s| s.field().into_owned()).collect()
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
            map.insert(fld.name, fld.dtype);
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
    /// If you know the number of fields you have ahead of time, using this is more efficient than using
    /// [`new`][Self::new]. Also consider using [`Schema::from_iter`] if you have the collection of fields available
    /// ahead of time.
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

    /// Create a new schema from this one, inserting a field with `name` and `dtype` at the given `index`
    ///
    /// If a field named `name` already exists, it is updated with the new dtype. Regardless, the field named `name` is
    /// always moved to the given index. Valid indices range from `0` (front of the schema) to `self.len()` (after the
    /// end of the schema).
    ///
    /// For a mutating version that doesn't clone, see [`insert_at_index`][Self::insert_at_index].
    ///
    /// Runtime: **O(m * n)** where `m` is the (average) length of the field names and `n` is the number of fields in
    /// the schema. This method clones every field in the schema.
    ///
    /// Returns: `Ok(new_schema)` if `index <= self.len()`, else `Err(PolarsError)`
    pub fn new_inserting_at_index(
        &self,
        index: usize,
        name: SmartString,
        dtype: DataType,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            index <= self.len(),
            OutOfBounds:
                "index {} is out of bounds for schema with length {} (the max index allowed is self.len())",
                    index,
                    self.len()
        );

        let mut new = Self::default();
        let mut iter = self.inner.iter().filter_map(|(fld_name, dtype)| {
            (fld_name != &name).then_some((fld_name.clone(), dtype.clone()))
        });
        new.inner.extend(iter.by_ref().take(index));
        new.inner.insert(name.clone(), dtype);
        new.inner.extend(iter);
        Ok(new)
    }

    /// Insert a field with `name` and `dtype` at the given `index` into this schema
    ///
    /// If a field named `name` already exists, it is updated with the new dtype. Regardless, the field named `name` is
    /// always moved to the given index. Valid indices range from `0` (front of the schema) to `self.len()` (after the
    /// end of the schema).
    ///
    /// For a non-mutating version that clones the schema, see [`new_inserting_at_index`][Self::new_inserting_at_index].
    ///
    /// Runtime: **O(n)** where `n` is the number of fields in the schema.
    ///
    /// Returns:
    /// - If index is out of bounds, `Err(PolarsError)`
    /// - Else if `name` was already in the schema, `Ok(Some(old_dtype))`
    /// - Else `Ok(None)`
    pub fn insert_at_index(
        &mut self,
        mut index: usize,
        name: SmartString,
        dtype: DataType,
    ) -> PolarsResult<Option<DataType>> {
        polars_ensure!(
            index <= self.len(),
            OutOfBounds:
                "index {} is out of bounds for schema with length {} (the max index allowed is self.len())",
                    index,
                    self.len()
        );

        let (old_index, old_dtype) = self.inner.insert_full(name, dtype);

        // If we're moving an existing field, one-past-the-end will actually be out of bounds. Also, self.len() won't
        // have changed after inserting, so `index == self.len()` is the same as it was before inserting.
        if old_dtype.is_some() && index == self.len() {
            index -= 1;
        }
        self.inner.move_index(old_index, index);
        Ok(old_dtype)
    }

    /// Get a reference to the dtype of the field named `name`, or `None` if the field doesn't exist
    pub fn get(&self, name: &str) -> Option<&DataType> {
        self.inner.get(name)
    }

    /// Get a reference to the dtype of the field named `name`, or `Err(PolarsErr)` if the field doesn't exist
    pub fn try_get(&self, name: &str) -> PolarsResult<&DataType> {
        self.get(name)
            .ok_or_else(|| polars_err!(SchemaFieldNotFound: "{}", name))
    }

    /// Get a mutable reference to the dtype of the field named `name`, or `Err(PolarsErr)` if the field doesn't exist
    pub fn try_get_mut(&mut self, name: &str) -> PolarsResult<&mut DataType> {
        self.inner
            .get_mut(name)
            .ok_or_else(|| polars_err!(SchemaFieldNotFound: "{}", name))
    }

    /// Return all data about the field named `name`: its index in the schema, its name, and its dtype
    ///
    /// Returns `Some((index, &name, &dtype))` if the field exists, `None` if it doesn't.
    pub fn get_full(&self, name: &str) -> Option<(usize, &SmartString, &DataType)> {
        self.inner.get_full(name)
    }

    /// Return all data about the field named `name`: its index in the schema, its name, and its dtype
    ///
    /// Returns `Ok((index, &name, &dtype))` if the field exists, `Err(PolarsErr)` if it doesn't.
    pub fn try_get_full(&self, name: &str) -> PolarsResult<(usize, &SmartString, &DataType)> {
        self.inner
            .get_full(name)
            .ok_or_else(|| polars_err!(SchemaFieldNotFound: "{}", name))
    }

    /// Look up the name in the schema and return an owned [`Field`] by cloning the data
    ///
    /// Returns `None` if the field does not exist.
    ///
    /// This method constructs the `Field` by cloning the name and dtype. For a version that returns references, see
    /// [`get`][Self::get] or [`get_full`][Self::get_full].
    pub fn get_field(&self, name: &str) -> Option<Field> {
        self.inner
            .get(name)
            .map(|dtype| Field::new(name, dtype.clone()))
    }

    /// Look up the name in the schema and return an owned [`Field`] by cloning the data
    ///
    /// Returns `Err(PolarsErr)` if the field does not exist.
    ///
    /// This method constructs the `Field` by cloning the name and dtype. For a version that returns references, see
    /// [`get`][Self::get] or [`get_full`][Self::get_full].
    pub fn try_get_field(&self, name: &str) -> PolarsResult<Field> {
        self.inner
            .get(name)
            .ok_or_else(|| polars_err!(SchemaFieldNotFound: "{}", name))
            .map(|dtype| Field::new(name, dtype.clone()))
    }

    /// Get references to the name and dtype of the field at `index`
    ///
    /// If `index` is inbounds, returns `Some((&name, &dtype))`, else `None`. See
    /// [`get_at_index_mut`][Self::get_at_index_mut] for a mutable version.
    pub fn get_at_index(&self, index: usize) -> Option<(&SmartString, &DataType)> {
        self.inner.get_index(index)
    }

    pub fn try_get_at_index(&self, index: usize) -> PolarsResult<(&SmartString, &DataType)> {
        self.inner.get_index(index).ok_or_else(|| polars_err!(ComputeError: "index {index} out of bounds with 'schema' of len: {}", self.len()))
    }

    /// Get mutable references to the name and dtype of the field at `index`
    ///
    /// If `index` is inbounds, returns `Some((&mut name, &mut dtype))`, else `None`. See
    /// [`get_at_index`][Self::get_at_index] for an immutable version.
    pub fn get_at_index_mut(&mut self, index: usize) -> Option<(&mut SmartString, &mut DataType)> {
        self.inner.get_index_mut2(index)
    }

    /// Swap-remove a field by name and, if the field existed, return its dtype
    ///
    /// If the field does not exist, the schema is not modified and `None` is returned.
    ///
    /// This method does a `swap_remove`, which is O(1) but **changes the order of the schema**: the field named `name`
    /// is replaced by the last field, which takes its position. For a slower, but order-preserving, method, use
    /// [`shift_remove`][Self::shift_remove].
    pub fn remove(&mut self, name: &str) -> Option<DataType> {
        self.inner.swap_remove(name)
    }

    /// Remove a field by name, preserving order, and, if the field existed, return its dtype
    ///
    /// If the field does not exist, the schema is not modified and `None` is returned.
    ///
    /// This method does a `shift_remove`, which preserves the order of the fields in the schema but **is O(n)**. For a
    /// faster, but not order-preserving, method, use [`remove`][Self::remove].
    pub fn shift_remove(&mut self, name: &str) -> Option<DataType> {
        self.inner.shift_remove(name)
    }

    /// Remove a field by name, preserving order, and, if the field existed, return its dtype
    ///
    /// If the field does not exist, the schema is not modified and `None` is returned.
    ///
    /// This method does a `shift_remove`, which preserves the order of the fields in the schema but **is O(n)**. For a
    /// faster, but not order-preserving, method, use [`remove`][Self::remove].
    pub fn shift_remove_index(&mut self, index: usize) -> Option<(SmartString, DataType)> {
        self.inner.shift_remove_index(index)
    }

    /// Whether the schema contains a field named `name`
    pub fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// Change the field named `name` to the given `dtype` and return the previous dtype
    ///
    /// If `name` doesn't already exist in the schema, the schema is not modified and `None` is returned. Otherwise
    /// returns `Some(old_dtype)`.
    ///
    /// This method only ever modifies an existing field and never adds a new field to the schema. To add a new field,
    /// use [`with_column`][Self::with_column] or [`insert_at_index`][Self::insert_at_index].
    pub fn set_dtype(&mut self, name: &str, dtype: DataType) -> Option<DataType> {
        let old_dtype = self.inner.get_mut(name)?;
        Some(std::mem::replace(old_dtype, dtype))
    }

    /// Change the field at the given index to the given `dtype` and return the previous dtype
    ///
    /// If the index is out of bounds, the schema is not modified and `None` is returned. Otherwise returns
    /// `Some(old_dtype)`.
    ///
    /// This method only ever modifies an existing index and never adds a new field to the schema. To add a new field,
    /// use [`with_column`][Self::with_column] or [`insert_at_index`][Self::insert_at_index].
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
    /// To enforce the index of the resulting field, use [`insert_at_index`][Self::insert_at_index].
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

    /// Iterates the [`Field`]s in this schema, constructing them anew by cloning each `(&name, &dtype)` pair
    ///
    /// Note that this clones each name and dtype in order to form an owned [`Field`]. For a clone-free version, use
    /// [`iter`][Self::iter], which returns `(&name, &dtype)`.
    pub fn iter_fields(&self) -> impl Iterator<Item = Field> + ExactSizeIterator + '_ {
        self.inner
            .iter()
            .map(|(name, dtype)| Field::new(name, dtype.clone()))
    }

    /// Iterates over references to the dtypes in this schema
    pub fn iter_dtypes(&self) -> impl Iterator<Item = &DataType> + '_ + ExactSizeIterator {
        self.inner.iter().map(|(_name, dtype)| dtype)
    }

    /// Iterates over mut references to the dtypes in this schema
    pub fn iter_dtypes_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut DataType> + '_ + ExactSizeIterator {
        self.inner.iter_mut().map(|(_name, dtype)| dtype)
    }

    /// Iterates over references to the names in this schema
    pub fn iter_names(&self) -> impl Iterator<Item = &SmartString> + '_ + ExactSizeIterator {
        self.inner.iter().map(|(name, _dtype)| name)
    }

    /// Iterates over the `(&name, &dtype)` pairs in this schema
    ///
    /// For an owned version, use [`iter_fields`][Self::iter_fields], which clones the data to iterate owned `Field`s
    pub fn iter(&self) -> impl Iterator<Item = (&SmartString, &DataType)> + '_ {
        self.inner.iter()
    }

    /// Take another [`Schema`] and try to find the supertypes between them.
    pub fn to_supertype(&mut self, other: &Schema) -> PolarsResult<bool> {
        polars_ensure!(self.len() == other.len(), ComputeError: "schema lengths differ");

        let mut changed = false;
        for ((k, dt), (other_k, other_dt)) in self.inner.iter_mut().zip(other.iter()) {
            polars_ensure!(k == other_k, ComputeError: "schema names differ: got {}, expected {}", k, other_k);

            let st = try_get_supertype(dt, other_dt)?;
            changed |= (&st != dt) || (&st != other_dt);
            *dt = st
        }
        Ok(changed)
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
pub trait IndexOfSchema: Debug {
    /// Get the index of a column by name.
    fn index_of(&self, name: &str) -> Option<usize>;

    /// Get a vector of all column names.
    fn get_names(&self) -> Vec<&str>;

    fn try_index_of(&self, name: &str) -> PolarsResult<usize> {
        self.index_of(name).ok_or_else(|| {
            polars_err!(
                ColumnNotFound:
                "unable to find column {:?}; valid columns: {:?}", name, self.get_names(),
            )
        })
    }
}

impl IndexOfSchema for Schema {
    fn index_of(&self, name: &str) -> Option<usize> {
        self.inner.get_index_of(name)
    }

    fn get_names(&self) -> Vec<&str> {
        self.iter_names().map(|name| name.as_str()).collect()
    }
}

impl IndexOfSchema for ArrowSchema {
    fn index_of(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }

    fn get_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }
}

impl From<&ArrowSchema> for Schema {
    fn from(value: &ArrowSchema) -> Self {
        Self::from_iter(value.fields.iter())
    }
}
impl From<ArrowSchema> for Schema {
    fn from(value: ArrowSchema) -> Self {
        Self::from(&value)
    }
}

impl From<ArrowSchemaRef> for Schema {
    fn from(value: ArrowSchemaRef) -> Self {
        Self::from(value.as_ref())
    }
}

impl From<&ArrowSchemaRef> for Schema {
    fn from(value: &ArrowSchemaRef) -> Self {
        Self::from(value.as_ref())
    }
}
