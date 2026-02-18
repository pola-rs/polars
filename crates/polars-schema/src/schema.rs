use core::fmt::Debug;
use core::hash::{Hash, Hasher};

use indexmap::map::MutableKeys;
use polars_error::{PolarsError, PolarsResult, polars_bail, polars_ensure, polars_err};
use polars_utils::aliases::{InitHashMaps, PlIndexMap};
use polars_utils::pl_str::PlSmallStr;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct Schema<Field, Metadata> {
    fields: PlIndexMap<PlSmallStr, Field>,
    metadata: Metadata,
}

impl<Field, Metadata: Default> Default for Schema<Field, Metadata> {
    fn default() -> Self {
        Self {
            fields: PlIndexMap::default(),
            metadata: Metadata::default(),
        }
    }
}

impl<Field: Eq, Metadata: Eq> Eq for Schema<Field, Metadata> {}

impl<Field, Metadata: Default> Schema<Field, Metadata> {
    pub fn with_capacity(capacity: usize) -> Self {
        let fields = PlIndexMap::with_capacity(capacity);
        Self {
            fields,
            metadata: Metadata::default(),
        }
    }

    pub fn from_iter_check_duplicates<I, F>(iter: I) -> PolarsResult<Self>
    where
        I: IntoIterator<Item = F>,
        F: Into<(PlSmallStr, Field)>,
    {
        Self::try_from_iter_check_duplicates(
            iter.into_iter().map(PolarsResult::Ok),
            |name: &str| polars_err!(Duplicate: "duplicate name when building schema '{}'", &name),
        )
    }

    pub fn try_from_iter_check_duplicates<I, F, E>(iter: I, err_func: E) -> PolarsResult<Self>
    where
        I: IntoIterator<Item = PolarsResult<F>>,
        F: Into<(PlSmallStr, Field)>,
        E: Fn(&str) -> PolarsError,
    {
        let iter = iter.into_iter();
        let mut slf = Self::with_capacity(iter.size_hint().1.unwrap_or(0));

        for v in iter {
            let (name, d) = v?.into();

            if slf.contains(&name) {
                return Err(err_func(&name));
            }

            slf.fields.insert(name, d);
        }

        Ok(slf)
    }
}

impl<Field, Metadata> Schema<Field, Metadata> {
    /// Reserve `additional` memory spaces in the schema.
    pub fn reserve(&mut self, additional: usize) {
        self.fields.reserve(additional);
    }

    /// The number of fields in the schema.
    #[inline]
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    pub fn metadata_mut(&mut self) -> &mut Metadata {
        &mut self.metadata
    }

    /// Rename field `old` to `new`, and return the (owned) old name.
    ///
    /// If `old` is not present in the schema, the schema is not modified and `None` is returned. Otherwise the schema
    /// is updated and `Some(old_name)` is returned.
    pub fn rename(&mut self, old: &str, new: PlSmallStr) -> Option<PlSmallStr> {
        // Remove `old`, get the corresponding index and dtype, and move the last item in the map to that position
        let (old_index, old_name, dtype) = self.fields.swap_remove_full(old)?;
        // Insert the same dtype under the new name at the end of the map and store that index
        let (new_index, _) = self.fields.insert_full(new, dtype);
        // Swap the two indices to move the originally last element back to the end and to move the new element back to
        // its original position
        self.fields.swap_indices(old_index, new_index);

        Some(old_name)
    }

    pub fn insert(&mut self, key: PlSmallStr, value: Field) -> Option<Field> {
        self.fields.insert(key, value)
    }

    /// Insert a field with `name` and `dtype` at the given `index` into this schema.
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
        name: PlSmallStr,
        dtype: Field,
    ) -> PolarsResult<Option<Field>> {
        polars_ensure!(
            index <= self.len(),
            OutOfBounds:
                "index {} is out of bounds for schema with length {} (the max index allowed is self.len())",
                    index,
                    self.len()
        );

        let (old_index, old_dtype) = self.fields.insert_full(name, dtype);

        // If we're moving an existing field, one-past-the-end will actually be out of bounds. Also, self.len() won't
        // have changed after inserting, so `index == self.len()` is the same as it was before inserting.
        if old_dtype.is_some() && index == self.len() {
            index -= 1;
        }
        self.fields.move_index(old_index, index);
        Ok(old_dtype)
    }

    /// Get a reference to the dtype of the field named `name`, or `None` if the field doesn't exist.
    pub fn get(&self, name: &str) -> Option<&Field> {
        self.fields.get(name)
    }

    /// Get a mutable reference to the dtype of the field named `name`, or `None` if the field doesn't exist.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Field> {
        self.fields.get_mut(name)
    }

    /// Get a reference to the dtype of the field named `name`, or `Err(PolarsErr)` if the field doesn't exist.
    pub fn try_get(&self, name: &str) -> PolarsResult<&Field> {
        self.get(name)
            .ok_or_else(|| polars_err!(SchemaFieldNotFound: "{name}"))
    }

    /// Get a mutable reference to the dtype of the field named `name`, or `Err(PolarsErr)` if the field doesn't exist.
    pub fn try_get_mut(&mut self, name: &str) -> PolarsResult<&mut Field> {
        self.fields
            .get_mut(name)
            .ok_or_else(|| polars_err!(SchemaFieldNotFound: "{name}"))
    }

    /// Return all data about the field named `name`: its index in the schema, its name, and its dtype.
    ///
    /// Returns `Some((index, &name, &dtype))` if the field exists, `None` if it doesn't.
    pub fn get_full(&self, name: &str) -> Option<(usize, &PlSmallStr, &Field)> {
        self.fields.get_full(name)
    }

    /// Return all data about the field named `name`: its index in the schema, its name, and its dtype.
    ///
    /// Returns `Ok((index, &name, &dtype))` if the field exists, `Err(PolarsErr)` if it doesn't.
    pub fn try_get_full(&self, name: &str) -> PolarsResult<(usize, &PlSmallStr, &Field)> {
        self.fields
            .get_full(name)
            .ok_or_else(|| polars_err!(SchemaFieldNotFound: "{name}"))
    }

    /// Get references to the name and dtype of the field at `index`.
    ///
    /// If `index` is inbounds, returns `Some((&name, &dtype))`, else `None`. See
    /// [`get_at_index_mut`][Self::get_at_index_mut] for a mutable version.
    pub fn get_at_index(&self, index: usize) -> Option<(&PlSmallStr, &Field)> {
        self.fields.get_index(index)
    }

    pub fn try_get_at_index(&self, index: usize) -> PolarsResult<(&PlSmallStr, &Field)> {
        self.fields.get_index(index).ok_or_else(|| polars_err!(ComputeError: "index {index} out of bounds with 'schema' of len: {}", self.len()))
    }

    /// Get mutable references to the name and dtype of the field at `index`.
    ///
    /// If `index` is inbounds, returns `Some((&mut name, &mut dtype))`, else `None`. See
    /// [`get_at_index`][Self::get_at_index] for an immutable version.
    pub fn get_at_index_mut(&mut self, index: usize) -> Option<(&mut PlSmallStr, &mut Field)> {
        self.fields.get_index_mut2(index)
    }

    /// Swap-remove a field by name and, if the field existed, return its dtype.
    ///
    /// If the field does not exist, the schema is not modified and `None` is returned.
    ///
    /// This method does a `swap_remove`, which is O(1) but **changes the order of the schema**: the field named `name`
    /// is replaced by the last field, which takes its position. For a slower, but order-preserving, method, use
    /// [`shift_remove`][Self::shift_remove].
    pub fn remove(&mut self, name: &str) -> Option<Field> {
        self.fields.swap_remove(name)
    }

    /// Remove a field by name, preserving order, and, if the field existed, return its dtype.
    ///
    /// If the field does not exist, the schema is not modified and `None` is returned.
    ///
    /// This method does a `shift_remove`, which preserves the order of the fields in the schema but **is O(n)**. For a
    /// faster, but not order-preserving, method, use [`remove`][Self::remove].
    pub fn shift_remove(&mut self, name: &str) -> Option<Field> {
        self.fields.shift_remove(name)
    }

    /// Remove a field by name, preserving order, and, if the field existed, return its dtype.
    ///
    /// If the field does not exist, the schema is not modified and `None` is returned.
    ///
    /// This method does a `shift_remove`, which preserves the order of the fields in the schema but **is O(n)**. For a
    /// faster, but not order-preserving, method, use [`remove`][Self::remove].
    pub fn shift_remove_index(&mut self, index: usize) -> Option<(PlSmallStr, Field)> {
        self.fields.shift_remove_index(index)
    }

    /// Whether the schema contains a field named `name`.
    pub fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// Change the field named `name` to the given `dtype` and return the previous dtype.
    ///
    /// If `name` doesn't already exist in the schema, the schema is not modified and `None` is returned. Otherwise
    /// returns `Some(old_dtype)`.
    ///
    /// This method only ever modifies an existing field and never adds a new field to the schema. To add a new field,
    /// use [`with_column`][Self::with_column] or [`insert_at_index`][Self::insert_at_index].
    pub fn set_dtype(&mut self, name: &str, dtype: Field) -> Option<Field> {
        let old_dtype = self.fields.get_mut(name)?;
        Some(std::mem::replace(old_dtype, dtype))
    }

    /// Change the field at the given index to the given `dtype` and return the previous dtype.
    ///
    /// If the index is out of bounds, the schema is not modified and `None` is returned. Otherwise returns
    /// `Some(old_dtype)`.
    ///
    /// This method only ever modifies an existing index and never adds a new field to the schema. To add a new field,
    /// use [`with_column`][Self::with_column] or [`insert_at_index`][Self::insert_at_index].
    pub fn set_dtype_at_index(&mut self, index: usize, dtype: Field) -> Option<Field> {
        let (_, old_dtype) = self.fields.get_index_mut(index)?;
        Some(std::mem::replace(old_dtype, dtype))
    }

    /// Insert a column into the [`Schema`].
    ///
    /// If the schema already has this column, this instead updates it with the new value and
    /// returns the old one. Otherwise, the column is inserted at the end.
    ///
    /// To enforce the index of the resulting field, use [`insert_at_index`][Self::insert_at_index].
    pub fn with_column(&mut self, name: PlSmallStr, dtype: Field) -> Option<Field> {
        self.fields.insert(name, dtype)
    }

    /// Raises DuplicateError if this column already exists in the schema.
    pub fn try_insert(&mut self, name: PlSmallStr, value: Field) -> PolarsResult<()> {
        if self.fields.contains_key(&name) {
            polars_bail!(Duplicate: "column '{}' is duplicate", name)
        }

        self.fields.insert(name, value);

        Ok(())
    }

    /// Performs [`Schema::try_insert`] for every column.
    ///
    /// Raises DuplicateError if a column already exists in the schema.
    pub fn hstack_mut(
        &mut self,
        columns: impl IntoIterator<Item = impl Into<(PlSmallStr, Field)>>,
    ) -> PolarsResult<()> {
        for v in columns {
            let (k, v) = v.into();
            self.try_insert(k, v)?;
        }

        Ok(())
    }

    /// Performs [`Schema::try_insert`] for every column.
    ///
    /// Raises DuplicateError if a column already exists in the schema.
    pub fn hstack(
        mut self,
        columns: impl IntoIterator<Item = impl Into<(PlSmallStr, Field)>>,
    ) -> PolarsResult<Self> {
        self.hstack_mut(columns)?;
        Ok(self)
    }

    pub fn sort_by_key<T, F>(&mut self, sort_key: F)
    where
        T: Ord,
        F: FnMut(&PlSmallStr, &Field) -> T,
    {
        self.fields.sort_by_key(sort_key);
    }

    /// Merge `other` into `self`.
    ///
    /// Merging logic:
    /// - Fields that occur in `self` but not `other` are unmodified
    /// - Fields that occur in `other` but not `self` are appended, in order, to the end of `self`
    /// - Fields that occur in both `self` and `other` are updated with the dtype from `other`, but keep their original
    ///   index
    pub fn merge(&mut self, other: Self) {
        self.fields.extend(other.fields)
    }

    /// Iterates over the `(&name, &dtype)` pairs in this schema.
    ///
    /// For an owned version, use [`iter_fields`][Self::iter_fields], which clones the data to iterate owned `Field`s
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&PlSmallStr, &Field)> + '_ {
        self.fields.iter()
    }

    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = (&PlSmallStr, &mut Field)> + '_ {
        self.fields.iter_mut()
    }

    /// Iterates over references to the names in this schema.
    pub fn iter_names(&self) -> impl '_ + ExactSizeIterator<Item = &PlSmallStr> {
        self.fields.iter().map(|(name, _dtype)| name)
    }

    pub fn iter_names_cloned(&self) -> impl '_ + ExactSizeIterator<Item = PlSmallStr> {
        self.iter_names().cloned()
    }

    /// Iterates over references to the dtypes in this schema.
    pub fn iter_values(&self) -> impl '_ + ExactSizeIterator<Item = &Field> {
        self.fields.iter().map(|(_name, dtype)| dtype)
    }

    pub fn into_iter_values(self) -> impl ExactSizeIterator<Item = Field> {
        self.fields.into_values()
    }

    /// Iterates over mut references to the dtypes in this schema.
    pub fn iter_values_mut(&mut self) -> impl '_ + ExactSizeIterator<Item = &mut Field> {
        self.fields.iter_mut().map(|(_name, dtype)| dtype)
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.fields.get_index_of(name)
    }

    pub fn try_index_of(&self, name: &str) -> PolarsResult<usize> {
        let Some(i) = self.fields.get_index_of(name) else {
            polars_bail!(
                ColumnNotFound:
                "unable to find column {:?}; valid columns: {:?}",
                name, self.iter_names().collect::<Vec<_>>(),
            )
        };

        Ok(i)
    }

    /// Compare the fields between two schema returning the additional columns that each schema has.
    pub fn field_compare<'a, 'b>(
        &'a self,
        other: &'b Self,
        self_extra: &mut Vec<(usize, (&'a PlSmallStr, &'a Field))>,
        other_extra: &mut Vec<(usize, (&'b PlSmallStr, &'b Field))>,
    ) {
        self_extra.extend(
            self.iter()
                .enumerate()
                .filter(|(_, (n, _))| !other.contains(n)),
        );
        other_extra.extend(
            other
                .iter()
                .enumerate()
                .filter(|(_, (n, _))| !self.contains(n)),
        );
    }
}

impl<Field, Metadata> Schema<Field, Metadata>
where
    Field: Clone,
    Metadata: Clone,
{
    /// Create a new schema from this one, inserting a field with `name` and `dtype` at the given `index`.
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
        name: PlSmallStr,
        field: Field,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            index <= self.len(),
            OutOfBounds:
                "index {} is out of bounds for schema with length {} (the max index allowed is self.len())",
                    index,
                    self.len()
        );

        let mut new = Self {
            fields: Default::default(),
            metadata: self.metadata().clone(),
        };
        let mut iter = self.fields.iter().filter_map(|(fld_name, dtype)| {
            (fld_name != &name).then_some((fld_name.clone(), dtype.clone()))
        });
        new.fields.extend(iter.by_ref().take(index));
        new.fields.insert(name.clone(), field);
        new.fields.extend(iter);
        Ok(new)
    }

    /// Merge borrowed `other` into `self`.
    ///
    /// Merging logic:
    /// - Fields that occur in `self` but not `other` are unmodified
    /// - Fields that occur in `other` but not `self` are appended, in order, to the end of `self`
    /// - Fields that occur in both `self` and `other` are updated with the dtype from `other`, but keep their original
    ///   index
    pub fn merge_from_ref(&mut self, other: &Self) {
        self.fields.extend(
            other
                .iter()
                .map(|(column, field)| (column.clone(), field.clone())),
        )
    }

    /// Generates another schema with just the specified columns selected from this one.
    pub fn try_project<I>(&self, columns: I) -> PolarsResult<Self>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let fields = columns
            .into_iter()
            .map(|c| {
                let name = c.as_ref();
                let (_, name, dtype) = self
                    .fields
                    .get_full(name)
                    .ok_or_else(|| polars_err!(col_not_found = name))?;
                PolarsResult::Ok((name.clone(), dtype.clone()))
            })
            .collect::<PolarsResult<PlIndexMap<PlSmallStr, _>>>()?;
        Ok(Self {
            fields,
            metadata: self.metadata().clone(),
        })
    }

    pub fn try_project_indices(&self, indices: &[usize]) -> PolarsResult<Self> {
        let fields = indices
            .iter()
            .map(|&i| {
                let Some((k, v)) = self.fields.get_index(i) else {
                    polars_bail!(
                        SchemaFieldNotFound:
                        "projection index {} is out of bounds for schema of length {}",
                        i, self.fields.len()
                    );
                };

                Ok((k.clone(), v.clone()))
            })
            .collect::<PolarsResult<PlIndexMap<_, _>>>()?;

        Ok(Self {
            fields,
            metadata: self.metadata().clone(),
        })
    }

    /// Returns a new [`Schema`] with a subset of all fields whose `predicate`
    /// evaluates to true.
    pub fn filter<F: Fn(usize, &Field) -> bool>(self, predicate: F) -> Self {
        let metadata = self.metadata().clone();
        let fields = self
            .fields
            .into_iter()
            .enumerate()
            .filter_map(|(index, (name, d))| {
                if (predicate)(index, &d) {
                    Some((name, d))
                } else {
                    None
                }
            })
            .collect();

        Self { fields, metadata }
    }
}

impl<Field: Hash, Metadata: Hash> Hash for Schema<Field, Metadata> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash(&SchemaHashEqWrap::from(self), state)
    }
}

// Schemas will only compare equal if they have the same fields in the same order. We can't use `self.inner ==
// other.inner` because [`IndexMap`] ignores order when checking equality, but we don't want to ignore it.
impl<Field: PartialEq, Metadata: PartialEq> PartialEq for Schema<Field, Metadata> {
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(
            &SchemaHashEqWrap::from(self),
            &SchemaHashEqWrap::from(other),
        )
    }
}

/// Specialization
/// * `IndexMap` eq impl does not consider key ordering, but we want key ordering.
/// * `IndexMap` does not impl Hash.
#[derive(Hash, PartialEq)]
struct SchemaHashEqWrap<'a, Field, Metadata> {
    fields: &'a indexmap::map::Slice<PlSmallStr, Field>,
    metadata: &'a Metadata,
}

impl<'a, Field, Metadata> From<&'a Schema<Field, Metadata>>
    for SchemaHashEqWrap<'a, Field, Metadata>
{
    fn from(value: &'a Schema<Field, Metadata>) -> Self {
        let Schema { fields, metadata } = value;

        Self {
            fields: fields.as_slice(),
            metadata,
        }
    }
}

impl<Field, Metadata: Default> From<PlIndexMap<PlSmallStr, Field>> for Schema<Field, Metadata> {
    fn from(fields: PlIndexMap<PlSmallStr, Field>) -> Self {
        Self {
            fields,
            metadata: Metadata::default(),
        }
    }
}

impl<F, Field, Metadata: Default> FromIterator<F> for Schema<Field, Metadata>
where
    F: Into<(PlSmallStr, Field)>,
{
    fn from_iter<I: IntoIterator<Item = F>>(iter: I) -> Self {
        let fields = PlIndexMap::from_iter(iter.into_iter().map(|x| x.into()));
        Self {
            fields,
            metadata: Metadata::default(),
        }
    }
}

impl<F, Field, Metadata> Extend<F> for Schema<Field, Metadata>
where
    F: Into<(PlSmallStr, Field)>,
{
    fn extend<T: IntoIterator<Item = F>>(&mut self, iter: T) {
        self.fields.extend(iter.into_iter().map(|x| x.into()))
    }
}

impl<Field, Metadata> IntoIterator for Schema<Field, Metadata> {
    type IntoIter = <PlIndexMap<PlSmallStr, Field> as IntoIterator>::IntoIter;
    type Item = (PlSmallStr, Field);

    fn into_iter(self) -> Self::IntoIter {
        self.fields.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::Schema;

    #[test]
    fn test_schema_eq_checks_key_order() {
        let lhs: Schema<(), ()> = Schema::from_iter([("a".into(), ()), ("b".into(), ())]);
        let rhs: Schema<(), ()> = Schema::from_iter([("b".into(), ()), ("a".into(), ())]);

        assert_ne!(lhs, rhs);
    }
}
