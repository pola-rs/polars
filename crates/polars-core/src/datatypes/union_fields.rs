use std::ops::Deref;

use polars_arrow::array::MAX_UNION_VARIANTS;

use super::*;

/// The variants of a [`DataType::Union`].
///
/// The kernels rely on every value of this type having between 1 and [`MAX_UNION_VARIANTS`]
/// uniquely named variants, none of which is stored as `Null`, `Object` or `Unknown`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(try_from = "Vec<Field>", into = "Vec<Field>")
)]
pub struct UnionFields(Vec<Field>);

impl UnionFields {
    pub fn try_new(fields: Vec<Field>) -> PolarsResult<Self> {
        polars_ensure!(
            !fields.is_empty() && fields.len() <= MAX_UNION_VARIANTS,
            InvalidOperation: "a Union must have between 1 and {MAX_UNION_VARIANTS} variants, got {}",
            fields.len()
        );
        let mut names = PlHashSet::with_capacity(fields.len());
        for fld in &fields {
            polars_ensure!(
                names.insert(fld.name()),
                Duplicate: "Union variant name `{}` is not unique", fld.name()
            );
            #[allow(unused_mut)]
            let mut storage = fld.dtype();
            #[cfg(feature = "dtype-extension")]
            while let DataType::Extension(_, inner) = storage {
                storage = inner;
            }
            polars_ensure!(
                !(storage.is_null() || storage.is_object() || storage.is_unknown()),
                InvalidOperation: "`{}` cannot be used as a Union variant dtype", fld.dtype()
            );
        }
        drop(names);
        Ok(Self(fields))
    }

    /// Replaces the dtype of every variant, keeping the names.
    pub fn try_map_dtypes(
        &self,
        mut f: impl FnMut(&DataType) -> PolarsResult<DataType>,
    ) -> PolarsResult<Self> {
        let fields = self
            .iter()
            .map(|fld| Ok(Field::new(fld.name().clone(), f(fld.dtype())?)))
            .collect::<PolarsResult<_>>()?;
        Self::try_new(fields)
    }

    pub fn into_fields(self) -> Vec<Field> {
        self.0
    }

    /// The dtype of `ChunkedArray<UnionType>` before it holds data, which no array can have.
    pub(crate) fn placeholder() -> Self {
        Self(Vec::new())
    }
}

impl Deref for UnionFields {
    type Target = [Field];

    fn deref(&self) -> &[Field] {
        &self.0
    }
}

impl<'a> IntoIterator for &'a UnionFields {
    type Item = &'a Field;
    type IntoIter = std::slice::Iter<'a, Field>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl TryFrom<Vec<Field>> for UnionFields {
    type Error = PolarsError;

    fn try_from(fields: Vec<Field>) -> PolarsResult<Self> {
        Self::try_new(fields)
    }
}

impl From<UnionFields> for Vec<Field> {
    fn from(fields: UnionFields) -> Self {
        fields.0
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn fields(dtypes: &[(&str, DataType)]) -> Vec<Field> {
        dtypes
            .iter()
            .map(|(n, dt)| Field::new((*n).into(), dt.clone()))
            .collect()
    }

    #[test]
    fn rejects_invalid_variants() {
        assert!(UnionFields::try_new(fields(&[("a", DataType::Int64)])).is_ok());
        assert!(UnionFields::try_new(vec![]).is_err());
        let dup = fields(&[("a", DataType::Int64), ("a", DataType::String)]);
        assert!(UnionFields::try_new(dup).is_err());
        let null = fields(&[("a", DataType::Int64), ("b", DataType::Null)]);
        assert!(UnionFields::try_new(null).is_err());
        let many = (0..=MAX_UNION_VARIANTS)
            .map(|i| Field::new(format!("v{i}").into(), DataType::Int64))
            .collect::<Vec<_>>();
        assert!(UnionFields::try_new(many[1..].to_vec()).is_ok());
        assert!(UnionFields::try_new(many).is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn deserialize_validates() {
        let valid =
            DataType::Union(UnionFields::try_new(fields(&[("a", DataType::Int64)])).unwrap());
        let json = serde_json::to_string(&valid).unwrap();
        assert_eq!(serde_json::from_str::<DataType>(&json).unwrap(), valid);
        let dup = json.replace("]", r#",{"name":"a","dtype":"String"}]"#);
        assert!(dup != json);
        let err = serde_json::from_str::<DataType>(&dup).unwrap_err();
        assert!(err.to_string().contains("not unique"), "{err}");
    }
}
