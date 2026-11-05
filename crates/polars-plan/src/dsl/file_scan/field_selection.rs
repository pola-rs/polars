use std::fmt;
use std::hash::{Hash, Hasher};

#[cfg(feature = "dtype-struct")]
use polars_core::prelude::Field;
use polars_core::prelude::{DataType, PlIndexMap};
use polars_utils::pl_str::PlSmallStr;

/// Which parts of one column's value are needed.
///
/// Built by projection pushdown, which reads struct fields by name; non-struct values are always
/// `All`. See `agent-docs/nested-projection-pushdown/01-field-pruning.md`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum FieldSelection {
    /// The whole value (every field, recursively). The default; all non-nested columns.
    #[default]
    All,
    /// Only these struct fields, each with its own selection. Empty: only the row count and the
    /// outer validity are needed (`s.is_null()`, `len`).
    Fields(Box<PlIndexMap<PlSmallStr, FieldSelection>>),
    /// The element of a `List` or `Array`, with its own selection, plus the offsets and validity
    /// of the list. Never `Element(All)`, which is `All`.
    Element(Box<FieldSelection>),
}

static ALL: FieldSelection = FieldSelection::All;

impl Hash for FieldSelection {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::All => 0u8.hash(state),
            Self::Fields(fields) => {
                1u8.hash(state);
                hash_unordered(fields, state);
            },
            Self::Element(element) => {
                2u8.hash(state);
                element.hash(state);
            },
        }
    }
}

/// Hashes a field map regardless of its order, as `PlIndexMap` compares regardless of order.
fn hash_unordered<H: Hasher>(fields: &PlIndexMap<PlSmallStr, FieldSelection>, state: &mut H) {
    fields.len().hash(state);
    let combined = fields.iter().fold(0u64, |acc, entry| {
        // A fixed-key hasher, so that equal maps hash equal in every process.
        let mut hasher = std::hash::DefaultHasher::new();
        entry.hash(&mut hasher);
        acc.wrapping_add(hasher.finish())
    });
    combined.hash(state);
}

impl FieldSelection {
    /// Only the row count and the outer validity.
    pub fn validity_only() -> Self {
        Self::Fields(Box::default())
    }

    /// Only the field `name`, with `selection`.
    pub fn field(name: PlSmallStr, selection: Self) -> Self {
        Self::Fields(Box::new(PlIndexMap::from_iter([(name, selection)])))
    }

    /// The element of a list or array with `selection`; `All` if `selection` is.
    pub fn element(selection: Self) -> Self {
        match selection {
            Self::All => Self::All,
            selection => Self::Element(Box::new(selection)),
        }
    }

    pub fn is_all(&self) -> bool {
        matches!(self, Self::All)
    }

    pub fn fields(&self) -> Option<&PlIndexMap<PlSmallStr, FieldSelection>> {
        match self {
            Self::Fields(fields) => Some(fields),
            Self::All | Self::Element(_) => None,
        }
    }

    /// The selection of field `name`: `All` under `All`, `None` if the field is not needed.
    /// `All` for an element selection, which has no fields.
    pub fn child(&self, name: &str) -> Option<&FieldSelection> {
        match self {
            Self::All | Self::Element(_) => Some(&ALL),
            Self::Fields(fields) => fields.get(name),
        }
    }

    /// The selection of the element of a list: `All` unless an element selection.
    pub fn element_selection(&self) -> &FieldSelection {
        match self {
            Self::Element(element) => element,
            Self::All | Self::Fields(_) => &ALL,
        }
    }

    /// Merges the needs of another consumer of the same value into `self`. `All` absorbs
    /// everything, field maps merge key by key and validity-only is absorbed by any selection.
    pub fn union(&mut self, other: FieldSelection) {
        match (&mut *self, other) {
            (Self::All, _) => {},
            (_, Self::All) => *self = Self::All,
            (Self::Fields(fields), Self::Fields(other)) => {
                if fields.is_empty() {
                    *fields = other;
                    return;
                }

                for (name, selection) in other.into_iter() {
                    insert_merged(fields, name, selection);
                }
            },
            (Self::Element(element), Self::Element(other)) => element.union(*other),
            // Validity-only is absorbed by an element selection, which reads the validity too.
            (Self::Element(_), Self::Fields(other)) if other.is_empty() => {},
            (Self::Fields(fields), other @ Self::Element(_)) if fields.is_empty() => *self = other,
            // Field and element selections of one value: not from one dtype, so read it whole.
            (Self::Element(_), Self::Fields(_)) | (Self::Fields(_), Self::Element(_)) => {
                *self = Self::All
            },
        }
    }

    /// Unions field `name` with `selection` into `self`. Does nothing under `All`; makes an
    /// element selection `All`, as it has no fields.
    pub fn insert_child(&mut self, name: PlSmallStr, selection: FieldSelection) {
        match self {
            Self::Fields(fields) => insert_merged(fields, name, selection),
            Self::Element(_) => *self = Self::All,
            Self::All => {},
        }
    }

    /// Makes the selection canonical for `dtype`: non-struct values and structs with every field
    /// selected whole become `All`, and fields that `dtype` does not have are dropped.
    pub fn normalize(self, dtype: &DataType) -> Self {
        // Validity-only of a list or array: any leaf under it carries its offsets and validity,
        // so its element is needed for validity only too.
        if let Self::Fields(fields) = &self
            && fields.is_empty()
            && (dtype.is_list() || dtype.is_array())
        {
            return Self::element(Self::validity_only()).normalize(dtype);
        }

        if let Self::Element(element) = &self {
            return match dtype {
                DataType::List(inner) => Self::element(element.as_ref().clone().normalize(inner)),
                #[cfg(feature = "dtype-array")]
                DataType::Array(inner, _) => {
                    Self::element(element.as_ref().clone().normalize(inner))
                },
                _ => Self::All,
            };
        }

        #[cfg(feature = "dtype-struct")]
        if let (Self::Fields(fields), DataType::Struct(dtype_fields)) = (&self, dtype) {
            let mut out = PlIndexMap::with_capacity_and_hasher(fields.len(), Default::default());
            let mut all_whole = true;

            for field in dtype_fields {
                match fields.get(field.name()) {
                    Some(selection) => {
                        let selection = selection.clone().normalize(field.dtype());
                        all_whole &= selection.is_all();
                        out.insert(field.name().clone(), selection);
                    },
                    None => all_whole = false,
                }
            }

            return if all_whole {
                Self::All
            } else {
                Self::Fields(Box::new(out))
            };
        }

        let _ = dtype;
        Self::All
    }

    /// The dtype with only the selected fields, in the order of `dtype`.
    pub fn prune_dtype(&self, dtype: &DataType) -> DataType {
        match (self, dtype) {
            #[cfg(feature = "dtype-struct")]
            (Self::Fields(fields), DataType::Struct(dtype_fields)) => DataType::Struct(
                dtype_fields
                    .iter()
                    .filter_map(|field| {
                        let selection = fields.get(field.name())?;
                        Some(Field::new(
                            field.name().clone(),
                            selection.prune_dtype(field.dtype()),
                        ))
                    })
                    .collect(),
            ),
            (Self::Element(element), DataType::List(inner)) => {
                DataType::List(Box::new(element.prune_dtype(inner)))
            },
            #[cfg(feature = "dtype-array")]
            (Self::Element(element), DataType::Array(inner, width)) => {
                DataType::Array(Box::new(element.prune_dtype(inner)), *width)
            },
            _ => dtype.clone(),
        }
    }

    /// `inner` applied within `self`: the selection relative to the unpruned value, for a
    /// selection `inner` made on a value already pruned to `self`.
    pub fn compose(&self, inner: &FieldSelection) -> FieldSelection {
        match (self, inner) {
            (Self::All, inner) => inner.clone(),
            (outer, Self::All) => outer.clone(),
            (Self::Fields(outer), Self::Fields(inner)) => Self::Fields(Box::new(
                inner
                    .iter()
                    .filter_map(|(name, inner)| {
                        Some((name.clone(), outer.get(name)?.compose(inner)))
                    })
                    .collect(),
            )),
            (Self::Element(outer), Self::Element(inner)) => Self::element(outer.compose(inner)),
            // Not from one dtype: `self` is at least what is needed.
            (outer @ (Self::Fields(_) | Self::Element(_)), _) => outer.clone(),
        }
    }
}

fn insert_merged(
    fields: &mut PlIndexMap<PlSmallStr, FieldSelection>,
    name: PlSmallStr,
    selection: FieldSelection,
) {
    match fields.entry(name) {
        indexmap::map::Entry::Occupied(mut entry) => entry.get_mut().union(selection),
        indexmap::map::Entry::Vacant(entry) => {
            entry.insert(selection);
        },
    }
}

impl fmt::Display for FieldSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fields = match self {
            Self::All => return Ok(()),
            Self::Element(element) => return write!(f, "[]{element}"),
            Self::Fields(fields) => fields,
        };

        f.write_str("{")?;
        for (i, (name, selection)) in fields.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{name}{selection}")?;
        }
        f.write_str("}")
    }
}

/// The struct fields a scan reads: an entry for each column that is not read whole.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct ScanFieldSelection(pub PlIndexMap<PlSmallStr, FieldSelection>);

impl Hash for ScanFieldSelection {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_unordered(&self.0, state);
    }
}

impl ScanFieldSelection {
    /// The selection of column `name`.
    pub fn get(&self, name: &str) -> &FieldSelection {
        self.0.get(name).unwrap_or(&ALL)
    }
}

impl fmt::Display for ScanFieldSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (name, selection)) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{name}{selection}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields(names: &[(&str, FieldSelection)]) -> FieldSelection {
        FieldSelection::Fields(Box::new(
            names
                .iter()
                .map(|(n, s)| (PlSmallStr::from_str(n), s.clone()))
                .collect(),
        ))
    }

    #[test]
    fn test_union() {
        use FieldSelection as FS;

        let a = fields(&[("a", FS::All)]);
        let b = fields(&[("b", FS::All)]);

        let mut s = a.clone();
        s.union(FS::All);
        assert_eq!(s, FS::All);

        let mut s = FS::All;
        s.union(a.clone());
        assert_eq!(s, FS::All);

        let mut s = a.clone();
        s.union(b.clone());
        assert_eq!(s, fields(&[("a", FS::All), ("b", FS::All)]));

        let mut s = FS::validity_only();
        s.union(a.clone());
        assert_eq!(s, a);

        let mut s = a.clone();
        s.union(FS::validity_only());
        assert_eq!(s, a);

        let mut s = fields(&[("a", a.clone())]);
        s.union(fields(&[("a", FS::All)]));
        assert_eq!(s, fields(&[("a", FS::All)]));
    }

    #[cfg(feature = "dtype-struct")]
    #[test]
    fn test_normalize_and_prune() {
        use FieldSelection as FS;

        let inner = DataType::Struct(vec![
            Field::new("x".into(), DataType::Int64),
            Field::new("y".into(), DataType::String),
        ]);
        let dtype = DataType::Struct(vec![
            Field::new("a".into(), DataType::Int64),
            Field::new("b".into(), inner.clone()),
        ]);

        let s = fields(&[("b", fields(&[("y", FS::All)]))]);
        assert_eq!(
            s.prune_dtype(&dtype),
            DataType::Struct(vec![Field::new(
                "b".into(),
                DataType::Struct(vec![Field::new("y".into(), DataType::String)])
            )])
        );

        let s = fields(&[
            ("b", fields(&[("x", FS::All), ("y", FS::All)])),
            ("a", FS::All),
        ]);
        assert_eq!(s.normalize(&dtype), FS::All);

        assert_eq!(FS::validity_only().normalize(&DataType::Int64), FS::All);
        assert_eq!(FS::validity_only().normalize(&dtype), FS::validity_only());
        assert_eq!(
            FS::validity_only().prune_dtype(&dtype),
            DataType::Struct(vec![])
        );
    }

    #[test]
    fn test_hash_ignores_field_order() {
        use std::hash::BuildHasher;

        use FieldSelection as FS;

        let ab = fields(&[("a", FS::All), ("b", fields(&[("x", FS::All)]))]);
        let ba = fields(&[("b", fields(&[("x", FS::All)])), ("a", FS::All)]);
        assert_eq!(ab, ba);
        let state = std::hash::RandomState::new();
        assert_eq!(state.hash_one(&ab), state.hash_one(&ba));

        let to_scan = |s: FieldSelection| ScanFieldSelection(s.fields().unwrap().clone());
        let (ab, ba) = (to_scan(ab), to_scan(ba));
        assert_eq!(ab, ba);
        assert_eq!(state.hash_one(&ab), state.hash_one(&ba));
        assert_ne!(
            state.hash_one(&ab),
            state.hash_one(to_scan(fields(&[("a", FS::All), ("b", FS::All)])))
        );
    }

    #[cfg(feature = "dtype-struct")]
    #[test]
    fn test_element() {
        use FieldSelection as FS;

        let a = fields(&[("a", FS::All)]);
        let b = fields(&[("b", FS::All)]);
        assert_eq!(FS::element(FS::All), FS::All);

        // Union: element selections merge; `All` and validity-only as for fields.
        let mut s = FS::element(a.clone());
        s.union(FS::element(b.clone()));
        assert_eq!(s, FS::element(fields(&[("a", FS::All), ("b", FS::All)])));
        let mut s = FS::element(a.clone());
        s.union(FS::validity_only());
        assert_eq!(s, FS::element(a.clone()));
        let mut s = FS::validity_only();
        s.union(FS::element(a.clone()));
        assert_eq!(s, FS::element(a.clone()));
        let mut s = FS::element(a.clone());
        s.union(FS::All);
        assert_eq!(s, FS::All);
        let mut s = FS::element(a.clone());
        s.union(b.clone());
        assert_eq!(s, FS::All);
        let mut s = FS::element(a.clone());
        s.insert_child("b".into(), FS::All);
        assert_eq!(s, FS::All);

        let element = DataType::Struct(vec![
            Field::new("a".into(), DataType::Int64),
            Field::new("b".into(), DataType::String),
        ]);
        let list = DataType::List(Box::new(element.clone()));

        assert_eq!(
            FS::element(a.clone()).normalize(&list),
            FS::element(a.clone())
        );
        assert_eq!(
            FS::element(fields(&[("a", FS::All), ("b", FS::All)])).normalize(&list),
            FS::All
        );
        // Not a list, or a list of non-structs: read whole.
        assert_eq!(FS::element(a.clone()).normalize(&element), FS::All);
        assert_eq!(
            FS::element(FS::validity_only()).normalize(&DataType::List(Box::new(DataType::Int64))),
            FS::All
        );
        assert_eq!(a.clone().normalize(&list), FS::All);
        // Validity-only of a list: its element for validity only, down to the first struct.
        assert_eq!(
            FS::validity_only().normalize(&list),
            FS::element(FS::validity_only())
        );
        assert_eq!(
            FS::validity_only().normalize(&DataType::List(Box::new(list.clone()))),
            FS::element(FS::element(FS::validity_only()))
        );
        assert_eq!(
            FS::validity_only().normalize(&DataType::List(Box::new(DataType::Int64))),
            FS::All
        );

        assert_eq!(
            FS::element(b.clone()).prune_dtype(&list),
            DataType::List(Box::new(DataType::Struct(vec![Field::new(
                "b".into(),
                DataType::String
            )])))
        );
        #[cfg(feature = "dtype-array")]
        assert_eq!(
            FS::element(FS::validity_only()).prune_dtype(&DataType::Array(Box::new(element), 3)),
            DataType::Array(Box::new(DataType::Struct(vec![])), 3)
        );

        let outer = FS::element(fields(&[("a", FS::All), ("b", a.clone())]));
        assert_eq!(
            outer.compose(&FS::element(fields(&[("b", FS::All)]))),
            FS::element(fields(&[("b", a.clone())]))
        );
        assert_eq!(outer.compose(&b), outer);

        assert_eq!(FS::element(a.clone()).to_string(), "[]{a}");
        assert_eq!(
            fields(&[("s", FS::element(FS::validity_only()))]).to_string(),
            "{s[]{}}"
        );

        use std::hash::BuildHasher;
        let state = std::hash::RandomState::new();
        assert_ne!(state.hash_one(FS::element(a.clone())), state.hash_one(&a));
        assert_eq!(
            state.hash_one(FS::element(fields(&[("a", FS::All), ("b", FS::All)]))),
            state.hash_one(FS::element(fields(&[("b", FS::All), ("a", FS::All)])))
        );
    }

    #[test]
    fn test_compose() {
        use FieldSelection as FS;

        let outer = fields(&[("a", FS::All), ("b", fields(&[("x", FS::All)]))]);
        assert_eq!(outer.compose(&FS::All), outer);
        assert_eq!(
            outer.compose(&fields(&[("b", FS::All)])),
            fields(&[("b", fields(&[("x", FS::All)]))])
        );
    }
}
