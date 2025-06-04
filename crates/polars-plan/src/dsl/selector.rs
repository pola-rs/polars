use std::ops::{Add, BitAnd, BitXor, Sub};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[derive(Clone, PartialEq, Hash, Debug, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum Selector {
    Union(Arc<Selector>, Arc<Selector>),
    Difference(Arc<Selector>, Arc<Selector>),
    ExclusiveOr(Arc<Selector>, Arc<Selector>),
    Intersect(Arc<Selector>, Arc<Selector>),
    Exclude(Arc<Selector>, Arc<[Excluded]>),

    // Leaf nodes
    WithDataTypes(Arc<[DataType]>),
    ByName(Arc<[PlSmallStr]>),
    AtIndex(Arc<[i64]>),
    Regex(PlSmallStr),
    Wildcard,
}

impl Selector {
    pub fn into_columns(&self, schema: &Schema) -> PolarsResult<PlIndexSet<PlSmallStr>> {
        let mut out = PlIndexSet::default();
        self.into_columns_impl(schema, &mut out)?;
        Ok(out)
    }

    fn into_columns_impl(
        &self,
        schema: &Schema,
        out: &mut PlIndexSet<PlSmallStr>,
    ) -> PolarsResult<()> {
        match self {
            Selector::Union(lhs, rhs) => {
                lhs.into_columns_impl(schema, out)?;
                rhs.into_columns_impl(schema, out)?;
            },
            Selector::Difference(lhs, rhs) => {
                let lhs = lhs.into_columns(schema)?;
                let rhs = rhs.into_columns(schema)?;

                out.extend(lhs.into_iter().filter(|n| !rhs.contains(n)));
            },
            Selector::ExclusiveOr(lhs, rhs) => {
                let lhs = lhs.into_columns(schema)?;
                let rhs = rhs.into_columns(schema)?;

                out.extend(lhs.iter().filter(|n| !rhs.contains(n.as_str())).cloned());
                out.extend(rhs.into_iter().filter(|n| !lhs.contains(n)));
            },
            Selector::Intersect(lhs, rhs) => {
                let lhs = lhs.into_columns(schema)?;
                let rhs = rhs.into_columns(schema)?;

                out.extend(lhs.into_iter().filter(|n| rhs.contains(n)));
            },
            Selector::Exclude(input, excludes) => {
                let mut input = input.into_columns(schema)?;
                for exclude in excludes.iter() {
                    match exclude {
                        #[cfg(feature = "regex")]
                        Excluded::Name(regex_str) if is_regex_projection(regex_str) => {
                            let re = polars_utils::regex_cache::compile_regex(regex_str)
                                .map_err(|e| polars_err!(ComputeError: "invalid regex {}", e))?;
                            input.retain(|name| !re.is_match(name));
                        },
                        Excluded::Name(excluded_name) => input.retain(|name| name != excluded_name),
                        Excluded::Dtype(excluded_dt) => input
                            .retain(|name| !dtypes_match(schema.get(name).unwrap(), excluded_dt)),
                    }
                }
            },

            Selector::WithDataTypes(data_types) => {
                let datatypes = PlHashSet::from_iter(data_types.iter());
                out.extend(
                    schema
                        .iter()
                        .filter(|(_, dtype)| datatypes.contains(dtype))
                        .map(|(name, _)| name.clone()),
                );
            },
            Selector::ByName(names) => {
                out.reserve(names.len());
                for name in names.iter() {
                    polars_ensure!(schema.contains(name), col_not_found = name);
                    out.insert(name.clone());
                }
            },
            Selector::AtIndex(indices) => {
                out.reserve(indices.len());
                let mut set = PlHashSet::with_capacity(indices.len());
                for &idx in indices.iter() {
                    let Some(idx) = idx.negative_to_usize(schema.len()) else {
                        polars_bail!(InvalidOperation: "cannot get the {idx}-th column when schema has {} columns", schema.len());
                    };
                    let (name, _) = schema.get_at_index(idx).unwrap();
                    if !set.insert(idx) {
                        polars_bail!(InvalidOperation: "duplicate column name {name}");
                    }
                    out.insert(name.clone());
                }
            },
            Selector::Regex(regex_str) => {
                let re = polars_utils::regex_cache::compile_regex(&regex_str).unwrap();
                out.extend(
                    schema
                        .iter_names()
                        .filter(|name| re.is_match(name))
                        .cloned(),
                );
            },
            Selector::Wildcard => out.extend(schema.iter_names().cloned()),
        }
        Ok(())
    }
}

pub fn is_regex_projection(name: &str) -> bool {
    name.starts_with('^') && name.ends_with('$')
}

fn dtypes_match(d1: &DataType, d2: &DataType) -> bool {
    match (d1, d2) {
        // note: allow Datetime "*" wildcard for timezones...
        (DataType::Datetime(tu_l, tz_l), DataType::Datetime(tu_r, tz_r)) => {
            tu_l == tu_r
                && (tz_l == tz_r
                    || match (tz_l, tz_r) {
                        (Some(l), Some(r)) => TimeZone::eq_wildcard_aware(l, r),
                        _ => false,
                    })
        },
        // ...but otherwise require exact match
        _ => d1 == d2,
    }
}

impl Add for Selector {
    type Output = Selector;

    fn add(self, rhs: Self) -> Self::Output {
        Selector::Union(Arc::new(self), Arc::new(rhs))
    }
}

impl BitAnd for Selector {
    type Output = Selector;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn bitand(self, rhs: Self) -> Self::Output {
        Selector::Intersect(Arc::new(self), Arc::new(rhs))
    }
}

impl BitXor for Selector {
    type Output = Selector;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Selector::ExclusiveOr(Arc::new(self), Arc::new(rhs))
    }
}

impl Sub for Selector {
    type Output = Selector;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self::Output {
        Selector::Difference(Arc::new(self), Arc::new(rhs))
    }
}

impl From<&str> for Selector {
    fn from(value: &str) -> Self {
        Selector::ByName([PlSmallStr::from_str(value)].into())
    }
}

impl From<String> for Selector {
    fn from(value: String) -> Self {
        Selector::ByName([PlSmallStr::from(value)].into())
    }
}

impl From<PlSmallStr> for Selector {
    fn from(value: PlSmallStr) -> Self {
        Selector::ByName([value].into())
    }
}
