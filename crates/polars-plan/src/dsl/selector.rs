use std::fmt::{self, Write};
use std::ops::{Add, BitAnd, BitOr, BitXor, Sub};

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

fn merge_sorted_columns(
    lhs: PlIndexSet<PlSmallStr>,
    rhs: PlIndexSet<PlSmallStr>,
    schema: &Schema,
    xor: bool,
) -> PlIndexSet<PlSmallStr> {
    if lhs.is_empty() {
        return rhs;
    }
    if rhs.is_empty() {
        return lhs;
    }

    let mut out = PlIndexSet::with_capacity(lhs.len() + rhs.len());
    let mut li = lhs.into_iter();
    let mut ri = rhs.into_iter();

    let mut lv = li.next().unwrap();
    let mut rv = ri.next().unwrap();

    let mut l = schema.index_of(&lv).unwrap();
    let mut r = schema.index_of(&rv).unwrap();

    loop {
        while l == r {
            if !xor {
                out.insert(lv);
            }

            let Some(n) = li.next() else {
                out.insert(rv);
                out.extend(ri);
                return out;
            };
            lv = n;
            l = schema.index_of(&lv).unwrap();
            let Some(n) = ri.next() else {
                out.insert(lv);
                out.extend(li);
                return out;
            };
            rv = n;
            r = schema.index_of(&rv).unwrap();
        }

        if l < r {
            out.insert(lv);
            let Some(n) = li.next() else {
                out.insert(rv);
                out.extend(ri);
                return out;
            };
            lv = n;
            l = schema.index_of(&lv).unwrap();
        } else {
            out.insert(rv);
            let Some(n) = ri.next() else {
                out.insert(lv);
                out.extend(li);
                return out;
            };
            rv = n;
            r = schema.index_of(&rv).unwrap();
        }
    }
}

impl Selector {
    pub fn into_columns(&self, schema: &Schema) -> PolarsResult<PlIndexSet<PlSmallStr>> {
        let out = match self {
            Selector::Union(lhs, rhs) => {
                let lhs = lhs.into_columns(schema)?;
                let rhs = rhs.into_columns(schema)?;
                merge_sorted_columns(lhs, rhs, schema, false)
            },
            Selector::Difference(lhs, rhs) => {
                let mut lhs = lhs.into_columns(schema)?;
                let rhs = rhs.into_columns(schema)?;
                lhs.retain(|n| !rhs.contains(n));
                lhs
            },
            Selector::ExclusiveOr(lhs, rhs) => {
                let lhs = lhs.into_columns(schema)?;
                let rhs = rhs.into_columns(schema)?;
                merge_sorted_columns(lhs, rhs, schema, true)
            },
            Selector::Intersect(lhs, rhs) => {
                let mut lhs = lhs.into_columns(schema)?;
                let rhs = rhs.into_columns(schema)?;
                lhs.retain(|n| rhs.contains(n));
                lhs
            },
            Selector::Exclude(input, excludes) => {
                let mut out = input.into_columns(schema)?;
                dbg!(&out);
                dbg!(&excludes);
                for exclude in excludes.iter() {
                    // @PERF: This is quadratic
                    match exclude {
                        #[cfg(feature = "regex")]
                        Excluded::Name(regex_str) if is_regex_projection(regex_str) => {
                            let re = polars_utils::regex_cache::compile_regex(regex_str)
                                .map_err(|e| polars_err!(ComputeError: "invalid regex {}", e))?;
                            out.retain(|name| !re.is_match(name));
                        },
                        Excluded::Name(excluded_name) => out.retain(|name| name != excluded_name),
                        Excluded::Dtype(excluded_dt) => {
                            out.retain(|name| !dtypes_match(schema.get(name).unwrap(), excluded_dt))
                        },
                    }
                }
                dbg!(&out);
                out
            },

            Selector::WithDataTypes(data_types) => {
                // @PERF: This is quadratic
                PlIndexSet::from_iter(
                    schema
                        .iter()
                        .filter(|(_, dtype)| data_types.iter().any(|dt| dtypes_match(dtype, dt)))
                        .map(|(name, _)| name.clone()),
                )
            },
            Selector::ByName(names) => {
                let mut out = PlIndexSet::with_capacity(names.len());
                for name in names.iter() {
                    polars_ensure!(schema.contains(name), col_not_found = name);
                    out.insert(name.clone());
                }
                out.sort_unstable_by(|l, r| {
                    schema
                        .index_of(l)
                        .unwrap()
                        .cmp(&schema.index_of(r).unwrap())
                });
                out
            },
            Selector::AtIndex(indices) => {
                let mut out = PlIndexSet::with_capacity(indices.len());
                let mut set = PlIndexSet::with_capacity(indices.len());
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
                out.sort_unstable_by(|l, r| {
                    schema
                        .index_of(l)
                        .unwrap()
                        .cmp(&schema.index_of(r).unwrap())
                });
                out
            },
            Selector::Regex(regex_str) => {
                let re = polars_utils::regex_cache::compile_regex(&regex_str).unwrap();
                PlIndexSet::from_iter(
                    schema
                        .iter_names()
                        .filter(|name| re.is_match(name))
                        .cloned(),
                )
            },
            Selector::Wildcard => PlIndexSet::from_iter(schema.iter_names().cloned()),
        };
        Ok(out)
    }

    pub fn col(name: impl Into<PlSmallStr>) -> Self {
        Self::ByName([name.into()].into())
    }

    /// Exclude a column from a wildcard/regex selection.
    ///
    /// You may also use regexes in the exclude as long as they start with `^` and end with `$`.
    pub fn exclude_cols(self, columns: impl IntoVec<PlSmallStr>) -> Self {
        let v = columns.into_vec().into_iter().map(Excluded::Name).collect();
        Self::Exclude(Arc::new(self), v)
    }

    pub fn exclude_dtype<D: AsRef<[DataType]>>(self, dtypes: D) -> Self {
        let v = dtypes
            .as_ref()
            .iter()
            .map(|dt| Excluded::Dtype(dt.clone()))
            .collect();
        Self::Exclude(Arc::new(self), v)
    }

    pub fn into_expr(self) -> Expr {
        self.into()
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

impl BitOr for Selector {
    type Output = Selector;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn bitor(self, rhs: Self) -> Self::Output {
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

impl From<Selector> for Expr {
    fn from(value: Selector) -> Self {
        Expr::Selector(value)
    }
}

impl fmt::Display for Selector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Selector::Union(left, right) => write!(f, "[{left} | {right}]"),
            Selector::Difference(left, right) => write!(f, "[{left} - {right}]"),
            Selector::ExclusiveOr(left, right) => write!(f, "[{left} ^ {right}]"),
            Selector::Intersect(left, right) => write!(f, "[{left} & {right}]"),
            Selector::Exclude(input, excludes) => {
                write!(f, "{input}.exclude(")?;

                if let Some(e) = excludes.first() {
                    fmt::Display::fmt(e, f)?;
                    for e in &excludes[1..] {
                        write!(f, ", {e}")?;
                    }
                }

                f.write_char(')')
            },
            Selector::WithDataTypes(dtypes) => write!(f, "col({:?})", dtypes),
            Selector::ByName(names) => write!(f, "col({:?})", names),
            Selector::AtIndex(items) if items.as_ref() == &[0] => f.write_str("first()"),
            Selector::AtIndex(items) if items.as_ref() == &[-1] => f.write_str("last()"),
            Selector::AtIndex(items) if items.len() == 1 => write!(f, "nth({})", items[0]),
            Selector::AtIndex(items) => write!(f, "nth({:?})", items.as_ref()),
            Selector::Regex(s) => write!(f, "regex(\"{s}\")"),
            Selector::Wildcard => f.write_str("all()"),
        }
    }
}

impl fmt::Display for Excluded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Excluded::Name(name) => write!(f, "\"{name}\""),
            Excluded::Dtype(dtype) => fmt::Display::fmt(dtype, f),
        }
    }
}
