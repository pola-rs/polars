use std::fmt::{self, Write};
use std::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Sub, SubAssign,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[cfg(feature = "dsl-schema")]
impl schemars::JsonSchema for TimeUnitSet {
    fn schema_name() -> String {
        "TimeUnitSet".to_owned()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "TimeUnitSet"))
    }

    fn json_schema(_generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        use serde_json::{Map, Value};

        let name_to_bits: Map<String, Value> = Self::all()
            .iter_names()
            .map(|(name, flag)| (name.to_owned(), flag.bits().into()))
            .collect();

        schemars::schema::Schema::Object(schemars::schema::SchemaObject {
            instance_type: Some(schemars::schema::InstanceType::String.into()),
            format: Some("bitflags".to_owned()),
            extensions: schemars::Map::from_iter([
                // Add a map of flag names and bit patterns to detect schema changes
                ("bitflags".to_owned(), Value::Object(name_to_bits)),
            ]),
            ..Default::default()
        })
    }
}

bitflags::bitflags! {
    #[repr(transparent)]
    #[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
    #[cfg_attr(
        feature = "serde",
        derive(serde::Serialize, serde::Deserialize)
    )]
    pub struct TimeUnitSet: u8 {
        const NANO_SECONDS = 0x01;
        const MICRO_SECONDS = 0x02;
        const MILLI_SECONDS = 0x04;
    }
}

impl From<TimeUnit> for TimeUnitSet {
    fn from(value: TimeUnit) -> Self {
        match value {
            TimeUnit::Nanoseconds => TimeUnitSet::NANO_SECONDS,
            TimeUnit::Microseconds => TimeUnitSet::MICRO_SECONDS,
            TimeUnit::Milliseconds => TimeUnitSet::MILLI_SECONDS,
        }
    }
}

impl fmt::Display for TimeUnitSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_all() {
            f.write_str("*")?;
        } else {
            if self.bits().count_ones() != 1 {
                f.write_str("[")?;
            }

            if self.contains(TimeUnitSet::NANO_SECONDS) {
                f.write_str("'ns'")?;
                if self.intersects(TimeUnitSet::MICRO_SECONDS | TimeUnitSet::MILLI_SECONDS) {
                    f.write_str(", ")?;
                }
            }
            if self.contains(TimeUnitSet::MICRO_SECONDS) {
                f.write_str("'ms'")?;
                if self.contains(TimeUnitSet::MILLI_SECONDS) {
                    f.write_str(", ")?;
                }
            }
            if self.contains(TimeUnitSet::MILLI_SECONDS) {
                f.write_str("'us'")?;
            }

            if self.bits().count_ones() != 1 {
                f.write_str("]")?;
            }
        }

        Ok(())
    }
}

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
    
    // These 2 return their inputs in given order not in schema order.
    ByName {
        names: Arc<[PlSmallStr]>,
        strict: bool,
    },
    ByIndex {
        indices: Arc<[i64]>,
        strict: bool,
    },

    ByDType(Arc<[DataType]>),
    Matches(PlSmallStr),
    Wildcard,
    Empty,

    Integer,
    UnsignedInteger,
    SignedInteger,
    Float,
    Categorical,
    Decimal,
    Numeric,
    Temporal,
    /// Selector for `DataType::Datetime` with optional matching on TimeUnit and TimeZone.
    Datetime(TimeUnitSet, Option<Vec<Option<TimeZone>>>),
    /// Selector for `DataType::Duration` with optional matching on TimeUnit.
    Duration(TimeUnitSet),
    Object,
}

fn dtype_selector(
    schema: &Schema,
    ignored_columns: &PlHashSet<PlSmallStr>,
    f: impl Fn(&DataType) -> bool,
) -> PlIndexSet<PlSmallStr> {
    PlIndexSet::from_iter(
        schema
            .iter()
            .filter(|(name, dtype)| !ignored_columns.contains(*name) && f(dtype))
            .map(|(name, _)| name.clone()),
    )
}

impl Selector {
    /// Turns the selector into an ordered set of selected columns from the schema.
    ///
    /// - The order of the columns corresponds to the order in the schema.
    /// - Column names in `ignored_columns` are only used if they are explicitly mentioned by a
    /// `ByName` or `Nth`.
    pub fn into_columns(
        &self,
        schema: &Schema,
        ignored_columns: &PlHashSet<PlSmallStr>,
    ) -> PolarsResult<PlIndexSet<PlSmallStr>> {
        let out = match self {
            Selector::Union(lhs, rhs) => {
                let mut lhs = lhs.into_columns(schema, ignored_columns)?;
                let rhs = rhs.into_columns(schema, ignored_columns)?;
                lhs.extend(rhs);
                lhs.sort_unstable_by(|l, r| {
                    schema
                        .index_of(l)
                        .unwrap()
                        .cmp(&schema.index_of(r).unwrap())
                });
                lhs
            },
            Selector::Difference(lhs, rhs) => {
                let mut lhs = lhs.into_columns(schema, ignored_columns)?;
                let rhs = rhs.into_columns(schema, ignored_columns)?;
                lhs.retain(|n| !rhs.contains(n));
                lhs.sort_unstable_by(|l, r| {
                    schema
                        .index_of(l)
                        .unwrap()
                        .cmp(&schema.index_of(r).unwrap())
                });
                lhs
            },
            Selector::ExclusiveOr(lhs, rhs) => {
                let mut lhs = lhs.into_columns(schema, ignored_columns)?;
                let mut rhs = rhs.into_columns(schema, ignored_columns)?;
                rhs.retain(|n| !lhs.contains(n));
                lhs.extend(rhs);
                lhs.sort_unstable_by(|l, r| {
                    schema
                        .index_of(l)
                        .unwrap()
                        .cmp(&schema.index_of(r).unwrap())
                });
                lhs
            },
            Selector::Intersect(lhs, rhs) => {
                let mut lhs = lhs.into_columns(schema, ignored_columns)?;
                let rhs = rhs.into_columns(schema, ignored_columns)?;
                lhs.retain(|n| rhs.contains(n));
                lhs.sort_unstable_by(|l, r| {
                    schema
                        .index_of(l)
                        .unwrap()
                        .cmp(&schema.index_of(r).unwrap())
                });
                lhs
            },
            Selector::Exclude(input, excludes) => {
                let mut out = input.into_columns(schema, ignored_columns)?;
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
                            out.retain(|name| schema.get(name).unwrap() != excluded_dt)
                        },
                    }
                }
                out.sort_unstable_by(|l, r| {
                    schema
                        .index_of(l)
                        .unwrap()
                        .cmp(&schema.index_of(r).unwrap())
                });
                out
            },

            Selector::ByDType(data_types) => {
                let dtypes = PlHashSet::from_iter(data_types.iter().cloned());
                dtype_selector(schema, ignored_columns, |dtype| dtypes.contains(dtype))
            },
            Selector::ByName { names, strict } => {
                let mut out = PlIndexSet::with_capacity(names.len());
                for name in names.iter() {
                    polars_ensure!(!strict || schema.contains(name), col_not_found = name);
                    out.insert(name.clone());
                }
                out
            },
            Selector::ByIndex { indices, strict } => {
                let mut out = PlIndexSet::with_capacity(indices.len());
                let mut set = PlIndexSet::with_capacity(indices.len());
                for &idx in indices.iter() {
                    let Some(idx) = idx.negative_to_usize(schema.len()) else {
                        polars_ensure!(!strict, ColumnNotFound: "cannot get the {idx}-th column when schema has {} columns", schema.len());
                        continue;
                    };
                    let (name, _) = schema.get_at_index(idx).unwrap();
                    if !set.insert(idx) {
                        polars_bail!(Duplicate: "duplicate column name {name}");
                    }
                    out.insert(name.clone());
                }
                out
            },
            Selector::Matches(regex_str) => {
                let re = polars_utils::regex_cache::compile_regex(&regex_str).unwrap();
                PlIndexSet::from_iter(
                    schema
                        .iter_names()
                        .filter(|name| !ignored_columns.contains(*name) && re.is_match(name))
                        .cloned(),
                )
            },
            Selector::Wildcard => PlIndexSet::from_iter(
                schema
                    .iter_names()
                    .filter(|name| !ignored_columns.contains(*name))
                    .cloned(),
            ),
            Selector::Empty => Default::default(),

            Selector::Float => dtype_selector(schema, ignored_columns, |dtype| dtype.is_float()),
            Selector::Integer => {
                dtype_selector(schema, ignored_columns, |dtype| dtype.is_integer())
            },
            Selector::SignedInteger => {
                dtype_selector(schema, ignored_columns, |dtype| dtype.is_signed_integer())
            },
            Selector::UnsignedInteger => {
                dtype_selector(schema, ignored_columns, |dtype| dtype.is_unsigned_integer())
            },
            Selector::Categorical => {
                dtype_selector(schema, ignored_columns, |dtype| dtype.is_categorical())
            },
            Selector::Decimal => {
                dtype_selector(schema, ignored_columns, |dtype| dtype.is_decimal())
            },
            Selector::Numeric => {
                dtype_selector(schema, ignored_columns, |dtype| dtype.is_numeric())
            },
            Selector::Temporal => {
                dtype_selector(schema, ignored_columns, |dtype| dtype.is_temporal())
            },
            Selector::Datetime(selector_tu, selector_tz) => {
                let selector_tz = selector_tz.as_ref().map(|tz| PlIndexSet::from_iter(tz));
                dtype_selector(schema, ignored_columns, |dtype| {
                    let DataType::Datetime(tu, tz) = dtype else {
                        return false;
                    };

                    selector_tu.contains(TimeUnitSet::from(*tu))
                        && selector_tz.as_ref().is_none_or(|stz| stz.contains(tz))
                })
            },
            Selector::Duration(selector_tu) => dtype_selector(schema, ignored_columns, |dtype| {
                let DataType::Duration(tu) = dtype else {
                    return false;
                };

                selector_tu.contains(TimeUnitSet::from(*tu))
            }),
            Selector::Object => dtype_selector(schema, ignored_columns, |dtype| dtype.is_object()),
        };
        Ok(out)
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

    pub fn as_expr(self) -> Expr {
        self.into()
    }
}

pub fn is_regex_projection(name: &str) -> bool {
    name.starts_with('^') && name.ends_with('$')
}

impl BitOr for Selector {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Selector::Union(Arc::new(self), Arc::new(rhs))
    }
}

impl BitOrAssign for Selector {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = Selector::Union(
            Arc::new(std::mem::replace(self, Self::Empty)),
            Arc::new(rhs),
        )
    }
}

impl BitAnd for Selector {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Selector::Intersect(Arc::new(self), Arc::new(rhs))
    }
}

impl BitAndAssign for Selector {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = Selector::Intersect(
            Arc::new(std::mem::replace(self, Self::Empty)),
            Arc::new(rhs),
        )
    }
}

impl BitXor for Selector {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        Selector::ExclusiveOr(Arc::new(self), Arc::new(rhs))
    }
}

impl BitXorAssign for Selector {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = Selector::ExclusiveOr(
            Arc::new(std::mem::replace(self, Self::Empty)),
            Arc::new(rhs),
        )
    }
}

impl Sub for Selector {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Selector::Difference(Arc::new(self), Arc::new(rhs))
    }
}

impl SubAssign for Selector {
    fn sub_assign(&mut self, rhs: Self) {
        *self = Selector::Difference(
            Arc::new(std::mem::replace(self, Self::Empty)),
            Arc::new(rhs),
        )
    }
}

impl Not for Selector {
    type Output = Self;
    fn not(self) -> Self::Output {
        Self::Wildcard - self
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

            Selector::ByDType(dtypes) => {
                use DataType as D;
                match dtypes.as_ref() {
                    [D::Boolean] => f.write_str("cs.boolean()"),
                    [D::Binary] => f.write_str("cs.binary()"),
                    [D::Time] => f.write_str("cs.time()"),
                    [D::Date] => f.write_str("cs.date()"),
                    [D::String] => f.write_str("cs.string()"),
                    _ => write!(f, "cs.by_dtype({:?})", dtypes),
                }
            },
            Selector::ByName { names, strict } => {
                f.write_str("cs.by_name(")?;

                for e in names.iter() {
                    write!(f, "'{e}', ")?;
                }

                write!(f, "strict={strict})")
            },
            Selector::ByIndex { indices, strict } if indices.as_ref() == &[0] => {
                write!(f, "cs.first(strict={strict})")
            },
            Selector::ByIndex { indices, strict } if indices.as_ref() == &[-1] => {
                write!(f, "cs.last(strict={strict})")
            },
            Selector::ByIndex { indices, strict } if indices.len() == 1 => {
                write!(f, "cs.nth({}, strict={strict})", indices[0])
            },
            Selector::ByIndex { indices, strict } => {
                write!(f, "cs.by_index({:?}, strict={strict})", indices.as_ref())
            },
            Selector::Matches(s) => write!(f, "cs.matches(\"{s}\")"),

            Selector::Float => write!(f, "cs.float()"),
            Selector::Integer => write!(f, "cs.integer()"),
            Selector::SignedInteger => write!(f, "cs.signed_integer()"),
            Selector::UnsignedInteger => write!(f, "cs.unsigned_integer()"),
            Selector::Categorical => write!(f, "cs.categorical()"),
            Selector::Numeric => write!(f, "cs.numeric()"),
            Selector::Decimal => write!(f, "cs.decimal()"),
            Selector::Temporal => write!(f, "cs.temporal()"),
            Selector::Datetime(tu, tz) => {
                write!(f, "cs.datetime(time_unit={tu}, time_zone=")?;
                match tz {
                    None => f.write_str("*")?,
                    Some(tz) => {
                        if tz.len() > 1 {
                            f.write_str("[")?;
                        }

                        if let Some(e) = tz.first() {
                            match e {
                                None => f.write_str("None"),
                                Some(e) => write!(f, "'{e}'"),
                            }?;
                            for e in &tz[1..] {
                                match e {
                                    None => f.write_str("None"),
                                    Some(e) => write!(f, "'{e}'"),
                                }?;
                            }
                        }

                        if tz.len() > 1 {
                            f.write_str("]")?;
                        }
                    },
                }
                f.write_str(")")
            },
            Selector::Duration(tu) => {
                write!(f, "cs.duration(time_unit={tu})")
            },
            Selector::Object => write!(f, "cs.object()"),
            Selector::Wildcard => f.write_str("cs.all()"),
            Selector::Empty => f.write_str("cs.empty()"),
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
