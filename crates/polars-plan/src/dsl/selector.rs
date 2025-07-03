use std::fmt::{self, Write};
use std::ops::{BitAnd, BitOr, BitXor, Sub};

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
    WithDataTypes(Arc<[DataType]>),
    ByName(Arc<[PlSmallStr]>),
    AtIndex(Arc<[i64]>),
    Matches(PlSmallStr),
    Wildcard,

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

fn dtype_selector(schema: &Schema, f: impl Fn(&DataType) -> bool) -> PlIndexSet<PlSmallStr> {
    PlIndexSet::from_iter(
        schema
            .iter()
            .filter(|(_, dtype)| f(dtype))
            .map(|(name, _)| name.clone()),
    )
}

impl Selector {
    /// Turns the selector into an ordered set of selected columns from the schema.
    ///
    /// The order of the columns corresponds to the order in the schema.
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
                out
            },

            Selector::WithDataTypes(data_types) => {
                let dtypes = PlHashSet::from_iter(data_types.iter().cloned());
                PlIndexSet::from_iter(
                    schema
                        .iter()
                        .filter(|(_, dtype)| dtypes.contains(*dtype))
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
                        polars_bail!(ColumnNotFound: "cannot get the {idx}-th column when schema has {} columns", schema.len());
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
            Selector::Matches(regex_str) => {
                let re = polars_utils::regex_cache::compile_regex(&regex_str).unwrap();
                PlIndexSet::from_iter(
                    schema
                        .iter_names()
                        .filter(|name| re.is_match(name))
                        .cloned(),
                )
            },
            Selector::Wildcard => PlIndexSet::from_iter(schema.iter_names().cloned()),

            Selector::Float => dtype_selector(schema, |dtype| dtype.is_float()),
            Selector::Integer => dtype_selector(schema, |dtype| dtype.is_integer()),
            Selector::SignedInteger => dtype_selector(schema, |dtype| dtype.is_signed_integer()),
            Selector::UnsignedInteger => {
                dtype_selector(schema, |dtype| dtype.is_unsigned_integer())
            },
            Selector::Categorical => dtype_selector(schema, |dtype| dtype.is_categorical()),
            Selector::Decimal => dtype_selector(schema, |dtype| dtype.is_decimal()),
            Selector::Numeric => dtype_selector(schema, |dtype| dtype.is_numeric()),
            Selector::Temporal => dtype_selector(schema, |dtype| dtype.is_temporal()),
            Selector::Datetime(selector_tu, selector_tz) => {
                let selector_tz = selector_tz.as_ref().map(|tz| PlIndexSet::from_iter(tz));
                dtype_selector(schema, |dtype| {
                    let DataType::Datetime(tu, tz) = dtype else {
                        return false;
                    };

                    selector_tu.contains(TimeUnitSet::from(*tu))
                        && selector_tz.as_ref().is_none_or(|stz| stz.contains(tz))
                })
            },
            Selector::Duration(selector_tu) => dtype_selector(schema, |dtype| {
                let DataType::Duration(tu) = dtype else {
                    return false;
                };

                selector_tu.contains(TimeUnitSet::from(*tu))
            }),
            Selector::Object => dtype_selector(schema, |dtype| dtype.is_object()),
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
            Selector::WithDataTypes(dtypes) => {
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
            Selector::ByName(names) => {
                f.write_str("cs.by_name(")?;

                if let Some(e) = names.first() {
                    write!(f, "'{e}'")?;
                    for e in &names[1..] {
                        write!(f, ", '{e}'")?;
                    }
                }

                f.write_str(")")
            },
            Selector::AtIndex(items) if items.as_ref() == &[0] => f.write_str("cs.first()"),
            Selector::AtIndex(items) if items.as_ref() == &[-1] => f.write_str("cs.last()"),
            Selector::AtIndex(items) if items.len() == 1 => write!(f, "cs.nth({})", items[0]),
            Selector::AtIndex(items) => write!(f, "cs.nth({:?})", items.as_ref()),
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
