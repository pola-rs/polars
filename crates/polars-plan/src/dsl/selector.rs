use std::fmt;
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

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum TimeZoneSet {
    Any,
    AnySet,
    Unset,
    UnsetOrAnyOf(Arc<[TimeZone]>),
    AnyOf(Arc<[TimeZone]>),
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
                f.write_str("'us'")?;
                if self.contains(TimeUnitSet::MILLI_SECONDS) {
                    f.write_str(", ")?;
                }
            }
            if self.contains(TimeUnitSet::MILLI_SECONDS) {
                f.write_str("'ms'")?;
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
pub enum DataTypeSelector {
    Union(Arc<DataTypeSelector>, Arc<DataTypeSelector>),
    Difference(Arc<DataTypeSelector>, Arc<DataTypeSelector>),
    ExclusiveOr(Arc<DataTypeSelector>, Arc<DataTypeSelector>),
    Intersect(Arc<DataTypeSelector>, Arc<DataTypeSelector>),

    Wildcard,
    Empty,

    AnyOf(Arc<[DataType]>),

    Integer,
    UnsignedInteger,
    SignedInteger,
    Float,

    Enum,
    Categorical,

    Nested,
    List(Option<Arc<DataTypeSelector>>),
    Array(Option<Arc<DataTypeSelector>>, Option<usize>),
    Struct,

    Decimal,
    Numeric,
    Temporal,
    /// Selector for `DataType::Datetime` with optional matching on TimeUnit and TimeZone.
    Datetime(TimeUnitSet, TimeZoneSet),
    /// Selector for `DataType::Duration` with optional matching on TimeUnit.
    Duration(TimeUnitSet),
    Object,
}

#[derive(Clone, PartialEq, Hash, Debug, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum Selector {
    Union(Arc<Selector>, Arc<Selector>),
    Difference(Arc<Selector>, Arc<Selector>),
    ExclusiveOr(Arc<Selector>, Arc<Selector>),
    Intersect(Arc<Selector>, Arc<Selector>),

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

    Matches(PlSmallStr),
    ByDType(DataTypeSelector),

    Wildcard,
    Empty,
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
    ///   `ByName` or `Nth`.
    pub fn into_columns(
        &self,
        schema: &Schema,
        ignored_columns: &PlHashSet<PlSmallStr>,
    ) -> PolarsResult<PlIndexSet<PlSmallStr>> {
        let out = match self {
            Self::Union(lhs, rhs) => {
                let mut lhs = lhs.into_columns(schema, ignored_columns)?;
                let rhs = rhs.into_columns(schema, ignored_columns)?;
                lhs.extend(rhs);
                sort_schema_order(&mut lhs, schema);
                lhs
            },
            Self::Difference(lhs, rhs) => {
                let mut lhs = lhs.into_columns(schema, ignored_columns)?;
                let rhs = rhs.into_columns(schema, ignored_columns)?;
                lhs.retain(|n| !rhs.contains(n));
                sort_schema_order(&mut lhs, schema);
                lhs
            },
            Self::ExclusiveOr(lhs, rhs) => {
                let lhs = lhs.into_columns(schema, ignored_columns)?;
                let rhs = rhs.into_columns(schema, ignored_columns)?;
                let mut out = PlIndexSet::with_capacity(lhs.len() + rhs.len());
                out.extend(lhs.iter().filter(|n| !rhs.contains(*n)).cloned());
                out.extend(rhs.into_iter().filter(|n| !lhs.contains(n)));
                sort_schema_order(&mut out, schema);
                out
            },
            Self::Intersect(lhs, rhs) => {
                let mut lhs = lhs.into_columns(schema, ignored_columns)?;
                let rhs = rhs.into_columns(schema, ignored_columns)?;
                lhs.retain(|n| rhs.contains(n));
                sort_schema_order(&mut lhs, schema);
                lhs
            },

            Self::ByDType(dts) => dts.into_columns(schema, ignored_columns)?,
            Self::ByName { names, strict } => {
                let mut out = PlIndexSet::with_capacity(names.len());
                for name in names.iter() {
                    if schema.contains(name) {
                        out.insert(name.clone());
                    } else if *strict {
                        polars_bail!(col_not_found = name);
                    }
                }
                out
            },
            Self::ByIndex { indices, strict } => {
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
            Self::Matches(regex_str) => {
                let re = polars_utils::regex_cache::compile_regex(regex_str).map_err(
                    |_| polars_err!(InvalidOperation: "invalid regex in selector '{regex_str}'"),
                )?;
                PlIndexSet::from_iter(
                    schema
                        .iter_names()
                        .filter(|name| !ignored_columns.contains(*name) && re.is_match(name))
                        .cloned(),
                )
            },
            Self::Wildcard => PlIndexSet::from_iter(
                schema
                    .iter_names()
                    .filter(|name| !ignored_columns.contains(*name))
                    .cloned(),
            ),
            Self::Empty => Default::default(),
        };
        Ok(out)
    }

    pub fn as_expr(self) -> Expr {
        self.into()
    }

    pub fn to_dtype_selector(&self) -> Option<DataTypeSelector> {
        use DataTypeSelector as DS;
        match self {
            Self::Union(l, r) => Some(DS::Union(
                Arc::new(l.to_dtype_selector()?),
                Arc::new(r.to_dtype_selector()?),
            )),
            Self::Difference(l, r) => Some(DS::Difference(
                Arc::new(l.to_dtype_selector()?),
                Arc::new(r.to_dtype_selector()?),
            )),
            Self::ExclusiveOr(l, r) => Some(DS::ExclusiveOr(
                Arc::new(l.to_dtype_selector()?),
                Arc::new(r.to_dtype_selector()?),
            )),
            Self::Intersect(l, r) => Some(DS::ExclusiveOr(
                Arc::new(l.to_dtype_selector()?),
                Arc::new(r.to_dtype_selector()?),
            )),
            Self::Wildcard => Some(DS::Wildcard),
            Self::Empty => Some(DS::Empty),

            Self::ByDType(dts) => Some(dts.clone()),

            Self::ByName { .. } | Self::ByIndex { .. } | Self::Matches(_) => None,
        }
    }

    /// Exclude a column from a wildcard/regex selection.
    ///
    /// You may also use regexes in the exclude as long as they start with `^` and end with `$`.
    pub fn exclude_cols(self, columns: impl IntoVec<PlSmallStr>) -> Self {
        self - cols(columns.into_vec())
    }

    pub fn exclude_dtype<D: AsRef<[DataType]>>(self, dtypes: D) -> Self {
        self - DataTypeSelector::AnyOf(dtypes.as_ref().into()).as_selector()
    }
}

fn list_matches(inner_dts: Option<&DataTypeSelector>, dtype: &DataType) -> bool {
    matches!(dtype, DataType::List(inner) if inner_dts.is_none_or(|dts| dts.matches(inner.as_ref())))
}

fn array_matches(
    inner_dts: Option<&DataTypeSelector>,
    swidth: Option<usize>,
    dtype: &DataType,
) -> bool {
    #[cfg(feature = "dtype-array")]
    {
        matches!(dtype, DataType::Array(inner, width) if inner_dts.is_none_or(|dts| dts.matches(inner.as_ref())) && swidth.is_none_or(|w| w == *width))
    }

    #[cfg(not(feature = "dtype-array"))]
    {
        false
    }
}

fn datetime_matches(stu: TimeUnitSet, stz: &TimeZoneSet, dtype: &DataType) -> bool {
    let DataType::Datetime(tu, tz) = dtype else {
        return false;
    };

    if !stu.contains(TimeUnitSet::from(*tu)) {
        return false;
    }

    use TimeZoneSet as TZS;
    match (stz, tz) {
        (TZS::Any, _)
        | (TZS::Unset, None)
        | (TZS::UnsetOrAnyOf(_), None)
        | (TZS::AnySet, Some(_)) => true,
        (TZS::AnyOf(stz) | TZS::UnsetOrAnyOf(stz), Some(tz)) => stz.contains(tz),
        _ => false,
    }
}

fn sort_schema_order(set: &mut PlIndexSet<PlSmallStr>, schema: &Schema) {
    set.sort_unstable_by(|l, r| {
        schema
            .index_of(l)
            .unwrap()
            .cmp(&schema.index_of(r).unwrap())
    })
}

fn duration_matches(stu: TimeUnitSet, dtype: &DataType) -> bool {
    matches!(dtype, DataType::Duration(tu) if stu.contains(TimeUnitSet::from(*tu)))
}

impl DataTypeSelector {
    pub fn matches(&self, dtype: &DataType) -> bool {
        match self {
            Self::Union(lhs, rhs) => lhs.matches(dtype) || rhs.matches(dtype),
            Self::Difference(lhs, rhs) => lhs.matches(dtype) && !rhs.matches(dtype),
            Self::ExclusiveOr(lhs, rhs) => lhs.matches(dtype) ^ rhs.matches(dtype),
            Self::Intersect(lhs, rhs) => lhs.matches(dtype) && rhs.matches(dtype),
            Self::Wildcard => true,
            Self::Empty => false,
            Self::AnyOf(dtypes) => dtypes.iter().any(|dt| dt == dtype),
            Self::Integer => dtype.is_integer(),
            Self::UnsignedInteger => dtype.is_unsigned_integer(),
            Self::SignedInteger => dtype.is_signed_integer(),
            Self::Float => dtype.is_float(),
            Self::Enum => dtype.is_enum(),
            Self::Categorical => dtype.is_categorical(),
            Self::Nested => dtype.is_nested(),
            Self::List(inner_dts) => list_matches(inner_dts.as_deref(), dtype),
            Self::Array(inner_dts, swidth) => array_matches(inner_dts.as_deref(), *swidth, dtype),
            Self::Struct => dtype.is_struct(),
            Self::Decimal => dtype.is_decimal(),
            Self::Numeric => dtype.is_numeric(),
            Self::Temporal => dtype.is_temporal(),
            Self::Datetime(stu, stz) => datetime_matches(*stu, stz, dtype),
            Self::Duration(stu) => duration_matches(*stu, dtype),
            Self::Object => dtype.is_object(),
        }
    }

    #[allow(clippy::wrong_self_convention)]
    fn into_columns(
        &self,
        schema: &Schema,
        ignored_columns: &PlHashSet<PlSmallStr>,
    ) -> PolarsResult<PlIndexSet<PlSmallStr>> {
        Ok(match self {
            Self::Union(lhs, rhs) => {
                let mut lhs = lhs.into_columns(schema, ignored_columns)?;
                let rhs = rhs.into_columns(schema, ignored_columns)?;
                lhs.extend(rhs);
                sort_schema_order(&mut lhs, schema);
                lhs
            },
            Self::Difference(lhs, rhs) => {
                let mut lhs = lhs.into_columns(schema, ignored_columns)?;
                let rhs = rhs.into_columns(schema, ignored_columns)?;
                lhs.retain(|n| !rhs.contains(n));
                sort_schema_order(&mut lhs, schema);
                lhs
            },
            Self::ExclusiveOr(lhs, rhs) => {
                let lhs = lhs.into_columns(schema, ignored_columns)?;
                let rhs = rhs.into_columns(schema, ignored_columns)?;
                let mut out = PlIndexSet::with_capacity(lhs.len() + rhs.len());
                out.extend(lhs.iter().filter(|n| !rhs.contains(*n)).cloned());
                out.extend(rhs.into_iter().filter(|n| !lhs.contains(n)));
                sort_schema_order(&mut out, schema);
                out
            },
            Self::Intersect(lhs, rhs) => {
                let mut lhs = lhs.into_columns(schema, ignored_columns)?;
                let rhs = rhs.into_columns(schema, ignored_columns)?;
                lhs.retain(|n| rhs.contains(n));
                sort_schema_order(&mut lhs, schema);
                lhs
            },
            Self::Wildcard => schema
                .iter_names()
                .filter(|n| ignored_columns.contains(*n))
                .cloned()
                .collect(),
            Self::Empty => Default::default(),
            Self::AnyOf(dtypes) => {
                let dtypes = PlHashSet::from_iter(dtypes.iter().cloned());
                dtype_selector(schema, ignored_columns, |dtype| dtypes.contains(dtype))
            },
            Self::Integer => dtype_selector(schema, ignored_columns, |dtype| dtype.is_integer()),
            Self::UnsignedInteger => {
                dtype_selector(schema, ignored_columns, |dtype| dtype.is_unsigned_integer())
            },
            Self::SignedInteger => {
                dtype_selector(schema, ignored_columns, |dtype| dtype.is_signed_integer())
            },
            Self::Float => dtype_selector(schema, ignored_columns, |dtype| dtype.is_float()),
            Self::Enum => dtype_selector(schema, ignored_columns, |dtype| dtype.is_enum()),
            Self::Categorical => {
                dtype_selector(schema, ignored_columns, |dtype| dtype.is_categorical())
            },
            Self::Nested => dtype_selector(schema, ignored_columns, |dtype| dtype.is_nested()),
            Self::List(inner_dts) => dtype_selector(schema, ignored_columns, |dtype| {
                list_matches(inner_dts.as_deref(), dtype)
            }),
            Self::Array(inner_dts, swidth) => dtype_selector(schema, ignored_columns, |dtype| {
                array_matches(inner_dts.as_deref(), *swidth, dtype)
            }),
            Self::Struct => dtype_selector(schema, ignored_columns, |dtype| dtype.is_struct()),
            Self::Decimal => dtype_selector(schema, ignored_columns, |dtype| dtype.is_decimal()),
            Self::Numeric => dtype_selector(schema, ignored_columns, |dtype| dtype.is_numeric()),
            Self::Temporal => dtype_selector(schema, ignored_columns, |dtype| dtype.is_temporal()),
            Self::Datetime(stu, stz) => dtype_selector(schema, ignored_columns, |dtype| {
                datetime_matches(*stu, stz, dtype)
            }),
            Self::Duration(stu) => dtype_selector(schema, ignored_columns, |dtype| {
                duration_matches(*stu, dtype)
            }),
            Self::Object => dtype_selector(schema, ignored_columns, |dtype| dtype.is_object()),
        })
    }

    pub fn as_selector(self) -> Selector {
        Selector::ByDType(self)
    }
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

impl BitOr for DataTypeSelector {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self::Union(Arc::new(self), Arc::new(rhs))
    }
}

impl BitOrAssign for DataTypeSelector {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = Self::Union(
            Arc::new(std::mem::replace(self, Self::Empty)),
            Arc::new(rhs),
        )
    }
}

impl BitAnd for DataTypeSelector {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self::Intersect(Arc::new(self), Arc::new(rhs))
    }
}

impl BitAndAssign for DataTypeSelector {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = Self::Intersect(
            Arc::new(std::mem::replace(self, Self::Empty)),
            Arc::new(rhs),
        )
    }
}

impl BitXor for DataTypeSelector {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self::ExclusiveOr(Arc::new(self), Arc::new(rhs))
    }
}

impl BitXorAssign for DataTypeSelector {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = Self::ExclusiveOr(
            Arc::new(std::mem::replace(self, Self::Empty)),
            Arc::new(rhs),
        )
    }
}

impl Sub for DataTypeSelector {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::Difference(Arc::new(self), Arc::new(rhs))
    }
}

impl SubAssign for DataTypeSelector {
    fn sub_assign(&mut self, rhs: Self) {
        *self = Self::Difference(
            Arc::new(std::mem::replace(self, Self::Empty)),
            Arc::new(rhs),
        )
    }
}

impl Not for DataTypeSelector {
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
            Self::Union(left, right) => write!(f, "[{left} | {right}]"),
            Self::Difference(left, right) => write!(f, "[{left} - {right}]"),
            Self::ExclusiveOr(left, right) => write!(f, "[{left} ^ {right}]"),
            Self::Intersect(left, right) => write!(f, "[{left} & {right}]"),

            Self::ByDType(dst) => fmt::Display::fmt(dst, f),
            Self::ByName { names, strict } => {
                f.write_str("cs.by_name(")?;

                for e in names.iter() {
                    write!(f, "'{e}', ")?;
                }

                write!(f, "require_all={strict})")
            },
            Self::ByIndex { indices, strict } if indices.as_ref() == [0] => {
                write!(f, "cs.first(require={strict})")
            },
            Self::ByIndex { indices, strict } if indices.as_ref() == [-1] => {
                write!(f, "cs.last(require={strict})")
            },
            Self::ByIndex { indices, strict } if indices.len() == 1 => {
                write!(f, "cs.nth({}, require_all={strict})", indices[0])
            },
            Self::ByIndex { indices, strict } => {
                write!(
                    f,
                    "cs.by_index({:?}, require_all={strict})",
                    indices.as_ref()
                )
            },
            Self::Matches(s) => write!(f, "cs.matches(\"{s}\")"),
            Self::Wildcard => f.write_str("cs.all()"),
            Self::Empty => f.write_str("cs.empty()"),
        }
    }
}

impl fmt::Display for DataTypeSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Union(left, right) => write!(f, "[{left} | {right}]"),
            Self::Difference(left, right) => write!(f, "[{left} - {right}]"),
            Self::ExclusiveOr(left, right) => write!(f, "[{left} ^ {right}]"),
            Self::Intersect(left, right) => write!(f, "[{left} & {right}]"),

            Self::Float => f.write_str("cs.float()"),
            Self::Integer => f.write_str("cs.integer()"),
            Self::SignedInteger => f.write_str("cs.signed_integer()"),
            Self::UnsignedInteger => f.write_str("cs.unsigned_integer()"),

            Self::Enum => f.write_str("cs.enum()"),
            Self::Categorical => f.write_str("cs.categorical()"),

            Self::Nested => f.write_str("cs.nested()"),
            Self::List(inner_dst) => {
                f.write_str("cs.list(")?;
                if let Some(inner_dst) = inner_dst {
                    fmt::Display::fmt(inner_dst.as_ref(), f)?;
                }
                f.write_str(")")
            },
            Self::Array(inner_dst, swidth) => {
                f.write_str("cs.list(")?;
                if let Some(inner_dst) = inner_dst {
                    fmt::Display::fmt(inner_dst.as_ref(), f)?;
                }
                f.write_str(", width=")?;
                match swidth {
                    None => f.write_str("*")?,
                    Some(swidth) => write!(f, "{swidth}")?,
                }
                f.write_str(")")
            },
            Self::Struct => f.write_str("cs.struct()"),

            Self::Numeric => f.write_str("cs.numeric()"),
            Self::Decimal => f.write_str("cs.decimal()"),
            Self::Temporal => f.write_str("cs.temporal()"),
            Self::Datetime(tu, tz) => {
                write!(f, "cs.datetime(time_unit={tu}, time_zone=")?;
                use TimeZoneSet as TZS;
                match tz {
                    TZS::Any => f.write_str("*")?,
                    TZS::AnySet => f.write_str("*set")?,
                    TZS::Unset => f.write_str("None")?,
                    TZS::UnsetOrAnyOf(tz) => {
                        f.write_str("[None")?;
                        for e in tz.iter() {
                            write!(f, ", '{e}'")?;
                        }
                        f.write_str("]")?;
                    },
                    TZS::AnyOf(tz) => {
                        f.write_str("[")?;
                        if let Some(e) = tz.first() {
                            write!(f, "'{e}'")?;
                            for e in &tz[1..] {
                                write!(f, ", '{e}'")?;
                            }
                        }
                        f.write_str("]")?;
                    },
                }
                f.write_str(")")
            },
            Self::Duration(tu) => {
                write!(f, "cs.duration(time_unit={tu})")
            },
            Self::Object => f.write_str("cs.object()"),

            Self::AnyOf(dtypes) => {
                use DataType as D;
                match dtypes.as_ref() {
                    [D::Boolean] => f.write_str("cs.boolean()"),
                    [D::Binary] => f.write_str("cs.binary()"),
                    [D::Time] => f.write_str("cs.time()"),
                    [D::Date] => f.write_str("cs.date()"),
                    [D::String] => f.write_str("cs.string()"),
                    _ => write!(f, "cs.by_dtype({dtypes:?})"),
                }
            },

            Self::Wildcard => f.write_str("cs.all()"),
            Self::Empty => f.write_str("cs.empty()"),
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
