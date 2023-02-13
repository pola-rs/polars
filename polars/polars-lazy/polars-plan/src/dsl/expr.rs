use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use polars_core::prelude::*;
use polars_core::utils::get_supertype;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::dsl::function_expr::FunctionExpr;
use crate::prelude::*;

/// A wrapper trait for any closure `Fn(Vec<Series>) -> PolarsResult<Series>`
pub trait SeriesUdf: Send + Sync {
    fn call_udf(&self, s: &mut [Series]) -> PolarsResult<Option<Series>>;
}

impl<F> SeriesUdf for F
where
    F: Fn(&mut [Series]) -> PolarsResult<Option<Series>> + Send + Sync,
{
    fn call_udf(&self, s: &mut [Series]) -> PolarsResult<Option<Series>> {
        self(s)
    }
}

impl Debug for dyn SeriesUdf {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "SeriesUdf")
    }
}

/// A wrapper trait for any binary closure `Fn(Series, Series) -> PolarsResult<Series>`
pub trait SeriesBinaryUdf: Send + Sync {
    fn call_udf(&self, a: Series, b: Series) -> PolarsResult<Series>;
}

impl<F> SeriesBinaryUdf for F
where
    F: Fn(Series, Series) -> PolarsResult<Series> + Send + Sync,
{
    fn call_udf(&self, a: Series, b: Series) -> PolarsResult<Series> {
        self(a, b)
    }
}

impl Debug for dyn SeriesBinaryUdf {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "SeriesBinaryUdf")
    }
}

impl Default for SpecialEq<Arc<dyn SeriesBinaryUdf>> {
    fn default() -> Self {
        panic!("implementation error");
    }
}

impl Default for SpecialEq<Arc<dyn BinaryUdfOutputField>> {
    fn default() -> Self {
        let output_field = move |_: &Schema, _: Context, _: &Field, _: &Field| None;
        SpecialEq::new(Arc::new(output_field))
    }
}

pub trait RenameAliasFn: Send + Sync {
    fn call(&self, name: &str) -> PolarsResult<String>;
}

impl<F: Fn(&str) -> PolarsResult<String> + Send + Sync> RenameAliasFn for F {
    fn call(&self, name: &str) -> PolarsResult<String> {
        self(name)
    }
}

impl Debug for dyn RenameAliasFn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "RenameAliasFn")
    }
}

#[derive(Clone)]
/// Wrapper type that has special equality properties
/// depending on the inner type specialization
pub struct SpecialEq<T>(T);

impl<T> SpecialEq<T> {
    pub fn new(val: T) -> Self {
        SpecialEq(val)
    }
}

impl<T: ?Sized> PartialEq for SpecialEq<Arc<T>> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl PartialEq for SpecialEq<Series> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Debug for SpecialEq<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "no_eq")
    }
}

impl<T> Deref for SpecialEq<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub trait BinaryUdfOutputField: Send + Sync {
    fn get_field(
        &self,
        input_schema: &Schema,
        cntxt: Context,
        field_a: &Field,
        field_b: &Field,
    ) -> Option<Field>;
}

impl<F> BinaryUdfOutputField for F
where
    F: Fn(&Schema, Context, &Field, &Field) -> Option<Field> + Send + Sync,
{
    fn get_field(
        &self,
        input_schema: &Schema,
        cntxt: Context,
        field_a: &Field,
        field_b: &Field,
    ) -> Option<Field> {
        self(input_schema, cntxt, field_a, field_b)
    }
}

pub trait FunctionOutputField: Send + Sync {
    fn get_field(&self, input_schema: &Schema, cntxt: Context, fields: &[Field]) -> Field;
}

pub type GetOutput = SpecialEq<Arc<dyn FunctionOutputField>>;

impl Default for GetOutput {
    fn default() -> Self {
        SpecialEq::new(Arc::new(
            |_input_schema: &Schema, _cntxt: Context, fields: &[Field]| fields[0].clone(),
        ))
    }
}

impl GetOutput {
    pub fn same_type() -> Self {
        Default::default()
    }

    pub fn from_type(dt: DataType) -> Self {
        SpecialEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            Field::new(flds[0].name(), dt.clone())
        }))
    }

    pub fn map_field<F: 'static + Fn(&Field) -> Field + Send + Sync>(f: F) -> Self {
        SpecialEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            f(&flds[0])
        }))
    }

    pub fn map_fields<F: 'static + Fn(&[Field]) -> Field + Send + Sync>(f: F) -> Self {
        SpecialEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            f(flds)
        }))
    }

    pub fn map_dtype<F: 'static + Fn(&DataType) -> DataType + Send + Sync>(f: F) -> Self {
        SpecialEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            let mut fld = flds[0].clone();
            let new_type = f(fld.data_type());
            fld.coerce(new_type);
            fld
        }))
    }

    pub fn float_type() -> Self {
        Self::map_dtype(|dt| match dt {
            DataType::Float32 => DataType::Float32,
            _ => DataType::Float64,
        })
    }

    pub fn super_type() -> Self {
        Self::map_dtypes(|dtypes| {
            let mut st = dtypes[0].clone();
            for dt in &dtypes[1..] {
                st = get_supertype(&st, dt).unwrap();
            }
            st
        })
    }

    pub fn map_dtypes<F>(f: F) -> Self
    where
        F: 'static + Fn(&[&DataType]) -> DataType + Send + Sync,
    {
        SpecialEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            let mut fld = flds[0].clone();
            let dtypes = flds.iter().map(|fld| fld.data_type()).collect::<Vec<_>>();
            let new_type = f(&dtypes);
            fld.coerce(new_type);
            fld
        }))
    }
}

impl<F> FunctionOutputField for F
where
    F: Fn(&Schema, Context, &[Field]) -> Field + Send + Sync,
{
    fn get_field(&self, input_schema: &Schema, cntxt: Context, fields: &[Field]) -> Field {
        self(input_schema, cntxt, fields)
    }
}

#[derive(PartialEq, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AggExpr {
    Min {
        input: Box<Expr>,
        propagate_nans: bool,
    },
    Max {
        input: Box<Expr>,
        propagate_nans: bool,
    },
    Median(Box<Expr>),
    NUnique(Box<Expr>),
    First(Box<Expr>),
    Last(Box<Expr>),
    Mean(Box<Expr>),
    List(Box<Expr>),
    Count(Box<Expr>),
    Quantile {
        expr: Box<Expr>,
        quantile: Box<Expr>,
        interpol: QuantileInterpolOptions,
    },
    Sum(Box<Expr>),
    AggGroups(Box<Expr>),
    Std(Box<Expr>, u8),
    Var(Box<Expr>, u8),
}

impl AsRef<Expr> for AggExpr {
    fn as_ref(&self) -> &Expr {
        use AggExpr::*;
        match self {
            Min { input, .. } => input,
            Max { input, .. } => input,
            Median(e) => e,
            NUnique(e) => e,
            First(e) => e,
            Last(e) => e,
            Mean(e) => e,
            List(e) => e,
            Count(e) => e,
            Quantile { expr, .. } => expr,
            Sum(e) => e,
            AggGroups(e) => e,
            Std(e, _) => e,
            Var(e, _) => e,
        }
    }
}

/// Queries consists of multiple expressions.
#[derive(Clone, PartialEq)]
#[must_use]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Expr {
    Alias(Box<Expr>, Arc<str>),
    Column(Arc<str>),
    Columns(Vec<String>),
    DtypeColumn(Vec<DataType>),
    Literal(LiteralValue),
    BinaryExpr {
        left: Box<Expr>,
        op: Operator,
        right: Box<Expr>,
    },
    Cast {
        expr: Box<Expr>,
        data_type: DataType,
        strict: bool,
    },
    Sort {
        expr: Box<Expr>,
        options: SortOptions,
    },
    Take {
        expr: Box<Expr>,
        idx: Box<Expr>,
    },
    SortBy {
        expr: Box<Expr>,
        by: Vec<Expr>,
        reverse: Vec<bool>,
    },
    Agg(AggExpr),
    /// A ternary operation
    /// if true then "foo" else "bar"
    Ternary {
        predicate: Box<Expr>,
        truthy: Box<Expr>,
        falsy: Box<Expr>,
    },
    Function {
        /// function arguments
        input: Vec<Expr>,
        /// function to apply
        function: FunctionExpr,
        options: FunctionOptions,
    },
    Explode(Box<Expr>),
    Filter {
        input: Box<Expr>,
        by: Box<Expr>,
    },
    /// See postgres window functions
    Window {
        /// Also has the input. i.e. avg("foo")
        function: Box<Expr>,
        partition_by: Vec<Expr>,
        order_by: Option<Box<Expr>>,
        options: WindowOptions,
    },
    Wildcard,
    Slice {
        input: Box<Expr>,
        /// length is not yet known so we accept negative offsets
        offset: Box<Expr>,
        length: Box<Expr>,
    },
    /// Can be used in a select statement to exclude a column from selection
    Exclude(Box<Expr>, Vec<Excluded>),
    /// Set root name as Alias
    KeepName(Box<Expr>),
    /// Special case that does not need columns
    Count,
    /// Take the nth column in the `DataFrame`
    Nth(i64),
    // skipped fields must be last otherwise serde fails in pickle
    #[cfg_attr(feature = "serde", serde(skip))]
    RenameAlias {
        function: SpecialEq<Arc<dyn RenameAliasFn>>,
        expr: Box<Expr>,
    },
    #[cfg_attr(feature = "serde", serde(skip))]
    AnonymousFunction {
        /// function arguments
        input: Vec<Expr>,
        /// function to apply
        function: SpecialEq<Arc<dyn SeriesUdf>>,
        /// output dtype of the function
        output_type: GetOutput,
        options: FunctionOptions,
    },
}

// TODO! derive. This is only a temporary fix
// Because PartialEq will have a lot of `false`, e.g. on Function
// Types, this may lead to many file reads, as we use predicate comparison
// to check if we can cache a file
#[allow(clippy::derived_hash_with_manual_eq)]
impl Hash for Expr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let s = format!("{self:?}");
        s.hash(state)
    }
}

impl Eq for Expr {}

impl Default for Expr {
    fn default() -> Self {
        Expr::Literal(LiteralValue::Null)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]

pub enum Excluded {
    Name(Arc<str>),
    Dtype(DataType),
}

impl Expr {
    /// Get Field result of the expression. The schema is the input data.
    pub fn to_field(&self, schema: &Schema, ctxt: Context) -> PolarsResult<Field> {
        // this is not called much and th expression depth is typically shallow
        let mut arena = Arena::with_capacity(5);
        self.to_field_amortized(schema, ctxt, &mut arena)
    }
    pub(crate) fn to_field_amortized(
        &self,
        schema: &Schema,
        ctxt: Context,
        expr_arena: &mut Arena<AExpr>,
    ) -> PolarsResult<Field> {
        let root = to_aexpr(self.clone(), expr_arena);
        expr_arena.get(root).to_field(schema, ctxt, expr_arena)
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Operator {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Plus,
    Minus,
    Multiply,
    Divide,
    TrueDivide,
    FloorDivide,
    Modulus,
    And,
    Or,
    Xor,
}

impl Display for Operator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use Operator::*;
        let tkn = match self {
            Eq => "==",
            NotEq => "!=",
            Lt => "<",
            LtEq => "<=",
            Gt => ">",
            GtEq => ">=",
            Plus => "+",
            Minus => "-",
            Multiply => "*",
            Divide => "//",
            TrueDivide => "/",
            FloorDivide => "floor_div",
            Modulus => "%",
            And => "&",
            Or => "|",
            Xor => "^",
        };
        write!(f, "{tkn}")
    }
}

impl Operator {
    pub(crate) fn is_comparison(&self) -> bool {
        matches!(
            self,
            Self::Eq
                | Self::NotEq
                | Self::Lt
                | Self::LtEq
                | Self::Gt
                | Self::GtEq
                | Self::And
                | Self::Or
                | Self::Xor
        )
    }

    pub(crate) fn is_arithmetic(&self) -> bool {
        !(self.is_comparison())
    }
}
