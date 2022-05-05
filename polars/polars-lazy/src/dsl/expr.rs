use crate::prelude::*;
use polars_core::prelude::*;
use polars_core::utils::get_supertype;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;

use crate::dsl::function_expr::FunctionExpr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A wrapper trait for any closure `Fn(Vec<Series>) -> Result<Series>`
pub trait SeriesUdf: Send + Sync {
    fn call_udf(&self, s: &mut [Series]) -> Result<Series>;
}

impl<F> SeriesUdf for F
where
    F: Fn(&mut [Series]) -> Result<Series> + Send + Sync,
{
    fn call_udf(&self, s: &mut [Series]) -> Result<Series> {
        self(s)
    }
}

impl Debug for dyn SeriesUdf {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "SeriesUdf")
    }
}

/// A wrapper trait for any binary closure `Fn(Series, Series) -> Result<Series>`
pub trait SeriesBinaryUdf: Send + Sync {
    fn call_udf(&self, a: Series, b: Series) -> Result<Series>;
}

impl<F> SeriesBinaryUdf for F
where
    F: Fn(Series, Series) -> Result<Series> + Send + Sync,
{
    fn call_udf(&self, a: Series, b: Series) -> Result<Series> {
        self(a, b)
    }
}

impl Debug for dyn SeriesBinaryUdf {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "SeriesBinaryUdf")
    }
}

pub trait RenameAliasFn: Send + Sync {
    fn call(&self, name: &str) -> String;
}

impl<F: Fn(&str) -> String + Send + Sync> RenameAliasFn for F {
    fn call(&self, name: &str) -> String {
        self(name)
    }
}

impl Debug for dyn RenameAliasFn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "RenameAliasFn")
    }
}

#[derive(Clone)]
/// Wrapper type that indicates that the inner type is not equal to anything
pub struct NoEq<T>(T);

impl<T> NoEq<T> {
    pub fn new(val: T) -> Self {
        NoEq(val)
    }
}

impl<T> PartialEq for NoEq<T> {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

impl<T> Debug for NoEq<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "no_eq")
    }
}

impl<T> Deref for NoEq<T> {
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

pub type GetOutput = NoEq<Arc<dyn FunctionOutputField>>;

impl Default for GetOutput {
    fn default() -> Self {
        NoEq::new(Arc::new(
            |_input_schema: &Schema, _cntxt: Context, fields: &[Field]| fields[0].clone(),
        ))
    }
}

impl GetOutput {
    pub fn same_type() -> Self {
        Default::default()
    }

    pub fn from_type(dt: DataType) -> Self {
        NoEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            Field::new(flds[0].name(), dt.clone())
        }))
    }

    pub fn map_field<F: 'static + Fn(&Field) -> Field + Send + Sync>(f: F) -> Self {
        NoEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            f(&flds[0])
        }))
    }

    pub fn map_fields<F: 'static + Fn(&[Field]) -> Field + Send + Sync>(f: F) -> Self {
        NoEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            f(flds)
        }))
    }

    pub fn map_dtype<F: 'static + Fn(&DataType) -> DataType + Send + Sync>(f: F) -> Self {
        NoEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
            let mut fld = flds[0].clone();
            let new_type = f(fld.data_type());
            fld.coerce(new_type);
            fld
        }))
    }

    pub fn super_type() -> Self {
        Self::map_dtypes(|dtypes| {
            let mut st = dtypes[0].clone();
            for dt in &dtypes[1..] {
                st = get_supertype(&st, dt).unwrap()
            }
            st
        })
    }

    pub fn map_dtypes<F>(f: F) -> Self
    where
        F: 'static + Fn(&[&DataType]) -> DataType + Send + Sync,
    {
        NoEq::new(Arc::new(move |_: &Schema, _: Context, flds: &[Field]| {
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
#[cfg_attr(feature = "serde", serde(bound(deserialize = "'de: 'static")))]
pub enum AggExpr {
    Min(Box<Expr>),
    Max(Box<Expr>),
    Median(Box<Expr>),
    NUnique(Box<Expr>),
    First(Box<Expr>),
    Last(Box<Expr>),
    Mean(Box<Expr>),
    List(Box<Expr>),
    Count(Box<Expr>),
    Quantile {
        expr: Box<Expr>,
        quantile: f64,
        interpol: QuantileInterpolOptions,
    },
    Sum(Box<Expr>),
    AggGroups(Box<Expr>),
    Std(Box<Expr>),
    Var(Box<Expr>),
}

impl AsRef<Expr> for AggExpr {
    fn as_ref(&self) -> &Expr {
        use AggExpr::*;
        match self {
            Min(e) => e,
            Max(e) => e,
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
            Std(e) => e,
            Var(e) => e,
        }
    }
}

/// Queries consists of multiple expressions.
#[derive(Clone, PartialEq)]
#[must_use]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(
    all(feature = "serde", feature = "object"),
    serde(bound(deserialize = "'de: 'static"))
)]
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
    Not(Box<Expr>),
    IsNotNull(Box<Expr>),
    IsNull(Box<Expr>),
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
    #[cfg_attr(feature = "serde", serde(skip))]
    AnonymousFunction {
        /// function arguments
        input: Vec<Expr>,
        /// function to apply
        function: NoEq<Arc<dyn SeriesUdf>>,
        /// output dtype of the function
        output_type: GetOutput,
        options: FunctionOptions,
    },
    Function {
        /// function arguments
        input: Vec<Expr>,
        /// function to apply
        function: FunctionExpr,
        options: FunctionOptions,
    },
    Shift {
        input: Box<Expr>,
        periods: i64,
    },
    Reverse(Box<Expr>),
    Duplicated(Box<Expr>),
    IsUnique(Box<Expr>),
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
    #[cfg_attr(feature = "serde", serde(skip))]
    Exclude(Box<Expr>, Vec<Excluded>),
    /// Set root name as Alias
    KeepName(Box<Expr>),
    #[cfg_attr(feature = "serde", serde(skip))]
    RenameAlias {
        function: NoEq<Arc<dyn RenameAliasFn>>,
        expr: Box<Expr>,
    },
    /// Special case that does not need columns
    Count,
    /// Take the nth column in the `DataFrame`
    Nth(i64),
}

impl Default for Expr {
    fn default() -> Self {
        Expr::Literal(LiteralValue::Null)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Excluded {
    Name(Arc<str>),
    Dtype(DataType),
}

impl Expr {
    /// Get Field result of the expression. The schema is the input data.
    pub(crate) fn to_field(&self, schema: &Schema, ctxt: Context) -> Result<Field> {
        // this is not called much and th expression depth is typically shallow
        let mut arena = Arena::with_capacity(5);
        let root = to_aexpr(self.clone(), &mut arena);
        arena.get(root).to_field(schema, ctxt, &arena)
    }
}

#[derive(Copy, Clone, PartialEq)]
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
    Modulus,
    And,
    Or,
    Xor,
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
