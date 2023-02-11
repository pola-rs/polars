#[cfg(feature = "list_to_struct")]
use std::sync::RwLock;

use polars_core::prelude::*;
#[cfg(feature = "diff")]
use polars_core::series::ops::NullBehavior;
use polars_ops::prelude::*;

use crate::dsl::function_expr::FunctionExpr;
use crate::prelude::function_expr::ListFunction;
use crate::prelude::*;

/// Specialized expressions for [`Series`] of [`DataType::List`].
pub struct ListNameSpace(pub Expr);

impl ListNameSpace {
    /// Get lengths of the arrays in the List type.
    pub fn lengths(self) -> Expr {
        let function = |s: Series| {
            let ca = s.list()?;
            Ok(Some(ca.lst_lengths().into_series()))
        };
        self.0
            .map(function, GetOutput::from_type(IDX_DTYPE))
            .with_fmt("arr.len")
    }

    /// Compute the maximum of the items in every sublist.
    pub fn max(self) -> Expr {
        self.0
            .map(
                |s| Ok(Some(s.list()?.lst_max())),
                GetOutput::map_field(|f| {
                    if let DataType::List(adt) = f.data_type() {
                        Field::new(f.name(), *adt.clone())
                    } else {
                        // inner type
                        f.clone()
                    }
                }),
            )
            .with_fmt("arr.max")
    }

    /// Compute the minimum of the items in every sublist.
    pub fn min(self) -> Expr {
        self.0
            .map(
                |s| Ok(Some(s.list()?.lst_min())),
                GetOutput::map_field(|f| {
                    if let DataType::List(adt) = f.data_type() {
                        Field::new(f.name(), *adt.clone())
                    } else {
                        // inner type
                        f.clone()
                    }
                }),
            )
            .with_fmt("arr.min")
    }

    /// Compute the sum the items in every sublist.
    pub fn sum(self) -> Expr {
        self.0
            .map(
                |s| Ok(Some(s.list()?.lst_sum())),
                GetOutput::map_field(|f| {
                    if let DataType::List(adt) = f.data_type() {
                        Field::new(f.name(), *adt.clone())
                    } else {
                        // inner type
                        f.clone()
                    }
                }),
            )
            .with_fmt("arr.sum")
    }

    /// Compute the mean of every sublist and return a `Series` of dtype `Float64`
    pub fn mean(self) -> Expr {
        self.0
            .map(
                |s| Ok(Some(s.list()?.lst_mean().into_series())),
                GetOutput::from_type(DataType::Float64),
            )
            .with_fmt("arr.mean")
    }

    /// Sort every sublist.
    pub fn sort(self, options: SortOptions) -> Expr {
        self.0
            .map(
                move |s| Ok(Some(s.list()?.lst_sort(options).into_series())),
                GetOutput::same_type(),
            )
            .with_fmt("arr.sort")
    }

    /// Reverse every sublist
    pub fn reverse(self) -> Expr {
        self.0
            .map(
                move |s| Ok(Some(s.list()?.lst_reverse().into_series())),
                GetOutput::same_type(),
            )
            .with_fmt("arr.reverse")
    }

    /// Keep only the unique values in every sublist.
    pub fn unique(self) -> Expr {
        self.0
            .map(
                move |s| Ok(Some(s.list()?.lst_unique()?.into_series())),
                GetOutput::same_type(),
            )
            .with_fmt("arr.unique")
    }

    /// Get items in every sublist by index.
    pub fn get(self, index: Expr) -> Expr {
        self.0.apply_many_private(
            FunctionExpr::ListExpr(ListFunction::Get),
            &[index],
            false,
            false,
        )
    }

    /// Get items in every sublist by multiple indexes.
    ///
    /// # Arguments
    /// - `null_on_oob`: Return a null when an index is out of bounds.
    /// This behavior is more expensive than defaulting to returing an `Error`.
    #[cfg(feature = "list_take")]
    pub fn take(self, index: Expr, null_on_oob: bool) -> Expr {
        self.0.apply_many_private(
            FunctionExpr::ListExpr(ListFunction::Take(null_on_oob)),
            &[index],
            false,
            false,
        )
    }

    /// Get first item of every sublist.
    pub fn first(self) -> Expr {
        self.get(lit(0i64))
    }

    /// Get last item of every sublist.
    pub fn last(self) -> Expr {
        self.get(lit(-1i64))
    }

    /// Join all string items in a sublist and place a separator between them.
    /// # Error
    /// This errors if inner type of list `!= DataType::Utf8`.
    pub fn join(self, separator: &str) -> Expr {
        let separator = separator.to_string();
        self.0
            .apply(
                move |s| {
                    s.list()?
                        .lst_join(&separator)
                        .map(|ca| Some(ca.into_series()))
                },
                GetOutput::from_type(DataType::Utf8),
            )
            .with_fmt("arr.join")
    }

    /// Return the index of the minimal value of every sublist
    pub fn arg_min(self) -> Expr {
        self.0
            .map(
                |s| Ok(Some(s.list()?.lst_arg_min().into_series())),
                GetOutput::from_type(IDX_DTYPE),
            )
            .with_fmt("arr.arg_min")
    }

    /// Return the index of the maximum value of every sublist
    pub fn arg_max(self) -> Expr {
        self.0
            .map(
                |s| Ok(Some(s.list()?.lst_arg_max().into_series())),
                GetOutput::from_type(IDX_DTYPE),
            )
            .with_fmt("arr.arg_max")
    }

    /// Diff every sublist.
    #[cfg(feature = "diff")]
    pub fn diff(self, n: usize, null_behavior: NullBehavior) -> Expr {
        self.0
            .map(
                move |s| Ok(Some(s.list()?.lst_diff(n, null_behavior).into_series())),
                GetOutput::same_type(),
            )
            .with_fmt("arr.diff")
    }

    /// Shift every sublist.
    pub fn shift(self, periods: i64) -> Expr {
        self.0
            .map(
                move |s| Ok(Some(s.list()?.lst_shift(periods).into_series())),
                GetOutput::same_type(),
            )
            .with_fmt("arr.shift")
    }

    /// Slice every sublist.
    pub fn slice(self, offset: Expr, length: Expr) -> Expr {
        self.0.apply_many_private(
            FunctionExpr::ListExpr(ListFunction::Slice),
            &[offset, length],
            false,
            false,
        )
    }

    /// Get the head of every sublist
    pub fn head(self, n: Expr) -> Expr {
        self.slice(lit(0), n)
    }

    /// Get the tail of every sublist
    pub fn tail(self, n: Expr) -> Expr {
        self.slice(lit(0) - n.clone().cast(DataType::Int64), n)
    }

    #[cfg(feature = "list_to_struct")]
    #[allow(clippy::wrong_self_convention)]
    /// Convert this `List` to a `Series` of type `Struct`. The width will be determined according to
    /// `ListToStructWidthStrategy` and the names of the fields determined by the given `name_generator`.
    ///
    /// # Schema
    ///
    /// A polars [`LazyFrame`] needs to know the schema at all time. The caller therefore must provide
    /// an `upper_bound` of struct fields that will be set.
    /// If this is incorrectly downstream operation may fail. For instance an `all().sum()` expression
    /// will look in the current schema to determine which columns to select.
    pub fn to_struct(
        self,
        n_fields: ListToStructWidthStrategy,
        name_generator: Option<NameGenerator>,
        upper_bound: usize,
    ) -> Expr {
        // heap allocate the output type and fill it later
        let out_dtype = Arc::new(RwLock::new(None::<DataType>));

        self.0
            .map(
                move |s| {
                    s.list()?
                        .to_struct(n_fields, name_generator.clone())
                        .map(|s| Some(s.into_series()))
                },
                // we don't yet know the fields
                GetOutput::map_dtype(move |dt: &DataType| {
                    let out = out_dtype.read().unwrap();
                    match out.as_ref() {
                        // dtype already set
                        Some(dt) => dt.clone(),
                        // dtype still unknown, set it
                        None => {
                            drop(out);
                            let mut lock = out_dtype.write().unwrap();

                            let inner = dt.inner_dtype().unwrap();
                            let fields = (0..upper_bound)
                                .map(|i| {
                                    let name = _default_struct_name_gen(i);
                                    Field::from_owned(name, inner.clone())
                                })
                                .collect();
                            let dt = DataType::Struct(fields);

                            *lock = Some(dt.clone());
                            dt
                        }
                    }
                }),
            )
            .with_fmt("arr.to_struct")
    }

    #[cfg(feature = "is_in")]
    /// Check if the list array contain an element
    pub fn contains<E: Into<Expr>>(self, other: E) -> Expr {
        let other = other.into();

        Expr::Function {
            input: vec![self.0, other],
            function: FunctionExpr::ListExpr(ListFunction::Contains),
            options: FunctionOptions {
                collect_groups: ApplyOptions::ApplyFlat,
                input_wildcard_expansion: true,
                auto_explode: true,
                fmt_str: "arr.contains",
                ..Default::default()
            },
        }
    }
}
