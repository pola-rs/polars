#[cfg(feature = "list_to_struct")]
use std::sync::RwLock;

use polars_core::prelude::*;
#[cfg(feature = "diff")]
use polars_core::series::ops::NullBehavior;

use crate::dsl::function_expr::FunctionExpr;
use crate::prelude::function_expr::ListFunction;
use crate::prelude::*;

/// Specialized expressions for [`Series`] of [`DataType::List`].
pub struct ListNameSpace(pub Expr);

impl ListNameSpace {
    #[cfg(feature = "list_any_all")]
    pub fn any(self) -> Expr {
        self.0
            .apply_private(FunctionExpr::ListExpr(ListFunction::Any))
            .with_fmt("list.any")
    }

    #[cfg(feature = "list_any_all")]
    pub fn all(self) -> Expr {
        self.0
            .apply_private(FunctionExpr::ListExpr(ListFunction::All))
            .with_fmt("list.all")
    }

    #[cfg(feature = "list_drop_nulls")]
    pub fn drop_nulls(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ListExpr(ListFunction::DropNulls))
    }

    #[cfg(feature = "list_sample")]
    pub fn sample_n(
        self,
        n: Expr,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Expr {
        self.0.map_many_private(
            FunctionExpr::ListExpr(ListFunction::Sample {
                is_fraction: false,
                with_replacement,
                shuffle,
                seed,
            }),
            &[n],
            false,
            false,
        )
    }

    #[cfg(feature = "list_sample")]
    pub fn sample_fraction(
        self,
        fraction: Expr,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Expr {
        self.0.map_many_private(
            FunctionExpr::ListExpr(ListFunction::Sample {
                is_fraction: true,
                with_replacement,
                shuffle,
                seed,
            }),
            &[fraction],
            false,
            false,
        )
    }

    /// Return the number of elements in each list.
    ///
    /// Null values are treated like regular elements in this context.
    pub fn len(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ListExpr(ListFunction::Length))
    }

    /// Compute the maximum of the items in every sublist.
    pub fn max(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ListExpr(ListFunction::Max))
    }

    /// Compute the minimum of the items in every sublist.
    pub fn min(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ListExpr(ListFunction::Min))
    }

    /// Compute the sum the items in every sublist.
    pub fn sum(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ListExpr(ListFunction::Sum))
    }

    /// Compute the product of the items in every sublist.
    pub fn product(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ListExpr(ListFunction::Product))
    }

    /// Compute the mean of every sublist and return a `Series` of dtype `Float64`
    pub fn mean(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ListExpr(ListFunction::Mean))
    }

    /// Sort every sublist.
    pub fn sort(self, options: SortOptions) -> Expr {
        self.0
            .map_private(FunctionExpr::ListExpr(ListFunction::Sort(options)))
    }

    /// Reverse every sublist
    pub fn reverse(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ListExpr(ListFunction::Reverse))
    }

    /// Keep only the unique values in every sublist.
    pub fn unique(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ListExpr(ListFunction::Unique(false)))
    }

    /// Keep only the unique values in every sublist.
    pub fn unique_stable(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ListExpr(ListFunction::Unique(true)))
    }

    /// Get items in every sublist by index.
    pub fn get(self, index: Expr) -> Expr {
        self.0.map_many_private(
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
    /// This behavior is more expensive than defaulting to returning an `Error`.
    #[cfg(feature = "list_gather")]
    pub fn take(self, index: Expr, null_on_oob: bool) -> Expr {
        self.0.map_many_private(
            FunctionExpr::ListExpr(ListFunction::Gather(null_on_oob)),
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
    /// This errors if inner type of list `!= DataType::String`.
    pub fn join(self, separator: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::ListExpr(ListFunction::Join),
            &[separator],
            false,
            false,
        )
    }

    /// Return the index of the minimal value of every sublist
    pub fn arg_min(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ListExpr(ListFunction::ArgMin))
    }

    /// Return the index of the maximum value of every sublist
    pub fn arg_max(self) -> Expr {
        self.0
            .map_private(FunctionExpr::ListExpr(ListFunction::ArgMax))
    }

    /// Diff every sublist.
    #[cfg(feature = "diff")]
    pub fn diff(self, n: i64, null_behavior: NullBehavior) -> Expr {
        self.0
            .map_private(FunctionExpr::ListExpr(ListFunction::Diff {
                n,
                null_behavior,
            }))
    }

    /// Shift every sublist.
    pub fn shift(self, periods: Expr) -> Expr {
        self.0.map_many_private(
            FunctionExpr::ListExpr(ListFunction::Shift),
            &[periods],
            false,
            false,
        )
    }

    /// Slice every sublist.
    pub fn slice(self, offset: Expr, length: Expr) -> Expr {
        self.0.map_many_private(
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
        self.slice(lit(0i64) - n.clone().cast(DataType::Int64), n)
    }

    #[cfg(feature = "dtype-array")]
    /// Convert a List column into an Array column with the same inner data type.
    pub fn to_array(self, width: usize) -> Expr {
        self.0
            .map_private(FunctionExpr::ListExpr(ListFunction::ToArray(width)))
    }

    #[cfg(feature = "list_to_struct")]
    #[allow(clippy::wrong_self_convention)]
    /// Convert this `List` to a `Series` of type `Struct`. The width will be determined according to
    /// `ListToStructWidthStrategy` and the names of the fields determined by the given `name_generator`.
    ///
    /// # Schema
    ///
    /// A polars `LazyFrame` needs to know the schema at all time. The caller therefore must provide
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
                        },
                    }
                }),
            )
            .with_fmt("list.to_struct")
    }

    #[cfg(feature = "is_in")]
    /// Check if the list array contain an element
    pub fn contains<E: Into<Expr>>(self, other: E) -> Expr {
        let other = other.into();

        self.0
            .map_many_private(
                FunctionExpr::ListExpr(ListFunction::Contains),
                &[other],
                false,
                false,
            )
            .with_function_options(|mut options| {
                options.input_wildcard_expansion = true;
                options
            })
    }
    #[cfg(feature = "list_count")]
    /// Count how often the value produced by ``element`` occurs.
    pub fn count_matches<E: Into<Expr>>(self, element: E) -> Expr {
        let other = element.into();

        self.0
            .map_many_private(
                FunctionExpr::ListExpr(ListFunction::CountMatches),
                &[other],
                false,
                false,
            )
            .with_function_options(|mut options| {
                options.input_wildcard_expansion = true;
                options
            })
    }

    #[cfg(feature = "list_sets")]
    fn set_operation(self, other: Expr, set_operation: SetOperation) -> Expr {
        self.0
            .map_many_private(
                FunctionExpr::ListExpr(ListFunction::SetOperation(set_operation)),
                &[other],
                false,
                true,
            )
            .with_function_options(|mut options| {
                options.input_wildcard_expansion = true;
                options
            })
    }

    /// Return the SET UNION between both list arrays.
    #[cfg(feature = "list_sets")]
    pub fn union<E: Into<Expr>>(self, other: E) -> Expr {
        let other = other.into();
        self.set_operation(other, SetOperation::Union)
    }

    /// Return the SET DIFFERENCE between both list arrays.
    #[cfg(feature = "list_sets")]
    pub fn set_difference<E: Into<Expr>>(self, other: E) -> Expr {
        let other = other.into();
        self.set_operation(other, SetOperation::Difference)
    }

    /// Return the SET INTERSECTION between both list arrays.
    #[cfg(feature = "list_sets")]
    pub fn set_intersection<E: Into<Expr>>(self, other: E) -> Expr {
        let other = other.into();
        self.set_operation(other, SetOperation::Intersection)
    }

    /// Return the SET SYMMETRIC DIFFERENCE between both list arrays.
    #[cfg(feature = "list_sets")]
    pub fn set_symmetric_difference<E: Into<Expr>>(self, other: E) -> Expr {
        let other = other.into();
        self.set_operation(other, SetOperation::SymmetricDifference)
    }
}
