use crate::prelude::*;
use polars_core::prelude::*;
use polars_core::series::ops::NullBehavior;

/// Specialized expressions for [`Series`] of [`DataType::List`].
pub struct ListNameSpace(pub(crate) Expr);

impl ListNameSpace {
    /// Get lengths of the arrays in the List type.
    pub fn lengths(self) -> Expr {
        let function = |s: Series| {
            let ca = s.list()?;
            Ok(ca.lst_lengths().into_series())
        };
        self.0
            .map(function, GetOutput::from_type(DataType::UInt32))
            .with_fmt("arr.len")
    }

    /// Compute the maximum of the items in every sublist.
    pub fn max(self) -> Expr {
        self.0
            .map(
                |s| Ok(s.list()?.lst_max()),
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
                |s| Ok(s.list()?.lst_min()),
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
                |s| Ok(s.list()?.lst_sum()),
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
                |s| Ok(s.list()?.lst_mean().into_series()),
                GetOutput::from_type(DataType::Float64),
            )
            .with_fmt("arr.mean")
    }

    /// Sort every sublist.
    pub fn sort(self, reverse: bool) -> Expr {
        self.0
            .map(
                move |s| Ok(s.list()?.lst_sort(reverse).into_series()),
                GetOutput::same_type(),
            )
            .with_fmt("arr.sort")
    }

    /// Reverse every sublist
    pub fn reverse(self) -> Expr {
        self.0
            .map(
                move |s| Ok(s.list()?.lst_reverse().into_series()),
                GetOutput::same_type(),
            )
            .with_fmt("arr.reverse")
    }

    /// Keep only the unique values in every sublist.
    pub fn unique(self) -> Expr {
        self.0
            .map(
                move |s| Ok(s.list()?.lst_unique()?.into_series()),
                GetOutput::same_type(),
            )
            .with_fmt("arr.unique")
    }

    /// Get items in every sublist by index.
    pub fn get(self, index: i64) -> Expr {
        self.0.map(
            move |s| s.list()?.lst_get(index),
            GetOutput::map_field(|field| match field.data_type() {
                DataType::List(inner) => Field::new(field.name(), *inner.clone()),
                _ => panic!("should be a list type"),
            }),
        )
    }

    /// Get first item of every sublist.
    pub fn first(self) -> Expr {
        self.get(0)
    }

    /// Get last item of every sublist.
    pub fn last(self) -> Expr {
        self.get(-1)
    }

    /// Join all string items in a sublist and place a separator between them.
    /// # Error
    /// This errors if inner type of list `!= DataType::Utf8`.
    pub fn join(self, separator: &str) -> Expr {
        let separator = separator.to_string();
        self.0
            .map(
                move |s| s.list()?.lst_join(&separator).map(|ca| ca.into_series()),
                GetOutput::from_type(DataType::Utf8),
            )
            .with_fmt("arr.join")
    }

    /// Return the index of the minimal value of every sublist
    pub fn arg_min(self) -> Expr {
        self.0
            .map(
                |s| Ok(s.list()?.lst_arg_min().into_series()),
                GetOutput::from_type(IDX_DTYPE),
            )
            .with_fmt("arr.arg_min")
    }

    /// Return the index of the maximum value of every sublist
    pub fn arg_max(self) -> Expr {
        self.0
            .map(
                |s| Ok(s.list()?.lst_arg_max().into_series()),
                GetOutput::from_type(IDX_DTYPE),
            )
            .with_fmt("arr.arg_max")
    }

    /// Diff every sublist.
    #[cfg(feature = "diff")]
    #[cfg_attr(docsrs, doc(cfg(feature = "diff")))]
    pub fn diff(self, n: usize, null_behavior: NullBehavior) -> Expr {
        self.0
            .map(
                move |s| Ok(s.list()?.lst_diff(n, null_behavior).into_series()),
                GetOutput::same_type(),
            )
            .with_fmt("arr.diff")
    }

    /// Shift every sublist.
    pub fn shift(self, periods: i64) -> Expr {
        self.0
            .map(
                move |s| Ok(s.list()?.lst_shift(periods).into_series()),
                GetOutput::same_type(),
            )
            .with_fmt("arr.diff")
    }

    /// Slice every sublist.
    pub fn slice(self, offset: i64, length: usize) -> Expr {
        self.0
            .map(
                move |s| Ok(s.list()?.lst_slice(offset, length).into_series()),
                GetOutput::same_type(),
            )
            .with_fmt("arr.slice")
    }

    /// Get the head of every sublist
    pub fn head(self, n: usize) -> Expr {
        self.slice(0, n)
    }

    /// Get the tail of every sublist
    pub fn tail(self, n: usize) -> Expr {
        self.slice(-(n as i64), n)
    }
}
