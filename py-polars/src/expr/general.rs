use std::ops::Neg;

use polars::lazy::dsl;
use polars::prelude::*;
use polars::series::ops::NullBehavior;
use polars_core::chunked_array::cast::CastOptions;
use polars_core::series::IsSorted;
use pyo3::class::basic::CompareOp;
use pyo3::prelude::*;

use crate::conversion::{parse_fill_null_strategy, vec_extract_wrapped, Wrap};
use crate::map::lazy::map_single;
use crate::PyExpr;

#[pymethods]
impl PyExpr {
    fn __richcmp__(&self, other: Self, op: CompareOp) -> Self {
        match op {
            CompareOp::Eq => self.eq(other),
            CompareOp::Ne => self.neq(other),
            CompareOp::Gt => self.gt(other),
            CompareOp::Lt => self.lt(other),
            CompareOp::Ge => self.gt_eq(other),
            CompareOp::Le => self.lt_eq(other),
        }
    }

    fn __add__(&self, rhs: Self) -> PyResult<Self> {
        Ok(dsl::binary_expr(self.inner.clone(), Operator::Plus, rhs.inner).into())
    }
    fn __sub__(&self, rhs: Self) -> PyResult<Self> {
        Ok(dsl::binary_expr(self.inner.clone(), Operator::Minus, rhs.inner).into())
    }
    fn __mul__(&self, rhs: Self) -> PyResult<Self> {
        Ok(dsl::binary_expr(self.inner.clone(), Operator::Multiply, rhs.inner).into())
    }
    fn __truediv__(&self, rhs: Self) -> PyResult<Self> {
        Ok(dsl::binary_expr(self.inner.clone(), Operator::TrueDivide, rhs.inner).into())
    }
    fn __mod__(&self, rhs: Self) -> PyResult<Self> {
        Ok(dsl::binary_expr(self.inner.clone(), Operator::Modulus, rhs.inner).into())
    }
    fn __floordiv__(&self, rhs: Self) -> PyResult<Self> {
        Ok(dsl::binary_expr(self.inner.clone(), Operator::FloorDivide, rhs.inner).into())
    }
    fn __neg__(&self) -> PyResult<Self> {
        Ok(self.inner.clone().neg().into())
    }

    fn to_str(&self) -> String {
        format!("{:?}", self.inner)
    }
    fn eq(&self, other: Self) -> Self {
        self.inner.clone().eq(other.inner).into()
    }

    fn eq_missing(&self, other: Self) -> Self {
        self.inner.clone().eq_missing(other.inner).into()
    }
    fn neq(&self, other: Self) -> Self {
        self.inner.clone().neq(other.inner).into()
    }
    fn neq_missing(&self, other: Self) -> Self {
        self.inner.clone().neq_missing(other.inner).into()
    }
    fn gt(&self, other: Self) -> Self {
        self.inner.clone().gt(other.inner).into()
    }
    fn gt_eq(&self, other: Self) -> Self {
        self.inner.clone().gt_eq(other.inner).into()
    }
    fn lt_eq(&self, other: Self) -> Self {
        self.inner.clone().lt_eq(other.inner).into()
    }
    fn lt(&self, other: Self) -> Self {
        self.inner.clone().lt(other.inner).into()
    }

    fn alias(&self, name: &str) -> Self {
        self.inner.clone().alias(name).into()
    }
    fn not_(&self) -> Self {
        self.inner.clone().not().into()
    }
    fn is_null(&self) -> Self {
        self.inner.clone().is_null().into()
    }
    fn is_not_null(&self) -> Self {
        self.inner.clone().is_not_null().into()
    }

    fn is_infinite(&self) -> Self {
        self.inner.clone().is_infinite().into()
    }

    fn is_finite(&self) -> Self {
        self.inner.clone().is_finite().into()
    }

    fn is_nan(&self) -> Self {
        self.inner.clone().is_nan().into()
    }

    fn is_not_nan(&self) -> Self {
        self.inner.clone().is_not_nan().into()
    }

    fn min(&self) -> Self {
        self.inner.clone().min().into()
    }
    fn max(&self) -> Self {
        self.inner.clone().max().into()
    }
    #[cfg(feature = "propagate_nans")]
    fn nan_max(&self) -> Self {
        self.inner.clone().nan_max().into()
    }
    #[cfg(feature = "propagate_nans")]
    fn nan_min(&self) -> Self {
        self.inner.clone().nan_min().into()
    }
    fn mean(&self) -> Self {
        self.inner.clone().mean().into()
    }
    fn median(&self) -> Self {
        self.inner.clone().median().into()
    }
    fn sum(&self) -> Self {
        self.inner.clone().sum().into()
    }
    fn n_unique(&self) -> Self {
        self.inner.clone().n_unique().into()
    }
    fn arg_unique(&self) -> Self {
        self.inner.clone().arg_unique().into()
    }
    fn unique(&self) -> Self {
        self.inner.clone().unique().into()
    }
    fn unique_stable(&self) -> Self {
        self.inner.clone().unique_stable().into()
    }
    fn first(&self) -> Self {
        self.inner.clone().first().into()
    }
    fn last(&self) -> Self {
        self.inner.clone().last().into()
    }
    fn implode(&self) -> Self {
        self.inner.clone().implode().into()
    }
    fn quantile(&self, quantile: Self, interpolation: Wrap<QuantileInterpolOptions>) -> Self {
        self.inner
            .clone()
            .quantile(quantile.inner, interpolation.0)
            .into()
    }

    #[pyo3(signature = (breaks, labels, left_closed, include_breaks))]
    #[cfg(feature = "cutqcut")]
    fn cut(
        &self,
        breaks: Vec<f64>,
        labels: Option<Vec<String>>,
        left_closed: bool,
        include_breaks: bool,
    ) -> Self {
        self.inner
            .clone()
            .cut(breaks, labels, left_closed, include_breaks)
            .into()
    }
    #[pyo3(signature = (probs, labels, left_closed, allow_duplicates, include_breaks))]
    #[cfg(feature = "cutqcut")]
    fn qcut(
        &self,
        probs: Vec<f64>,
        labels: Option<Vec<String>>,
        left_closed: bool,
        allow_duplicates: bool,
        include_breaks: bool,
    ) -> Self {
        self.inner
            .clone()
            .qcut(probs, labels, left_closed, allow_duplicates, include_breaks)
            .into()
    }
    #[pyo3(signature = (n_bins, labels, left_closed, allow_duplicates, include_breaks))]
    #[cfg(feature = "cutqcut")]
    fn qcut_uniform(
        &self,
        n_bins: usize,
        labels: Option<Vec<String>>,
        left_closed: bool,
        allow_duplicates: bool,
        include_breaks: bool,
    ) -> Self {
        self.inner
            .clone()
            .qcut_uniform(
                n_bins,
                labels,
                left_closed,
                allow_duplicates,
                include_breaks,
            )
            .into()
    }

    #[cfg(feature = "rle")]
    fn rle(&self) -> Self {
        self.inner.clone().rle().into()
    }
    #[cfg(feature = "rle")]
    fn rle_id(&self) -> Self {
        self.inner.clone().rle_id().into()
    }

    fn agg_groups(&self) -> Self {
        self.inner.clone().agg_groups().into()
    }
    fn count(&self) -> Self {
        self.inner.clone().count().into()
    }
    fn len(&self) -> Self {
        self.inner.clone().len().into()
    }
    fn value_counts(&self, sort: bool, parallel: bool, name: String, normalize: bool) -> Self {
        self.inner
            .clone()
            .value_counts(sort, parallel, name, normalize)
            .into()
    }
    fn unique_counts(&self) -> Self {
        self.inner.clone().unique_counts().into()
    }
    fn null_count(&self) -> Self {
        self.inner.clone().null_count().into()
    }
    fn cast(&self, data_type: Wrap<DataType>, strict: bool, wrap_numerical: bool) -> Self {
        let dt = data_type.0;

        let options = if wrap_numerical {
            CastOptions::Overflowing
        } else if strict {
            CastOptions::Strict
        } else {
            CastOptions::NonStrict
        };

        let expr = self.inner.clone().cast_with_options(dt, options);
        expr.into()
    }
    fn sort_with(&self, descending: bool, nulls_last: bool) -> Self {
        self.inner
            .clone()
            .sort(SortOptions {
                descending,
                nulls_last,
                multithreaded: true,
                maintain_order: false,
            })
            .into()
    }

    fn arg_sort(&self, descending: bool, nulls_last: bool) -> Self {
        self.inner
            .clone()
            .arg_sort(SortOptions {
                descending,
                nulls_last,
                multithreaded: true,
                maintain_order: false,
            })
            .into()
    }

    #[cfg(feature = "top_k")]
    fn top_k(&self, k: Self) -> Self {
        self.inner.clone().top_k(k.inner).into()
    }

    #[cfg(feature = "top_k")]
    fn top_k_by(&self, by: Vec<Self>, k: Self, reverse: Vec<bool>) -> Self {
        let by = by.into_iter().map(|e| e.inner).collect::<Vec<_>>();
        self.inner.clone().top_k_by(k.inner, by, reverse).into()
    }

    #[cfg(feature = "top_k")]
    fn bottom_k(&self, k: Self) -> Self {
        self.inner.clone().bottom_k(k.inner).into()
    }

    #[cfg(feature = "top_k")]
    fn bottom_k_by(&self, by: Vec<Self>, k: Self, reverse: Vec<bool>) -> Self {
        let by = by.into_iter().map(|e| e.inner).collect::<Vec<_>>();
        self.inner.clone().bottom_k_by(k.inner, by, reverse).into()
    }

    #[cfg(feature = "peaks")]
    fn peak_min(&self) -> Self {
        self.inner.clone().peak_min().into()
    }

    #[cfg(feature = "peaks")]
    fn peak_max(&self) -> Self {
        self.inner.clone().peak_max().into()
    }

    fn arg_max(&self) -> Self {
        self.inner.clone().arg_max().into()
    }

    fn arg_min(&self) -> Self {
        self.inner.clone().arg_min().into()
    }

    #[cfg(feature = "search_sorted")]
    fn search_sorted(&self, element: Self, side: Wrap<SearchSortedSide>) -> Self {
        self.inner
            .clone()
            .search_sorted(element.inner, side.0)
            .into()
    }
    fn gather(&self, idx: Self) -> Self {
        self.inner.clone().gather(idx.inner).into()
    }

    fn get(&self, idx: Self) -> Self {
        self.inner.clone().get(idx.inner).into()
    }

    fn sort_by(
        &self,
        by: Vec<Self>,
        descending: Vec<bool>,
        nulls_last: Vec<bool>,
        multithreaded: bool,
        maintain_order: bool,
    ) -> Self {
        let by = by.into_iter().map(|e| e.inner).collect::<Vec<_>>();
        self.inner
            .clone()
            .sort_by(
                by,
                SortMultipleOptions {
                    descending,
                    nulls_last,
                    multithreaded,
                    maintain_order,
                },
            )
            .into()
    }

    fn backward_fill(&self, limit: FillNullLimit) -> Self {
        self.inner.clone().backward_fill(limit).into()
    }

    fn forward_fill(&self, limit: FillNullLimit) -> Self {
        self.inner.clone().forward_fill(limit).into()
    }

    fn shift(&self, n: Self, fill_value: Option<Self>) -> Self {
        let expr = self.inner.clone();
        let out = match fill_value {
            Some(v) => expr.shift_and_fill(n.inner, v.inner),
            None => expr.shift(n.inner),
        };
        out.into()
    }

    fn fill_null(&self, expr: Self) -> Self {
        self.inner.clone().fill_null(expr.inner).into()
    }

    fn fill_null_with_strategy(&self, strategy: &str, limit: FillNullLimit) -> PyResult<Self> {
        let strategy = parse_fill_null_strategy(strategy, limit)?;
        Ok(self.inner.clone().fill_null_with_strategy(strategy).into())
    }

    fn fill_nan(&self, expr: Self) -> Self {
        self.inner.clone().fill_nan(expr.inner).into()
    }

    fn drop_nulls(&self) -> Self {
        self.inner.clone().drop_nulls().into()
    }

    fn drop_nans(&self) -> Self {
        self.inner.clone().drop_nans().into()
    }

    fn filter(&self, predicate: Self) -> Self {
        self.inner.clone().filter(predicate.inner).into()
    }

    fn reverse(&self) -> Self {
        self.inner.clone().reverse().into()
    }

    fn std(&self, ddof: u8) -> Self {
        self.inner.clone().std(ddof).into()
    }

    fn var(&self, ddof: u8) -> Self {
        self.inner.clone().var(ddof).into()
    }

    fn is_unique(&self) -> Self {
        self.inner.clone().is_unique().into()
    }

    fn is_between(&self, lower: Self, upper: Self, closed: Wrap<ClosedInterval>) -> Self {
        self.inner
            .clone()
            .is_between(lower.inner, upper.inner, closed.0)
            .into()
    }

    fn approx_n_unique(&self) -> Self {
        self.inner.clone().approx_n_unique().into()
    }

    fn is_first_distinct(&self) -> Self {
        self.inner.clone().is_first_distinct().into()
    }

    fn is_last_distinct(&self) -> Self {
        self.inner.clone().is_last_distinct().into()
    }

    fn explode(&self) -> Self {
        self.inner.clone().explode().into()
    }

    fn gather_every(&self, n: usize, offset: usize) -> Self {
        self.inner.clone().gather_every(n, offset).into()
    }

    fn slice(&self, offset: Self, length: Self) -> Self {
        self.inner.clone().slice(offset.inner, length.inner).into()
    }

    fn head(&self, n: usize) -> Self {
        self.inner.clone().head(Some(n)).into()
    }

    fn tail(&self, n: usize) -> Self {
        self.inner.clone().tail(Some(n)).into()
    }

    fn append(&self, other: Self, upcast: bool) -> Self {
        self.inner.clone().append(other.inner, upcast).into()
    }

    fn rechunk(&self) -> Self {
        self.inner
            .clone()
            .map(|s| Ok(Some(s.rechunk())), GetOutput::same_type())
            .into()
    }

    fn round(&self, decimals: u32) -> Self {
        self.inner.clone().round(decimals).into()
    }

    fn round_sig_figs(&self, digits: i32) -> Self {
        self.clone().inner.round_sig_figs(digits).into()
    }

    fn floor(&self) -> Self {
        self.inner.clone().floor().into()
    }

    fn ceil(&self) -> Self {
        self.inner.clone().ceil().into()
    }

    fn clip(&self, min: Option<Self>, max: Option<Self>) -> Self {
        let expr = self.inner.clone();
        let out = match (min, max) {
            (Some(min), Some(max)) => expr.clip(min.inner, max.inner),
            (Some(min), None) => expr.clip_min(min.inner),
            (None, Some(max)) => expr.clip_max(max.inner),
            (None, None) => expr,
        };
        out.into()
    }

    fn abs(&self) -> Self {
        self.inner.clone().abs().into()
    }

    #[cfg(feature = "trigonometry")]
    fn sin(&self) -> Self {
        self.inner.clone().sin().into()
    }

    #[cfg(feature = "trigonometry")]
    fn cos(&self) -> Self {
        self.inner.clone().cos().into()
    }

    #[cfg(feature = "trigonometry")]
    fn tan(&self) -> Self {
        self.inner.clone().tan().into()
    }

    #[cfg(feature = "trigonometry")]
    fn cot(&self) -> Self {
        self.inner.clone().cot().into()
    }

    #[cfg(feature = "trigonometry")]
    fn arcsin(&self) -> Self {
        self.inner.clone().arcsin().into()
    }

    #[cfg(feature = "trigonometry")]
    fn arccos(&self) -> Self {
        self.inner.clone().arccos().into()
    }

    #[cfg(feature = "trigonometry")]
    fn arctan(&self) -> Self {
        self.inner.clone().arctan().into()
    }

    #[cfg(feature = "trigonometry")]
    fn arctan2(&self, y: Self) -> Self {
        self.inner.clone().arctan2(y.inner).into()
    }

    #[cfg(feature = "trigonometry")]
    fn sinh(&self) -> Self {
        self.inner.clone().sinh().into()
    }

    #[cfg(feature = "trigonometry")]
    fn cosh(&self) -> Self {
        self.inner.clone().cosh().into()
    }

    #[cfg(feature = "trigonometry")]
    fn tanh(&self) -> Self {
        self.inner.clone().tanh().into()
    }

    #[cfg(feature = "trigonometry")]
    fn arcsinh(&self) -> Self {
        self.inner.clone().arcsinh().into()
    }

    #[cfg(feature = "trigonometry")]
    fn arccosh(&self) -> Self {
        self.inner.clone().arccosh().into()
    }

    #[cfg(feature = "trigonometry")]
    fn arctanh(&self) -> Self {
        self.inner.clone().arctanh().into()
    }

    #[cfg(feature = "trigonometry")]
    pub fn degrees(&self) -> Self {
        self.inner.clone().degrees().into()
    }

    #[cfg(feature = "trigonometry")]
    pub fn radians(&self) -> Self {
        self.inner.clone().radians().into()
    }

    #[cfg(feature = "sign")]
    fn sign(&self) -> Self {
        self.inner.clone().sign().into()
    }

    fn is_duplicated(&self) -> Self {
        self.inner.clone().is_duplicated().into()
    }

    #[pyo3(signature = (partition_by, order_by, order_by_descending, order_by_nulls_last, mapping_strategy))]
    fn over(
        &self,
        partition_by: Vec<Self>,
        order_by: Option<Vec<Self>>,
        order_by_descending: bool,
        order_by_nulls_last: bool,
        mapping_strategy: Wrap<WindowMapping>,
    ) -> Self {
        let partition_by = partition_by
            .into_iter()
            .map(|e| e.inner)
            .collect::<Vec<Expr>>();

        let order_by = order_by.map(|order_by| {
            (
                order_by.into_iter().map(|e| e.inner).collect::<Vec<Expr>>(),
                SortOptions {
                    descending: order_by_descending,
                    nulls_last: order_by_nulls_last,
                    maintain_order: false,
                    ..Default::default()
                },
            )
        });

        self.inner
            .clone()
            .over_with_options(partition_by, order_by, mapping_strategy.0)
            .into()
    }

    fn rolling(
        &self,
        index_column: &str,
        period: &str,
        offset: &str,
        closed: Wrap<ClosedWindow>,
    ) -> Self {
        let options = RollingGroupOptions {
            index_column: index_column.into(),
            period: Duration::parse(period),
            offset: Duration::parse(offset),
            closed_window: closed.0,
        };

        self.inner.clone().rolling(options).into()
    }

    fn and_(&self, expr: Self) -> Self {
        self.inner.clone().and(expr.inner).into()
    }

    fn or_(&self, expr: Self) -> Self {
        self.inner.clone().or(expr.inner).into()
    }

    fn xor_(&self, expr: Self) -> Self {
        self.inner.clone().xor(expr.inner).into()
    }

    #[cfg(feature = "is_in")]
    fn is_in(&self, expr: Self) -> Self {
        self.inner.clone().is_in(expr.inner).into()
    }

    #[cfg(feature = "repeat_by")]
    fn repeat_by(&self, by: Self) -> Self {
        self.inner.clone().repeat_by(by.inner).into()
    }

    fn pow(&self, exponent: Self) -> Self {
        self.inner.clone().pow(exponent.inner).into()
    }

    fn sqrt(&self) -> Self {
        self.inner.clone().sqrt().into()
    }

    fn cbrt(&self) -> Self {
        self.inner.clone().cbrt().into()
    }

    fn cum_sum(&self, reverse: bool) -> Self {
        self.inner.clone().cum_sum(reverse).into()
    }
    fn cum_max(&self, reverse: bool) -> Self {
        self.inner.clone().cum_max(reverse).into()
    }
    fn cum_min(&self, reverse: bool) -> Self {
        self.inner.clone().cum_min(reverse).into()
    }
    fn cum_prod(&self, reverse: bool) -> Self {
        self.inner.clone().cum_prod(reverse).into()
    }
    fn cum_count(&self, reverse: bool) -> Self {
        self.inner.clone().cum_count(reverse).into()
    }

    fn cumulative_eval(&self, expr: Self, min_periods: usize, parallel: bool) -> Self {
        self.inner
            .clone()
            .cumulative_eval(expr.inner, min_periods, parallel)
            .into()
    }

    fn product(&self) -> Self {
        self.inner.clone().product().into()
    }

    fn shrink_dtype(&self) -> Self {
        self.inner.clone().shrink_dtype().into()
    }

    #[pyo3(signature = (lambda, output_type, agg_list, is_elementwise, returns_scalar))]
    fn map_batches(
        &self,
        lambda: PyObject,
        output_type: Option<Wrap<DataType>>,
        agg_list: bool,
        is_elementwise: bool,
        returns_scalar: bool,
    ) -> Self {
        map_single(
            self,
            lambda,
            output_type,
            agg_list,
            is_elementwise,
            returns_scalar,
        )
    }

    fn dot(&self, other: Self) -> Self {
        self.inner.clone().dot(other.inner).into()
    }

    fn reinterpret(&self, signed: bool) -> Self {
        self.inner.clone().reinterpret(signed).into()
    }
    fn mode(&self) -> Self {
        self.inner.clone().mode().into()
    }
    fn exclude(&self, columns: Vec<String>) -> Self {
        self.inner.clone().exclude(columns).into()
    }
    fn exclude_dtype(&self, dtypes: Vec<Wrap<DataType>>) -> Self {
        let dtypes = vec_extract_wrapped(dtypes);
        self.inner.clone().exclude_dtype(&dtypes).into()
    }
    fn interpolate(&self, method: Wrap<InterpolationMethod>) -> Self {
        self.inner.clone().interpolate(method.0).into()
    }
    fn interpolate_by(&self, by: PyExpr) -> Self {
        self.inner.clone().interpolate_by(by.inner).into()
    }

    fn lower_bound(&self) -> Self {
        self.inner.clone().lower_bound().into()
    }

    fn upper_bound(&self) -> Self {
        self.inner.clone().upper_bound().into()
    }

    fn rank(&self, method: Wrap<RankMethod>, descending: bool, seed: Option<u64>) -> Self {
        let options = RankOptions {
            method: method.0,
            descending,
        };
        self.inner.clone().rank(options, seed).into()
    }

    fn diff(&self, n: i64, null_behavior: Wrap<NullBehavior>) -> Self {
        self.inner.clone().diff(n, null_behavior.0).into()
    }

    #[cfg(feature = "pct_change")]
    fn pct_change(&self, n: Self) -> Self {
        self.inner.clone().pct_change(n.inner).into()
    }

    fn skew(&self, bias: bool) -> Self {
        self.inner.clone().skew(bias).into()
    }
    fn kurtosis(&self, fisher: bool, bias: bool) -> Self {
        self.inner.clone().kurtosis(fisher, bias).into()
    }

    fn reshape(&self, dims: Vec<i64>) -> Self {
        self.inner.clone().reshape(&dims, NestedType::Array).into()
    }

    fn to_physical(&self) -> Self {
        self.inner.clone().to_physical().into()
    }

    #[pyo3(signature = (seed))]
    fn shuffle(&self, seed: Option<u64>) -> Self {
        self.inner.clone().shuffle(seed).into()
    }

    #[pyo3(signature = (n, with_replacement, shuffle, seed))]
    fn sample_n(&self, n: Self, with_replacement: bool, shuffle: bool, seed: Option<u64>) -> Self {
        self.inner
            .clone()
            .sample_n(n.inner, with_replacement, shuffle, seed)
            .into()
    }

    #[pyo3(signature = (frac, with_replacement, shuffle, seed))]
    fn sample_frac(
        &self,
        frac: Self,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Self {
        self.inner
            .clone()
            .sample_frac(frac.inner, with_replacement, shuffle, seed)
            .into()
    }

    fn ewm_mean(&self, alpha: f64, adjust: bool, min_periods: usize, ignore_nulls: bool) -> Self {
        let options = EWMOptions {
            alpha,
            adjust,
            bias: false,
            min_periods,
            ignore_nulls,
        };
        self.inner.clone().ewm_mean(options).into()
    }
    fn ewm_mean_by(&self, times: PyExpr, half_life: &str) -> Self {
        let half_life = Duration::parse(half_life);
        self.inner
            .clone()
            .ewm_mean_by(times.inner, half_life)
            .into()
    }

    fn ewm_std(
        &self,
        alpha: f64,
        adjust: bool,
        bias: bool,
        min_periods: usize,
        ignore_nulls: bool,
    ) -> Self {
        let options = EWMOptions {
            alpha,
            adjust,
            bias,
            min_periods,
            ignore_nulls,
        };
        self.inner.clone().ewm_std(options).into()
    }
    fn ewm_var(
        &self,
        alpha: f64,
        adjust: bool,
        bias: bool,
        min_periods: usize,
        ignore_nulls: bool,
    ) -> Self {
        let options = EWMOptions {
            alpha,
            adjust,
            bias,
            min_periods,
            ignore_nulls,
        };
        self.inner.clone().ewm_var(options).into()
    }
    fn extend_constant(&self, value: PyExpr, n: PyExpr) -> Self {
        self.inner
            .clone()
            .extend_constant(value.inner, n.inner)
            .into()
    }

    fn any(&self, ignore_nulls: bool) -> Self {
        self.inner.clone().any(ignore_nulls).into()
    }
    fn all(&self, ignore_nulls: bool) -> Self {
        self.inner.clone().all(ignore_nulls).into()
    }

    fn log(&self, base: f64) -> Self {
        self.inner.clone().log(base).into()
    }

    fn log1p(&self) -> Self {
        self.inner.clone().log1p().into()
    }

    fn exp(&self) -> Self {
        self.inner.clone().exp().into()
    }

    fn entropy(&self, base: f64, normalize: bool) -> Self {
        self.inner.clone().entropy(base, normalize).into()
    }
    fn hash(&self, seed: u64, seed_1: u64, seed_2: u64, seed_3: u64) -> Self {
        self.inner.clone().hash(seed, seed_1, seed_2, seed_3).into()
    }
    fn set_sorted_flag(&self, descending: bool) -> Self {
        let is_sorted = if descending {
            IsSorted::Descending
        } else {
            IsSorted::Ascending
        };
        self.inner.clone().set_sorted_flag(is_sorted).into()
    }

    fn replace(&self, old: PyExpr, new: PyExpr) -> Self {
        self.inner.clone().replace(old.inner, new.inner).into()
    }

    fn replace_strict(
        &self,
        old: PyExpr,
        new: PyExpr,
        default: Option<PyExpr>,
        return_dtype: Option<Wrap<DataType>>,
    ) -> Self {
        self.inner
            .clone()
            .replace_strict(
                old.inner,
                new.inner,
                default.map(|e| e.inner),
                return_dtype.map(|dt| dt.0),
            )
            .into()
    }

    #[cfg(feature = "hist")]
    #[pyo3(signature = (bins, bin_count, include_category, include_breakpoint))]
    fn hist(
        &self,
        bins: Option<PyExpr>,
        bin_count: Option<usize>,
        include_category: bool,
        include_breakpoint: bool,
    ) -> Self {
        let bins = bins.map(|e| e.inner);
        self.inner
            .clone()
            .hist(bins, bin_count, include_category, include_breakpoint)
            .into()
    }
}
