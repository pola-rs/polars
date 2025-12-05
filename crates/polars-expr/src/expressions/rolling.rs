use arrow::array::PrimitiveArray;
use polars_time::prelude::RollingWindower;
use polars_time::{ClosedWindow, Duration, PolarsTemporalGroupby, RollingGroupOptions};
use polars_utils::UnitVec;

use super::*;

pub(crate) struct RollingExpr {
    /// the root column that the Function will be applied on.
    /// This will be used to create a smaller DataFrame to prevent taking unneeded columns by index
    /// TODO! support keys?
    /// The challenge is that the group_by will reorder the results and the
    /// keys, and time index would need to be updated, or the result should be joined back
    /// For now, don't support it.
    ///
    /// A function Expr. i.e. Mean, Median, Max, etc.
    pub(crate) phys_function: Arc<dyn PhysicalExpr>,
    pub(crate) index_column: Arc<dyn PhysicalExpr>,
    pub(crate) period: Duration,
    pub(crate) offset: Duration,
    pub(crate) closed_window: ClosedWindow,
    pub(crate) expr: Expr,
    pub(crate) output_field: Field,
}

impl PhysicalExpr for RollingExpr {
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let groups = if let Some(index_column_name) = self.index_column.as_column() {
            let options = RollingGroupOptions {
                index_column: index_column_name.clone(),
                period: self.period,
                offset: self.offset,
                closed_window: self.closed_window,
            };
            let groups_key = format!("{options:?}");
            let groups = {
                // Groups must be set by expression runner.
                state.window_cache.get_groups(&groups_key)
            };

            // There can be multiple rolling expressions in a single expr.
            // E.g. `min().rolling() + max().rolling()`
            // So if we hit that we will compute them here.
            match groups {
                Some(groups) => groups,
                None => {
                    let (_time_key, groups) = df.rolling(None, &options)?;
                    state.window_cache.insert_groups(groups_key, groups.clone());
                    groups
                },
            }
        } else {
            let index_column_name = PlSmallStr::from_static("__PL_INDEX_COL");
            let options = RollingGroupOptions {
                index_column: index_column_name.clone(),
                period: self.period,
                offset: self.offset,
                closed_window: self.closed_window,
            };

            let index_column = self.index_column.evaluate(df, state)?;

            let mut df = df.clone();
            df.with_column(index_column.with_name(index_column_name))?;
            let (_time_key, groups) = df.rolling(None, &options)?;
            groups
        };

        let out = self
            .phys_function
            .evaluate_on_groups(df, &groups, state)?
            .finalize();
        polars_ensure!(out.len() == groups.len(), agg_len = out.len(), groups.len());
        Ok(out.into_column())
    }

    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut index_column = self.index_column.evaluate_on_groups(df, groups, state)?;

        index_column.groups();

        let mut index_column_data = index_column.flat_naive();
        use DataType as DT;
        let (time_unit, time_zone): (TimeUnit, Option<TimeZone>) = match index_column_data.dtype() {
            DT::Datetime(tu, tz) => (*tu, tz.clone()),
            DT::Date => (TimeUnit::Microseconds, None),
            DT::UInt32 | DT::UInt64 | DT::Int32 => {
                index_column_data = Cow::Owned(index_column_data.cast(&DT::Int64)?);
                (TimeUnit::Nanoseconds, None)
            },
            DT::Int64 => (TimeUnit::Nanoseconds, None),
            dt => polars_bail!(
                ComputeError:
                "expected any of the following dtypes: {{ Date, Datetime, Int32, Int64, UInt32, UInt64 }}, got {}",
                dt
            ),
        };
        let index_column_data =
            index_column_data.cast(&DataType::Datetime(time_unit, time_zone.clone()))?;

        // @NOTE: This is a bit strange since it ignores errors, but it mirrors the in-memory
        // engine.
        let tz = time_zone.and_then(|tz| tz.parse::<chrono_tz::Tz>().ok());

        polars_ensure!(
            index_column_data.null_count() == 0,
            ComputeError: "null values in `rolling` not supported, fill nulls."
        );
        let index_column_data = index_column_data.rechunk_to_arrow(CompatLevel::newest());
        let index_column_data = index_column_data
            .as_any()
            .downcast_ref::<PrimitiveArray<i64>>()
            .unwrap();
        let mut index_column_data = Cow::Borrowed(index_column_data.values().as_slice());
        let mut rolling =
            RollingWindower::new(self.period, self.offset, self.closed_window, time_unit, tz);

        let num_elements = groups.num_elements();

        // Convert the index groups to slices.
        //
        // This is not strictly necessary but allows us to reuse the existing `RollingWindower`
        // struct.
        let (slice_groups, overlapping, monotonic) = match &**index_column.groups {
            GroupsType::Idx(idx) => {
                let mut data = Vec::with_capacity(num_elements);
                let mut slices = Vec::with_capacity(groups.len());
                for i in idx.all() {
                    slices.push([data.len() as IdxSize, i.len() as IdxSize]);
                    data.extend(i.iter().map(|i| index_column_data[*i as usize]));
                }
                index_column_data = Cow::Owned(data);
                (Cow::Owned(slices), false, true)
            },
            GroupsType::Slice {
                groups,
                overlapping,
                monotonic,
            } => (Cow::Borrowed(groups), *overlapping, *monotonic),
        };

        // We need to make sure there are no length mismatches, otherwise we will have problems
        // down the line.
        assert_eq!(slice_groups.len(), groups.len());
        let length_mismatch = match &**groups {
            GroupsType::Idx(idx) => idx
                .all()
                .iter()
                .zip(slice_groups.iter())
                .map(|(i, [_, s])| (i.len(), *s as usize))
                .find(|(l, r)| *l != *r),
            GroupsType::Slice {
                groups,
                overlapping: _,
                monotonic: _,
            } => groups
                .iter()
                .zip(slice_groups.iter())
                .map(|([_, s1], [_, s2])| (*s1 as usize, *s2 as usize))
                .find(|(l, r)| *l != *r),
        };
        if let Some((l, r)) = length_mismatch {
            polars_bail!(length_mismatch = "rolling", l, r);
        }

        // Get the subslices within each group.
        let mut windows = Vec::with_capacity(num_elements);
        for [start, length] in slice_groups.as_ref() {
            rolling.reset();
            let time = &index_column_data[*start as usize..][..*length as usize];
            let offset = rolling.insert(&[time], &mut windows)?;
            let time = &time[offset as usize..];
            rolling.finalize(&[time], &mut windows);
        }

        // Create new groups as subgroups of the existing groups.
        let nested_groups = match &**groups {
            GroupsType::Idx(idx) => {
                let mut nested_groups = Vec::with_capacity(num_elements);
                let mut i = 0;
                for idx in idx.all() {
                    nested_groups.extend(windows[i..][..idx.len()].iter().map(|[s, l]| {
                        (
                            idx[*s as usize],
                            UnitVec::from_iter(idx[*s as usize..][..*l as usize].iter().copied()),
                        )
                    }));
                    i += idx.len();
                }
                GroupsType::Idx(nested_groups.into())
            },
            GroupsType::Slice {
                groups,
                overlapping: _,
                monotonic,
            } => {
                let mut nested_groups = Vec::with_capacity(num_elements);
                let mut i = 0;
                for [start, length] in groups {
                    nested_groups.extend(
                        windows[i..][..*length as usize]
                            .iter()
                            .map(|[s, l]| [*start + *s, *l]),
                    );
                    i += *length as usize;
                }
                GroupsType::new_slice(nested_groups, true, *monotonic)
            },
        };

        let nested_groups = nested_groups.into_sliceable();
        let out = self
            .phys_function
            .evaluate_on_groups(df, &nested_groups, state)?
            .finalize();
        polars_ensure!(
            out.len() == nested_groups.len(),
            agg_len = out.len(),
            nested_groups.len()
        );

        let out = AggregationContext {
            state: AggState::NotAggregated(out.into_column()),
            groups: Cow::Owned(
                GroupsType::new_slice(slice_groups.into_owned(), overlapping, monotonic)
                    .into_sliceable(),
            ),
            update_groups: UpdateGroups::No,
            original_len: false,
        };
        Ok(out)
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(self.output_field.clone())
    }

    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn is_scalar(&self) -> bool {
        false
    }
}
