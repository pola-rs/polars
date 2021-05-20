use crate::logical_plan::Context;
use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupBy;
use polars_core::prelude::*;
use std::sync::Arc;

pub struct WindowExpr {
    /// the root column that the Function will be applied on.
    /// This will be used to create a smaller DataFrame to prevent taking unneeded columns by index
    pub(crate) group_column: Arc<dyn PhysicalExpr>,
    pub(crate) apply_column: Arc<String>,
    pub(crate) out_name: Option<Arc<String>>,
    /// A function Expr. i.e. Mean, Median, Max, etc.
    pub(crate) function: Expr,
}

impl PhysicalExpr for WindowExpr {
    // Note: this was first implemented with expression evaluation but this performed really bad.
    // Therefore we choose the groupby -> apply -> self join approach
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        // This method does the following:
        // 1. determine groupby tuples based on the group_column
        // 2. apply an aggregation function
        // 3. join the results back to the original dataframe
        //    this stores all group values on the original df size
        // 4. select the final column and return

        // We create a key to store in the state cache
        // assume 32 digits per ptr.
        let mut key = String::with_capacity(df.width() * 32);
        df.get_columns()
            .iter()
            .for_each(|s| key.push_str(&format!("{}", s.get_data_ptr())));

        let groupby_column = self.group_column.evaluate(df, state)?;
        key.push_str(groupby_column.name());

        // 1. get the group tuples
        // We keep the lock for the entire window expression, we want those to be sequential
        // The utilize parallelism enough in groupby and join operation
        let mut groups_lock;

        // We have got this spin-lock because we can deadlock here. That's because in the gb.aggregations
        // below, like `sum`, `min` etc. the work is put on a rayon threadpool, stopping this thread
        // letting other threads do the aggregation, but rayon may also start a new window expression,
        // trying to acquire the lock that is held here. Therefore we spin while trying to lock,
        // and if it's held by another thread we release this thread.
        loop {
            match state.group_tuples.try_lock() {
                Ok(lock) => {
                    groups_lock = lock;
                    break;
                }
                Err(_) => {
                    // thread yield could still cause a dead lock, maybe because it remained
                    // high priority?
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        }
        let groups = match groups_lock.get_mut(&key) {
            Some(groups) => std::mem::take(groups),
            None => {
                let mut gb = df.groupby_with_series(vec![groupby_column.clone()], true)?;
                std::mem::take(gb.get_groups_mut())
            }
        };

        // 2. create GroupBy object and apply aggregation
        let mut gb = GroupBy::new(
            df,
            vec![groupby_column.clone()],
            groups,
            Some(vec![&self.apply_column]),
        );

        let out = match &self.function {
            Expr::Udf { function, .. } => {
                let mut df = gb.agg_list()?;
                df.may_apply_at_idx(1, |s| function.call_udf(s.clone()))?;
                Ok(df)
            }
            Expr::Agg(agg) => match agg {
                AggExpr::Median(_) => gb.median(),
                AggExpr::Mean(_) => gb.mean(),
                AggExpr::Max(_) => gb.max(),
                AggExpr::Min(_) => gb.min(),
                AggExpr::Sum(_) => gb.sum(),
                AggExpr::First(_) => gb.first(),
                AggExpr::Last(_) => gb.last(),
                AggExpr::Count(_) => gb.count(),
                AggExpr::NUnique(_) => gb.n_unique(),
                AggExpr::Quantile { quantile, .. } => gb.quantile(*quantile),
                AggExpr::List(_) => gb.agg_list(),
                AggExpr::AggGroups(_) => gb.groups(),
                AggExpr::Std(_) => gb.std(),
                AggExpr::Var(_) => gb.var(),
            },
            _ => Err(PolarsError::Other(
                format!(
                    "{:?} function not supported in window operation.\
                Note that you should use an aggregation",
                    self.function
                )
                .into(),
            )),
        }?;
        // store the group tuples and drop the lock so other threads may use them
        groups_lock.insert(key.clone(), std::mem::take(gb.get_groups_mut()));
        drop(groups_lock);

        // 3. get the join tuples and use them to take the new Series
        let out_column = out.select_at_idx(1).unwrap();
        let mut join_tuples_lock = state.join_tuples.lock().unwrap();
        let opt_join_tuples = match join_tuples_lock.get_mut(&key) {
            Some(t) => std::mem::take(t),
            None => {
                // group key from right column
                let right = out.select_at_idx(0).unwrap();
                groupby_column.hash_join_left(right)
            }
        };

        let mut iter = opt_join_tuples
            .iter()
            .map(|(_left, right)| right.map(|i| i as usize));
        let mut out = unsafe { out_column.take_opt_iter_unchecked(&mut iter) };
        join_tuples_lock.insert(key, opt_join_tuples);
        if let Some(name) = &self.out_name {
            out.rename(name.as_str());
        }
        Ok(out)
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.function.to_field(input_schema, Context::Default)
    }
}
