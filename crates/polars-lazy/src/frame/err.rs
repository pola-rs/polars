/// Helper to delay a failing method until the query plan is collected
#[macro_export]
macro_rules! fallible {
    ($e:expr, $lf:expr) => {{
        use $crate::prelude::*;
        match $e {
            Ok(e) => e,
            Err(err) => {
                let lf: LazyFrame = LogicalPlanBuilder::from($lf.clone().logical_plan)
                    .add_err(err)
                    .0
                    .into();
                return lf;
            },
        }
    }};
}
