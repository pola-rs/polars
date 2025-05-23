use polars_core::prelude::*;
use polars_plan::dsl::function_expr::cat::get_cat_phys_map;
use polars_plan::dsl::*;

use crate::physical_plan::exotic::prepare_expression_for_context;
use crate::prelude::*;

pub trait IntoCategoricalNameSpace {
    fn into_categorical_name_space(self) -> CategoricalNameSpace;
}

impl IntoCategoricalNameSpace for CategoricalNameSpace {
    fn into_categorical_name_space(self) -> CategoricalNameSpace {
        self
    }
}

pub trait CategoricalNameSpaceExtension: IntoCategoricalNameSpace + Sized {
    fn cat_str_eval(self, expr: Expr) -> Expr {
        let this = self.into_categorical_name_space();
        let expr2 = expr.clone();

        let func = move |c: Column| {
            for e in expr.into_iter() {
                if let Expr::Column(name) = e {
                    polars_ensure!(
                        name.is_empty(),
                        InvalidOperation:
                        "named columns are not allowed in 'cat.eval'; consider using 'element' or 'col(\"\")'"
                    );
                }
            }

            // Extract the categories and the corresponding local index.
            let ca = c.categorical()?;
            let (categories, phys) = get_cat_phys_map(ca);
            let n_categories = categories.len();

            // Evaluate the expression on the categories.
            let state = ExecutionState::new();
            let df = categories
                .with_name(PlSmallStr::EMPTY)
                .into_series()
                .into_frame();
            let phys_expr = prepare_expression_for_context(
                PlSmallStr::EMPTY,
                &expr,
                &DataType::String,
                Context::Default,
            )?;
            let result = phys_expr.evaluate(&df, &state)?;
            polars_ensure!(
                n_categories == result.len(),
                InvalidOperation: "'eval' expression must not change series length",
            );

            // Re-broadcast the result back to the original idx.
            let out = unsafe { result.take_unchecked(phys.idx().unwrap()) };
            Ok(Some(out.into_column()))
        };

        this.0
            .map(
                func,
                GetOutput::map_field(move |f| Ok(eval_field_to_dtype(f, &expr2, true))),
            )
            .with_fmt("eval")
    }
}

impl CategoricalNameSpaceExtension for CategoricalNameSpace {}
