use polars_core::prelude::*;

use crate::prelude::*;

pub(crate) fn det_join_schema(
    schema_left: &SchemaRef,
    schema_right: &SchemaRef,
    left_on: &[Expr],
    right_on: &[String],
    options: &JoinOptions,
) -> PolarsResult<SchemaRef> {
    match options.how {
        // semi and anti joins are just filtering operations
        // the schema will never change.
        #[cfg(feature = "semi_anti_join")]
        JoinType::Semi | JoinType::Anti => Ok(schema_left.clone()),
        _ => {
            // column names of left table
            let mut names: PlHashSet<&str> =
                PlHashSet::with_capacity(schema_left.len() + schema_right.len());
            let mut new_schema = Schema::with_capacity(schema_left.len() + schema_right.len());

            for (name, dtype) in schema_left.iter() {
                names.insert(name.as_str());
                new_schema.with_column(name.to_string(), dtype.clone())
            }

            // make sure that expression are assigned to the schema
            // an expression can have an alias, and change a dtype.
            // we only do this for the left hand side as the right hand side
            // is dropped.
            let mut arena = Arena::with_capacity(8);
            for e in left_on {
                let field = e.to_field_amortized(schema_left, Context::Default, &mut arena)?;
                new_schema.with_column(field.name, field.dtype);
                arena.clear();
            }

            let right_names: PlHashSet<_> = right_on.iter().map(|s| s.as_str()).collect();

            for (name, dtype) in schema_right.iter() {
                if !right_names.contains(name.as_str()) {
                    if names.contains(name.as_str()) {
                        #[cfg(feature = "asof_join")]
                        if let JoinType::AsOf(asof_options) = &options.how {
                            if let (Some(left_by), Some(right_by)) =
                                (&asof_options.left_by, &asof_options.right_by)
                            {
                                {
                                    // Do not add suffix. The column of the left table will be used
                                    if left_by.contains(name) && right_by.contains(name) {
                                        continue;
                                    }
                                }
                            }
                        }

                        let new_name = format!("{}{}", name, options.suffix.as_ref());
                        new_schema.with_column(new_name, dtype.clone());
                    } else {
                        new_schema.with_column(name.to_string(), dtype.clone());
                    }
                }
            }

            Ok(Arc::new(new_schema))
        }
    }
}
