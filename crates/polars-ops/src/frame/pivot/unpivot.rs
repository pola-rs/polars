use arrow::array::{MutableArray, MutablePlString};
use arrow::compute::concatenate::concatenate_unchecked;
use polars_core::datatypes::{DataType, PlSmallStr};
use polars_core::frame::DataFrame;
use polars_core::frame::column::Column;
use polars_core::prelude::{IntoVec, Series, UnpivotArgsIR};
use polars_core::utils::merge_dtypes_many;
use polars_error::{PolarsResult, polars_err};
use polars_utils::aliases::PlHashSet;

use crate::frame::IntoDf;

pub trait UnpivotDF: IntoDf {
    /// Unpivot a `DataFrame` from wide to long format.
    ///
    /// # Example
    ///
    /// # Arguments
    ///
    /// * `on` - String slice that represent the columns to use as value variables.
    /// * `index` - String slice that represent the columns to use as id variables.
    ///
    /// If `on` is empty all columns that are not in `index` will be used.
    ///
    /// ```ignore
    /// # use polars_core::prelude::*;
    /// let df = df!("A" => &["a", "b", "a"],
    ///              "B" => &[1, 3, 5],
    ///              "C" => &[10, 11, 12],
    ///              "D" => &[2, 4, 6]
    ///     )?;
    ///
    /// let unpivoted = df.unpivot(&["A", "B"], &["C", "D"])?;
    /// println!("{:?}", df);
    /// println!("{:?}", unpivoted);
    /// # Ok::<(), PolarsError>(())
    /// ```
    /// Outputs:
    /// ```text
    ///  +-----+-----+-----+-----+
    ///  | A   | B   | C   | D   |
    ///  | --- | --- | --- | --- |
    ///  | str | i32 | i32 | i32 |
    ///  +=====+=====+=====+=====+
    ///  | "a" | 1   | 10  | 2   |
    ///  +-----+-----+-----+-----+
    ///  | "b" | 3   | 11  | 4   |
    ///  +-----+-----+-----+-----+
    ///  | "a" | 5   | 12  | 6   |
    ///  +-----+-----+-----+-----+
    ///
    ///  +-----+-----+----------+-------+
    ///  | A   | B   | variable | value |
    ///  | --- | --- | ---      | ---   |
    ///  | str | i32 | str      | i32   |
    ///  +=====+=====+==========+=======+
    ///  | "a" | 1   | "C"      | 10    |
    ///  +-----+-----+----------+-------+
    ///  | "b" | 3   | "C"      | 11    |
    ///  +-----+-----+----------+-------+
    ///  | "a" | 5   | "C"      | 12    |
    ///  +-----+-----+----------+-------+
    ///  | "a" | 1   | "D"      | 2     |
    ///  +-----+-----+----------+-------+
    ///  | "b" | 3   | "D"      | 4     |
    ///  +-----+-----+----------+-------+
    ///  | "a" | 5   | "D"      | 6     |
    ///  +-----+-----+----------+-------+
    /// ```
    fn unpivot<I, J>(&self, on: I, index: J) -> PolarsResult<DataFrame>
    where
        I: IntoVec<PlSmallStr>,
        J: IntoVec<PlSmallStr>,
    {
        let index = index.into_vec();
        let on = on.into_vec();
        self.unpivot2(UnpivotArgsIR {
            on,
            index,
            ..Default::default()
        })
    }

    /// Similar to unpivot, but without generics. This may be easier if you want to pass
    /// an empty `index` or empty `on`.
    fn unpivot2(&self, args: UnpivotArgsIR) -> PolarsResult<DataFrame> {
        let self_ = self.to_df();
        let index = args.index;
        let mut on = args.on;

        let variable_name = args
            .variable_name
            .unwrap_or_else(|| PlSmallStr::from_static("variable"));
        let value_name = args
            .value_name
            .unwrap_or_else(|| PlSmallStr::from_static("value"));

        if self_.get_columns().is_empty() {
            return DataFrame::new(vec![
                Column::new_empty(variable_name, &DataType::String),
                Column::new_empty(value_name, &DataType::Null),
            ]);
        }

        let len = self_.height();

        // If value vars is empty we take all columns that are not in id_vars.
        if on.is_empty() {
            // Return empty frame if there are no columns available to use as value vars.
            if index.len() == self_.width() {
                let variable_col = Column::new_empty(variable_name, &DataType::String);
                let value_col = Column::new_empty(value_name, &DataType::Null);

                let mut out = self_.select(index).unwrap().clear().take_columns();

                out.push(variable_col);
                out.push(value_col);

                return Ok(unsafe { DataFrame::new_no_checks(0, out) });
            }

            let index_set = PlHashSet::from_iter(index.iter().cloned());
            on = self_
                .get_columns()
                .iter()
                .filter_map(|s| {
                    if index_set.contains(s.name()) {
                        None
                    } else {
                        Some(s.name().clone())
                    }
                })
                .collect();
        }

        // Values will all be placed in single column, so we must find their supertype
        let schema = self_.schema();
        let dtypes = on
            .iter()
            .map(|v| schema.get(v).ok_or_else(|| polars_err!(col_not_found = v)))
            .collect::<PolarsResult<Vec<_>>>()?;

        let st = merge_dtypes_many(dtypes.iter())?;

        // The column name of the variable that is unpivoted
        let mut variable_col = MutablePlString::with_capacity(len * on.len() + 1);
        // prepare ids
        let ids_ = self_.select_with_schema_unchecked(index, schema)?;
        let mut ids = ids_.clone();
        if ids.width() > 0 {
            for _ in 0..on.len() - 1 {
                ids.vstack_mut_unchecked(&ids_)
            }
        }
        ids.as_single_chunk_par();
        drop(ids_);

        let mut values = Vec::with_capacity(on.len());
        let columns = self_.get_columns();

        for value_column_name in &on {
            variable_col.extend_constant(len, Some(value_column_name.as_str()));
            // ensure we go via the schema so we are O(1)
            // self.column() is linear
            // together with this loop that would make it O^2 over `on`
            let (pos, _name, _dtype) = schema.try_get_full(value_column_name)?;
            let col = &columns[pos];
            let value_col = col.cast(&st).map_err(
                |_| polars_err!(InvalidOperation: "'unpivot' not supported for dtype: {}\n\nConsider casting to String.", col.dtype()),
            )?;
            values.extend_from_slice(value_col.as_materialized_series().chunks())
        }
        let values_arr = concatenate_unchecked(&values)?;
        // SAFETY:
        // The give dtype is correct
        let values =
            unsafe { Series::from_chunks_and_dtype_unchecked(value_name, vec![values_arr], &st) }
                .into();

        let variable_col = variable_col.as_box();
        // SAFETY:
        // The given dtype is correct
        let variables = unsafe {
            Series::from_chunks_and_dtype_unchecked(
                variable_name,
                vec![variable_col],
                &DataType::String,
            )
        }
        .into();

        ids.hstack_mut(&[variables, values])?;

        Ok(ids)
    }
}

impl UnpivotDF for DataFrame {}

#[cfg(test)]
mod test {
    use polars_core::df;
    use polars_core::utils::Container;

    use super::*;

    #[test]
    fn test_unpivot() -> PolarsResult<()> {
        let df = df!("A" => &["a", "b", "a"],
         "B" => &[1, 3, 5],
         "C" => &[10, 11, 12],
         "D" => &[2, 4, 6]
        )
        .unwrap();

        // Specify on and index
        let unpivoted = df.unpivot(["C", "D"], ["A", "B"])?;
        assert_eq!(
            unpivoted.get_column_names(),
            &["A", "B", "variable", "value"]
        );
        assert_eq!(
            Vec::from(unpivoted.column("value")?.i32()?),
            &[Some(10), Some(11), Some(12), Some(2), Some(4), Some(6)]
        );

        // Specify custom column names
        let args = UnpivotArgsIR {
            on: vec!["C".into(), "D".into()],
            index: vec!["A".into(), "B".into()],
            variable_name: Some("custom_variable".into()),
            value_name: Some("custom_value".into()),
        };
        let unpivoted = df.unpivot2(args).unwrap();
        assert_eq!(
            unpivoted.get_column_names(),
            &["A", "B", "custom_variable", "custom_value"]
        );

        // Specify neither on nor index
        let args = UnpivotArgsIR {
            on: vec![],
            index: vec![],
            ..Default::default()
        };

        let unpivoted = df.unpivot2(args).unwrap();
        assert_eq!(unpivoted.get_column_names(), &["variable", "value"]);
        let value = unpivoted.column("value")?;
        // String because of supertype
        let value = value.str()?;
        let value = value.into_no_null_iter().collect::<Vec<_>>();
        assert_eq!(
            value,
            &[
                "a", "b", "a", "1", "3", "5", "10", "11", "12", "2", "4", "6"
            ]
        );

        // Specify index but not on
        let args = UnpivotArgsIR {
            on: vec![],
            index: vec!["A".into()],
            ..Default::default()
        };

        let unpivoted = df.unpivot2(args).unwrap();
        assert_eq!(unpivoted.get_column_names(), &["A", "variable", "value"]);
        let value = unpivoted.column("value")?;
        let value = value.i32()?;
        let value = value.into_no_null_iter().collect::<Vec<_>>();
        assert_eq!(value, &[1, 3, 5, 10, 11, 12, 2, 4, 6]);
        let variable = unpivoted.column("variable")?;
        let variable = variable.str()?;
        let variable = variable.into_no_null_iter().collect::<Vec<_>>();
        assert_eq!(variable, &["B", "B", "B", "C", "C", "C", "D", "D", "D"]);
        assert!(unpivoted.column("A").is_ok());

        // Specify all columns in index
        let args = UnpivotArgsIR {
            on: vec![],
            index: vec!["A".into(), "B".into(), "C".into(), "D".into()],
            ..Default::default()
        };
        let unpivoted = df.unpivot2(args).unwrap();
        assert_eq!(
            unpivoted.get_column_names(),
            &["A", "B", "C", "D", "variable", "value"]
        );
        assert_eq!(unpivoted.len(), 0);

        Ok(())
    }
}
