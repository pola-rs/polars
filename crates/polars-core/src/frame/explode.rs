use arrow::legacy::kernels::concatenate::concatenate_owned_unchecked;
use arrow::offset::OffsetsBuffer;
use rayon::prelude::*;
#[cfg(feature = "serde-lazy")]
use serde::{Deserialize, Serialize};
use smartstring::alias::String as SmartString;

use crate::chunked_array::ops::explode::offsets_to_indexes;
use crate::prelude::*;
use crate::series::IsSorted;
use crate::utils::try_get_supertype;
use crate::POOL;

fn get_exploded(series: &Series) -> PolarsResult<(Series, OffsetsBuffer<i64>)> {
    match series.dtype() {
        DataType::List(_) => series.list().unwrap().explode_and_offsets(),
        #[cfg(feature = "dtype-array")]
        DataType::Array(_, _) => series.array().unwrap().explode_and_offsets(),
        _ => polars_bail!(opq = explode, series.dtype()),
    }
}

/// Arguments for `[DataFrame::unpivot]` function
#[derive(Clone, Default, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde-lazy", derive(Serialize, Deserialize))]
pub struct UnpivotArgs {
    pub on: Vec<SmartString>,
    pub index: Vec<SmartString>,
    pub variable_name: Option<SmartString>,
    pub value_name: Option<SmartString>,
    /// Whether the unpivot may be done
    /// in the streaming engine
    /// This will not have a stable ordering
    pub streamable: bool,
}

impl DataFrame {
    pub fn explode_impl(&self, mut columns: Vec<Series>) -> PolarsResult<DataFrame> {
        polars_ensure!(!columns.is_empty(), InvalidOperation: "no columns provided in explode");
        let mut df = self.clone();
        if self.is_empty() {
            for s in &columns {
                df.with_column(s.explode()?)?;
            }
            return Ok(df);
        }
        columns.sort_by(|sa, sb| {
            self.check_name_to_idx(sa.name())
                .expect("checked above")
                .partial_cmp(&self.check_name_to_idx(sb.name()).expect("checked above"))
                .expect("cmp usize -> Ordering")
        });

        // first remove all the exploded columns
        for s in &columns {
            df = df.drop(s.name())?;
        }

        let exploded_columns = POOL.install(|| {
            columns
                .par_iter()
                .map(get_exploded)
                .collect::<PolarsResult<Vec<_>>>()
        })?;

        fn process_column(
            original_df: &DataFrame,
            df: &mut DataFrame,
            exploded: Series,
        ) -> PolarsResult<()> {
            if exploded.len() == df.height() || df.width() == 0 {
                let col_idx = original_df.check_name_to_idx(exploded.name())?;
                df.columns.insert(col_idx, exploded);
            } else {
                polars_bail!(
                    ShapeMismatch: "exploded column(s) {:?} doesn't have the same length: {} \
                    as the dataframe: {}", exploded.name(), exploded.name(), df.height(),
                );
            }
            Ok(())
        }

        let check_offsets = || {
            let first_offsets = exploded_columns[0].1.as_slice();
            for (_, offsets) in &exploded_columns[1..] {
                polars_ensure!(first_offsets == offsets.as_slice(),
                    ShapeMismatch: "exploded columns must have matching element counts"
                )
            }
            Ok(())
        };
        let process_first = || {
            let (exploded, offsets) = &exploded_columns[0];

            let row_idx = offsets_to_indexes(offsets.as_slice(), exploded.len());
            let mut row_idx = IdxCa::from_vec("", row_idx);
            row_idx.set_sorted_flag(IsSorted::Ascending);

            // SAFETY:
            // We just created indices that are in bounds.
            let mut df = unsafe { df.take_unchecked(&row_idx) };
            process_column(self, &mut df, exploded.clone())?;
            PolarsResult::Ok(df)
        };
        let (df, result) = POOL.join(process_first, check_offsets);
        let mut df = df?;
        result?;

        for (exploded, _) in exploded_columns.into_iter().skip(1) {
            process_column(self, &mut df, exploded)?
        }

        Ok(df)
    }
    /// Explode `DataFrame` to long format by exploding a column with Lists.
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use polars_core::prelude::*;
    /// let s0 = Series::new("a", &[1i64, 2, 3]);
    /// let s1 = Series::new("b", &[1i64, 1, 1]);
    /// let s2 = Series::new("c", &[2i64, 2, 2]);
    /// let list = Series::new("foo", &[s0, s1, s2]);
    ///
    /// let s0 = Series::new("B", [1, 2, 3]);
    /// let s1 = Series::new("C", [1, 1, 1]);
    /// let df = DataFrame::new(vec![list, s0, s1])?;
    /// let exploded = df.explode(["foo"])?;
    ///
    /// println!("{:?}", df);
    /// println!("{:?}", exploded);
    /// # Ok::<(), PolarsError>(())
    /// ```
    /// Outputs:
    ///
    /// ```text
    ///  +-------------+-----+-----+
    ///  | foo         | B   | C   |
    ///  | ---         | --- | --- |
    ///  | list [i64]  | i32 | i32 |
    ///  +=============+=====+=====+
    ///  | "[1, 2, 3]" | 1   | 1   |
    ///  +-------------+-----+-----+
    ///  | "[1, 1, 1]" | 2   | 1   |
    ///  +-------------+-----+-----+
    ///  | "[2, 2, 2]" | 3   | 1   |
    ///  +-------------+-----+-----+
    ///
    ///  +-----+-----+-----+
    ///  | foo | B   | C   |
    ///  | --- | --- | --- |
    ///  | i64 | i32 | i32 |
    ///  +=====+=====+=====+
    ///  | 1   | 1   | 1   |
    ///  +-----+-----+-----+
    ///  | 2   | 1   | 1   |
    ///  +-----+-----+-----+
    ///  | 3   | 1   | 1   |
    ///  +-----+-----+-----+
    ///  | 1   | 2   | 1   |
    ///  +-----+-----+-----+
    ///  | 1   | 2   | 1   |
    ///  +-----+-----+-----+
    ///  | 1   | 2   | 1   |
    ///  +-----+-----+-----+
    ///  | 2   | 3   | 1   |
    ///  +-----+-----+-----+
    ///  | 2   | 3   | 1   |
    ///  +-----+-----+-----+
    ///  | 2   | 3   | 1   |
    ///  +-----+-----+-----+
    /// ```
    pub fn explode<I, S>(&self, columns: I) -> PolarsResult<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        // We need to sort the column by order of original occurrence. Otherwise the insert by index
        // below will panic
        let columns = self.select_series(columns)?;
        self.explode_impl(columns)
    }

    ///
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
    pub fn unpivot<I, J>(&self, on: I, index: J) -> PolarsResult<Self>
    where
        I: IntoVec<SmartString>,
        J: IntoVec<SmartString>,
    {
        let index = index.into_vec();
        let on = on.into_vec();
        self.unpivot2(UnpivotArgs {
            on,
            index,
            ..Default::default()
        })
    }

    /// Similar to unpivot, but without generics. This may be easier if you want to pass
    /// an empty `index` or empty `on`.
    pub fn unpivot2(&self, args: UnpivotArgs) -> PolarsResult<Self> {
        let index = args.index;
        let mut on = args.on;

        let variable_name = args.variable_name.as_deref().unwrap_or("variable");
        let value_name = args.value_name.as_deref().unwrap_or("value");

        let len = self.height();

        // if value vars is empty we take all columns that are not in id_vars.
        if on.is_empty() {
            // return empty frame if there are no columns available to use as value vars
            if index.len() == self.width() {
                let variable_col = Series::new_empty(variable_name, &DataType::String);
                let value_col = Series::new_empty(variable_name, &DataType::Null);

                let mut out = self.select(index).unwrap().clear().columns;
                out.push(variable_col);
                out.push(value_col);

                return Ok(unsafe { DataFrame::new_no_checks(out) });
            }

            let index_set = PlHashSet::from_iter(index.iter().map(|s| s.as_str()));
            on = self
                .get_columns()
                .iter()
                .filter_map(|s| {
                    if index_set.contains(s.name()) {
                        None
                    } else {
                        Some(s.name().into())
                    }
                })
                .collect();
        }

        // values will all be placed in single column, so we must find their supertype
        let schema = self.schema();
        let mut iter = on.iter().map(|v| {
            schema
                .get(v)
                .ok_or_else(|| polars_err!(ColumnNotFound: "{}", v))
        });
        let mut st = iter.next().unwrap()?.clone();
        for dt in iter {
            st = try_get_supertype(&st, dt?)?;
        }

        // The column name of the variable that is unpivoted
        let mut variable_col = MutableBinaryViewArray::<str>::with_capacity(len * on.len() + 1);
        // prepare ids
        let ids_ = self.select_with_schema_unchecked(index, &schema)?;
        let mut ids = ids_.clone();
        if ids.width() > 0 {
            for _ in 0..on.len() - 1 {
                ids.vstack_mut_unchecked(&ids_)
            }
        }
        ids.as_single_chunk_par();
        drop(ids_);

        let mut values = Vec::with_capacity(on.len());

        for value_column_name in &on {
            variable_col.extend_constant(len, Some(value_column_name.as_str()));
            // ensure we go via the schema so we are O(1)
            // self.column() is linear
            // together with this loop that would make it O^2 over `on`
            let (pos, _name, _dtype) = schema.try_get_full(value_column_name)?;
            let col = &self.columns[pos];
            let value_col = col.cast(&st).map_err(
                |_| polars_err!(InvalidOperation: "'unpivot' not supported for dtype: {}", col.dtype()),
            )?;
            values.extend_from_slice(value_col.chunks())
        }
        let values_arr = concatenate_owned_unchecked(&values)?;
        // SAFETY:
        // The give dtype is correct
        let values =
            unsafe { Series::from_chunks_and_dtype_unchecked(value_name, vec![values_arr], &st) };

        let variable_col = variable_col.as_box();
        // SAFETY:
        // The given dtype is correct
        let variables = unsafe {
            Series::from_chunks_and_dtype_unchecked(
                variable_name,
                vec![variable_col],
                &DataType::String,
            )
        };

        ids.hstack_mut(&[variables, values])?;

        Ok(ids)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    #[cfg(feature = "dtype-i8")]
    #[cfg_attr(miri, ignore)]
    fn test_explode() {
        let s0 = Series::new("a", &[1i8, 2, 3]);
        let s1 = Series::new("b", &[1i8, 1, 1]);
        let s2 = Series::new("c", &[2i8, 2, 2]);
        let list = Series::new("foo", &[s0, s1, s2]);

        let s0 = Series::new("B", [1, 2, 3]);
        let s1 = Series::new("C", [1, 1, 1]);
        let df = DataFrame::new(vec![list, s0.clone(), s1.clone()]).unwrap();
        let exploded = df.explode(["foo"]).unwrap();
        assert_eq!(exploded.shape(), (9, 3));
        assert_eq!(exploded.column("C").unwrap().i32().unwrap().get(8), Some(1));
        assert_eq!(exploded.column("B").unwrap().i32().unwrap().get(8), Some(3));
        assert_eq!(
            exploded.column("foo").unwrap().i8().unwrap().get(8),
            Some(2)
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_explode_df_empty_list() -> PolarsResult<()> {
        let s0 = Series::new("a", &[1, 2, 3]);
        let s1 = Series::new("b", &[1, 1, 1]);
        let list = Series::new("foo", &[s0, s1.clone(), s1.clear()]);
        let s0 = Series::new("B", [1, 2, 3]);
        let s1 = Series::new("C", [1, 1, 1]);
        let df = DataFrame::new(vec![list, s0.clone(), s1.clone()])?;

        let out = df.explode(["foo"])?;
        let expected = df![
            "foo" => [Some(1), Some(2), Some(3), Some(1), Some(1), Some(1), None],
            "B" => [1, 1, 1, 2, 2, 2, 3],
            "C" => [1, 1, 1, 1, 1, 1, 1],
        ]?;

        assert!(out.equals_missing(&expected));

        let list = Series::new("foo", [s0.clone(), s1.clear(), s1.clone()]);
        let df = DataFrame::new(vec![list, s0, s1])?;
        let out = df.explode(["foo"])?;
        let expected = df![
            "foo" => [Some(1), Some(2), Some(3), None, Some(1), Some(1), Some(1)],
            "B" => [1, 1, 1, 2, 3, 3, 3],
            "C" => [1, 1, 1, 1, 1, 1, 1],
        ]?;

        assert!(out.equals_missing(&expected));
        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_explode_single_col() -> PolarsResult<()> {
        let s0 = Series::new("a", &[1i32, 2, 3]);
        let s1 = Series::new("b", &[1i32, 1, 1]);
        let list = Series::new("foo", &[s0, s1]);
        let df = DataFrame::new(vec![list])?;

        let out = df.explode(["foo"])?;
        let out = out
            .column("foo")?
            .i32()?
            .into_no_null_iter()
            .collect::<Vec<_>>();
        assert_eq!(out, &[1i32, 2, 3, 1, 1, 1]);

        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_unpivot() -> PolarsResult<()> {
        let df = df!("A" => &["a", "b", "a"],
         "B" => &[1, 3, 5],
         "C" => &[10, 11, 12],
         "D" => &[2, 4, 6]
        )
        .unwrap();

        let unpivoted = df.unpivot(["C", "D"], ["A", "B"])?;
        assert_eq!(
            Vec::from(unpivoted.column("value")?.i32()?),
            &[Some(10), Some(11), Some(12), Some(2), Some(4), Some(6)]
        );

        let args = UnpivotArgs {
            on: vec![],
            index: vec![],
            ..Default::default()
        };

        let unpivoted = df.unpivot2(args).unwrap();
        let value = unpivoted.column("value")?;
        // String because of supertype
        let value = value.str()?;
        let value = value.into_no_null_iter().collect::<Vec<_>>();
        assert_eq!(
            value,
            &["a", "b", "a", "1", "3", "5", "10", "11", "12", "2", "4", "6"]
        );

        let args = UnpivotArgs {
            on: vec![],
            index: vec!["A".into()],
            ..Default::default()
        };

        let unpivoted = df.unpivot2(args).unwrap();
        let value = unpivoted.column("value")?;
        let value = value.i32()?;
        let value = value.into_no_null_iter().collect::<Vec<_>>();
        assert_eq!(value, &[1, 3, 5, 10, 11, 12, 2, 4, 6]);
        let variable = unpivoted.column("variable")?;
        let variable = variable.str()?;
        let variable = variable.into_no_null_iter().collect::<Vec<_>>();
        assert_eq!(variable, &["B", "B", "B", "C", "C", "C", "D", "D", "D"]);
        assert!(unpivoted.column("A").is_ok());
        Ok(())
    }
}
