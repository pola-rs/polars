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

/// Arguments for `[DataFrame::melt]` function
#[derive(Clone, Default, Debug, PartialEq)]
#[cfg_attr(feature = "serde-lazy", derive(Serialize, Deserialize))]
pub struct MeltArgs {
    pub id_vars: Vec<SmartString>,
    pub value_vars: Vec<SmartString>,
    pub variable_name: Option<SmartString>,
    pub value_name: Option<SmartString>,
    /// Whether the melt may be done
    /// in the streaming engine
    /// This will not have a stable ordering
    pub streamable: bool,
}

impl DataFrame {
    pub fn explode_impl(&self, mut columns: Vec<Series>) -> PolarsResult<DataFrame> {
        polars_ensure!(!columns.is_empty(), InvalidOperation: "no columns provided in explode");
        let mut df = self.clone();
        if self.height() == 0 {
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

            // Safety
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
    /// * `id_vars` - String slice that represent the columns to use as id variables.
    /// * `value_vars` - String slice that represent the columns to use as value variables.
    ///
    /// If `value_vars` is empty all columns that are not in `id_vars` will be used.
    ///
    /// ```ignore
    /// # use polars_core::prelude::*;
    /// let df = df!("A" => &["a", "b", "a"],
    ///              "B" => &[1, 3, 5],
    ///              "C" => &[10, 11, 12],
    ///              "D" => &[2, 4, 6]
    ///     )?;
    ///
    /// let melted = df.melt(&["A", "B"], &["C", "D"])?;
    /// println!("{:?}", df);
    /// println!("{:?}", melted);
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
    pub fn melt<I, J>(&self, id_vars: I, value_vars: J) -> PolarsResult<Self>
    where
        I: IntoVec<SmartString>,
        J: IntoVec<SmartString>,
    {
        let id_vars = id_vars.into_vec();
        let value_vars = value_vars.into_vec();
        self.melt2(MeltArgs {
            id_vars,
            value_vars,
            ..Default::default()
        })
    }

    /// Similar to melt, but without generics. This may be easier if you want to pass
    /// an empty `id_vars` or empty `value_vars`.
    pub fn melt2(&self, args: MeltArgs) -> PolarsResult<Self> {
        let id_vars = args.id_vars;
        let mut value_vars = args.value_vars;

        let value_name = args.value_name.as_deref().unwrap_or("value");
        let variable_name = args.variable_name.as_deref().unwrap_or("variable");

        let len = self.height();

        // if value vars is empty we take all columns that are not in id_vars.
        if value_vars.is_empty() {
            let id_vars_set = PlHashSet::from_iter(id_vars.iter().map(|s| s.as_str()));
            value_vars = self
                .get_columns()
                .iter()
                .filter_map(|s| {
                    if id_vars_set.contains(s.name()) {
                        None
                    } else {
                        Some(s.name().into())
                    }
                })
                .collect();
        }

        // values will all be placed in single column, so we must find their supertype
        let schema = self.schema();
        let mut iter = value_vars.iter().map(|v| {
            schema
                .get(v)
                .ok_or_else(|| polars_err!(ColumnNotFound: "{}", v))
        });
        let mut st = iter.next().unwrap()?.clone();
        for dt in iter {
            st = try_get_supertype(&st, dt?)?;
        }

        let values_len = value_vars.iter().map(|name| name.len()).sum::<usize>();

        // The column name of the variable that is melted
        let mut variable_col = MutableBinaryViewArray::<str>::with_capacity(
            len * value_vars.len() + 1,
        );
        // prepare ids
        let ids_ = self.select_with_schema_unchecked(id_vars, &schema)?;
        let mut ids = ids_.clone();
        if ids.width() > 0 {
            for _ in 0..value_vars.len() - 1 {
                ids.vstack_mut_unchecked(&ids_)
            }
        }
        ids.as_single_chunk_par();
        drop(ids_);

        let mut values = Vec::with_capacity(value_vars.len());

        for value_column_name in &value_vars {
            variable_col.extend_constant(len, Some(value_column_name.as_str()));
            // ensure we go via the schema so we are O(1)
            // self.column() is linear
            // together with this loop that would make it O^2 over value_vars
            let (pos, _name, _dtype) = schema.try_get_full(value_column_name)?;
            let value_col = self.columns[pos].cast(&st).unwrap();
            values.extend_from_slice(value_col.chunks())
        }
        let values_arr = concatenate_owned_unchecked(&values)?;
        // Safety
        // The give dtype is correct
        let values =
            unsafe { Series::from_chunks_and_dtype_unchecked(value_name, vec![values_arr], &st) };

        let variable_col = variable_col.as_box();
        // Safety
        // The give dtype is correct
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
    use crate::frame::explode::MeltArgs;
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

        let str = Series::new("foo", &["abc", "de", "fg"]);
        let df = DataFrame::new(vec![str, s0, s1]).unwrap();
        let exploded = df.explode(["foo"]).unwrap();
        assert_eq!(exploded.column("C").unwrap().i32().unwrap().get(6), Some(1));
        assert_eq!(exploded.column("B").unwrap().i32().unwrap().get(6), Some(3));
        assert_eq!(
            exploded.column("foo").unwrap().str().unwrap().get(6),
            Some("g")
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
    fn test_melt() -> PolarsResult<()> {
        let df = df!("A" => &["a", "b", "a"],
         "B" => &[1, 3, 5],
         "C" => &[10, 11, 12],
         "D" => &[2, 4, 6]
        )
        .unwrap();

        let melted = df.melt(["A", "B"], ["C", "D"])?;
        assert_eq!(
            Vec::from(melted.column("value")?.i32()?),
            &[Some(10), Some(11), Some(12), Some(2), Some(4), Some(6)]
        );

        let args = MeltArgs {
            id_vars: vec![],
            value_vars: vec![],
            ..Default::default()
        };

        let melted = df.melt2(args).unwrap();
        let value = melted.column("value")?;
        // String because of supertype
        let value = value.str()?;
        let value = value.into_no_null_iter().collect::<Vec<_>>();
        assert_eq!(
            value,
            &["a", "b", "a", "1", "3", "5", "10", "11", "12", "2", "4", "6"]
        );

        let args = MeltArgs {
            id_vars: vec!["A".into()],
            value_vars: vec![],
            ..Default::default()
        };

        let melted = df.melt2(args).unwrap();
        let value = melted.column("value")?;
        let value = value.i32()?;
        let value = value.into_no_null_iter().collect::<Vec<_>>();
        assert_eq!(value, &[1, 3, 5, 10, 11, 12, 2, 4, 6]);
        let variable = melted.column("variable")?;
        let variable = variable.str()?;
        let variable = variable.into_no_null_iter().collect::<Vec<_>>();
        assert_eq!(variable, &["B", "B", "B", "C", "C", "C", "D", "D", "D"]);
        assert!(melted.column("A").is_ok());
        Ok(())
    }
}
