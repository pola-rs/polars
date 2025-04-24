problem: jumps in this branch, last branch when we dont have upper bound specified 
(weird stuff)

if max fields (upper bound) is set we get weird stuff as well in list to struct
-> in this case fails with default field not found, example 

```python
    import polars as pl 

    df = pl.DataFrame({"x": [1, 2, 3], "y": [[], [], []]}, schema={"x": pl.Int64, "y": pl.List(pl.Int64)})
    empty_df = df.select(
        pl.col("y").list.to_struct(n_field_strategy="max_width")
    )
    print(empty_df)
    empty_df = df.select(
        pl.col("y").list.to_struct(n_field_strategy="max_width", upper_bound=1).struct.unnest()
    )
    print(empty_df)
```

if it is not set, we jump into the datatype struct from infer width and fail that way because datatype is not known

impl ListToStructArgs {
    pub fn get_output_dtype(&self, input_dtype: &DataType) -> PolarsResult<DataType> {
        let DataType::List(inner_dtype) = input_dtype else {
            polars_bail!(
                InvalidOperation:
                "attempted list to_struct on non-list dtype: {}",
                input_dtype
            );
        };
        let inner_dtype = inner_dtype.as_ref();

        match self {
            Self::FixedWidth(names) => Ok(DataType::Struct(
                names
                    .iter()
                    .map(|x| Field::new(x.clone(), inner_dtype.clone()))
                    .collect::<Vec<_>>(),
            )),
            Self::InferWidth {
                get_index_name,
                max_fields,
                ..
            } if *max_fields > 0 => {
                let get_index_name_func = get_index_name.as_ref().map_or(
                    &_default_struct_name_gen as &dyn Fn(usize) -> PlSmallStr,
                    |x| x.0.as_ref(),
                );
                Ok(DataType::Struct(
                    (0..*max_fields)
                        .map(|i| Field::new(get_index_name_func(i), inner_dtype.clone()))
                        .collect::<Vec<_>>(),
                ))
            },
            Self::InferWidth { .. } => Ok(DataType::Unknown(UnknownKind::Any)),
        }
    }
