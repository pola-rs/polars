use crate::frame::row::Row;
use crate::prelude::*;
use indexmap::map::IndexMap as HashMap;
use indexmap::set::IndexSet as HashSet;

use std::borrow::Borrow;

type Tracker = HashMap<String, HashSet<DataType>>;

pub fn infer_schema(rows: &[Row]) -> Schema {
    let mut values: Tracker = Tracker::new();
    let max_infer = std::cmp::min(rows.len(), 50);

    rows.iter().take(max_infer).for_each(|row| {
        row.0.iter().enumerate().for_each(|(i, av)| {
            let dt: DataType = av.into();
            let col_name = format!("column_{}", i);
            add_or_insert(&mut values, &col_name, dt);
        });
    });

    let fields = resolve_fields(values);
    Schema::new(fields)

}

fn add_or_insert(values: &mut Tracker, key: &str, data_type: DataType) {
    if data_type == DataType::Null {
        return;
    }
    if values.contains_key(key) {
        let x = values.get_mut(key).unwrap();
        x.insert(data_type);
    } else {
        // create hashset and add value type
        let mut hs = HashSet::new();
        hs.insert(data_type);
        values.insert(key.to_string(), hs);
    }
}

fn resolve_fields(spec: HashMap<String, HashSet<DataType>>) -> Vec<Field> {
    spec.iter()
        .map(|(k, hs)| {
            let v: Vec<&DataType> = hs.iter().collect();
            Field::new(k, coerce_data_type(&v))
        })
        .collect()
}

fn coerce_data_type<A: Borrow<DataType>>(datatypes: &[A]) -> DataType {
    use DataType::*;

    let are_all_equal = datatypes.windows(2).all(|w| w[0].borrow() == w[1].borrow());

    if are_all_equal {
        return datatypes[0].borrow().clone();
    }

    let (lhs, rhs) = (datatypes[0].borrow(), datatypes[1].borrow());

    return match (lhs, rhs) {
        (lhs, rhs) if lhs == rhs => lhs.clone(),
        (List(lhs), List(rhs)) => {
            
            let inner = coerce_data_type(&[lhs.as_ref(), rhs.as_ref()]);
            List(Box::new(inner))
        }
        (scalar, List(list)) => {
            let inner = coerce_data_type(&[scalar, list.as_ref()]);
            List(Box::new(inner))
        }
        (List(list), scalar) => {
            let inner = coerce_data_type(&[scalar, list.as_ref()]);
            List(Box::new(inner))
        }
        (Float64, Int64) => Float64,
        (Int64, Float64) => Float64,
        (Int64, Boolean) => Int64,
        (Boolean, Int64) => Int64,
        (_, _) => Utf8,
    };
}
