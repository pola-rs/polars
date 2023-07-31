use polars_error::PolarsError;
use serde::de::Error;
use serde::*;

use crate::prelude::{DataFrame, Series};

// utility to ensure we serde to a struct
// {
//  columns: Vec<Series>
// }
// that ensures it differentiates between Vec<Series>
// and is backwards compatible
#[derive(Deserialize)]
struct Util {
    columns: Vec<Series>,
}

#[derive(Serialize)]
struct UtilBorrowed<'a> {
    columns: &'a [Series],
}

impl<'de> Deserialize<'de> for DataFrame {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let parsed = <Util>::deserialize(deserializer)?;
        DataFrame::new(parsed.columns).map_err(|e| {
            let e = PolarsError::ComputeError(format!("successful parse invalid data: {e}").into());
            D::Error::custom::<PolarsError>(e)
        })
    }
}

impl Serialize for DataFrame {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        UtilBorrowed {
            columns: &self.columns,
        }
        .serialize(serializer)
    }
}
