use std::sync::Arc;

use polars_core::series::Series;

use super::{AnonymousColumnsUdf, SpecialEq};
use crate::dsl::LazySerde;

impl<T: schemars::JsonSchema> schemars::JsonSchema for SpecialEq<Arc<T>> {
    fn schema_name() -> String {
        T::schema_name()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        T::schema_id()
    }

    fn json_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        T::json_schema(generator)
    }
}

impl<T: schemars::JsonSchema + Clone> schemars::JsonSchema for LazySerde<T> {
    fn schema_name() -> String {
        T::schema_name()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        T::schema_id()
    }

    fn json_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        Vec::<u8>::json_schema(generator)
    }
}

impl schemars::JsonSchema for SpecialEq<Series> {
    fn schema_name() -> String {
        Series::schema_name()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        Series::schema_id()
    }

    fn json_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        Series::json_schema(generator)
    }
}

impl schemars::JsonSchema for SpecialEq<Arc<dyn AnonymousColumnsUdf>> {
    fn schema_name() -> String {
        "AnonymousColumnsUdf".to_owned()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "AnonymousColumnsUdf"))
    }

    fn json_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        Vec::<u8>::json_schema(generator)
    }
}
