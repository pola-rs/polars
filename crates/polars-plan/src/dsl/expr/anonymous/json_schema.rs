use std::sync::Arc;

use polars_core::series::Series;

use super::{AnonymousAgg, AnonymousColumnsUdf, SpecialEq};
use crate::dsl::LazySerde;

impl<T: schemars::JsonSchema> schemars::JsonSchema for SpecialEq<Arc<T>> {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        T::schema_name()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        T::schema_id()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        T::json_schema(generator)
    }
}

impl<T: schemars::JsonSchema + Clone> schemars::JsonSchema for LazySerde<T> {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        T::schema_name()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        T::schema_id()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        Vec::<u8>::json_schema(generator)
    }
}

impl schemars::JsonSchema for SpecialEq<Series> {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        Series::schema_name()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        Series::schema_id()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        Series::json_schema(generator)
    }
}

impl schemars::JsonSchema for SpecialEq<Arc<dyn AnonymousAgg>> {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "AnonymousAgg".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "AnonymousAgg"))
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        Vec::<u8>::json_schema(generator)
    }
}

impl schemars::JsonSchema for SpecialEq<Arc<dyn AnonymousColumnsUdf>> {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "AnonymousColumnsUdf".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "AnonymousColumnsUdf"))
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        Vec::<u8>::json_schema(generator)
    }
}
