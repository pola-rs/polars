use polars_core::chunked_array::metadata::Metadata;
use polars_core::datatypes::{
    BinaryType, BooleanType, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type,
    PolarsDataType, StringType, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
};
use polars_parquet::parquet::statistics::{
    BinaryStatistics, BooleanStatistics, PrimitiveStatistics,
};

pub trait ToMetadata<D: PolarsDataType + 'static>: Sized + 'static {
    fn to_metadata(&self) -> Metadata<D>;
}

impl ToMetadata<BooleanType> for BooleanStatistics {
    fn to_metadata(&self) -> Metadata<BooleanType> {
        let mut md = Metadata::default();

        md.set_distinct_count(self.distinct_count.and_then(|v| v.try_into().ok()));
        md.set_min_value(self.min_value);
        md.set_max_value(self.max_value);

        md
    }
}

impl ToMetadata<BinaryType> for BinaryStatistics {
    fn to_metadata(&self) -> Metadata<BinaryType> {
        let mut md = Metadata::default();

        md.set_distinct_count(self.distinct_count.and_then(|v| v.try_into().ok()));
        md.set_min_value(
            self.min_value
                .as_ref()
                .map(|v| v.clone().into_boxed_slice()),
        );
        md.set_max_value(
            self.max_value
                .as_ref()
                .map(|v| v.clone().into_boxed_slice()),
        );

        md
    }
}

impl ToMetadata<StringType> for BinaryStatistics {
    fn to_metadata(&self) -> Metadata<StringType> {
        let mut md = Metadata::default();

        md.set_distinct_count(self.distinct_count.and_then(|v| v.try_into().ok()));
        md.set_min_value(
            self.min_value
                .as_ref()
                .and_then(|s| String::from_utf8(s.clone()).ok()),
        );
        md.set_max_value(
            self.max_value
                .as_ref()
                .and_then(|s| String::from_utf8(s.clone()).ok()),
        );

        md
    }
}

macro_rules! prim_statistics {
    ($(($bstore:ty, $pltype:ty),)+) => {
        $(
        impl ToMetadata<$pltype> for PrimitiveStatistics<$bstore> {
            fn to_metadata(&self) -> Metadata<$pltype> {
                let mut md = Metadata::default();

                md.set_distinct_count(self.distinct_count.and_then(|v| v.try_into().ok()));
                md.set_min_value(self.min_value.map(|v| v as <$pltype as PolarsDataType>::OwnedPhysical));
                md.set_max_value(self.max_value.map(|v| v as <$pltype as PolarsDataType>::OwnedPhysical));

                md
            }
        }
        )+
    }
}

prim_statistics! {
    (i32, Int8Type),
    (i32, Int16Type),
    (i32, Int32Type),
    (i64, Int64Type),

    (i32, UInt8Type),
    (i32, UInt16Type),
    (i32, UInt32Type),
    (i64, UInt64Type),

    (f32, Float32Type),
    (f64, Float64Type),
}
