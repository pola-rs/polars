mod basic;
mod integer;

pub(crate) use basic::{
    AsDecoderFunction, DecoderFunction, IntoDecoderFunction, PrimitiveDecoder, UnitDecoderFunction,
};
pub(crate) use integer::IntDecoder;
