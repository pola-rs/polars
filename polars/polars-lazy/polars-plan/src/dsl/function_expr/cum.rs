use super::*;

pub(super) fn cumcount(s: &Series, reverse: bool) -> PolarsResult<Series> {
    if reverse {
        let ca: NoNull<UInt32Chunked> = (0u32..s.len() as u32).rev().collect();
        let mut ca = ca.into_inner();
        ca.rename(s.name());
        Ok(ca.into_series())
    } else {
        let ca: NoNull<UInt32Chunked> = (0u32..s.len() as u32).collect();
        let mut ca = ca.into_inner();
        ca.rename(s.name());
        Ok(ca.into_series())
    }
}

pub(super) fn cumsum(s: &Series, reverse: bool) -> PolarsResult<Series> {
    Ok(s.cumsum(reverse))
}

pub(super) fn cumprod(s: &Series, reverse: bool) -> PolarsResult<Series> {
    Ok(s.cumprod(reverse))
}

pub(super) fn cummin(s: &Series, reverse: bool) -> PolarsResult<Series> {
    Ok(s.cummin(reverse))
}

pub(super) fn cummax(s: &Series, reverse: bool) -> PolarsResult<Series> {
    Ok(s.cummax(reverse))
}

pub(super) mod dtypes {
    use DataType::*;

    use super::*;

    pub fn cumsum(dt: &DataType) -> DataType {
        if dt.is_logical() {
            dt.clone()
        } else {
            match dt {
                Boolean => UInt32,
                Int32 => Int32,
                UInt32 => UInt32,
                UInt64 => UInt64,
                Float32 => Float32,
                Float64 => Float64,
                _ => Int64,
            }
        }
    }

    pub fn cumprod(dt: &DataType) -> DataType {
        match dt {
            Boolean => Int64,
            UInt64 => UInt64,
            Float32 => Float32,
            Float64 => Float64,
            _ => Int64,
        }
    }
}
