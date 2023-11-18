//! Traits to provide fast Random access to ChunkedArrays data.
//! This prevents downcasting every iteration.

use crate::prelude::*;
use crate::utils::NoNull;

mod take_chunked;
#[cfg(feature = "chunked_ids")]
pub(crate) use take_chunked::*;
