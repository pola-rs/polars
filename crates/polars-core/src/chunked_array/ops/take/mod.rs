//! Traits to provide fast Random access to ChunkedArrays data.
//! This prevents downcasting every iteration.
//! IntoTakeRandom provides structs that implement the TakeRandom trait.
//! There are several structs that implement the fastest path for random access.
//!

use crate::prelude::*;
use crate::utils::NoNull;

mod take_chunked;
#[cfg(feature = "chunked_ids")]
pub(crate) use take_chunked::*;
