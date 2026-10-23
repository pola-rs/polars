//! zstd compression and decompression using state that is reused per thread.
//!
//! Creating a zstd context allocates and initializes its internal state, which
//! can cost more than compressing or decompressing a small buffer.

use std::cell::RefCell;
use std::io::Cursor;

use ::zstd::zstd_safe::{self, CCtx, CParameter, DCtx, InBuffer, OutBuffer, ResetDirective};
use polars_error::{PolarsError, PolarsResult, polars_ensure, polars_err};

/// Upper bound on the size of each zstd context kept per thread between calls.
/// Contexts for high compression levels or large windows exceed it and are not
/// kept, but for those the cost of creating a context is small relative to the
/// work.
const MAX_CACHED_CONTEXT_BYTES: usize = 2 * 1000 * 1000;

thread_local! {
    static CCTX: RefCell<Option<CCtx<'static>>> = const { RefCell::new(None) };
    static DCTX: RefCell<Option<DCtx<'static>>> = const { RefCell::new(None) };
}

fn zstd_err(code: zstd_safe::ErrorCode) -> PolarsError {
    polars_err!(ComputeError: "zstd error: {}", zstd_safe::get_error_name(code))
}

fn with_cctx<R>(
    level: i32,
    f: impl FnOnce(&mut CCtx<'static>) -> PolarsResult<R>,
) -> PolarsResult<R> {
    let run = |ctx: &mut CCtx<'static>| {
        ctx.set_parameter(CParameter::CompressionLevel(level))
            .map_err(zstd_err)?;
        f(ctx)
    };
    CCTX.with(|cell| {
        let Ok(mut cached) = cell.try_borrow_mut() else {
            // Re-entrant use.
            return run(&mut CCtx::create());
        };
        let ctx = cached.get_or_insert_with(CCtx::create);
        let out = run(ctx);
        if ctx.sizeof() > MAX_CACHED_CONTEXT_BYTES {
            *cached = None;
        }
        out
    })
}

fn with_dctx<R>(f: impl FnOnce(&mut DCtx<'static>) -> PolarsResult<R>) -> PolarsResult<R> {
    DCTX.with(|cell| {
        let Ok(mut cached) = cell.try_borrow_mut() else {
            // Re-entrant use.
            return f(&mut DCtx::create());
        };
        let ctx = cached.get_or_insert_with(DCtx::create);
        let out = f(ctx);
        if ctx.sizeof() > MAX_CACHED_CONTEXT_BYTES {
            *cached = None;
        }
        out
    })
}

/// Compresses `src` into a single zstd frame appended to `dst`.
pub fn compress(src: &[u8], dst: &mut Vec<u8>, level: i32) -> PolarsResult<()> {
    with_cctx(level, |ctx| {
        dst.reserve(zstd_safe::compress_bound(src.len()));
        let mut out = Cursor::new(&mut *dst);
        out.set_position(out.get_ref().len() as u64);
        ctx.compress2(&mut out, src).map_err(zstd_err)?;
        Ok(())
    })
}

/// Decompresses the zstd frames in `src`, appending them to `dst`, and returns
/// the number of bytes appended. Fails if they do not fit in the spare
/// capacity of `dst`.
pub fn decompress(src: &[u8], dst: &mut Vec<u8>) -> PolarsResult<usize> {
    with_dctx(|ctx| {
        let mut out = Cursor::new(&mut *dst);
        out.set_position(out.get_ref().len() as u64);
        ctx.decompress(&mut out, src).map_err(zstd_err)
    })
}

/// Decompresses the zstd frames in `src`, appending the first `len` decompressed
/// bytes to `dst`. Fails if they decompress to fewer bytes.
pub fn decompress_prefix(src: &[u8], dst: &mut Vec<u8>, len: usize) -> PolarsResult<()> {
    let end = dst.len() + len;
    dst.reserve_exact(len);

    // A single frame that records exactly `len` bytes is decompressed in one shot.
    // zstd verifies the recorded size while decompressing.
    if matches!(zstd_safe::find_frame_compressed_size(src), Ok(n) if n == src.len())
        && matches!(zstd_safe::get_frame_content_size(src), Ok(Some(n)) if n == len as u64)
    {
        decompress(src, dst)?;
        return Ok(());
    }

    with_dctx(|ctx| {
        ctx.reset(ResetDirective::SessionOnly).map_err(zstd_err)?;
        let mut out = Cursor::new(&mut *dst);
        out.set_position(out.get_ref().len() as u64);
        let mut output = OutBuffer::around(&mut out);
        let mut input = InBuffer::around(src);
        while output.pos() < len {
            let progress = (input.pos(), output.pos());
            ctx.decompress_stream(&mut output, &mut input)
                .map_err(zstd_err)?;
            polars_ensure!(
                (input.pos(), output.pos()) != progress,
                ComputeError: "zstd data decompresses to fewer than {len} bytes"
            );
        }
        Ok(())
    })?;
    // The spare capacity may have received more than `len` bytes.
    dst.truncate(end);
    Ok(())
}
