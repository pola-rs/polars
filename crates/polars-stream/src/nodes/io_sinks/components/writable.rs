use std::io;

use polars_io::ExternalCompression;
use polars_io::utils::compression::CompressedWriter;
use polars_io::utils::file::Writable as WriteTargetVariant;
use polars_io::utils::sync_on_close::SyncOnCloseType;
use polars_utils::file::close_file;

use crate::nodes::io_sinks::components::sinked_path_info_list::SinkedPathInfoEntry;

/// Represents an owned, opened write target.
pub struct WriteTarget {
    variant: WriteTargetVariant,
    inner: WriteTargetInner,
}

struct WriteTargetInner {
    bytes_written: u64,
    path_info: Option<SinkedPathInfoEntry>,
}

impl WriteTarget {
    pub fn new(
        variant: polars_io::utils::file::Writable,
        path_info: Option<SinkedPathInfoEntry>,
    ) -> WriteTarget {
        WriteTarget {
            variant,
            inner: WriteTargetInner {
                bytes_written: 0,
                path_info,
            },
        }
    }

    pub fn as_writable(&mut self) -> Writable<'_> {
        Writable(WritableVariant::TargetRef(WriteTargetWrap::new_ref(self)))
    }

    pub fn as_buffered_writable(&mut self) -> Box<dyn io::Write + Send + '_> {
        let is_cloud = matches!(&self.variant, WriteTargetVariant::Cloud(_));

        let wrap = WriteTargetWrap::new_ref(self);

        if is_cloud {
            Box::new(wrap)
        } else {
            Box::new(io::BufWriter::new(wrap))
        }
    }

    #[expect(clippy::type_complexity)]
    pub fn as_compressed_writable(
        &mut self,
        compression: ExternalCompression,
    ) -> io::Result<(
        Writable<'_>,
        Option<fn(Writable<'_>) -> io::Result<Writable<'_>>>,
    )> {
        let compressed_writer = match compression {
            ExternalCompression::Uncompressed => return Ok((self.as_writable(), None)),
            ExternalCompression::Gzip { level } => {
                CompressedWriter::gzip(WriteTargetWrap::new_ref(self), level)
            },
            ExternalCompression::Zstd { level } => {
                CompressedWriter::zstd(WriteTargetWrap::new_ref(self), level)?
            },
        };

        let ret = Writable(WritableVariant::Compressed(Box::new(compressed_writer)));

        fn finish_compressed_writer(w: Writable<'_>) -> io::Result<Writable<'_>> {
            let Writable(WritableVariant::Compressed(mut compressed_writer)) = w else {
                unreachable!()
            };

            let w = compressed_writer.finish()?;
            Ok(Writable(WritableVariant::TargetRef(w)))
        }

        Ok((ret, Some(finish_compressed_writer)))
    }

    pub async fn close(mut self, sync: SyncOnCloseType) -> io::Result<()> {
        match sync {
            SyncOnCloseType::All => self.sync_all().await?,
            SyncOnCloseType::Data => self.sync_data().await?,
            SyncOnCloseType::None => {},
        }

        match self.variant {
            WriteTargetVariant::Cloud(mut x) => tokio::io::AsyncWriteExt::shutdown(&mut x).await?,
            WriteTargetVariant::Dyn(mut x) => {
                x.flush()?;
                x.close()?;
            },
            WriteTargetVariant::Local(x) => close_file(x)?,
        }

        if let Some(path_info) = self.inner.path_info {
            path_info.set_num_bytes(self.inner.bytes_written);
        }

        Ok(())
    }

    async fn sync_all(&mut self) -> io::Result<()> {
        match &self.variant {
            WriteTargetVariant::Cloud(_) => Ok(()),
            WriteTargetVariant::Dyn(x) => x.sync_all(),
            WriteTargetVariant::Local(x) => x.sync_all(),
        }
    }

    async fn sync_data(&mut self) -> io::Result<()> {
        match &self.variant {
            WriteTargetVariant::Cloud(_) => Ok(()),
            WriteTargetVariant::Dyn(x) => x.sync_data(),
            WriteTargetVariant::Local(x) => x.sync_data(),
        }
    }
}

impl WriteTargetInner {
    fn add_bytes_written(&mut self, n: u64) {
        self.bytes_written = u64::saturating_add(self.bytes_written, n);
    }
}

enum WritableVariant<'a> {
    TargetRef(&'a mut WriteTargetWrap),
    Compressed(Box<CompressedWriter<'a, WriteTargetWrap>>),
}

pub struct Writable<'a>(WritableVariant<'a>);

impl<'a> Writable<'a> {
    pub async fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        match &mut self.0 {
            WritableVariant::TargetRef(x) => x.write_all(buf).await,
            WritableVariant::Compressed(x) => io::Write::write_all(x, buf),
        }
    }

    pub async fn write_all_owned<T>(&mut self, src: &mut T) -> io::Result<()>
    where
        T: AsRef<[u8]> + Default + Drop,
        bytes::Bytes: From<T>,
    {
        match &mut self.0 {
            WritableVariant::TargetRef(x) => x.write_all_owned(src).await,
            WritableVariant::Compressed(x) => io::Write::write_all(x, src.as_ref()),
        }
    }

    pub async fn flush(&mut self) -> io::Result<()> {
        match &mut self.0 {
            WritableVariant::TargetRef(x) => x.flush().await,
            WritableVariant::Compressed(x) => io::Write::flush(x),
        }
    }

    pub fn is_cloud(&self) -> bool {
        matches!(
            &self.0,
            WritableVariant::TargetRef(WriteTargetWrap(x)) if matches!(x.variant, WriteTargetVariant::Cloud(_))
        )
    }
}

/// `write()` dispatch with tracking of written bytes.
#[repr(transparent)]
struct WriteTargetWrap(WriteTarget);

impl WriteTargetWrap {
    fn new_ref<'a>(target: &'a mut WriteTarget) -> &'a mut WriteTargetWrap {
        // Safety: repr(transparent)
        unsafe { std::mem::transmute::<&'a mut WriteTarget, &'a mut WriteTargetWrap>(target) }
    }

    async fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        let n = buf.len();

        match &mut self.0.variant {
            WriteTargetVariant::Cloud(x) => tokio::io::AsyncWriteExt::write_all(x, buf).await,
            // Don't dispatch to async for local files; tokio files aren't truly async (they do spawn_blocking),
            // but this is not needed for us since we already consider tokio async threads to
            // be non-computational threads dedicated to I/O.
            WriteTargetVariant::Local(x) => io::Write::write_all(x, buf),
            WriteTargetVariant::Dyn(x) => io::Write::write_all(x, buf),
        }?;

        self.0.inner.add_bytes_written(n as u64);

        Ok(())
    }

    async fn write_all_owned<T>(&mut self, src: &mut T) -> io::Result<()>
    where
        T: AsRef<[u8]> + Default + Drop,
        bytes::Bytes: From<T>,
    {
        let n = src.as_ref().len();
        self.0.variant.write_all_owned(src).await?;
        self.0.inner.add_bytes_written(n as u64);

        Ok(())
    }

    async fn flush(&mut self) -> io::Result<()> {
        match &mut self.0.variant {
            WriteTargetVariant::Cloud(x) => tokio::io::AsyncWriteExt::flush(x).await,
            WriteTargetVariant::Local(x) => io::Write::flush(x),
            WriteTargetVariant::Dyn(x) => io::Write::flush(x),
        }
    }
}

impl io::Write for WriteTargetWrap {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.0.variant.write(buf)?;
        self.0.inner.add_bytes_written(n as u64);
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.variant.flush()
    }
}
