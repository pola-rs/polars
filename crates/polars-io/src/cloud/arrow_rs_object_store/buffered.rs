// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Utilities for performing tokio-style buffered IO

use std::io::{Error, ErrorKind};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::future::{BoxFuture, FutureExt};
use futures::ready;
use object_store::path::Path;
use object_store::{
    Attributes, Extensions, ObjectStore, PutMultipartOptions, PutOptions, PutPayloadMut, TagSet,
};
use tokio::io::AsyncWrite;

use super::upload::WriteMultipart;

/// An async buffered writer compatible with the tokio IO traits
///
/// This writer adaptively uses [`ObjectStore::put_opts`] or
/// [`ObjectStore::put_multipart_opts`] depending on the amount of data that has
/// been written.
///
/// Up to `capacity` bytes will be buffered in memory, and flushed on shutdown
/// using [`ObjectStore::put_opts`]. If `capacity` is exceeded, data will instead be
/// streamed using [`ObjectStore::put_multipart_opts`].
pub struct BufWriter {
    capacity: usize,
    max_concurrency: usize,
    attributes: Option<Attributes>,
    tags: Option<TagSet>,
    extensions: Option<Extensions>,
    state: BufWriterState,
    store: Arc<dyn ObjectStore>,
}

impl std::fmt::Debug for BufWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufWriter")
            .field("capacity", &self.capacity)
            .finish()
    }
}

enum BufWriterState {
    /// Buffer up to capacity bytes
    Buffer(Path, PutPayloadMut),
    /// [`ObjectStore::put_multipart_opts`]
    Prepare(BoxFuture<'static, object_store::Result<WriteMultipart>>),
    /// Write to a multipart upload
    Write(Option<WriteMultipart>),
    /// [`ObjectStore::put_opts`]
    Flush(BoxFuture<'static, object_store::Result<()>>),
}

impl BufWriter {
    /// Create a new [`BufWriter`] from the provided [`ObjectStore`] and [`Path`]
    pub fn new(store: Arc<dyn ObjectStore>, path: Path) -> Self {
        Self::with_capacity(store, path, 10 * 1024 * 1024)
    }

    /// Create a new [`BufWriter`] from the provided [`ObjectStore`], [`Path`] and `capacity`
    pub fn with_capacity(store: Arc<dyn ObjectStore>, path: Path, capacity: usize) -> Self {
        Self {
            capacity,
            store,
            max_concurrency: 8,
            attributes: None,
            tags: None,
            extensions: None,
            state: BufWriterState::Buffer(path, PutPayloadMut::new()),
        }
    }

    /// Override the maximum number of in-flight requests for this writer
    ///
    /// Defaults to 8
    pub fn with_max_concurrency(self, max_concurrency: usize) -> Self {
        Self {
            max_concurrency,
            ..self
        }
    }

    /// Set the attributes of the uploaded object
    pub fn with_attributes(self, attributes: Attributes) -> Self {
        Self {
            attributes: Some(attributes),
            ..self
        }
    }

    /// Set the tags of the uploaded object
    pub fn with_tags(self, tags: TagSet) -> Self {
        Self {
            tags: Some(tags),
            ..self
        }
    }

    /// Set the extensions of the uploaded object
    ///
    /// Implementation-specific extensions. Intended for use by [`ObjectStore`] implementations
    /// that need to pass context-specific information (like tracing spans) via trait methods.
    ///
    /// These extensions are ignored entirely by backends offered through this crate.
    pub fn with_extensions(self, extensions: Extensions) -> Self {
        Self {
            extensions: Some(extensions),
            ..self
        }
    }

    /// Write data to the writer in [`Bytes`].
    ///
    /// Unlike [`AsyncWrite::poll_write`], `put` can write data without extra copying.
    ///
    /// This API is recommended while the data source generates [`Bytes`].
    pub async fn put(&mut self, bytes: Bytes) -> object_store::Result<()> {
        loop {
            return match &mut self.state {
                BufWriterState::Write(Some(write)) => {
                    write.wait_for_capacity(self.max_concurrency).await?;
                    write.put(bytes);
                    Ok(())
                },
                BufWriterState::Write(None) | BufWriterState::Flush(_) => {
                    panic!("Already shut down")
                },
                // NOTE
                //
                // This case should never happen in practice, but rust async API does
                // make it possible for users to call `put` before `poll_write` returns `Ready`.
                //
                // We allow such usage by `await` the future and continue the loop.
                BufWriterState::Prepare(f) => {
                    self.state = BufWriterState::Write(f.await?.into());
                    continue;
                },
                BufWriterState::Buffer(path, b) => {
                    if b.content_length().saturating_add(bytes.len()) < self.capacity {
                        b.push(bytes);
                        Ok(())
                    } else {
                        let buffer = std::mem::take(b);
                        let path = std::mem::take(path);
                        let opts = PutMultipartOptions {
                            attributes: self.attributes.take().unwrap_or_default(),
                            tags: self.tags.take().unwrap_or_default(),
                            extensions: self.extensions.take().unwrap_or_default(),
                        };
                        let upload = self.store.put_multipart_opts(&path, opts).await?;
                        let mut chunked =
                            WriteMultipart::new_with_chunk_size(upload, self.capacity);
                        for chunk in buffer.freeze() {
                            chunked.put(chunk);
                        }
                        chunked.put(bytes);
                        self.state = BufWriterState::Write(Some(chunked));
                        Ok(())
                    }
                },
            };
        }
    }

    /// Abort this writer, cleaning up any partially uploaded state
    ///
    /// # Panic
    ///
    /// Panics if this writer has already been shutdown or aborted
    pub async fn abort(&mut self) -> object_store::Result<()> {
        match &mut self.state {
            BufWriterState::Buffer(_, _) | BufWriterState::Prepare(_) => Ok(()),
            BufWriterState::Flush(_) => panic!("Already shut down"),
            BufWriterState::Write(x) => x.take().unwrap().abort().await,
        }
    }
}

impl AsyncWrite for BufWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, Error>> {
        let cap = self.capacity;
        let max_concurrency = self.max_concurrency;
        loop {
            return match &mut self.state {
                BufWriterState::Write(Some(write)) => {
                    ready!(write.poll_for_capacity(cx, max_concurrency))?;
                    write.write(buf);
                    Poll::Ready(Ok(buf.len()))
                },
                BufWriterState::Write(None) | BufWriterState::Flush(_) => {
                    panic!("Already shut down")
                },
                BufWriterState::Prepare(f) => {
                    self.state = BufWriterState::Write(ready!(f.poll_unpin(cx)?).into());
                    continue;
                },
                BufWriterState::Buffer(path, b) => {
                    if b.content_length().saturating_add(buf.len()) >= cap {
                        let buffer = std::mem::take(b);
                        let path = std::mem::take(path);
                        let opts = PutMultipartOptions {
                            attributes: self.attributes.take().unwrap_or_default(),
                            tags: self.tags.take().unwrap_or_default(),
                            extensions: self.extensions.take().unwrap_or_default(),
                        };
                        let store = Arc::clone(&self.store);
                        self.state = BufWriterState::Prepare(Box::pin(async move {
                            let upload = store.put_multipart_opts(&path, opts).await?;
                            let mut chunked = WriteMultipart::new_with_chunk_size(upload, cap);
                            for chunk in buffer.freeze() {
                                chunked.put(chunk);
                            }
                            Ok(chunked)
                        }));
                        continue;
                    }
                    b.extend_from_slice(buf);
                    Poll::Ready(Ok(buf.len()))
                },
            };
        }
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Error>> {
        loop {
            return match &mut self.state {
                BufWriterState::Write(_) | BufWriterState::Buffer(_, _) => Poll::Ready(Ok(())),
                BufWriterState::Flush(_) => panic!("Already shut down"),
                BufWriterState::Prepare(f) => {
                    self.state = BufWriterState::Write(ready!(f.poll_unpin(cx)?).into());
                    continue;
                },
            };
        }
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Error>> {
        loop {
            match &mut self.state {
                BufWriterState::Prepare(f) => {
                    self.state = BufWriterState::Write(ready!(f.poll_unpin(cx)?).into());
                },
                BufWriterState::Buffer(p, b) => {
                    let buf = std::mem::take(b);
                    let path = std::mem::take(p);
                    let opts = PutOptions {
                        attributes: self.attributes.take().unwrap_or_default(),
                        tags: self.tags.take().unwrap_or_default(),
                        ..Default::default()
                    };
                    let store = Arc::clone(&self.store);
                    self.state = BufWriterState::Flush(Box::pin(async move {
                        store.put_opts(&path, buf.into(), opts).await?;
                        Ok(())
                    }));
                },
                BufWriterState::Flush(f) => return f.poll_unpin(cx).map_err(std::io::Error::from),
                BufWriterState::Write(x) => {
                    let upload = x.take().ok_or_else(|| {
                        std::io::Error::new(
                            ErrorKind::InvalidInput,
                            "Cannot shutdown a writer that has already been shut down",
                        )
                    })?;
                    self.state = BufWriterState::Flush(
                        async move {
                            upload.finish().await?;
                            Ok(())
                        }
                        .boxed(),
                    )
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use object_store::{Attribute, GetOptions};
    use polars_utils::itertools::Itertools as _;
    use tokio::io::AsyncWriteExt;

    use super::*;

    // Note: `BufWriter::with_tags` functionality is tested in `object_store::tests::tagging`
    #[tokio::test]
    async fn test_buf_writer() {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let path = Path::from("file.txt");
        let attributes = Attributes::from_iter([
            (Attribute::ContentType, "text/html"),
            (Attribute::CacheControl, "max-age=604800"),
        ]);

        // Test put
        let mut writer = BufWriter::with_capacity(Arc::clone(&store), path.clone(), 30)
            .with_attributes(attributes.clone());
        writer.write_all(&[0; 20]).await.unwrap();
        writer.flush().await.unwrap();
        writer.write_all(&[0; 5]).await.unwrap();
        writer.shutdown().await.unwrap();
        let response = store
            .get_opts(&path, GetOptions::new().with_head(true))
            .await
            .unwrap();
        assert_eq!(response.meta.size, 25);
        assert_eq!(response.attributes, attributes);

        // Test multipart
        let mut writer = BufWriter::with_capacity(Arc::clone(&store), path.clone(), 30)
            .with_attributes(attributes.clone());
        writer.write_all(&[0; 20]).await.unwrap();
        writer.flush().await.unwrap();
        writer.write_all(&[0; 20]).await.unwrap();
        writer.shutdown().await.unwrap();
        let response = store
            .get_opts(&path, GetOptions::new().with_head(true))
            .await
            .unwrap();
        assert_eq!(response.meta.size, 40);
        assert_eq!(response.attributes, attributes);
    }

    #[tokio::test]
    async fn test_buf_writer_with_put() {
        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let path = Path::from("file.txt");

        // Test put
        let mut writer = BufWriter::with_capacity(Arc::clone(&store), path.clone(), 30);
        writer
            .put(Bytes::from((0..20).collect_vec()))
            .await
            .unwrap();
        writer
            .put(Bytes::from((20..25).collect_vec()))
            .await
            .unwrap();
        writer.shutdown().await.unwrap();
        let response = store
            .get_opts(&path, GetOptions::new().with_head(true))
            .await
            .unwrap();
        assert_eq!(response.meta.size, 25);
        assert_eq!(response.bytes().await.unwrap(), (0..25).collect_vec());

        // Test multipart
        let mut writer = BufWriter::with_capacity(Arc::clone(&store), path.clone(), 30);
        writer
            .put(Bytes::from((0..20).collect_vec()))
            .await
            .unwrap();
        writer
            .put(Bytes::from((20..40).collect_vec()))
            .await
            .unwrap();
        writer.shutdown().await.unwrap();
        let response = store
            .get_opts(&path, GetOptions::new().with_head(true))
            .await
            .unwrap();
        assert_eq!(response.meta.size, 40);
        assert_eq!(response.bytes().await.unwrap(), (0..40).collect_vec());
    }
}
