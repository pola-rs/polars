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

use std::task::{Context, Poll};

use bytes::Bytes;
use futures::ready;
use object_store::{MultipartUpload, PutPayload, PutPayloadMut, PutResult, Result};
use tokio::task::JoinSet;

/// A synchronous write API for uploading data in parallel in fixed size chunks
///
/// Uses multiple tokio tasks in a [`JoinSet`] to multiplex upload tasks in parallel
///
/// The design also takes inspiration from [`Sink`] with [`WriteMultipart::wait_for_capacity`]
/// allowing back pressure on producers, prior to buffering the next part. However, unlike
/// [`Sink`] this back pressure is optional, allowing integration with synchronous producers
///
/// [`Sink`]: futures::sink::Sink
#[derive(Debug)]
pub struct WriteMultipart {
    upload: Box<dyn MultipartUpload>,

    buffer: PutPayloadMut,

    chunk_size: usize,

    tasks: JoinSet<Result<()>>,
}

impl WriteMultipart {
    /// Create a new [`WriteMultipart`] that will upload in fixed `chunk_size` sized chunks
    pub fn new_with_chunk_size(upload: Box<dyn MultipartUpload>, chunk_size: usize) -> Self {
        Self {
            upload,
            chunk_size,
            buffer: PutPayloadMut::new(),
            tasks: Default::default(),
        }
    }

    /// Polls for there to be less than `max_concurrency` [`UploadPart`] in progress
    ///
    /// See [`Self::wait_for_capacity`] for an async version of this function
    pub fn poll_for_capacity(
        &mut self,
        cx: &mut Context<'_>,
        max_concurrency: usize,
    ) -> Poll<Result<()>> {
        while !self.tasks.is_empty() && self.tasks.len() >= max_concurrency {
            ready!(self.tasks.poll_join_next(cx)).unwrap()??
        }
        Poll::Ready(Ok(()))
    }

    /// Wait until there are less than `max_concurrency` [`UploadPart`] in progress
    ///
    /// See [`Self::poll_for_capacity`] for a [`Poll`] version of this function
    pub async fn wait_for_capacity(&mut self, max_concurrency: usize) -> Result<()> {
        futures::future::poll_fn(|cx| self.poll_for_capacity(cx, max_concurrency)).await
    }

    /// Write data to this [`WriteMultipart`]
    ///
    /// Data is buffered using [`PutPayloadMut::extend_from_slice`]. Implementations looking to
    /// write data from owned buffers may prefer [`Self::put`] as this avoids copying.
    ///
    /// Note this method is synchronous (not `async`) and will immediately
    /// start new uploads as soon as the internal `chunk_size` is hit,
    /// regardless of how many outstanding uploads are already in progress.
    ///
    /// Back pressure can optionally be applied to producers by calling
    /// [`Self::wait_for_capacity`] prior to calling this method
    pub fn write(&mut self, mut buf: &[u8]) {
        while !buf.is_empty() {
            let remaining = self.chunk_size - self.buffer.content_length();
            let to_read = buf.len().min(remaining);
            self.buffer.extend_from_slice(&buf[..to_read]);
            if to_read == remaining {
                let buffer = std::mem::take(&mut self.buffer);
                self.put_part(buffer.into())
            }
            buf = &buf[to_read..]
        }
    }

    /// Put a chunk of data into this [`WriteMultipart`] without copying
    ///
    /// Data is buffered using [`PutPayloadMut::push`]. Implementations looking to
    /// perform writes from non-owned buffers should prefer [`Self::write`] as this
    /// will allow multiple calls to share the same underlying allocation.
    ///
    /// See [`Self::write`] for information on backpressure
    pub fn put(&mut self, mut bytes: Bytes) {
        while !bytes.is_empty() {
            let remaining = self.chunk_size - self.buffer.content_length();
            if bytes.len() < remaining {
                self.buffer.push(bytes);
                return;
            }
            self.buffer.push(bytes.split_to(remaining));
            let buffer = std::mem::take(&mut self.buffer);
            self.put_part(buffer.into())
        }
    }

    pub(crate) fn put_part(&mut self, part: PutPayload) {
        self.tasks.spawn(self.upload.put_part(part));
    }

    /// Abort this upload, attempting to clean up any successfully uploaded parts
    pub async fn abort(mut self) -> Result<()> {
        self.tasks.shutdown().await;
        self.upload.abort().await
    }

    /// Flush final chunk, and await completion of all in-flight requests
    pub async fn finish(mut self) -> Result<PutResult> {
        if !self.buffer.is_empty() {
            let part = std::mem::take(&mut self.buffer);
            self.put_part(part.into())
        }

        self.wait_for_capacity(0).await?;

        match self.upload.complete().await {
            Err(e) => {
                self.tasks.shutdown().await;
                self.upload.abort().await?;
                Err(e)
            },
            Ok(result) => Ok(result),
        }
    }
}
