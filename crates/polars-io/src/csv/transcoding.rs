use std::io::Write;

use encoding_rs::{Encoder, EncoderResult, Encoding};

pub(super) struct TranscodingWriter<'a, W> {
    sink: &'a mut W,
    encoder: Encoder,
    buffer: [u8; 1024],
}

impl<'a, W> TranscodingWriter<'a, W>
where
    W: Write,
{
    pub(super) fn new(sink: &'a mut W, encoding: &'static Encoding) -> Self {
        return Self {
            sink,
            encoder: encoding.new_encoder(),
            buffer: [0; 1024],
        };
    }
}

impl<'a, W> Write for TranscodingWriter<'a, W>
where
    W: Write,
{
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // safety: provided buffer is known to be UTF8
        let src = unsafe { std::str::from_utf8_unchecked(buf) };
        let (result, n_bytes_read, n_bytes_written) = self
            .encoder
            .encode_from_utf8_without_replacement(src, &mut self.buffer, false);
        match result {
            EncoderResult::InputEmpty | EncoderResult::OutputFull => self
                .sink
                .write_all(&mut self.buffer[..n_bytes_written])
                .and(Ok(n_bytes_read)),
            EncoderResult::Unmappable(c) => Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("failed to encode character '{}'", c),
            )),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if self.encoder.has_pending_state() {
            let (result, _, _) =
                self.encoder
                    .encode_from_utf8_without_replacement("", &mut self.buffer, true);
            match result {
                EncoderResult::OutputFull => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("failed to finalize encoding"),
                    ))
                }
                _ => {}
            };
            self.sink.write_all(&mut self.buffer[..])?;
        }
        self.sink.flush()
    }
}
