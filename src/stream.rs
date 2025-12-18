use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;

use crate::error::LLMError;
use crate::http::HttpBodyStream;

/// Standardized SSE event yielded by [`StreamDecoder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    /// Raw `data:` payload emitted by the provider.
    Data(String),
    /// Terminal marker reported via `[DONE]`.
    Done,
}

/// Normalizes provider SSE feeds into [`StreamEvent`] values.
pub struct StreamDecoder {
    body: HttpBodyStream,
    buffer: Vec<u8>,
    data_lines: Vec<Vec<u8>>,
    pending: VecDeque<Result<StreamEvent, LLMError>>,
    provider: &'static str,
    stream_closed: bool,
    done_received: bool,
}

impl StreamDecoder {
    /// Wraps a raw HTTP body stream and prepares it for SSE decoding.
    pub fn new(body: HttpBodyStream, provider: &'static str) -> Self {
        Self {
            body,
            buffer: Vec::new(),
            data_lines: Vec::new(),
            pending: VecDeque::new(),
            provider,
            stream_closed: false,
            done_received: false,
        }
    }

    fn handle_line(&mut self, line: Vec<u8>) {
        if line.starts_with(b"data:") {
            let mut data = line[5..].to_vec();
            if let Some(first) = data.first() {
                if *first == b' ' {
                    data.remove(0);
                }
            }
            self.data_lines.push(data);
        }
    }

    fn flush_event(&mut self) -> Result<(), LLMError> {
        if self.data_lines.is_empty() {
            return Ok(());
        }

        let mut joined = Vec::new();
        for (idx, mut segment) in self.data_lines.drain(..).enumerate() {
            if idx > 0 {
                joined.push(b'\n');
            }
            joined.append(&mut segment);
        }

        if joined.is_empty() {
            return Ok(());
        }

        let data = String::from_utf8(joined).map_err(|err| LLMError::Provider {
            provider: self.provider,
            message: format!("invalid UTF-8 in stream chunk: {err}"),
        })?;

        if data.trim() == "[DONE]" {
            if !self.done_received {
                self.done_received = true;
                self.pending.push_back(Ok(StreamEvent::Done));
            }
        } else {
            self.pending.push_back(Ok(StreamEvent::Data(data)));
        }

        Ok(())
    }

    fn drain_line(buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
        buffer.iter().position(|b| *b == b'\n').map(|pos| {
            let mut line: Vec<u8> = buffer.drain(..=pos).collect();
            if line.last() == Some(&b'\n') {
                line.pop();
            }
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            line
        })
    }
}

impl Stream for StreamDecoder {
    type Item = Result<StreamEvent, LLMError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if let Some(event) = this.pending.pop_front() {
            return Poll::Ready(Some(event));
        }

        if this.done_received && this.pending.is_empty() {
            return Poll::Ready(None);
        }

        loop {
            if this.stream_closed {
                if !this.buffer.is_empty() {
                    let line = this.buffer.drain(..).collect::<Vec<u8>>();
                    this.handle_line(line);
                }
                if let Err(err) = this.flush_event() {
                    return Poll::Ready(Some(Err(err)));
                }
                return this
                    .pending
                    .pop_front()
                    .map_or(Poll::Ready(None), |event| Poll::Ready(Some(event)));
            }

            match this.body.as_mut().poll_next(cx) {
                Poll::Ready(Some(chunk_result)) => match chunk_result {
                    Ok(bytes) => {
                        this.buffer.extend_from_slice(&bytes);
                        while let Some(line) = Self::drain_line(&mut this.buffer) {
                            if line.is_empty() {
                                if let Err(err) = this.flush_event() {
                                    return Poll::Ready(Some(Err(err)));
                                }
                                if let Some(event) = this.pending.pop_front() {
                                    return Poll::Ready(Some(event));
                                }
                            } else {
                                this.handle_line(line);
                            }
                        }
                        if let Some(event) = this.pending.pop_front() {
                            return Poll::Ready(Some(event));
                        }
                    }
                    Err(err) => return Poll::Ready(Some(Err(err))),
                },
                Poll::Ready(None) => {
                    this.stream_closed = true;
                    continue;
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use futures_util::StreamExt;
    use futures_util::stream;

    use super::*;

    fn build_body(chunks: Vec<Result<Vec<u8>, LLMError>>) -> HttpBodyStream {
        Box::pin(stream::iter(chunks))
    }

    #[tokio::test]
    async fn decoder_emits_data_and_done_events() {
        let chunks = vec![
            Ok(b"data: {\"text\":\"hi\"}\n\n".to_vec()),
            Ok(b"data: [DONE]\n\n".to_vec()),
        ];
        let mut decoder = StreamDecoder::new(build_body(chunks), "test_provider");

        let first = decoder.next().await.expect("event").expect("ok");
        assert_eq!(first, StreamEvent::Data("{\"text\":\"hi\"}".to_string()));

        let second = decoder.next().await.expect("event").expect("ok");
        assert_eq!(second, StreamEvent::Done);

        assert!(decoder.next().await.is_none());
    }

    #[tokio::test]
    async fn decoder_combines_multiline_payloads() {
        let chunks = vec![
            Ok(b"data: line one\n".to_vec()),
            Ok(b"data: line two\n\n".to_vec()),
        ];
        let mut decoder = StreamDecoder::new(build_body(chunks), "test_provider");
        let event = decoder.next().await.expect("event").expect("ok");
        assert_eq!(event, StreamEvent::Data("line one\nline two".to_string()));
        assert!(decoder.next().await.is_none());
    }

    #[tokio::test]
    async fn decoder_reports_utf8_errors() {
        let chunks = vec![Ok(b"data: \xff\n\n".to_vec())];
        let mut decoder = StreamDecoder::new(build_body(chunks), "test_provider");
        let err = decoder.next().await.expect("event").unwrap_err();
        match err {
            LLMError::Provider { provider, .. } => assert_eq!(provider, "test_provider"),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
