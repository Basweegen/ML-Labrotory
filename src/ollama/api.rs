// Copyright 2026 Sean M. Stow. All rights reserved.
use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use futures::StreamExt;

fn deserialize_null_default<'de, D, T>(deserializer: D) -> std::result::Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    let opt = Option::deserialize(deserializer)?;
    Ok(opt.unwrap_or_default())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub name: String,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub modified_at: String,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub size: u64,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub digest: String,
    #[serde(default)]
    pub details: Option<ModelDetails>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDetails {
    // Tolerant parsing: older/newer Ollama servers may omit fields or return null;
    // a single missing or null field must not fail the entire model list (empty list disables
    // slot assignment and the Send button).
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub parent_model: String,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub format: String,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub family: String,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub families: Vec<String>,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub parameter_size: String,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub quantization_level: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_length: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Vec<String>>,
}


#[derive(Debug, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<ChatOptions>,
    // Bounds how long Ollama keeps the model resident after the reply.
    // Short retention matters on small-RAM machines (this box: 7 GB).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,
    /// Max context window. Small = less RAM + faster on old machines.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
    /// Number of CPU threads for matrix multiplication.
    /// Setting this optimally (e.g. to physical P-cores instead of all 22 hybrid logical threads)
    /// avoids spinning on slow Low-Power Island E-cores, boosting inference speed by 3x+.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_thread: Option<u32>,
    /// Number of GPU layers to offload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_gpu: Option<u32>,
    /// Number of tokens to process in parallel during prompt evaluation.
    /// Setting to 256 bounds transient RAM usage during context evaluation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_batch: Option<u32>,
    /// Memory map model weights to allow OS paging instead of full physical RAM allocation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_mmap: Option<bool>,
}

impl ChatOptions {
    /// Bounds for old / small-RAM hardware: short context, capped output.
    /// Automatically applies optimal CPU thread configuration.
    pub fn lowram() -> Self {
        Self::lowram_with_threads(None)
    }

    /// Low-RAM profile with optional explicit thread count.
    /// If `threads` is None or 0, auto-calculates optimal P-core thread allocation.
    pub fn lowram_with_threads(threads: Option<u32>) -> Self {
        let t = match threads {
            Some(n) if n > 0 => n,
            _ => Self::optimal_threads(),
        };
        Self {
            temperature: None,
            top_p: None,
            top_k: None,
            num_predict: Some(1024),
            num_ctx: Some(2048),
            num_thread: Some(t),
            num_gpu: None,
            num_batch: Some(256),
            use_mmap: Some(true),
        }
    }

    /// Calculate optimal CPU threads for llama.cpp / Ollama matrix multiplication.
    /// On modern hybrid architectures (e.g. Intel Core Ultra / Meteor Lake / Raptor Lake with P + E + LP-E cores),
    /// spreading inference threads across all logical cores (e.g. 22 threads) causes severe lock
    /// contention and latency spikes due to slow Low-Power Island cores (~1.0 GHz).
    /// Clamping threads to P-cores + standard threads (e.g. 8-12) achieves >3x tok/sec speedup.
    pub fn optimal_threads() -> u32 {
        let total = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) as u32;
        if total > 16 {
            12
        } else if total > 8 {
            8
        } else if total > 4 {
            total - 2
        } else {
            total.max(1)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: Message,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

#[derive(Debug, thiserror::Error)]
pub enum OllamaError {
    #[error("Request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("API error: {0}")]
    Api(String),
}

#[derive(Clone)]
pub struct OllamaClient {
    client: Client,
    base_url: String,
}

/// Host part of an http(s) URL (lowercased, port/userinfo stripped).
fn url_host(url: &str) -> Option<String> {
    let rest = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))?;
    let host_port = rest.split('/').next().unwrap_or("");
    if let Some(bracketed) = host_port.strip_prefix('[') {
        return bracketed
            .split(']')
            .next()
            .filter(|h| !h.is_empty())
            .map(|h| h.to_ascii_lowercase());
    }
    let host = host_port
        .split('@')
        .next_back()
        .unwrap_or("")
        .split(':')
        .next()
        .unwrap_or("");
    if host.is_empty() {
        None
    } else {
        Some(host.to_ascii_lowercase())
    }
}

fn host_is_loopback(host: &str) -> bool {
    host == "localhost" || host == "::1" || host.starts_with("127.")
}

impl OllamaClient {
    pub fn new(base_url: impl Into<String>, allow_remote: bool) -> Result<Self> {
        let base_url: String = base_url.into();
        // SSRF guard: settings-stored URL must be http(s); reject file:// etc.
        if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
            anyhow::bail!("refusing non-http ollama url: {:?}", base_url);
        }
        // Default-deny remote servers: a stray/compromised URL must not turn
        // the dashboard into a relay into the LAN or cloud metadata endpoints.
        if !allow_remote {
            match url_host(&base_url) {
                Some(h) if host_is_loopback(&h) => {}
                _ => anyhow::bail!(
                    "refusing non-local ollama url {:?} (enable Settings → Allow remote Ollama to override)",
                    base_url
                ),
            }
        }
        // Local inference on CPU can take many minutes for a long reply:
        // only the connect phase gets a short timeout, the body streams
        // until Ollama finishes (or the user hits Stop).
        // TCP nodelay disables Nagle's algorithm, eliminating 10-40ms buffering delays on loopback streaming chunks.
        let client = Client::builder()
            .tcp_nodelay(true)
            .tcp_keepalive(Some(Duration::from_secs(60)))
            .pool_idle_timeout(Some(Duration::from_secs(90)))
            .pool_max_idle_per_host(10)
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(1800))
            .build()
            .context("Failed to create HTTP client")?;
        Ok(Self {
            client,
            base_url,
        })
    }

    pub async fn list_models(&self) -> Result<Vec<Model>> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = self.client.get(&url).send().await?;
        if !resp.status().is_success() {
            return Err(OllamaError::Api(format!("Status: {}", resp.status())).into());
        }
        let body: serde_json::Value = resp.json().await?;
        let mut models = Vec::new();
        if let Some(items) = body.get("models").and_then(|m| m.as_array()) {
            for item in items {
                match serde_json::from_value::<Model>(item.clone()) {
                    Ok(model) => models.push(model),
                    Err(e) => {
                        eprintln!("[ML Laboratory] Warning: skipping unparseable model: {e}");
                    }
                }
            }
        }
        Ok(models)
    }

    /// Server version string (`/api/version` -> {"version": "0.1.2"}).
    pub async fn version(&self) -> Result<String> {
        let url = format!("{}/api/version", self.base_url);
        let resp = self.client.get(&url).send().await?;
        if !resp.status().is_success() {
            return Err(OllamaError::Api(format!("Status: {}", resp.status())).into());
        }
        let v: serde_json::Value = resp.json().await?;
        v.get("version")
            .and_then(|x| x.as_str())
            .map(|x| x.to_string())
            .ok_or_else(|| OllamaError::Api("version missing".to_string()).into())
    }

    pub async fn pull_model(&self, name: &str) -> Result<()> {
        let url = format!("{}/api/pull", self.base_url);
        let body = serde_json::json!({ "name": name });
        let resp = self.client.post(&url).json(&body).send().await?;
        if !resp.status().is_success() {
            let err = resp.text().await.unwrap_or_default();
            return Err(OllamaError::Api(err).into());
        }
        Ok(())
    }

    pub async fn delete_model(&self, name: &str) -> Result<()> {
        let url = format!("{}/api/delete", self.base_url);
        let body = serde_json::json!({ "name": name });
        let resp = self.client.delete(&url).json(&body).send().await?;
        if !resp.status().is_success() {
            let err = resp.text().await.unwrap_or_default();
            return Err(OllamaError::Api(err).into());
        }
        Ok(())
    }

    /// Streaming chat. Calls `on_chunk` with each content piece as it
    /// arrives; resolves to the fully assembled response when done.
    pub async fn chat_stream(
        &self,
        req: ChatRequest,
        mut on_chunk: impl FnMut(&str) + Send,
    ) -> Result<ChatResponse> {
        let url = format!("{}/api/chat", self.base_url);
        let mut req = req;
        req.stream = true;
        let resp = self.client.post(&url).json(&req).send().await?;
        if !resp.status().is_success() {
            let err = resp.text().await.unwrap_or_default();
            return Err(OllamaError::Api(err).into());
        }
        let mut stream = resp.bytes_stream();
        // Buffer raw bytes across TCP chunk boundaries so multi-byte UTF-8
        // sequences are never severed mid-character, preventing replacement
        // character artifacts.
        let mut pending_bytes: Vec<u8> = Vec::new();
        let mut assembled = String::new();
        let mut last: Option<ChatResponse> = None;
        let feed_line = |line: &str,
                         assembled: &mut String,
                         last: &mut Option<ChatResponse>,
                         on_chunk: &mut dyn FnMut(&str)|
         -> Result<()> {
            let line = line.trim();
            if line.is_empty() {
                return Ok(());
            }
            let msg: ChatResponse = serde_json::from_str(line)
                .map_err(|e| OllamaError::Api(format!("bad stream line: {e}")))?;
            if !msg.message.content.is_empty() {
                assembled.push_str(&msg.message.content);
                on_chunk(&msg.message.content);
            }
            *last = Some(msg);
            Ok(())
        };
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(OllamaError::Request)?;
            pending_bytes.extend_from_slice(&chunk);
            while let Some(pos) = pending_bytes.iter().position(|&b| b == b'\n') {
                if let Ok(line_str) = std::str::from_utf8(&pending_bytes[..pos]) {
                    feed_line(line_str, &mut assembled, &mut last, &mut on_chunk)?;
                } else {
                    let line_str = String::from_utf8_lossy(&pending_bytes[..pos]);
                    feed_line(&line_str, &mut assembled, &mut last, &mut on_chunk)?;
                }
                pending_bytes.drain(..=pos);
            }
        }
        if !pending_bytes.is_empty() {
            let tail = std::mem::take(&mut pending_bytes);
            let line_str = String::from_utf8_lossy(&tail);
            feed_line(&line_str, &mut assembled, &mut last, &mut on_chunk)?;
        }
        match last {
            Some(mut fin) => {
                fin.message.content = assembled;
                fin.done = true;
                Ok(fin)
            }
            None => Err(OllamaError::Api("empty stream".into()).into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_policy() {
        assert!(OllamaClient::new("http://localhost:11434", false).is_ok());
        assert!(OllamaClient::new("http://LOCALHOST:11434", false).is_ok());
        assert!(OllamaClient::new("http://127.0.0.1:11434", false).is_ok());
        assert!(OllamaClient::new("http://127.9.9.9:11434", false).is_ok());
        assert!(OllamaClient::new("http://[::1]:11434", false).is_ok());
        assert!(OllamaClient::new("https://localhost:11434", false).is_ok());
    }

    #[test]
    fn remote_denied_by_default() {
        assert!(OllamaClient::new("http://192.168.1.5:11434", false).is_err());
        assert!(OllamaClient::new("http://10.0.0.2:11434", false).is_err());
        assert!(OllamaClient::new("http://ollama.lan:11434", false).is_err());
        assert!(OllamaClient::new("http://169.254.169.254/", false).is_err());
        assert!(OllamaClient::new("http://example.com/", false).is_err());
        assert!(OllamaClient::new("ftp://localhost/x", false).is_err());
        assert!(OllamaClient::new("not a url", false).is_err());
    }

    #[test]
    fn remote_allowed_with_flag() {
        assert!(OllamaClient::new("http://192.168.1.5:11434", true).is_ok());
        assert!(OllamaClient::new("http://ollama.lan:11434", true).is_ok());
        assert!(OllamaClient::new("ftp://x/", true).is_err());
    }

    #[test]
    fn parse_model_with_null_families() {
        let raw = r#"{
            "name": "blackgrg26/WORMGPT-14:latest",
            "modified_at": "2026-07-26T01:26:29.795731271-07:00",
            "size": 37282,
            "digest": "f9809643910c",
            "details": {
                "parent_model": "",
                "format": "",
                "family": "",
                "families": null,
                "parameter_size": "",
                "quantization_level": ""
            }
        }"#;
        let m: Result<Model, _> = serde_json::from_str(raw);
        assert!(m.is_ok(), "Failed to parse model with null families: {:?}", m.err());
        let details = m.unwrap().details.unwrap();
        assert!(details.families.is_empty());
    }

    #[test]
    fn chat_options_threading_and_serialization() {
        let opts = ChatOptions::lowram();
        assert!(opts.num_thread.is_some());
        let threads = opts.num_thread.unwrap();
        assert!(threads >= 1 && threads <= 32);
        assert_eq!(opts.num_batch, Some(256));
        assert_eq!(opts.use_mmap, Some(true));

        let json = serde_json::to_string(&opts).unwrap();
        assert!(json.contains("\"num_thread\":"));
        assert!(json.contains("\"num_ctx\":2048"));
        assert!(json.contains("\"num_batch\":256"));
        assert!(json.contains("\"use_mmap\":true"));

        let custom = ChatOptions::lowram_with_threads(Some(6));
        assert_eq!(custom.num_thread, Some(6));
        assert_eq!(custom.num_batch, Some(256));
        assert_eq!(custom.use_mmap, Some(true));
    }
}

