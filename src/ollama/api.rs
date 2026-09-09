use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub name: String,
    #[serde(default)]
    pub modified_at: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub digest: String,
    pub details: Option<ModelDetails>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDetails {
    // Tolerant parsing: older/newer Ollama servers may omit fields; a single
    // missing field must not fail the entire model list (empty list disables
    // slot assignment and the Send button).
    #[serde(default)]
    pub parent_model: String,
    #[serde(default)]
    pub format: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub families: Vec<String>,
    #[serde(default)]
    pub parameter_size: String,
    #[serde(default)]
    pub quantization_level: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_length: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelsResponse {
    pub models: Vec<Model>,
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
}

impl ChatOptions {
    /// Bounds for old / small-RAM hardware: short context, capped output.
    /// Keeps 1-4B models responsive instead of swapping.
    pub fn lowram() -> Self {
        Self {
            temperature: None,
            top_p: None,
            top_k: None,
            num_predict: Some(1024),
            num_ctx: Some(2048),
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

impl OllamaClient {
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        let base_url: String = base_url.into();
        // SSRF guard: settings-stored URL must be local http(s); reject file:// etc.
        if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
            anyhow::bail!("refusing non-http ollama url: {:?}", base_url);
        }
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
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
        let data: ModelsResponse = resp.json().await?;
        Ok(data.models)
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

    pub async fn chat(&self, req: ChatRequest) -> Result<ChatResponse> {
        let url = format!("{}/api/chat", self.base_url);
        let resp = self.client.post(&url).json(&req).send().await?;
        if !resp.status().is_success() {
            let err = resp.text().await.unwrap_or_default();
            return Err(OllamaError::Api(err).into());
        }
        let data: ChatResponse = resp.json().await?;
        Ok(data)
    }

}
