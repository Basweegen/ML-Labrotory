use anyhow::{Context, Result};
use async_trait::async_trait;
use crate::ollama::api::{Model, OllamaClient as ApiClient};
use crate::ollama::cli::{OllamaCli, CliModel};

#[async_trait]
pub trait OllamaService: Send + Sync {
    async fn list_models(&self) -> Result<Vec<Model>>;
    async fn pull_model(&self, name: &str, progress_tx: Option<tokio::sync::mpsc::Sender<String>>) -> Result<()>;
    async fn delete_model(&self, name: &str) -> Result<()>;
    async fn is_available(&self) -> bool;
}

#[derive(Clone)]
pub struct ApiOllamaService {
    client: ApiClient,
}

impl ApiOllamaService {
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        Ok(Self {
            client: ApiClient::new(base_url)?,
        })
    }
}

#[async_trait]
impl OllamaService for ApiOllamaService {
    async fn list_models(&self) -> Result<Vec<Model>> {
        self.client.list_models().await
    }

    async fn pull_model(&self, name: &str, _progress_tx: Option<tokio::sync::mpsc::Sender<String>>) -> Result<()> {
        self.client.pull_model(name).await
    }

    async fn delete_model(&self, name: &str) -> Result<()> {
        self.client.delete_model(name).await
    }

    async fn is_available(&self) -> bool {
        self.client.list_models().await.is_ok()
    }
}

#[derive(Clone)]
pub struct CliOllamaService {
    cli: OllamaCli,
}

impl CliOllamaService {
    pub fn new() -> Result<Self> {
        Ok(Self {
            cli: OllamaCli::new()?,
        })
    }
}

#[async_trait]
impl OllamaService for CliOllamaService {
    async fn list_models(&self) -> Result<Vec<Model>> {
        let cli_models = self.cli.list_models().await?;
        Ok(cli_models.into_iter().map(|m| Model {
            name: m.name,
            modified_at: m.modified,
            // CLI sizes may not parse; never report 0 or the RAM guard
            // would treat the model as free and over-assign.
            size: match m.size.parse() {
                Ok(0) | Err(_) => crate::resources::ESTIMATED_MODEL_BYTES,
                Ok(n) => n,
            },
            digest: m.digest,
            details: None,
        }).collect())
    }

    async fn pull_model(&self, name: &str, progress_tx: Option<tokio::sync::mpsc::Sender<String>>) -> Result<()> {
        use std::process::Stdio;
        use tokio::io::{AsyncBufReadExt, BufReader};
        use tokio::process::Command;
        use tokio::sync::Mutex;
        use std::sync::Arc;

        let mut child = Command::new(&self.cli.get_ollama_path())
            .args(["pull", name])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdout = child.stdout.take().context("Failed to capture stdout")?;
        let stderr = child.stderr.take().context("Failed to capture stderr")?;

        let mut stdout_reader = BufReader::new(stdout).lines();
        let mut stderr_reader = BufReader::new(stderr).lines();

        let last_error = Arc::new(Mutex::new(String::new()));
        let last_error_stdout = last_error.clone();
        let last_error_stderr = last_error.clone();

        let progress_tx_stdout = progress_tx.clone();
        let progress_tx_stderr = progress_tx.clone();

        let stdout_task = async move {
            while let Some(line) = stdout_reader.next_line().await? {
                if let Some(tx) = &progress_tx_stdout {
                    let _ = tx.send(line.clone()).await;
                }
                if line.contains("error") || line.contains("Error") {
                    let mut err = last_error_stdout.lock().await;
                    err.push_str(&line);
                    err.push('\n');
                }
            }
            Ok::<_, anyhow::Error>(())
        };

        let stderr_task = async move {
            while let Some(line) = stderr_reader.next_line().await? {
                if let Some(tx) = &progress_tx_stderr {
                    let _ = tx.send(line.clone()).await;
                }
                let mut err = last_error_stderr.lock().await;
                err.push_str(&line);
                err.push('\n');
            }
            Ok::<_, anyhow::Error>(())
        };

        tokio::try_join!(stdout_task, stderr_task)?;

        let status = child.wait().await.map_err(|e| anyhow::anyhow!(e.to_string()))?;
        if !status.success() {
            let err = last_error.lock().await;
            return Err(anyhow::anyhow!(err.clone()));
        }
        Ok(())
    }

    async fn delete_model(&self, name: &str) -> Result<()> {
        self.cli.delete_model(name).await
    }

    async fn is_available(&self) -> bool {
        self.cli.list_models().await.is_ok()
    }
}

pub struct ModelService {
    api_service: Option<ApiOllamaService>,
    cli_service: Option<CliOllamaService>,
    cache: tokio::sync::RwLock<Option<(Vec<Model>, std::time::Instant)>>,
    cache_ttl: std::time::Duration,
}

impl ModelService {
    pub fn new(ollama_url: &str) -> Result<Self> {
        let api_service = ApiOllamaService::new(ollama_url).ok();
        let cli_service = CliOllamaService::new().ok();
        
        Ok(Self {
            api_service,
            cli_service,
            cache: tokio::sync::RwLock::new(None),
            cache_ttl: std::time::Duration::from_secs(30),
        })
    }

    pub async fn list_models(&self, force_refresh: bool) -> Result<Vec<Model>> {
        if !force_refresh {
            let cache = self.cache.read().await;
            if let Some((models, timestamp)) = cache.as_ref() {
                if timestamp.elapsed() < self.cache_ttl {
                    return Ok(models.clone());
                }
            }
        }

        let models = if let Some(api) = &self.api_service {
            api.list_models().await?
        } else if let Some(cli) = &self.cli_service {
            cli.list_models().await?
        } else {
            return Err(anyhow::anyhow!("No Ollama service available"));
        };

        let mut cache = self.cache.write().await;
        *cache = Some((models.clone(), std::time::Instant::now()));
        
        Ok(models)
    }

    pub async fn pull_model(&self, name: &str, progress_tx: Option<tokio::sync::mpsc::Sender<String>>) -> Result<()> {
        if let Some(cli) = &self.cli_service {
            cli.pull_model(name, progress_tx).await
        } else if let Some(api) = &self.api_service {
            api.pull_model(name, progress_tx).await
        } else {
            Err(anyhow::anyhow!("No Ollama service available"))
        }
    }

    pub async fn delete_model(&self, name: &str) -> Result<()> {
        if let Some(cli) = &self.cli_service {
            cli.delete_model(name).await
        } else if let Some(api) = &self.api_service {
            api.delete_model(name).await
        } else {
            Err(anyhow::anyhow!("No Ollama service available"))
        }
    }

    pub async fn is_available(&self) -> bool {
        if let Some(s) = self.api_service.as_ref() {
            if s.is_available().await {
                return true;
            }
        }
        if let Some(s) = self.cli_service.as_ref() {
            return s.is_available().await;
        }
        false
    }

    pub fn invalidate_cache(&self) {
        let cache = self.cache.try_write();
        if let Ok(mut c) = cache {
            *c = None;
        }
    }
}
