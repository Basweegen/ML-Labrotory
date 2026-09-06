use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliModel {
    pub name: String,
    pub size: String,
    pub modified: String,
    pub digest: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CliListResponse {
    pub models: Vec<CliModel>,
}

#[derive(Debug, thiserror::Error)]
pub enum CliError {
    #[error("CLI not found: {0}")]
    NotFound(String),
    #[error("Command failed: {0}")]
    Command(String),
    #[error("Parse error: {0}")]
    Parse(String),
}

pub struct OllamaCli {
    ollama_path: String,
}

impl Clone for OllamaCli {
    fn clone(&self) -> Self {
        Self {
            ollama_path: self.ollama_path.clone(),
        }
    }
}

impl OllamaCli {
    pub fn get_ollama_path(&self) -> &str {
        &self.ollama_path
    }
    
    pub fn new() -> Result<Self> {
        let ollama_path = which::which("ollama")
            .context("ollama CLI not found in PATH")?
            .to_string_lossy()
            .to_string();
        Ok(Self { ollama_path })
    }

    pub async fn list_models(&self) -> Result<Vec<CliModel>> {
        let output = Command::new(&self.ollama_path)
            .args(["list", "--json"])
            .output()
            .await
            .map_err(|e| CliError::Command(e.to_string()))?;

        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            return Err(CliError::Command(err.to_string()).into());
        }

        let text = String::from_utf8_lossy(&output.stdout);
        let response: CliListResponse = serde_json::from_str(&text)
            .map_err(|e| CliError::Parse(e.to_string()))?;
        Ok(response.models)
    }

    pub async fn pull_model(&self, name: &str) -> Result<()> {
        let mut child = Command::new(&self.ollama_path)
            .args(["pull", name])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| CliError::Command(e.to_string()))?;

        let stdout = child.stdout.take().context("Failed to capture stdout")?;
        let stderr = child.stderr.take().context("Failed to capture stderr")?;

        let mut stdout_reader = BufReader::new(stdout).lines();
        let mut stderr_reader = BufReader::new(stderr).lines();

        let last_error = Arc::new(Mutex::new(String::new()));

        let last_error_stdout = last_error.clone();
        let last_error_stderr = last_error.clone();

        let stdout_task = async move {
            while let Some(line) = stdout_reader.next_line().await? {
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
                let mut err = last_error_stderr.lock().await;
                err.push_str(&line);
                err.push('\n');
            }
            Ok::<_, anyhow::Error>(())
        };

        tokio::try_join!(stdout_task, stderr_task)?;

        let status = child.wait().await.map_err(|e| CliError::Command(e.to_string()))?;
        if !status.success() {
            let err = last_error.lock().await;
            return Err(CliError::Command(err.clone()).into());
        }
        Ok(())
    }

    pub async fn delete_model(&self, name: &str) -> Result<()> {
        let output = Command::new(&self.ollama_path)
            .args(["rm", name])
            .output()
            .await
            .map_err(|e| CliError::Command(e.to_string()))?;

        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            return Err(CliError::Command(err.to_string()).into());
        }
        Ok(())
    }

    pub async fn run_model(&self, name: &str, prompt: &str) -> Result<String> {
        let output = Command::new(&self.ollama_path)
            .args(["run", name, prompt])
            .output()
            .await
            .map_err(|e| CliError::Command(e.to_string()))?;

        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            return Err(CliError::Command(err.to_string()).into());
        }

        let text = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(text)
    }
}
