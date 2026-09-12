// Copyright 2026 Sean M. Stow. All rights reserved.
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

#[derive(Debug, thiserror::Error)]
pub enum CliError {
    #[error("Command failed: {0}")]
    Command(String),
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

/// Model names go to `ollama` as argv entries (no shell), but a name like
/// `--help` or `-f/path` would still be parsed as a flag. Restrict the charset.
pub fn validate_model_name(name: &str) -> Result<()> {
    if name.is_empty() || name.len() > 128 {
        anyhow::bail!("bad model name: empty or >128 chars");
    }
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/' | ':')) {
        anyhow::bail!("bad model name: {:?} (allowed: A-Za-z0-9 ._-/:)", name);
    }
    if name.starts_with('-') {
        anyhow::bail!("bad model name: must not start with '-'");
    }
    Ok(())
}

impl OllamaCli {
    pub fn new() -> Result<Self> {
        let ollama_path = which::which("ollama")
            .context("ollama CLI not found in PATH")?
            .to_string_lossy()
            .to_string();
        Ok(Self { ollama_path })
    }

    pub async fn list_models(&self) -> Result<Vec<CliModel>> {
        let output = Command::new(&self.ollama_path)
            .arg("list")
            .output()
            .await
            .map_err(|e| CliError::Command(e.to_string()))?;

        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            return Err(CliError::Command(err.to_string()).into());
        }

        let text = String::from_utf8_lossy(&output.stdout);
        let mut models = Vec::new();
        for line in text.lines().skip(1) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                let name = parts[0].to_string();
                let digest = parts[1].to_string();
                let size = if parts.len() >= 4 {
                    format!("{} {}", parts[2], parts[3])
                } else {
                    parts[2].to_string()
                };
                let modified = parts.get(4..).map(|p| p.join(" ")).unwrap_or_default();
                models.push(CliModel {
                    name,
                    size,
                    modified,
                    digest,
                });
            }
        }
        Ok(models)
    }

    pub async fn pull_model(
        &self,
        name: &str,
        progress: Option<tokio::sync::mpsc::Sender<String>>,
    ) -> Result<()> {
        validate_model_name(name)?;
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
        let progress_stdout = progress.clone();
        let progress_stderr = progress.clone();

        let stdout_task = async move {
            while let Some(line) = stdout_reader.next_line().await? {
                if let Some(tx) = &progress_stdout {
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
                if let Some(tx) = &progress_stderr {
                    let _ = tx.send(line.clone()).await;
                }
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
        validate_model_name(name)?;
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

}
