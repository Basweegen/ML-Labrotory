// Copyright 2026 Sean M. Stow. All rights reserved.
use anyhow::Result;
use std::path::PathBuf;
use std::process::Command;
use tokio::process::Command as TokioCommand;

#[derive(Clone)]
pub struct VoiceEngine {
    piper_path: PathBuf,
    whisper_path: PathBuf,
    tts_voice: String,
    stt_model: String,
    /// PID of the live `aplay` child, if any (shared across clones so any
    /// handle can stop playback; the speak task reaps the child on wait).
    player_pid: std::sync::Arc<std::sync::Mutex<Option<u32>>>,
}

impl VoiceEngine {
    pub fn new(tts_voice: String, stt_model: String) -> Result<Self> {
        let piper_path = Self::find_piper()?;
        let whisper_path = Self::find_whisper()?;
        
        Ok(Self {
            piper_path,
            whisper_path,
            tts_voice: Self::resolve_voice(&tts_voice),
            stt_model: Self::resolve_model(&stt_model),
            player_pid: std::sync::Arc::new(std::sync::Mutex::new(None)),
        })
    }

    /// Bare names resolve against the app data dir (old settings keep working);
    /// absolute/existing paths pass through untouched.
    fn data_file(subdir: &str, name: &str) -> Option<PathBuf> {
        let base = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("ai-dashboard")
            .join(subdir);
        let direct = base.join(name);
        if direct.is_file() {
            return Some(direct);
        }
        let with_ext = base.join(format!("{}.onnx", name));
        if with_ext.is_file() {
            return Some(with_ext);
        }
        None
    }

    fn resolve_voice(name: &str) -> String {
        if PathBuf::from(name).is_file() {
            return name.to_string();
        }
        if let Some(p) = Self::data_file("voices", name) {
            return p.to_string_lossy().to_string();
        }
        name.to_string()
    }

    fn resolve_model(name: &str) -> String {
        if PathBuf::from(name).is_file() {
            return name.to_string();
        }
        if let Some(p) = Self::data_file("whisper", name) {
            return p.to_string_lossy().to_string();
        }
        name.to_string()
    }

    fn local_bin(name: &str) -> Option<PathBuf> {
        dirs::home_dir().map(|h| h.join(".local/bin").join(name))
    }

    /// Absolute path for a bare binary name via `which`, if present on PATH.
    fn on_path(name: &str) -> Option<PathBuf> {
        let out = Command::new("which").arg(name).output().ok()?;
        if !out.status.success() {
            return None;
        }
        let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if p.is_empty() {
            None
        } else {
            Some(PathBuf::from(p))
        }
    }

    fn find_piper() -> Result<PathBuf> {
        let mut paths: Vec<PathBuf> = vec![
            PathBuf::from("/usr/bin/piper"),
            PathBuf::from("/usr/local/bin/piper"),
        ];
        if let Some(p) = Self::on_path("piper") {
            paths.push(p);
        }
        if let Some(p) = Self::local_bin("piper") {
            paths.push(p);
        }
        
        for path in &paths {
            if Command::new(path).arg("--help").output().is_ok() {
                return Ok(path.clone());
            }
        }

        if let Ok(output) = Command::new("which").arg("piper").output() {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path.is_empty() {
                    return Ok(PathBuf::from(path));
                }
            }
        }
        
        Err(anyhow::anyhow!("Piper TTS not found. Install with: pip install piper-tts"))
    }

    fn find_whisper() -> Result<PathBuf> {
        let mut paths: Vec<PathBuf> = vec![
            PathBuf::from("/usr/bin/whisper.cpp"),
            PathBuf::from("/usr/local/bin/whisper.cpp"),
            PathBuf::from("/usr/bin/whisper-cli"),
            PathBuf::from("/usr/local/bin/whisper-cli"),
        ];
        for name in ["whisper-cli", "whisper.cpp"] {
            if let Some(p) = Self::on_path(name) {
                paths.push(p);
            }
            if let Some(p) = Self::local_bin(name) {
                paths.push(p);
            }
        }
        
        for path in &paths {
            if Command::new(path).arg("--help").output().is_ok() {
                return Ok(path.clone());
            }
        }

        if let Ok(output) = Command::new("which").arg("whisper-cli").output() {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path.is_empty() {
                    return Ok(PathBuf::from(path));
                }
            }
        }
        
        Err(anyhow::anyhow!("Whisper.cpp not found. Install with: cargo install whisper-rs or build from source"))
    }

    pub async fn speak(&self, text: &str) -> Result<()> {
        let mut output = TokioCommand::new(&self.piper_path)
            .args([
                "--model", &self.tts_voice,
                "--output_raw",
            ])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()?;
        
        if let Some(mut stdin) = output.stdin.take() {
            use tokio::io::AsyncWriteExt;
            stdin.write_all(text.as_bytes()).await?;
        }
        
        let output = output.wait_with_output().await?;
        
        if !output.status.success() {
            return Err(anyhow::anyhow!("Piper TTS failed: {}", String::from_utf8_lossy(&output.stderr)));
        }
        
        let mut play = TokioCommand::new("aplay")
            .args(["-f", "S16_LE", "-r", "22050", "-c", "1", "-"])
            .stdin(std::process::Stdio::piped())
            .spawn()?;
        
        if let Some(id) = play.id() {
            *self.player_pid.lock().unwrap() = Some(id);
        }

        if let Some(mut stdin) = play.stdin.take() {
            use tokio::io::AsyncWriteExt;
            stdin.write_all(&output.stdout).await?;
        }
        
        let status = play.wait().await;
        *self.player_pid.lock().unwrap() = None;
        status?;
        
        Ok(())
    }

    /// Kill live playback, if any. The speak task reaps the child on wait.
    pub async fn stop(&self) {
        let pid = self.player_pid.lock().unwrap().take();
        if let Some(pid) = pid {
            let _ = TokioCommand::new("kill")
                .args(["-9", &pid.to_string()])
                .status()
                .await;
        }
    }

    pub async fn listen(&self) -> Result<String> {
        let record = TokioCommand::new("arecord")
            .args([
                "-f", "S16_LE",
                "-r", "16000",
                "-c", "1",
                "-d", "5",
                "-t", "wav",
                "-",
            ])
            .stdout(std::process::Stdio::piped())
            .spawn()?;
        
        let record_output = record.wait_with_output().await?;
        
        if !record_output.status.success() {
            return Err(anyhow::anyhow!("Audio recording failed"));
        }
        
        // Unique temp name: a fixed name in shared /tmp is a symlink-attack target.
        let tag = format!("{}_{}", std::process::id(), chrono::Utc::now().timestamp_millis());
        let temp_audio = std::env::temp_dir().join(format!("voice_input_{tag}.wav"));
        tokio::fs::write(&temp_audio, &record_output.stdout).await?;
        
        let temp_audio_str = temp_audio.to_string_lossy().into_owned();
        let whisper_output = TokioCommand::new(&self.whisper_path)
            .args([
                "-m",
                self.stt_model.as_str(),
                "-f",
                temp_audio_str.as_str(),
                "-otxt",
            ])
            .output()
            .await?;
        
        let _ = tokio::fs::remove_file(&temp_audio).await;
        
        if !whisper_output.status.success() {
            return Err(anyhow::anyhow!("Whisper STT failed: {}", String::from_utf8_lossy(&whisper_output.stderr)));
        }
        
        let temp_txt = std::env::temp_dir().join(format!("voice_input_{tag}.txt"));
        if temp_txt.exists() {
            let text = tokio::fs::read_to_string(&temp_txt).await?;
            let _ = tokio::fs::remove_file(&temp_txt).await;
            Ok(text.trim().to_string())
        } else {
            Ok(String::from_utf8_lossy(&whisper_output.stdout).trim().to_string())
        }
    }

    pub fn is_available(&self) -> bool {
        self.piper_path.exists() && self.whisper_path.exists()
    }
}

impl Default for VoiceEngine {
    fn default() -> Self {
        Self::new(
            "en_US-lessac-medium".to_string(),
            "ggml-base.en.bin".to_string(),
        ).unwrap_or_else(|_| Self {
            piper_path: PathBuf::from("piper"),
            whisper_path: PathBuf::from("whisper-cli"),
            tts_voice: "en_US-lessac-medium".to_string(),
            player_pid: std::sync::Arc::new(std::sync::Mutex::new(None)),
            stt_model: "ggml-base.en.bin".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_passes_existing_paths_through() {
        assert_eq!(VoiceEngine::resolve_voice("/tmp/x.onnx"), "/tmp/x.onnx");
        assert_eq!(VoiceEngine::resolve_model("/tmp/y.bin"), "/tmp/y.bin");
    }

    #[test]
    fn resolve_passes_unknown_names_through() {
        assert_eq!(
            VoiceEngine::resolve_voice("definitely-not-a-voice-xyz"),
            "definitely-not-a-voice-xyz"
        );
    }

    #[test]
    fn local_bin_points_at_home() {
        let p = VoiceEngine::local_bin("piper").expect("home dir");
        assert!(p.ends_with(".local/bin/piper"));
    }
}
