use anyhow::Result;
use std::path::PathBuf;
use std::process::Command;
use tokio::process::Command as TokioCommand;

pub struct VoiceEngine {
    piper_path: PathBuf,
    whisper_path: PathBuf,
    tts_voice: String,
    stt_model: String,
}

impl VoiceEngine {
    pub fn new(tts_voice: String, stt_model: String) -> Result<Self> {
        let piper_path = Self::find_piper()?;
        let whisper_path = Self::find_whisper()?;
        
        Ok(Self {
            piper_path,
            whisper_path,
            tts_voice,
            stt_model,
        })
    }

    fn find_piper() -> Result<PathBuf> {
        let paths = [
            "/usr/bin/piper",
            "/usr/local/bin/piper",
            "piper",
        ];
        
        for path in paths {
            if Command::new(path).arg("--version").output().is_ok() {
                return Ok(PathBuf::from(path));
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
        let paths = [
            "/usr/bin/whisper.cpp",
            "/usr/local/bin/whisper.cpp",
            "/usr/bin/whisper-cli",
            "/usr/local/bin/whisper-cli",
            "whisper-cli",
            "whisper.cpp",
        ];
        
        for path in paths {
            if Command::new(path).arg("--help").output().is_ok() {
                return Ok(PathBuf::from(path));
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
        
        if let Some(mut stdin) = play.stdin.take() {
            use tokio::io::AsyncWriteExt;
            stdin.write_all(&output.stdout).await?;
        }
        
        play.wait().await?;
        
        Ok(())
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
        
        let temp_audio = std::env::temp_dir().join("voice_input.wav");
        tokio::fs::write(&temp_audio, &record_output.stdout).await?;
        
        let whisper_output = TokioCommand::new(&self.whisper_path)
            .args([
                "-m", &self.stt_model,
                "-f", temp_audio.to_str().unwrap(),
                "-otxt",
            ])
            .output()
            .await?;
        
        let _ = tokio::fs::remove_file(&temp_audio).await;
        
        if !whisper_output.status.success() {
            return Err(anyhow::anyhow!("Whisper STT failed: {}", String::from_utf8_lossy(&whisper_output.stderr)));
        }
        
        let temp_txt = std::env::temp_dir().join("voice_input.txt");
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
            stt_model: "ggml-base.en.bin".to_string(),
        })
    }
}
