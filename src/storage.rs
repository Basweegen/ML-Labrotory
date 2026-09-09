use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sled::Db;
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    pub id: Uuid,
    pub name: String,
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Theme {
    Dark,
    Light,
    System,
}

impl Default for Theme {
    fn default() -> Self {
        Theme::Dark
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    pub theme: Theme,
    pub font_size: f32,
    pub ollama_url: String,
    pub voice_enabled: bool,
    pub tts_voice: String,
    pub stt_model: String,
    /// Global identity every model speaks as. Prepended to each slot's role.
    /// Same persona no matter which model is loaded.
    pub persona: String,
    /// Long-term facts the assistant remembers across models and restarts.
    pub memory: String,
    /// Slot layout restored on launch (model + role per slot).
    #[serde(default)]
    pub slot_layout: Vec<SlotConfig>,
}

/// One persisted model slot: assignment + role.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SlotConfig {
    pub model: Option<String>,
    pub role: String,
    pub custom_role: String,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            theme: Theme::Dark,
            font_size: 14.0,
            ollama_url: "http://localhost:11434".to_string(),
            voice_enabled: false,
            tts_voice: "en_US-lessac-medium".to_string(),
            stt_model: "ggml-base.en.bin".to_string(),
            persona: "You are ML Lab, a calm and direct assistant. Be concise, plain-spoken, and practical. Never mention model names unless asked.".to_string(),
            memory: String::new(),
            slot_layout: Vec::new(),
        }
    }
}

pub struct Storage {
    // Kept alive for the trees; never read directly.
    _db: Db,
    sessions_tree: sled::Tree,
    config_tree: sled::Tree,
}

impl Storage {
    pub fn new() -> Result<Self> {
        let data_dir = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("ai-dashboard");
        std::fs::create_dir_all(&data_dir)?;
        let db_path = data_dir.join("storage");
        let db = sled::open(db_path)?;
        let sessions_tree = db.open_tree("sessions")?;
        let config_tree = db.open_tree("config")?;
        Ok(Self { _db: db, sessions_tree, config_tree })
    }

    pub fn save_session(&self, session: &ChatSession) -> Result<()> {
        let key = session.id.as_bytes().to_vec();
        let value = bincode::serialize(session)?;
        self.sessions_tree.insert(key, value)?;
        self.sessions_tree.flush()?;
        Ok(())
    }

    /// Max messages kept per session when loading (bounds RAM + render cost).
    /// Max chars per message (bounds a runaway model blob in the db).
    pub const MAX_SESSION_MESSAGES: usize = 500;
    pub const MAX_MESSAGE_CHARS: usize = 50_000;

    fn sanitize_session(mut s: ChatSession) -> ChatSession {
        if s.messages.len() > Self::MAX_SESSION_MESSAGES {
            let overflow = s.messages.len() - Self::MAX_SESSION_MESSAGES;
            s.messages.drain(0..overflow);
        }
        for m in &mut s.messages {
            if m.content.len() > Self::MAX_MESSAGE_CHARS {
                let mut end = Self::MAX_MESSAGE_CHARS;
                while !m.content.is_char_boundary(end) { end -= 1; }
                m.content.truncate(end);
                m.content.push_str("…[truncated]");
            }
        }
        s.messages.shrink_to_fit();
        s
    }

    pub fn load_sessions(&self) -> Result<Vec<ChatSession>> {
        let mut sessions = Vec::new();
        for entry in self.sessions_tree.iter() {
            // One corrupt entry must not wipe the whole history (was: `?` aborted all).
            let (_, value) = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            if value.len() > 10 * 1024 * 1024 {
                continue; // absurdly large blob: skip rather than OOM
            }
            match bincode::deserialize::<ChatSession>(&value) {
                Ok(s) => sessions.push(Self::sanitize_session(s)),
                Err(_) => continue,
            }
        }
        sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(sessions)
    }

    pub fn delete_session(&self, id: Uuid) -> Result<()> {
        self.sessions_tree.remove(id.as_bytes())?;
        self.sessions_tree.flush()?;
        Ok(())
    }

    pub fn save_settings(&self, settings: &AppSettings) -> Result<()> {
        let value = bincode::serialize(settings)?;
        self.config_tree.insert("settings", value)?;
        self.config_tree.flush()?;
        Ok(())
    }

    pub fn load_settings(&self) -> Result<AppSettings> {
        if let Some(value) = self.config_tree.get("settings")? {
            match bincode::deserialize::<AppSettings>(&value) {
                Ok(settings) => Ok(settings),
                Err(_) => Ok(AppSettings::default()),
            }
        } else {
            Ok(AppSettings::default())
        }
    }

}
