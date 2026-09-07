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
        }
    }
}

pub struct Storage {
    db: Db,
    sessions_tree: sled::Tree,
    config_tree: sled::Tree,
    profiles_tree: sled::Tree,
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
        let profiles_tree = db.open_tree("profiles")?;
        Ok(Self { db, sessions_tree, config_tree, profiles_tree })
    }

    pub fn save_session(&self, session: &ChatSession) -> Result<()> {
        let key = session.id.as_bytes().to_vec();
        let value = bincode::serialize(session)?;
        self.sessions_tree.insert(key, value)?;
        self.sessions_tree.flush()?;
        Ok(())
    }

    pub fn load_sessions(&self) -> Result<Vec<ChatSession>> {
        let mut sessions = Vec::new();
        for entry in self.sessions_tree.iter() {
            let (_, value) = entry?;
            let session: ChatSession = bincode::deserialize(&value)?;
            sessions.push(session);
        }
        sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(sessions)
    }

    pub fn delete_session(&self, id: Uuid) -> Result<()> {
        self.sessions_tree.remove(id.as_bytes())?;
        self.sessions_tree.flush()?;
        Ok(())
    }

    pub fn save_ollama_url(&self, url: &str) -> Result<()> {
        self.config_tree.insert("ollama_url", url.as_bytes())?;
        self.config_tree.flush()?;
        Ok(())
    }

    pub fn get_ollama_url(&self) -> Option<String> {
        self.config_tree.get("ollama_url").ok().flatten().map(|v| String::from_utf8(v.to_vec()).ok()).flatten()
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

    pub fn save_profile(&self, profile: &ModelProfile) -> Result<()> {
        let key = profile.id.as_bytes().to_vec();
        let value = bincode::serialize(profile)?;
        self.profiles_tree.insert(key, value)?;
        self.profiles_tree.flush()?;
        Ok(())
    }

    pub fn load_profiles(&self) -> Result<Vec<ModelProfile>> {
        let mut profiles = Vec::new();
        for entry in self.profiles_tree.iter() {
            let (_, value) = entry?;
            let profile: ModelProfile = bincode::deserialize(&value)?;
            profiles.push(profile);
        }
        profiles.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(profiles)
    }

    pub fn delete_profile(&self, id: Uuid) -> Result<()> {
        self.profiles_tree.remove(id.as_bytes())?;
        self.profiles_tree.flush()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub id: Uuid,
    pub name: String,
    pub model_name: String,
    pub system_prompt: String,
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: u32,
    pub skills: Vec<Skill>,
    pub usage_count: u64,
    pub avg_rating: f32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Default for ModelProfile {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "Default".to_string(),
            model_name: "llama3.1".to_string(),
            system_prompt: "You are a helpful AI assistant.".to_string(),
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 4096,
            skills: Vec::new(),
            usage_count: 0,
            avg_rating: 0.0,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub proficiency: f32,
    pub last_used: DateTime<Utc>,
    pub usage_count: u32,
}

impl Skill {
    pub fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            proficiency: 0.0,
            last_used: Utc::now(),
            usage_count: 0,
        }
    }
}
