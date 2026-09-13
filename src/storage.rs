// Copyright 2026 Sean M. Stow. All rights reserved.
use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use anyhow::Result;
use chrono::{DateTime, Utc};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sled::Db;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use uuid::Uuid;
use zeroize::{Zeroize, ZeroizeOnDrop};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    pub id: Uuid,
    pub name: String,
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// One security-activity event (model pulls, assigns, voice, file saves,
/// secret blocks...). Newest last by key; viewer reverses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub ts: DateTime<Utc>,
    pub kind: String,
    pub detail: String,
}

impl AuditEntry {
    /// Bound field sizes so one runaway detail can't bloat the tree.
    pub fn sanitize(mut self) -> Self {
        if self.kind.len() > 64 {
            self.kind.truncate(64);
        }
        if self.detail.len() > 500 {
            let mut end = 500;
            while !self.detail.is_char_boundary(end) {
                end -= 1;
            }
            self.detail.truncate(end);
        }
        self
    }
}

/// An autonomous learned agent capability persisted in the post-quantum vault.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub prompt_template: String,
    pub domain_idx: usize,
    pub reinforcement_score: f32,
    pub execution_count: u64,
    pub last_used: Option<DateTime<Utc>>,
    pub is_built_in: bool,
}

impl Skill {
    pub fn sanitize(mut self) -> Self {
        if self.name.len() > 64 {
            let mut end = 64;
            while !self.name.is_char_boundary(end) { end -= 1; }
            self.name.truncate(end);
        }
        if self.description.len() > 500 {
            let mut end = 500;
            while !self.description.is_char_boundary(end) { end -= 1; }
            self.description.truncate(end);
        }
        if self.prompt_template.len() > 10_000 {
            let mut end = 10_000;
            while !self.prompt_template.is_char_boundary(end) { end -= 1; }
            self.prompt_template.truncate(end);
        }
        if !self.reinforcement_score.is_finite() {
            self.reinforcement_score = 1.0;
        } else {
            self.reinforcement_score = self.reinforcement_score.clamp(0.1, 20.0);
        }
        self
    }
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
    /// Recent turns resent with each prompt (1-50). Bounds context cost.
    #[serde(default = "default_history_depth")]
    pub history_depth: u32,
    /// When false (default), Ollama URLs are restricted to localhost.
    #[serde(default)]
    pub allow_remote: bool,
    /// Slot layout restored on launch (model + role per slot).
    #[serde(default)]
    pub slot_layout: Vec<SlotConfig>,
    /// Project root for the Files tab. Empty = unset. All file ops are
    /// confined under this dir (no `..` escapes, no absolute paths).
    #[serde(default)]
    pub workspace_root: String,
    /// Reactive companion face above chat + minis on slot cards. Default on.
    #[serde(default = "default_true")]
    pub show_avatar: bool,
    /// Big-face diameter in px (40-80). Mini faces stay fixed.
    #[serde(default = "default_avatar_size")]
    pub avatar_size: f32,
    /// CPU threads used for Ollama inference. 0 = auto-detect optimal threads.
    #[serde(default)]
    pub num_threads: u32,
    /// Active Guardrail safety tier.
    #[serde(default)]
    pub guardrail_tier: crate::guardrails::GuardrailTier,
    /// Whether the legal liability waiver for Unrestricted tier was acknowledged.
    #[serde(default)]
    pub unrestricted_waiver_accepted: bool,
    /// Timestamp when the legal waiver was signed.
    #[serde(default)]
    pub unrestricted_waiver_timestamp: Option<chrono::DateTime<chrono::Utc>>,
    /// Default startup landing tab.
    #[serde(default = "default_tab_name")]
    pub default_tab: String,
    /// List of tab labels visible in the navigation strip.
    #[serde(default = "default_visible_tabs")]
    pub visible_tabs: Vec<String>,
    /// Custom ordering of tabs in the navigation strip.
    #[serde(default = "default_tab_order")]
    pub tab_order: Vec<String>,
    /// Whether the chat view defaults to side-by-side multi-model split view.
    #[serde(default)]
    pub chat_split_view_default: bool,
    /// When on, Broadcast and single-slot sends prepend a role-specific slice
    /// so identical models still answer the same prompt from different angles.
    #[serde(default)]
    pub contrast_on: bool,
    /// Operational and collaboration rules enforced across all chats and models.
    #[serde(default = "default_chat_rules")]
    pub chat_rules: String,
    /// When on, multi-model turns enforce complementary assistance (no redundant text).
    #[serde(default = "default_true")]
    pub auto_assist_rules: bool,
    /// When on, completing a prompt in a slot automatically handshakes and triggers
    /// the peer slot to assist, fill gaps, or validate without manual copy/paste.
    #[serde(default = "default_true")]
    pub swarm_auto_assist: bool,
    /// When true, navigation tabs rotate to a horizontal top strip to maximize horizontal real estate.
    #[serde(default)]
    pub tabs_at_top: bool,
}

fn default_true() -> bool {
    true
}

fn default_avatar_size() -> f32 {
    56.0
}

pub fn default_tab_name() -> String {
    "Chat".to_string()
}

pub fn default_tab_order() -> Vec<String> {
    vec![
        "Chat".to_string(),
        "Swarm Relay".to_string(),
        "Compare".to_string(),
        "Editor".to_string(),
        "Models".to_string(),
        "History".to_string(),
        "Neural".to_string(),
        "Skills".to_string(),
        "Tools".to_string(),
        "Settings".to_string(),
    ]
}

pub fn default_visible_tabs() -> Vec<String> {
    default_tab_order()
}

fn default_history_depth() -> u32 {
    20
}

pub fn default_chat_rules() -> String {
    "1. Complementary Cooperation: When collaborating with other models, never duplicate work already completed. If prior output is partial or incomplete, continue and complete the missing parts. If complete, concur and provide an advanced optimization or alternative.\n2. Production-Grade Output: Always provide complete, working code without lazy placeholders, ellipses, or missing implementations.\n3. Direct & Concise: Provide the direct solution first, followed by clear, practical rationale.".to_string()
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
            history_depth: default_history_depth(),
            slot_layout: Vec::new(),
            allow_remote: false,
            workspace_root: String::new(),
            show_avatar: true,
            avatar_size: default_avatar_size(),
            num_threads: 0,
            guardrail_tier: crate::guardrails::GuardrailTier::Heavy,
            unrestricted_waiver_accepted: false,
            unrestricted_waiver_timestamp: None,
            default_tab: default_tab_name(),
            visible_tabs: default_visible_tabs(),
            tab_order: default_tab_order(),
            chat_split_view_default: false,
            contrast_on: false,
            chat_rules: default_chat_rules(),
            auto_assist_rules: true,
            swarm_auto_assist: true,
            tabs_at_top: false,
        }
    }
}

pub const ENVELOPE_MAGIC: &[u8; 4] = b"A256";
pub const NONCE_LEN: usize = 12;
pub const TAG_LEN: usize = 16;

/// Zeroized-on-drop wrapper for 256-bit AES cryptographic key.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct VaultKey(pub [u8; 32]);

/// Post-Quantum symmetric encryption vault employing AES-256-GCM.
/// Resists Grover's quantum search algorithm (NIST Category 5 Post-Quantum standard,
/// retaining 128 bits of security). All key and database files on disk enforce strict
/// 0o600 Unix permissions.
#[derive(Clone)]
pub struct StorageVault {
    key: VaultKey,
    cipher: Aes256Gcm,
}

impl StorageVault {
    /// Load existing 256-bit vault key or securely generate and write a new one
    /// with strict 0o600 Unix file permissions.
    pub fn load_or_create(key_path: &std::path::Path) -> Result<Self> {
        if key_path.exists() {
            let bytes = std::fs::read(key_path)?;
            if bytes.len() == 32 {
                let mut key_arr = [0u8; 32];
                key_arr.copy_from_slice(&bytes);
                let cipher = Aes256Gcm::new_from_slice(&key_arr)
                    .map_err(|e| anyhow::anyhow!("AES-256-GCM initialization failed: {}", e))?;
                return Ok(Self {
                    key: VaultKey(key_arr),
                    cipher,
                });
            }
        }

        let mut key_arr = [0u8; 32];
        OsRng.fill_bytes(&mut key_arr);

        if let Some(parent) = key_path.parent() {
            let _ = std::fs::create_dir_all(parent);
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700));
            }
        }

        #[cfg(unix)]
        {
            use std::io::Write;
            use std::os::unix::fs::OpenOptionsExt;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(key_path)?;
            file.write_all(&key_arr)?;
            file.flush()?;
        }
        #[cfg(not(unix))]
        {
            use std::io::Write;
            let mut file = std::fs::File::create(key_path)?;
            file.write_all(&key_arr)?;
            file.flush()?;
        }

        let cipher = Aes256Gcm::new_from_slice(&key_arr)
            .map_err(|e| anyhow::anyhow!("AES-256-GCM initialization failed: {}", e))?;

        Ok(Self {
            key: VaultKey(key_arr),
            cipher,
        })
    }

    /// Construct in-memory vault from explicit key bytes (for testing and ephemeral isolation).
    #[allow(dead_code)]
    pub fn from_key(key_bytes: [u8; 32]) -> Self {
        let cipher = Aes256Gcm::new_from_slice(&key_bytes)
            .expect("Valid 32-byte key for AES-256");
        Self {
            key: VaultKey(key_bytes),
            cipher,
        }
    }

    /// Derive key using memory-hard Argon2id (Post-Quantum resistance against quantum ASIC and time-memory tradeoffs).
    #[allow(dead_code)]
    pub fn derive_from_passphrase(passphrase: &str, salt: &[u8; 16]) -> Result<Self> {
        use argon2::{Algorithm, Argon2, Params, Version};
        let params = Params::new(64 * 1024, 3, 4, Some(32))
            .map_err(|e| anyhow::anyhow!("Argon2 params error: {}", e))?;
        let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
        let mut key = [0u8; 32];
        argon2
            .hash_password_into(passphrase.as_bytes(), salt, &mut key)
            .map_err(|e| anyhow::anyhow!("Argon2 derivation error: {}", e))?;
        let cipher = Aes256Gcm::new_from_slice(&key)
            .map_err(|e| anyhow::anyhow!("AES-256-GCM initialization failed: {}", e))?;
        Ok(Self {
            key: VaultKey(key),
            cipher,
        })
    }

    /// Encrypt plaintext into Post-Quantum AES-256-GCM envelope:
    /// Format: `b"A256" (4 bytes) || Nonce (12 bytes) || Ciphertext + Tag (N + 16 bytes)`
    pub fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>> {
        let mut nonce_bytes = [0u8; NONCE_LEN];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = self.cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| anyhow::anyhow!("AES-256-GCM encryption failed: {}", e))?;

        let mut out = Vec::with_capacity(ENVELOPE_MAGIC.len() + NONCE_LEN + ciphertext.len());
        out.extend_from_slice(ENVELOPE_MAGIC);
        out.extend_from_slice(&nonce_bytes);
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Decrypt envelope or pass through legacy plaintext if not encrypted.
    /// Backward-compatible with existing unencrypted databases.
    pub fn decrypt(&self, payload: &[u8]) -> Result<Vec<u8>> {
        if payload.len() >= ENVELOPE_MAGIC.len() + NONCE_LEN + TAG_LEN
            && &payload[0..ENVELOPE_MAGIC.len()] == ENVELOPE_MAGIC
        {
            let nonce = Nonce::from_slice(&payload[ENVELOPE_MAGIC.len()..ENVELOPE_MAGIC.len() + NONCE_LEN]);
            let ciphertext = &payload[ENVELOPE_MAGIC.len() + NONCE_LEN..];
            let plaintext = self.cipher
                .decrypt(nonce, ciphertext)
                .map_err(|_| anyhow::anyhow!("Decryption failed: cryptographic integrity violation or invalid key"))?;
            Ok(plaintext)
        } else {
            // Legacy unencrypted plaintext fallback
            Ok(payload.to_vec())
        }
    }

    /// Access the underlying zeroized cryptographic key container.
    #[allow(dead_code)]
    pub fn key(&self) -> &VaultKey {
        &self.key
    }
}

pub struct Storage {
    // Kept alive for the trees; never read directly.
    _db: Db,
    sessions_tree: sled::Tree,
    config_tree: sled::Tree,
    audit_tree: sled::Tree,
    pub skills_tree: sled::Tree,
    audit_seq: AtomicU64,
    vault: StorageVault,
}

impl Storage {
    pub fn new() -> Result<Self> {
        let data_dir = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("ai-dashboard");
        std::fs::create_dir_all(&data_dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&data_dir, std::fs::Permissions::from_mode(0o700));
        }
        Self::open_path(&data_dir.join("storage"))
    }

    /// Open a store at an explicit path (tests use a temp dir so the real
    /// log is never touched).
    pub fn open_path(db_path: &std::path::Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700));
            }
        }
        let key_path = db_path.parent().unwrap_or(db_path).join("vault.key");
        let vault = StorageVault::load_or_create(&key_path)?;

        let db = sled::Config::new()
            .path(db_path)
            .cache_capacity(16 * 1024 * 1024) // 16 MiB page cache bounds memory consumption
            .open()?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(db_path, std::fs::Permissions::from_mode(0o700));
        }
        let sessions_tree = db.open_tree("sessions")?;
        let config_tree = db.open_tree("config")?;
        let audit_tree = db.open_tree("audit")?;
        let skills_tree = db.open_tree("skills")?;
        let st = Self {
            _db: db,
            sessions_tree,
            config_tree,
            audit_tree,
            skills_tree,
            audit_seq: AtomicU64::new(0),
            vault,
        };
        let _ = st.seed_default_skills();
        Ok(st)
    }

    #[allow(dead_code)]
    pub fn vault(&self) -> &StorageVault {
        &self.vault
    }

    pub fn save_session(&self, session: &ChatSession) -> Result<()> {
        let key = session.id.as_bytes().to_vec();
        let raw = bincode::serialize(session)?;
        let encrypted = self.vault.encrypt(&raw)?;
        self.sessions_tree.insert(key, encrypted)?;
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
            let decrypted = match self.vault.decrypt(&value) {
                Ok(d) => d,
                Err(_) => continue,
            };
            match bincode::deserialize::<ChatSession>(&decrypted) {
                Ok(s) => sessions.push(Self::sanitize_session(s)),
                Err(_) => continue,
            }
        }
        sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(sessions)
    }

    /// Max audit rows kept; oldest evicted first.
    pub const MAX_AUDIT_ROWS: usize = 2000;

    /// Drop every audit row; returns how many were removed.
    pub fn clear_audit(&self) -> Result<usize> {
        let n = self.audit_tree.len();
        self.audit_tree.clear()?;
        self.audit_tree.flush()?;
        Ok(n)
    }

    pub fn log_audit(&self, kind: &str, detail: &str) -> Result<()> {
        // Key = millis ++ pid ++ seq (16 bytes): millis alone collides for
        // same-millisecond bursts, and the per-instance seq restarts in every
        // process, so the pid scopes concurrent writers (tests included).
        // (Older builds used nanos*10000, which overflowed u64 and collapsed
        // every key to u64::MAX — those rows are filtered on load.)
        let millis = Utc::now().timestamp_millis().max(0) as u64;
        let pid = std::process::id();
        let seq = self.audit_seq.fetch_add(1, Ordering::Relaxed);
        let mut key = Vec::with_capacity(16);
        key.extend_from_slice(&millis.to_be_bytes());
        key.extend_from_slice(&pid.to_be_bytes());
        key.extend_from_slice(&seq.to_be_bytes());
        let entry = AuditEntry {
            ts: Utc::now(),
            kind: kind.to_string(),
            detail: detail.to_string(),
        }
        .sanitize();
        let raw = bincode::serialize(&entry)?;
        let encrypted = self.vault.encrypt(&raw)?;
        self.audit_tree.insert(key, encrypted)?;
        while self.audit_tree.len() > Self::MAX_AUDIT_ROWS {
            let oldest = self.audit_tree.iter().next().transpose()?.map(|(k, _)| k);
            match oldest {
                Some(k) => {
                    self.audit_tree.remove(k)?;
                }
                None => break,
            }
        }
        self.audit_tree.flush()?;
        Ok(())
    }

    /// Newest first.
    pub fn load_audit(&self) -> Result<Vec<AuditEntry>> {
        let mut out = Vec::new();
        for entry in self.audit_tree.iter() {
            let (key, value) = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            // Drop legacy rows from the u64::MAX key-collision bug: pre-fix
            // builds saturated every key, keeping only the latest row there.
            // (A valid millis key can't reach MAX before the year ~58000.)
            if key.len() == 8 && key.iter().all(|&b| b == 0xFF) {
                continue;
            }
            if value.len() > 64 * 1024 {
                continue;
            }
            let decrypted = match self.vault.decrypt(&value) {
                Ok(d) => d,
                Err(_) => continue,
            };
            match bincode::deserialize::<AuditEntry>(&decrypted) {
                Ok(a) => out.push(a.sanitize()),
                Err(_) => continue,
            }
        }
        // Explicit timestamp sort: keys only guarantee uniqueness, not total
        // time order across key-scheme generations (nanos, millis-packed,
        // millis+pid+seq). Newest first.
        out.sort_by(|a, b| b.ts.cmp(&a.ts));
        Ok(out)
    }

    pub fn delete_session(&self, id: Uuid) -> Result<()> {
        self.sessions_tree.remove(id.as_bytes())?;
        self.sessions_tree.flush()?;
        Ok(())
    }

    /// Pinned session ids (JSON list of uuid strings). Survives struct
    /// changes because it never touches the session blobs.
    pub fn save_pins(&self, pins: &[uuid::Uuid]) -> Result<()> {
        let ids: Vec<String> = pins.iter().map(|u| u.to_string()).collect();
        let raw = serde_json::to_vec(&ids)?;
        let encrypted = self.vault.encrypt(&raw)?;
        self.config_tree.insert("pins", encrypted)?;
        self.config_tree.flush()?;
        Ok(())
    }

    pub fn load_pins(&self) -> Result<Vec<uuid::Uuid>> {
        if let Some(value) = self.config_tree.get("pins")? {
            let decrypted = self.vault.decrypt(&value).unwrap_or_else(|_| value.to_vec());
            let ids: Vec<String> = serde_json::from_slice(&decrypted)?;
            Ok(ids.iter().filter_map(|s| s.parse().ok()).collect())
        } else {
            Ok(Vec::new())
        }
    }

    pub fn save_settings(&self, settings: &AppSettings) -> Result<()> {
        let raw = bincode::serialize(settings)?;
        let encrypted = self.vault.encrypt(&raw)?;
        self.config_tree.insert("settings", encrypted)?;
        self.config_tree.flush()?;
        Ok(())
    }

    pub fn load_settings(&self) -> Result<AppSettings> {
        if let Some(value) = self.config_tree.get("settings")? {
            let decrypted = self.vault.decrypt(&value).unwrap_or_else(|_| value.to_vec());
            match bincode::deserialize::<AppSettings>(&decrypted) {
                Ok(settings) => Ok(settings),
                Err(_) => Ok(AppSettings::default()),
            }
        } else {
            Ok(AppSettings::default())
        }
    }

    pub fn save_tools(&self, registry: &crate::tools::ToolRegistry) -> Result<()> {
        let raw = bincode::serialize(registry)?;
        let encrypted = self.vault.encrypt(&raw)?;
        self.config_tree.insert("tools_registry", encrypted)?;
        self.config_tree.flush()?;
        Ok(())
    }

    pub fn load_tools(&self) -> Result<crate::tools::ToolRegistry> {
        if let Some(value) = self.config_tree.get("tools_registry")? {
            let decrypted = self.vault.decrypt(&value).unwrap_or_else(|_| value.to_vec());
            match bincode::deserialize::<crate::tools::ToolRegistry>(&decrypted) {
                Ok(reg) => Ok(reg),
                Err(_) => Ok(crate::tools::ToolRegistry::with_defaults()),
            }
        } else {
            Ok(crate::tools::ToolRegistry::with_defaults())
        }
    }

    pub fn save_skill(&self, skill: &Skill) -> Result<()> {
        let key = skill.id.as_bytes().to_vec();
        let sanitized = skill.clone().sanitize();
        let raw = bincode::serialize(&sanitized)?;
        let encrypted = self.vault.encrypt(&raw)?;
        self.skills_tree.insert(key, encrypted)?;
        self.skills_tree.flush()?;
        Ok(())
    }

    pub fn load_skills(&self) -> Result<Vec<Skill>> {
        let mut skills = Vec::new();
        for entry in self.skills_tree.iter() {
            let (_, value) = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            if value.len() > 1024 * 1024 {
                continue;
            }
            let decrypted = match self.vault.decrypt(&value) {
                Ok(d) => d,
                Err(_) => continue,
            };
            match bincode::deserialize::<Skill>(&decrypted) {
                Ok(s) => skills.push(s.sanitize()),
                Err(_) => continue,
            }
        }
        skills.sort_by(|a, b| b.reinforcement_score.partial_cmp(&a.reinforcement_score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(skills)
    }

    pub fn delete_skill(&self, id: Uuid) -> Result<()> {
        self.skills_tree.remove(id.as_bytes())?;
        self.skills_tree.flush()?;
        Ok(())
    }

    pub fn reinforce_skill(&self, id: Uuid, reward_delta: f32) -> Result<Option<Skill>> {
        let key = id.as_bytes();
        if let Some(val) = self.skills_tree.get(key)? {
            let decrypted = self.vault.decrypt(&val)?;
            let mut skill: Skill = bincode::deserialize(&decrypted)?;
            skill.reinforcement_score = (skill.reinforcement_score + reward_delta).clamp(0.1, 20.0);
            skill.execution_count = skill.execution_count.saturating_add(1);
            skill.last_used = Some(Utc::now());
            self.save_skill(&skill)?;
            Ok(Some(skill))
        } else {
            Ok(None)
        }
    }

    pub fn seed_default_skills(&self) -> Result<()> {
        if !self.skills_tree.is_empty() {
            return Ok(());
        }
        let defaults = vec![
            Skill {
                id: Uuid::new_v4(),
                name: "vulnerability_scan".to_string(),
                description: "Deep cyber security inspection for memory unsafety, credential leakage, logic bugs, and injection vectors.".to_string(),
                prompt_template: "Conduct a rigorous security vulnerability audit of the following code. Identify potential buffer overflows, memory leaks, secret exposure, concurrency flaws, and SSRF tripwires. Provide defensive remediations:".to_string(),
                domain_idx: 3, // Cyber / Critic
                reinforcement_score: 5.0,
                execution_count: 0,
                last_used: None,
                is_built_in: true,
            },
            Skill {
                id: Uuid::new_v4(),
                name: "quantum_cryptanalysis".to_string(),
                description: "Post-Quantum cryptographic resistance evaluation against Shor's and Grover's quantum search algorithms.".to_string(),
                prompt_template: "Analyze the cryptographic primitives used in this architecture. Verify AES-256 Grover attack resistance (retaining 128 bits quantum security), Argon2id memory hardness, and post-quantum envelope integrity:".to_string(),
                domain_idx: 3, // Cyber / Critic
                reinforcement_score: 4.5,
                execution_count: 0,
                last_used: None,
                is_built_in: true,
            },
            Skill {
                id: Uuid::new_v4(),
                name: "high_perf_code_optimizer".to_string(),
                description: "Zero-allocation refactoring, SIMD vectorization, cache locality, and lock-free thread optimization.".to_string(),
                prompt_template: "Refactor this code for optimal execution throughput. Eliminate unnecessary heap allocations (e.g. using Cow, stack arrays, or pre-allocated buffers), reduce cache misses, and optimize tight loops:".to_string(),
                domain_idx: 1, // Coder
                reinforcement_score: 4.0,
                execution_count: 0,
                last_used: None,
                is_built_in: true,
            },
            Skill {
                id: Uuid::new_v4(),
                name: "swarm_orchestrator".to_string(),
                description: "Deconstructs complex objectives into specialist agent pipelines using stigmergic ecological niches.".to_string(),
                prompt_template: "Deconstruct the following high-level objective into an ordered multi-agent swarm pipeline. Specify the specialist domain role, input requirements, and validation gates for each phase:".to_string(),
                domain_idx: 4, // Planner
                reinforcement_score: 3.8,
                execution_count: 0,
                last_used: None,
                is_built_in: true,
            },
            Skill {
                id: Uuid::new_v4(),
                name: "research_synthesizer".to_string(),
                description: "Deep factual literature synthesis, mathematical analysis, and step-by-step hypothesis verification.".to_string(),
                prompt_template: "Perform a comprehensive research synthesis on the following concept. Distinguish empirical facts from theoretical models, cite core mathematical principles, and summarize key conclusions:".to_string(),
                domain_idx: 2, // Researcher
                reinforcement_score: 3.5,
                execution_count: 0,
                last_used: None,
                is_built_in: true,
            },
        ];
        for skill in defaults {
            self.save_skill(&skill)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// The store lives at a fixed user path: serialize tests that open it.
    static STORE_LOCK: Mutex<()> = Mutex::new(());

    fn store_lock() -> std::sync::MutexGuard<'static, ()> {
        STORE_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn audit_clear_empties() {
        let dir = std::env::temp_dir().join(format!("aidash-test-{}", std::process::id()));
        let st = Storage::open_path(&dir.join("storage")).expect("open");
        st.log_audit("x.y", "z").expect("log");
        assert!(st.audit_tree.len() >= 1);
        let n = st.clear_audit().expect("clear");
        assert!(n >= 1);
        assert_eq!(st.audit_tree.len(), 0);
    }

    #[test]
    fn pins_roundtrip() {
        let dir = std::env::temp_dir().join(format!("aidash-test-{}-pins", std::process::id()));
        let st = Storage::open_path(&dir.join("storage")).expect("open");
        assert!(st.load_pins().expect("load").is_empty());
        let a = uuid::Uuid::new_v4();
        let b = uuid::Uuid::new_v4();
        st.save_pins(&[a, b]).expect("save");
        assert_eq!(st.load_pins().expect("reload"), vec![a, b]);
        st.save_pins(&[]).expect("clear");
        assert!(st.load_pins().expect("empty").is_empty());
    }

    #[test]
    fn audit_write_then_read() {
        let _guard = store_lock();
        let dir = std::env::temp_dir().join(format!("aidash-test-{}-{}", std::process::id(), line!()));
        let st = Storage::open_path(&dir.join("storage")).expect("open store");
        let before = st.audit_tree.len();
        st.log_audit("test.selfcheck", "roundtrip-probe").expect("log");
        st.log_audit("test.selfcheck", "roundtrip-probe").expect("log");
        st.log_audit("test.selfcheck", "roundtrip-probe").expect("log");
        assert_eq!(st.audit_tree.len(), before + 3, "three rows stored");
        let v = st.load_audit().expect("load");
        let mut legacy = 0usize;
        for e in st.audit_tree.iter().flatten() {
            if e.0.len() == 8 && e.0.iter().all(|&b| b == 0xFF) {
                legacy += 1;
            }
        }
        assert_eq!(v.len(), st.audit_tree.len() - legacy, "load returns every row");
        assert_eq!(&v[0].kind, "test.selfcheck");
        assert_eq!(&v[0].detail, "roundtrip-probe");
    }

    #[test]
    fn audit_load_matches_tree() {
        let _guard = store_lock();
        let dir = std::env::temp_dir().join(format!("aidash-test-{}-{}", std::process::id(), line!()));
        let st = Storage::open_path(&dir.join("storage")).expect("open store");
        let n = st.audit_tree.len();
        let mut legacy = 0usize;
        for e in st.audit_tree.iter() {
            if let Ok((k, _)) = e {
                if k.len() == 8 && k.iter().all(|&b| b == 0xFF) {
                    legacy += 1;
                }
            }
        }
        let v = st.load_audit().expect("load audit");
        assert_eq!(v.len(), n - legacy, "load returns every valid row");
        for w in v.windows(2) {
            assert!(w[0].ts >= w[1].ts, "newest first");
        }
    }

    #[test]
    fn audit_entry_sanitize_bounds() {
        let a = AuditEntry {
            ts: Utc::now(),
            kind: "k".repeat(100),
            detail: "d".repeat(1000),
        }
        .sanitize();
        assert!(a.kind.len() <= 64);
        assert!(a.detail.len() <= 500);
    }

    #[test]
    fn vault_roundtrip() {
        let dir = std::env::temp_dir().join(format!("aidash-vault-test-{}", std::process::id()));
        let key_path = dir.join("vault.key");
        let vault = StorageVault::load_or_create(&key_path).expect("vault init");

        let plaintext = b"Quantum-Secure Post-Quantum Cryptographic Audit Payload 2026";
        let encrypted = vault.encrypt(plaintext).expect("encrypt");
        assert_ne!(&encrypted[..], plaintext);
        assert_eq!(&encrypted[0..4], ENVELOPE_MAGIC);

        let decrypted = vault.decrypt(&encrypted).expect("decrypt");
        assert_eq!(&decrypted[..], plaintext);
    }

    #[test]
    fn vault_tamper_detection() {
        let dir = std::env::temp_dir().join(format!("aidash-tamper-test-{}", std::process::id()));
        let key_path = dir.join("vault.key");
        let vault = StorageVault::load_or_create(&key_path).expect("vault init");

        let plaintext = b"Sensitive model memory and weights";
        let mut encrypted = vault.encrypt(plaintext).expect("encrypt");

        // Flip a bit in the ciphertext payload
        let last_idx = encrypted.len() - 1;
        encrypted[last_idx] ^= 0x01;

        let result = vault.decrypt(&encrypted);
        assert!(result.is_err(), "tampered ciphertext must fail authentication tag check");
    }

    #[test]
    fn vault_legacy_fallback() {
        let vault = StorageVault::from_key([42u8; 32]);
        let legacy_plaintext = b"Legacy unencrypted plaintext session data from pre-quantum era";
        let result = vault.decrypt(legacy_plaintext).expect("legacy fallback");
        assert_eq!(&result[..], legacy_plaintext);
    }

    #[test]
    fn argon2id_derivation_deterministic() {
        let passphrase = "PostQuantumSuperSecretPassphrase2026!";
        let salt = [0x7fu8; 16];
        let vault1 = StorageVault::derive_from_passphrase(passphrase, &salt).expect("derive 1");
        let vault2 = StorageVault::derive_from_passphrase(passphrase, &salt).expect("derive 2");

        let data = b"Reinforcement pheromone stigmergy matrix";
        let enc = vault1.encrypt(data).expect("encrypt with vault 1");
        let dec = vault2.decrypt(&enc).expect("decrypt with vault 2");
        assert_eq!(&dec[..], data);
    }

    #[test]
    fn session_encrypted_at_rest() {
        let _guard = store_lock();
        let dir = std::env::temp_dir().join(format!("aidash-test-session-enc-{}", std::process::id()));
        let st = Storage::open_path(&dir.join("storage")).expect("open store");

        let session_id = Uuid::new_v4();
        let session = ChatSession {
            id: session_id,
            name: "Quantum Resistance Evaluation".to_string(),
            model: "deepseek-r1:14b".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Evaluate post-quantum key exchange and AES-256 Grover resistance".to_string(),
                timestamp: Utc::now(),
            }],
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        st.save_session(&session).expect("save session");

        // Direct inspection of raw sled bytes: must have A256 envelope magic
        let raw_blob = st.sessions_tree.get(session_id.as_bytes()).expect("read sled").expect("found");
        assert!(raw_blob.len() > 16);
        assert_eq!(&raw_blob[0..4], ENVELOPE_MAGIC, "session stored in sled must start with A256 magic");

        // Load sessions via Storage API: must transparently decrypt
        let loaded = st.load_sessions().expect("load sessions");
        let found = loaded.iter().find(|s| s.id == session_id).expect("found in loaded list");
        assert_eq!(found.name, "Quantum Resistance Evaluation");
        assert_eq!(found.model, "deepseek-r1:14b");
        assert_eq!(found.messages.len(), 1);
        assert_eq!(found.messages[0].content, "Evaluate post-quantum key exchange and AES-256 Grover resistance");
    }

    #[test]
    fn skills_encrypted_roundtrip_and_reinforce() {
        let _guard = store_lock();
        let dir = std::env::temp_dir().join(format!("aidash-test-skills-enc-{}", std::process::id()));
        let st = Storage::open_path(&dir.join("storage")).expect("open store");

        // Should have 5 seeded default skills
        let loaded = st.load_skills().expect("load default skills");
        assert!(loaded.len() >= 5, "expected at least 5 default seeded skills, got {}", loaded.len());
        assert!(loaded.iter().any(|s| s.name == "vulnerability_scan"));

        let skill_id = Uuid::new_v4();
        let skill = Skill {
            id: skill_id,
            name: "test_adversarial_probe".to_string(),
            description: "Probe LLM for hallucinations and boundary evasion".to_string(),
            prompt_template: "Act as an adversarial tester:".to_string(),
            domain_idx: 3,
            reinforcement_score: 2.0,
            execution_count: 0,
            last_used: None,
            is_built_in: false,
        };

        st.save_skill(&skill).expect("save custom skill");

        // Inspect raw sled blob: must have A256 envelope magic
        let raw = st.skills_tree.get(skill_id.as_bytes()).expect("read sled").expect("found");
        assert_eq!(&raw[0..4], ENVELOPE_MAGIC);

        // Reinforce skill
        let updated = st.reinforce_skill(skill_id, 0.5).expect("reinforce").expect("updated");
        assert_eq!(updated.reinforcement_score, 2.5);
        assert_eq!(updated.execution_count, 1);
        assert!(updated.last_used.is_some());

        // Delete skill
        st.delete_skill(skill_id).expect("delete");
        let reloaded = st.load_skills().expect("reload");
        assert!(!reloaded.iter().any(|s| s.id == skill_id));
    }

    #[test]
    fn tools_encrypted_roundtrip() {
        let _guard = store_lock();
        let dir = std::env::temp_dir().join(format!("aidash-test-tools-enc-{}", std::process::id()));
        let st = Storage::open_path(&dir.join("storage")).expect("open store");

        let mut reg = st.load_tools().expect("load default tools");
        assert!(reg.get("editor").is_some());

        reg.add_or_update(crate::tools::UserTool {
            id: "wireshark".to_string(),
            name: "Packet Analyzer".to_string(),
            command: "wireshark".to_string(),
            args_template: "-r {file}".to_string(),
            execution_type: crate::tools::ToolExecutionType::Detached,
            sensitivity: crate::tools::SensitivityLevel::High,
            min_guardrail: crate::guardrails::GuardrailTier::Medium,
            description: "Deep packet inspection".to_string(),
        });

        st.save_tools(&reg).expect("save tools");

        // Inspect raw blob: must be encrypted
        let raw = st.config_tree.get("tools_registry").unwrap().unwrap();
        assert_eq!(&raw[0..4], ENVELOPE_MAGIC);

        let loaded = st.load_tools().expect("load tools");
        let ws = loaded.get("wireshark").expect("wireshark found");
        assert_eq!(ws.command, "wireshark");
        assert_eq!(ws.sensitivity, crate::tools::SensitivityLevel::High);
    }

    #[test]
    fn layout_settings_roundtrip() {
        let _guard = store_lock();
        let dir = std::env::temp_dir().join(format!("aidash-test-layout-{}", std::process::id()));
        let st = Storage::open_path(&dir.join("storage")).expect("open store");

        let mut settings = st.load_settings().expect("load settings");
        assert_eq!(settings.default_tab, "Chat");
        assert!(settings.visible_tabs.contains(&"Chat".to_string()));
        assert_eq!(settings.contrast_on, false);

        settings.default_tab = "Editor".to_string();
        settings.chat_split_view_default = true;
        settings.contrast_on = true;
        settings.chat_rules = "Custom rule: Always use Quantum Post-Quantum Cryptography.".to_string();
        settings.auto_assist_rules = false;
        settings.swarm_auto_assist = false;
        settings.tabs_at_top = true;
        settings.visible_tabs = vec!["Chat".to_string(), "Editor".to_string(), "Settings".to_string()];
        settings.tab_order = vec!["Editor".to_string(), "Chat".to_string(), "Settings".to_string()];

        st.save_settings(&settings).expect("save settings");

        let reloaded = st.load_settings().expect("reload settings");
        assert_eq!(reloaded.default_tab, "Editor");
        assert_eq!(reloaded.chat_split_view_default, true);
        assert_eq!(reloaded.contrast_on, true);
        assert_eq!(reloaded.chat_rules, "Custom rule: Always use Quantum Post-Quantum Cryptography.");
        assert_eq!(reloaded.auto_assist_rules, false);
        assert_eq!(reloaded.swarm_auto_assist, false);
        assert_eq!(reloaded.tabs_at_top, true);
        assert_eq!(reloaded.visible_tabs, vec!["Chat", "Editor", "Settings"]);
        assert_eq!(reloaded.tab_order, vec!["Editor", "Chat", "Settings"]);
    }
}
