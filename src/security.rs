// Copyright 2026 Sean M. Stow. All rights reserved.
//! Outbound secrets guard: everything the user (or a pasted blob) sends to a
//! model is scanned for credential-shaped strings first. Local models don\'t
//! exfiltrate, but chats persist to disk and get copied around — a leaked key
//! in history is still a leak. Patterns + a Shannon-entropy tripwire.

use std::time::{Duration, Instant};

/// One finding. `preview` is redacted (first 3 + last 2 chars max).
#[derive(Debug, Clone, PartialEq)]
pub struct SecretHit {
    pub kind: &'static str,
    pub preview: String,
}

fn redact(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= 8 {
        return "•••…•••".to_string();
    }
    format!(
        "{}{}{}…{}{}",
        chars[0], chars[1], chars[2],
        chars[chars.len() - 2],
        chars[chars.len() - 1]
    )
}

/// Byte Shannon entropy. Random base64 sits ~5.5-6; English ~4.0-4.5.
fn entropy(s: &str) -> f64 {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return 0.0;
    }
    let mut counts = [0u64; 256];
    for &b in bytes {
        counts[b as usize] += 1;
    }
    let n = bytes.len() as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum()
}

/// Token-ish runs worth entropy-checking.
fn token_runs(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '=' | '_' | '-' | '.')))
        .filter(|w| w.len() >= 32)
        .map(|w| w.to_string())
        .collect()
}

/// Literal credential patterns: (kind, marker or prefix rule).
/// Markers matched case-sensitively except where noted.
fn pattern_hits(text: &str) -> Vec<SecretHit> {
    let mut out = Vec::new();
    // (kind, needle): report the token containing the needle, or the needle.
    // Lowercase needles: matched case-insensitively (past tense blobs get
    // lowercased by tooling; real prefixes are rarely valid English).
    let needles: &[(&str, &str)] = &[
        ("aws_access_key", "akia"),
        ("github_token", "ghp_"),
        ("github_token", "gho_"),
        ("github_token", "github_pat_"),
        ("openai_key", "sk-"),
        ("anthropic_key", "sk-ant-"),
        ("xai_key", "xai-"),
        ("huggingface_token", "hf_"),
        ("cohere_key", "co-"),
        ("discord_token", "mfa."),
        ("slack_token", "xoxb-"),
        ("slack_token", "xoxp-"),
        ("slack_token", "xoxa-"),
        ("slack_token", "xoxr-"),
        ("slack_token", "xoxs-"),
        ("google_key", "aiza"),
        ("stripe_key", "sk_live_"),
        ("stripe_key", "rk_live_"),
    ];
    // Multi-word needles hit the raw text (case-insensitive): tokenizers
    // split on the space inside "PRIVATE KEY".
    let raw_upper = text.to_uppercase();
    if raw_upper.contains("PRIVATE KEY") || raw_upper.contains("BEGIN OPENSSH") || raw_upper.contains("BEGIN RSA") {
        out.push(SecretHit {
            kind: "private_key",
            preview: "\u{2022}\u{2022}\u{2022}\u{2026}\u{2022}\u{2022}\u{2022}".to_string(),
        });
    }
    // Candidate tokens: runs that could hold a credential.
    let cands: Vec<&str> = text
        .split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '=' | '_' | '-' | '.' | ' ')))
        .flat_map(|chunk| chunk.split_whitespace())
        .filter(|w| w.len() >= 8)
        .collect();
    for (kind, needle) in needles {
        for c in &cands {
            if c.to_ascii_lowercase().contains(needle) {
                out.push(SecretHit {
                    kind,
                    preview: redact(c),
                });
                break;
            }
        }
    }
    // Bearer tokens: `Bearer <20+ chars>`.
    let words: Vec<&str> = text.split_whitespace().collect();
    for w in words.windows(2) {
        if w[0].eq_ignore_ascii_case("bearer") && w[1].len() >= 20 && w[1].len() <= 200 {
            out.push(SecretHit {
                kind: "bearer_token",
                preview: redact(w[1]),
            });
            break;
        }
    }
    out
}

/// Scan text for credential-shaped strings. Empty = clean.
pub fn find_secrets(text: &str) -> Vec<SecretHit> {
    let mut hits = pattern_hits(text);
    // Entropy tripwire for pattern-less tokens (only when no pattern hit:
    // avoids double-reporting the same blob).
    if hits.is_empty() {
        for tok in token_runs(text) {
            if entropy(&tok) >= 4.7 {
                hits.push(SecretHit {
                    kind: "high_entropy_token",
                    preview: redact(&tok),
                });
                break;
            }
        }
    }
    hits
}

/// Resend-to-confirm gate. First send carrying secrets is BLOCKED and arms a
/// 60s window; sending the identical text again inside the window proceeds.
/// One gate per input surface (chat slot, editor).
pub struct ConfirmGate {
    until: Option<Instant>,
    armed_for: Option<String>,
}

impl ConfirmGate {
    pub fn new() -> Self {
        Self {
            until: None,
            armed_for: None,
        }
    }

    /// Ok(()) = send allowed. Err(hits) = blocked, window armed for a resend
    /// of the exact same text.
    pub fn check(&mut self, text: &str) -> Result<(), Vec<SecretHit>> {
        let hits = find_secrets(text);
        if hits.is_empty() {
            return Ok(());
        }
        let now = Instant::now();
        if let (Some(until), Some(armed)) = (self.until, self.armed_for.as_deref()) {
            if now < until && armed == text {
                self.until = None;
                self.armed_for = None;
                return Ok(());
            }
        }
        self.until = Some(now + Duration::from_secs(60));
        self.armed_for = Some(text.to_string());
        Err(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_insensitive_prefix() {
        assert!(find_secrets("akiaiosfodnn7example leaked").iter().any(|h| h.kind == "aws_access_key"));
        assert!(find_secrets("GHP_abcdefghijklmnopqrstu1234567890 here").iter().any(|h| h.kind == "github_token"));
    }

    #[test]
    fn catches_known_prefixes() {
        assert!(find_secrets("key is AKIAIOSFODNN7EXAMPLE ok").iter().any(|h| h.kind == "aws_access_key"));
        assert!(find_secrets("token ghp_abcdefghijklmnopqrstu1234567890").iter().any(|h| h.kind == "github_token"));
        assert!(find_secrets("hf_abcdefghijklmnopqrstuvwxyz12345678").iter().any(|h| h.kind == "huggingface_token"));
        assert!(find_secrets("-----BEGIN OPENSSH PRIVATE KEY-----\nb3Bl...").iter().any(|h| h.kind == "private_key"));
        assert!(find_secrets("sk-abcdefghijklmnopqrstuvwxyZ0123456789abcd").iter().any(|h| h.kind == "openai_key"));
        assert!(find_secrets("-----BEGIN RSA PRIVATE KEY-----\nMIIE...").iter().any(|h| h.kind == "private_key"));
        assert!(find_secrets("Authorization: Bearer abcdefghijklmnopqrstuvwx").iter().any(|h| h.kind == "bearer_token"));
    }

    #[test]
    fn entropy_tripwire() {
        // Random-looking blob, no known prefix.
        let blob = "k7Qm9vX2pL4nR8sT1wY5uI3oP6aZ0xC2vB4nM6qW8eR0tY2uI4oP6aS8dF0";
        assert!(find_secrets(blob).iter().any(|h| h.kind == "high_entropy_token"));
    }

    #[test]
    fn clean_prose_passes() {
        assert!(find_secrets("explain how neural networks learn from data").is_empty());
        assert!(find_secrets("fn main() { println!(\"hi\"); }").is_empty());
        assert!(find_secrets("the meeting is tomorrow at noon").is_empty());
    }

    #[test]
    fn preview_is_redacted() {
        let hits = find_secrets("sk-abcdefghijklmnopqrstuvwxyZ0123456789abcd");
        assert!(!hits.is_empty());
        let p = &hits[0].preview;
        assert!(p.len() < 12, "preview must stay short: {p}");
        assert!(!p.contains("mnop"), "must not leak middle chars");
    }

    #[test]
    fn gate_blocks_once_then_allows_resend() {
        let mut g = ConfirmGate::new();
        let evil = "deploy with AKIAIOSFODNN7EXAMPLE now";
        assert!(g.check(evil).is_err(), "first send blocked");
        assert!(g.check(evil).is_ok(), "identical resend allowed");
        // Window consumed: blocks again.
        assert!(g.check(evil).is_err());
        // Clean text always passes.
        assert!(g.check("hello world").is_ok());
    }

    #[test]
    fn gate_ignores_different_text() {
        let mut g = ConfirmGate::new();
        assert!(g.check("first AKIAIOSFODNN7EXAMPLE").is_err());
        assert!(g.check("other ghp_abcdefghijklmnopqrstu1234567890").is_err());
    }
}
