// Copyright 2026 Sean M. Stow. All rights reserved.
//! Extensible Tool Execution Engine (Bring-Your-Own-Tool / BYOT).
//! Allows developers, programmers, and SOC specialists to register, configure,
//! and dispatch custom CLI utilities, scripts, binary tools, and GUI text editors.

use crate::guardrails::GuardrailTier;
use serde::{Deserialize, Serialize};
use std::process::{Command, Stdio};

/// Execution mode for a registered tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolExecutionType {
    /// Runs asynchronously inside the integrated IDE Terminal Dock, streaming stdout/stderr.
    TerminalDock,
    /// Spawns a detached process (ideal for GUI text editors, IDEs, or visual analyzers).
    Detached,
}

impl ToolExecutionType {
    pub fn label(&self) -> &'static str {
        match self {
            ToolExecutionType::TerminalDock => "Terminal Dock",
            ToolExecutionType::Detached => "Detached (GUI/Background)",
        }
    }

    pub fn short_badge(&self) -> &'static str {
        match self {
            ToolExecutionType::TerminalDock => "CLI",
            ToolExecutionType::Detached => "GUI",
        }
    }
}

/// Operational sensitivity level determining execution flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensitivityLevel {
    /// Lower-sensitivity tools (e.g. text editors, code formatters) support 1-click execution chips.
    Low,
    /// High-sensitivity tools (e.g. scanners, compilers, system binaries) require confirmation.
    High,
}

impl SensitivityLevel {
    pub fn label(&self) -> &'static str {
        match self {
            SensitivityLevel::Low => "Low (1-Click Run)",
            SensitivityLevel::High => "High (Requires Confirmation)",
        }
    }
}

/// A user-registered tool configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserTool {
    pub id: String,
    pub name: String,
    pub command: String,
    pub args_template: String,
    pub execution_type: ToolExecutionType,
    pub sensitivity: SensitivityLevel,
    pub min_guardrail: GuardrailTier,
    pub description: String,
}

impl UserTool {
    /// Formats the final command line by substituting `{file}`, `{workspace}`, and `{input}` variables.
    pub fn format_command_line(
        &self,
        file_path: Option<&str>,
        workspace_root: Option<&str>,
        user_input: Option<&str>,
    ) -> String {
        let mut formatted_args = self.args_template.clone();

        if let Some(f) = file_path {
            formatted_args = formatted_args.replace("{file}", f);
        } else {
            formatted_args = formatted_args.replace("{file}", "");
        }

        if let Some(ws) = workspace_root {
            formatted_args = formatted_args.replace("{workspace}", ws);
        } else {
            formatted_args = formatted_args.replace("{workspace}", ".");
        }

        if let Some(inp) = user_input {
            formatted_args = formatted_args.replace("{input}", inp);
        } else {
            formatted_args = formatted_args.replace("{input}", "");
        }

        let trimmed_args = formatted_args.trim();
        if trimmed_args.is_empty() {
            self.command.clone()
        } else {
            format!("{} {}", self.command, trimmed_args)
        }
    }
}

/// Registry managing all established user tools.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolRegistry {
    pub tools: Vec<UserTool>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    /// Pre-populates clean starter tool configurations for common workflows.
    pub fn with_defaults() -> Self {
        Self {
            tools: vec![
                UserTool {
                    id: "editor".to_string(),
                    name: "System Text Editor".to_string(),
                    command: "gedit".to_string(),
                    args_template: "{file}".to_string(),
                    execution_type: ToolExecutionType::Detached,
                    sensitivity: SensitivityLevel::Low,
                    min_guardrail: GuardrailTier::Heavy,
                    description: "Launches the system graphical text editor on the targeted file.".to_string(),
                },
                UserTool {
                    id: "code".to_string(),
                    name: "VS Code Editor".to_string(),
                    command: "code".to_string(),
                    args_template: "{file}".to_string(),
                    execution_type: ToolExecutionType::Detached,
                    sensitivity: SensitivityLevel::Low,
                    min_guardrail: GuardrailTier::Heavy,
                    description: "Opens targeted file or directory in Visual Studio Code.".to_string(),
                },
                UserTool {
                    id: "hexdump".to_string(),
                    name: "Hexadecimal Inspector".to_string(),
                    command: "xxd".to_string(),
                    args_template: "{file}".to_string(),
                    execution_type: ToolExecutionType::TerminalDock,
                    sensitivity: SensitivityLevel::Low,
                    min_guardrail: GuardrailTier::Heavy,
                    description: "Generates hex dump of binary files or payload data in the terminal dock.".to_string(),
                },
                UserTool {
                    id: "format".to_string(),
                    name: "Code Formatter".to_string(),
                    command: "rustfmt".to_string(),
                    args_template: "{file}".to_string(),
                    execution_type: ToolExecutionType::TerminalDock,
                    sensitivity: SensitivityLevel::Low,
                    min_guardrail: GuardrailTier::Heavy,
                    description: "Formats Rust or source files according to style guidelines.".to_string(),
                },
            ],
        }
    }

    pub fn get(&self, id: &str) -> Option<&UserTool> {
        self.tools.iter().find(|t| t.id.eq_ignore_ascii_case(id))
    }

    pub fn add_or_update(&mut self, tool: UserTool) {
        if let Some(pos) = self.tools.iter().position(|t| t.id.eq_ignore_ascii_case(&tool.id)) {
            self.tools[pos] = tool;
        } else {
            self.tools.push(tool);
        }
    }

    pub fn remove(&mut self, id: &str) -> bool {
        let initial_len = self.tools.len();
        self.tools.retain(|t| !t.id.eq_ignore_ascii_case(id));
        self.tools.len() < initial_len
    }

    /// Spawns a detached tool process (e.g. GUI editor or external visualizer).
    pub fn spawn_detached(
        &self,
        id: &str,
        file_path: Option<&str>,
        workspace_root: Option<&str>,
        user_input: Option<&str>,
    ) -> Result<u32, String> {
        let tool = self.get(id).ok_or_else(|| format!("Tool '{}' not found in registry.", id))?;
        let cmd_line = tool.format_command_line(file_path, workspace_root, user_input);

        let parts: Vec<&str> = cmd_line.split_whitespace().collect();
        if parts.is_empty() {
            return Err("Empty command line.".to_string());
        }

        let mut cmd = Command::new(parts[0]);
        if parts.len() > 1 {
            cmd.args(&parts[1..]);
        }

        if let Some(ws) = workspace_root {
            cmd.current_dir(ws);
        }

        cmd.stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        match cmd.spawn() {
            Ok(child) => Ok(child.id()),
            Err(e) => Err(format!("Failed to spawn detached tool '{}': {}", tool.name, e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_substitution() {
        let tool = UserTool {
            id: "test".to_string(),
            name: "Test Tool".to_string(),
            command: "python3".to_string(),
            args_template: "run.py --file {file} --dir {workspace} --data {input}".to_string(),
            execution_type: ToolExecutionType::TerminalDock,
            sensitivity: SensitivityLevel::Low,
            min_guardrail: GuardrailTier::Heavy,
            description: "A test runner".to_string(),
        };

        let cmd = tool.format_command_line(
            Some("src/main.rs"),
            Some("/home/user/project"),
            Some("payload_123"),
        );
        assert_eq!(
            cmd,
            "python3 run.py --file src/main.rs --dir /home/user/project --data payload_123"
        );
    }

    #[test]
    fn test_registry_add_remove() {
        let mut reg = ToolRegistry::new();
        assert!(reg.tools.is_empty());

        let t = UserTool {
            id: "custom".to_string(),
            name: "Custom Runner".to_string(),
            command: "bash".to_string(),
            args_template: "script.sh".to_string(),
            execution_type: ToolExecutionType::TerminalDock,
            sensitivity: SensitivityLevel::High,
            min_guardrail: GuardrailTier::Medium,
            description: "Runs custom script".to_string(),
        };

        reg.add_or_update(t.clone());
        assert_eq!(reg.tools.len(), 1);
        assert!(reg.get("custom").is_some());

        assert!(reg.remove("custom"));
        assert_eq!(reg.tools.len(), 0);
    }

    #[test]
    fn test_default_presets() {
        let reg = ToolRegistry::with_defaults();
        assert!(reg.get("editor").is_some());
        assert!(reg.get("code").is_some());
        assert!(reg.get("hexdump").is_some());
        assert!(reg.get("format").is_some());
    }
}
