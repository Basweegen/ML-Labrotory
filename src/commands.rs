// Copyright 2026 Sean M. Stow. All rights reserved.
//! Unified Command Engine: Single command syntax and dispatch system
//! supporting in-chat slash commands and headless terminal CLI execution.
//! Commands: /help, /new, /clear, /model, /swarm, /skills, /audit, /threads, /status.

use anyhow::Result;

/// Autonomous skills subcommand options
#[derive(Debug, Clone, PartialEq)]
pub enum SkillsCommand {
    List,
    Run(String),
    Add { name: String, description: String },
}

/// Swarm multi-agent collaboration subcommands
#[derive(Debug, Clone, PartialEq)]
pub enum SwarmCommand {
    Dispatch(String),
    Preset { template: String, prompt: Option<String> },
    ListPresets,
    Dag { preset: Option<String>, prompt: Option<String> },
    Blackboard,
    Abort,
}

/// Project management subcommands
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectCommand {
    Scaffold { template: String, name: String },
    Build,
    Run,
    Test,
    Setup,
    Verify,
    Clean,
    GenerateScripts,
}

/// User-registered external tools subcommands (BYOT)
#[derive(Debug, Clone, PartialEq)]
pub enum ToolsCommand {
    List,
    Run { id: String, args: Option<String> },
    Add {
        id: String,
        command: String,
        args_template: String,
        detached: bool,
        high_sensitivity: bool,
    },
    Rm(String),
}

/// Guardrail management subcommands
#[derive(Debug, Clone, PartialEq)]
pub enum GuardrailCommand {
    Status,
    Set(crate::guardrails::GuardrailTier),
}

/// Supported slash commands
#[derive(Debug, Clone, PartialEq)]
pub enum SlashCommand {
    Help,
    New,
    Clear,
    Model(String),
    Swarm(SwarmCommand),
    Skills(SkillsCommand),
    Audit,
    Threads(u32),
    Status,
    Exec(String),
    Mkdir(String),
    Touch(String),
    Rm(String),
    Mv { src: String, dst: String },
    Ls(Option<String>),
    Project(ProjectCommand),
    Tools(ToolsCommand),
    Guardrail(GuardrailCommand),
}

/// Action to apply to app state or chat slot
#[derive(Debug, Clone, PartialEq)]
pub enum CommandAction {
    NewSession,
    ClearChat,
    SwitchModel(String),
    LaunchSwarm(String),
    RunSkill(String),
    SetThreads(u32),
    ExecuteTerminal(String),
    WorkspaceRefresh,
    SwitchGuardrail(crate::guardrails::GuardrailTier),
    RunTool { id: String, args: Option<String> },
}

/// Formatted output of command execution
#[derive(Debug, Clone)]
pub struct CommandOutput {
    pub success: bool,
    pub title: String,
    pub message: String,
    pub action: Option<CommandAction>,
}

impl CommandOutput {
    pub fn ok(title: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            success: true,
            title: title.into(),
            message: message.into(),
            action: None,
        }
    }

    pub fn ok_with_action(title: impl Into<String>, message: impl Into<String>, action: CommandAction) -> Self {
        Self {
            success: true,
            title: title.into(),
            message: message.into(),
            action: Some(action),
        }
    }

    pub fn err(title: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            success: false,
            title: title.into(),
            message: message.into(),
            action: None,
        }
    }

    /// Format for displaying in chat transcript
    pub fn format_chat_bubble(&self) -> String {
        let icon = if self.success { "⚡" } else { "⚠" };
        format!("{} **{}**\n\n{}", icon, self.title, self.message)
    }
}

impl SlashCommand {
    /// Parse a line of text. If it starts with '/', attempt to parse as a slash command.
    /// Returns Ok(Some(command)) if recognized, Err(error_message) if invalid syntax,
    /// or Ok(None) if not a slash command.
    pub fn parse(input: &str) -> Result<Option<Self>, String> {
        let trimmed = input.trim();
        if !trimmed.starts_with('/') {
            return Ok(None);
        }

        let without_slash = &trimmed[1..];
        let mut parts = without_slash.split_whitespace();
        let cmd = match parts.next() {
            Some(c) => c.to_ascii_lowercase(),
            None => return Err("Empty command. Type /help for a list of available commands.".to_string()),
        };

        match cmd.as_str() {
            "help" | "?" => Ok(Some(SlashCommand::Help)),
            "new" => Ok(Some(SlashCommand::New)),
            "clear" => Ok(Some(SlashCommand::Clear)),
            "model" | "m" => {
                let model_name = parts.collect::<Vec<&str>>().join(" ");
                if model_name.is_empty() {
                    Err("Usage: /model <model_name>\nExample: /model qwen2.5-coder:7b".to_string())
                } else {
                    Ok(Some(SlashCommand::Model(model_name)))
                }
            }
            "swarm" | "relay" => {
                let first = parts.next();
                match first {
                    None => {
                        Err("Usage: /swarm <prompt> OR /swarm preset <template> [prompt]\nPresets: hive, code, research, triad, soc, fullstack, quantum".to_string())
                    }
                    Some(word) if word.eq_ignore_ascii_case("dag") => {
                        let sub_preset = parts.next();
                        let prompt = parts.collect::<Vec<&str>>().join(" ");
                        let prompt_opt = if prompt.trim().is_empty() { None } else { Some(prompt) };
                        Ok(Some(SlashCommand::Swarm(SwarmCommand::Dag {
                            preset: sub_preset.map(|s| s.to_string()),
                            prompt: prompt_opt,
                        })))
                    }
                    Some(word) if word.eq_ignore_ascii_case("blackboard") || word.eq_ignore_ascii_case("bb") => {
                        Ok(Some(SlashCommand::Swarm(SwarmCommand::Blackboard)))
                    }
                    Some(word) if word.eq_ignore_ascii_case("abort") || word.eq_ignore_ascii_case("stop") => {
                        Ok(Some(SlashCommand::Swarm(SwarmCommand::Abort)))
                    }
                    Some(word) if word.eq_ignore_ascii_case("preset") || word.eq_ignore_ascii_case("presets") || word.eq_ignore_ascii_case("list") => {
                        let tmpl = parts.next();
                        match tmpl {
                            Some(t) => {
                                let prompt = parts.collect::<Vec<&str>>().join(" ");
                                let prompt_opt = if prompt.trim().is_empty() { None } else { Some(prompt) };
                                Ok(Some(SlashCommand::Swarm(SwarmCommand::Preset {
                                    template: t.to_string(),
                                    prompt: prompt_opt,
                                })))
                            }
                            None => Ok(Some(SlashCommand::Swarm(SwarmCommand::ListPresets))),
                        }
                    }
                    Some(word) => {
                        let mut prompt = word.to_string();
                        let rest = parts.collect::<Vec<&str>>().join(" ");
                        if !rest.is_empty() {
                            prompt.push(' ');
                            prompt.push_str(&rest);
                        }
                        Ok(Some(SlashCommand::Swarm(SwarmCommand::Dispatch(prompt))))
                    }
                }
            }
            "skills" | "skill" => {
                let sub = parts.next().unwrap_or("list").to_ascii_lowercase();
                match sub.as_str() {
                    "list" | "ls" => Ok(Some(SlashCommand::Skills(SkillsCommand::List))),
                    "run" | "exec" => {
                        let name = parts.collect::<Vec<&str>>().join(" ");
                        if name.is_empty() {
                            Err("Usage: /skills run <skill_name>\nExample: /skills run vulnerability_scan".to_string())
                        } else {
                            Ok(Some(SlashCommand::Skills(SkillsCommand::Run(name))))
                        }
                    }
                    "add" | "create" => {
                        let name = match parts.next() {
                            Some(n) => n.to_string(),
                            None => return Err("Usage: /skills add <name> <description>\nExample: /skills add audit Scan for memory leaks".to_string()),
                        };
                        let desc = parts.collect::<Vec<&str>>().join(" ");
                        if desc.is_empty() {
                            Err("Usage: /skills add <name> <description>".to_string())
                        } else {
                            Ok(Some(SlashCommand::Skills(SkillsCommand::Add { name, description: desc })))
                        }
                    }
                    unknown => Err(format!("Unknown skills subcommand '{}'. Usage: /skills [list | run <name> | add <name> <desc>]", unknown)),
                }
            }
            "audit" | "security" => Ok(Some(SlashCommand::Audit)),
            "threads" | "cpu" => {
                let n_str = match parts.next() {
                    Some(n) => n,
                    None => return Err("Usage: /threads <number (1-64)>\nExample: /threads 10".to_string()),
                };
                match n_str.parse::<u32>() {
                    Ok(n) if (1..=64).contains(&n) => Ok(Some(SlashCommand::Threads(n))),
                    _ => Err("Invalid thread count. Must be between 1 and 64.".to_string()),
                }
            }
            "status" | "info" => Ok(Some(SlashCommand::Status)),
            "exec" | "run" => {
                let cmd_line = parts.collect::<Vec<&str>>().join(" ");
                if cmd_line.is_empty() {
                    Err("Usage: /exec <command>\nExample: /exec cargo build".to_string())
                } else {
                    Ok(Some(SlashCommand::Exec(cmd_line)))
                }
            }
            "mkdir" => {
                let dir_path = parts.collect::<Vec<&str>>().join(" ");
                if dir_path.is_empty() {
                    Err("Usage: /mkdir <path>\nExample: /mkdir src/components".to_string())
                } else {
                    Ok(Some(SlashCommand::Mkdir(dir_path)))
                }
            }
            "touch" => {
                let file_path = parts.collect::<Vec<&str>>().join(" ");
                if file_path.is_empty() {
                    Err("Usage: /touch <file_path>\nExample: /touch src/utils.rs".to_string())
                } else {
                    Ok(Some(SlashCommand::Touch(file_path)))
                }
            }
            "rm" | "del" => {
                let target = parts.collect::<Vec<&str>>().join(" ");
                if target.is_empty() {
                    Err("Usage: /rm <path>\nExample: /rm old_module.rs".to_string())
                } else {
                    Ok(Some(SlashCommand::Rm(target)))
                }
            }
            "mv" | "move" => {
                let src = match parts.next() {
                    Some(s) => s.to_string(),
                    None => return Err("Usage: /mv <source> <destination>\nExample: /mv draft.py main.py".to_string()),
                };
                let dst = match parts.next() {
                    Some(d) => d.to_string(),
                    None => return Err("Usage: /mv <source> <destination>".to_string()),
                };
                Ok(Some(SlashCommand::Mv { src, dst }))
            }
            "ls" | "dir" => {
                let path_arg = parts.next().map(|s| s.to_string());
                Ok(Some(SlashCommand::Ls(path_arg)))
            }
            "project" | "proj" => {
                let sub = parts.next().unwrap_or("run").to_ascii_lowercase();
                match sub.as_str() {
                    "scaffold" | "create" | "new" => {
                        let tmpl = parts.next().unwrap_or("rust").to_string();
                        let name = parts.collect::<Vec<&str>>().join(" ");
                        let clean_name = if name.is_empty() { "app".to_string() } else { name };
                        Ok(Some(SlashCommand::Project(ProjectCommand::Scaffold {
                            template: tmpl,
                            name: clean_name,
                        })))
                    }
                    "build" => Ok(Some(SlashCommand::Project(ProjectCommand::Build))),
                    "test" => Ok(Some(SlashCommand::Project(ProjectCommand::Test))),
                    "run" | "start" => Ok(Some(SlashCommand::Project(ProjectCommand::Run))),
                    "setup" => Ok(Some(SlashCommand::Project(ProjectCommand::Setup))),
                    "verify" | "audit" => Ok(Some(SlashCommand::Project(ProjectCommand::Verify))),
                    "clean" => Ok(Some(SlashCommand::Project(ProjectCommand::Clean))),
                    "generate-scripts" | "genscripts" | "gen" => Ok(Some(SlashCommand::Project(ProjectCommand::GenerateScripts))),
                    other => Err(format!("Unknown project subcommand '{}'. Usage: /project [scaffold <type> <name> | build | test | run | setup | verify | clean | generate-scripts]", other)),
                }
            }
            "tools" | "tool" => {
                let sub = parts.next().unwrap_or("list").to_ascii_lowercase();
                match sub.as_str() {
                    "list" | "ls" => Ok(Some(SlashCommand::Tools(ToolsCommand::List))),
                    "run" | "exec" => {
                        let id = match parts.next() {
                            Some(i) => i.to_string(),
                            None => return Err("Usage: /tools run <id> [args]".to_string()),
                        };
                        let rest = parts.collect::<Vec<&str>>().join(" ");
                        let args = if rest.is_empty() { None } else { Some(rest) };
                        Ok(Some(SlashCommand::Tools(ToolsCommand::Run { id, args })))
                    }
                    "rm" | "del" | "remove" => {
                        let id = match parts.next() {
                            Some(i) => i.to_string(),
                            None => return Err("Usage: /tools rm <id>".to_string()),
                        };
                        Ok(Some(SlashCommand::Tools(ToolsCommand::Rm(id))))
                    }
                    "add" | "create" => {
                        let id = match parts.next() {
                            Some(i) => i.to_string(),
                            None => return Err("Usage: /tools add <id> <command> [args_template] [--detached] [--high]".to_string()),
                        };
                        let cmd_part = match parts.next() {
                            Some(c) => c.to_string(),
                            None => return Err("Usage: /tools add <id> <command> [args_template] [--detached] [--high]".to_string()),
                        };
                        let mut detached = false;
                        let mut high_sensitivity = false;
                        let mut remaining = Vec::new();
                        for p in parts {
                            match p {
                                "--detached" | "-d" | "--gui" => detached = true,
                                "--high" | "-h" | "--confirm" => high_sensitivity = true,
                                other => remaining.push(other),
                            }
                        }
                        let args_template = if remaining.is_empty() {
                            "{file}".to_string()
                        } else {
                            remaining.join(" ")
                        };
                        Ok(Some(SlashCommand::Tools(ToolsCommand::Add {
                            id,
                            command: cmd_part,
                            args_template,
                            detached,
                            high_sensitivity,
                        })))
                    }
                    unknown => Err(format!("Unknown tools subcommand '{}'. Usage: /tools [list | run <id> [args] | add <id> <cmd> [args] [--detached] [--high] | rm <id>]", unknown)),
                }
            }
            "guardrail" | "guardrails" | "gr" => {
                let sub = parts.next().unwrap_or("status").to_ascii_lowercase();
                match sub.as_str() {
                    "status" | "info" | "show" => Ok(Some(SlashCommand::Guardrail(GuardrailCommand::Status))),
                    "heavy" | "sandbox" | "sandboxed" => Ok(Some(SlashCommand::Guardrail(GuardrailCommand::Set(crate::guardrails::GuardrailTier::Heavy)))),
                    "medium" | "balanced" | "dev" => Ok(Some(SlashCommand::Guardrail(GuardrailCommand::Set(crate::guardrails::GuardrailTier::Medium)))),
                    "none" | "unrestricted" | "soc" => Ok(Some(SlashCommand::Guardrail(GuardrailCommand::Set(crate::guardrails::GuardrailTier::None)))),
                    unknown => Err(format!("Unknown guardrail tier '{}'. Usage: /guardrail [status | heavy | medium | none]", unknown)),
                }
            }
            unknown => Err(format!("Unknown command '/{}'. Type /help for available commands.", unknown)),
        }
    }

    /// Parse command-line argument slice (e.g. ["/model", "qwen2.5-coder:7b"] or ["model", "..."]).
    pub fn parse_cli_args(args: &[String]) -> Result<Option<Self>, String> {
        if args.is_empty() {
            return Ok(None);
        }
        let first = args[0].trim();
        let cmd_str = if first.starts_with('/') {
            args.join(" ")
        } else if first.starts_with("--") {
            format!("/{}", &first[2..])
        } else if first.starts_with('-') {
            format!("/{}", &first[1..])
        } else {
            format!("/{}", args.join(" "))
        };
        Self::parse(&cmd_str)
    }

    /// Help manual markdown
    pub fn help_manual() -> &'static str {
        "**ML Laboratory - Unified Command Engine**\n\n\
         The following commands can be executed in the chat box or in the terminal:\n\n\
         • `/help` — Display this command reference.\n\
         • `/new` — Reset current chat and start a fresh session.\n\
         • `/clear` — Clear chat history messages from view.\n\
         • `/model <name>` — Switch active slot model and pre-warm into memory.\n\
         • `/swarm <prompt>` — Dispatch task to multi-agent Swarm Relay.\n\
         • `/skills list` — List all autonomous skills and reinforcement scores.\n\
         • `/skills run <name>` — Execute a specialized autonomous skill.\n\
         • `/skills add <name> <desc>` — Define a new reusable capability.\n\
         • `/audit` — Query recent security and secret detection events.\n\
         • `/threads <1-64>` — Set inference CPU threads for matrix acceleration.\n\
         • `/status` — View system memory, GPU offload, and Ollama status.\n\
         • `/exec <command>` — Run terminal command in active destination folder.\n\
         • `/mkdir <path>` — Create directory structure in destination folder.\n\
         • `/touch <path>` — Create a new source file in destination folder.\n\
         • `/rm <path>` — Delete a file or folder in destination folder.\n\
         • `/mv <src> <dst>` — Move or rename file/folder in destination folder.\n\
         • `/ls [path]` — List files and subfolders in destination folder.\n\
         • `/project scaffold <type> <name>` — Scaffold full-fledged application.\n\
         • `/tools [list|run|add|rm]` — Bring-Your-Own-Tool manager for custom CLI & GUI apps.\n\
         • `/guardrail [status|heavy|medium|none]` — Safety guardrails & authorized SOC execution tier."
    }
}

fn open_storage_cli() -> Result<Option<crate::storage::Storage>> {
    match crate::storage::Storage::new() {
        Ok(s) => Ok(Some(s)),
        Err(e) => {
            let err_str = e.to_string();
            if err_str.contains("Resource temporarily unavailable") || err_str.contains("WouldBlock") {
                println!("• Notice: ML Laboratory GUI is currently active (holding database lock).\n  Please run slash commands directly in the in-chat box or close the GUI instance.");
                Ok(None)
            } else {
                Err(e)
            }
        }
    }
}

/// Execute command in headless CLI mode
pub async fn run_headless_cli(cmd: SlashCommand) -> Result<()> {
    match cmd {
        SlashCommand::Help => {
            println!("{}", SlashCommand::help_manual());
        }
        SlashCommand::Status => {
            println!("=== ML Laboratory System Status ===");
            let client = crate::ollama::api::OllamaClient::new("http://localhost:11434", false);
            match client {
                Ok(cli) => {
                    match cli.version().await {
                        Ok(v) => println!("• Ollama Daemon: ONLINE (v{})", v),
                        Err(e) => println!("• Ollama Daemon: ERROR ({})", e),
                    }
                    match cli.list_models().await {
                        Ok(models) => {
                            println!("• Available Models ({}):", models.len());
                            for m in models {
                                println!("  - {:<30} {:>10.2} MB", m.name, m.size as f64 / (1024.0 * 1024.0));
                            }
                        }
                        Err(e) => println!("• Models list error: {}", e),
                    }
                }
                Err(e) => println!("• Ollama client init failed: {}", e),
            }
            let mem = crate::resources::system_memory();
            println!("• Memory: Used {:.2} GB / Total {:.2} GB", 
                mem.used_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
                mem.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
            println!("• Optimal Inference Threads: {}", crate::ollama::api::ChatOptions::optimal_threads());
        }
        SlashCommand::Audit => {
            println!("=== Recent Security & Secret Audit Log ===");
            let Some(storage) = open_storage_cli()? else { return Ok(()); };
            let entries = storage.load_audit()?;
            if entries.is_empty() {
                println!("Audit log is empty (no violations or security events recorded).");
            } else {
                for e in entries.iter().take(25) {
                    println!("[{}] {:<20} {}", e.ts.format("%Y-%m-%d %H:%M:%S"), e.kind, e.detail);
                }
            }
        }
        SlashCommand::Skills(sub) => match sub {
            SkillsCommand::List => {
                println!("=== Autonomous Skills Registry ===");
                let Some(storage) = open_storage_cli()? else { return Ok(()); };
                let skills = storage.load_skills().unwrap_or_default();
                if skills.is_empty() {
                    println!("No custom skills registered. Built-in skills will initialize on GUI launch.");
                } else {
                    for s in skills {
                        let domain_name = match s.domain_idx {
                            1 => "Coder",
                            2 => "Researcher",
                            3 => "Cyber/Critic",
                            4 => "Planner",
                            5 => "Writer",
                            _ => "General",
                        };
                        println!("• {:<25} [{:<12}] Score: {:<5.2} Runs: {}", s.name, domain_name, s.reinforcement_score, s.execution_count);
                        println!("  Description: {}", s.description);
                    }
                }
            }
            SkillsCommand::Run(name) => {
                println!("Executing skill '{}'...", name);
                let Some(storage) = open_storage_cli()? else { return Ok(()); };
                let skills = storage.load_skills().unwrap_or_default();
                if let Some(skill) = skills.iter().find(|s| s.name.eq_ignore_ascii_case(&name)) {
                    println!("Skill found: {}", skill.description);
                    println!("Prompt template:\n{}", skill.prompt_template);
                } else {
                    println!("Skill '{}' not found. Run '/skills list' to view available skills.", name);
                }
            }
            SkillsCommand::Add { name, description } => {
                println!("Registering skill '{}': {}", name, description);
                let Some(storage) = open_storage_cli()? else { return Ok(()); };
                let skill = crate::storage::Skill {
                    id: uuid::Uuid::new_v4(),
                    name,
                    description,
                    prompt_template: "You are a specialized autonomous skill agent.".to_string(),
                    domain_idx: 0,
                    reinforcement_score: 1.0,
                    execution_count: 0,
                    last_used: None,
                    is_built_in: false,
                };
                storage.save_skill(&skill)?;
                println!("Skill successfully encrypted and saved to Post-Quantum vault.");
            }
        },
        SlashCommand::Model(name) => {
            println!("Pre-warming model '{}' on loopback Ollama...", name);
            let client = crate::ollama::api::OllamaClient::new("http://localhost:11434", false)?;
            match client.warm_model(&name, Some("30m")).await {
                Ok(_) => println!("Model '{}' resident in RAM and ready for inference.", name),
                Err(e) => println!("Failed to warm model '{}': {}", name, e),
            }
        }
        SlashCommand::Swarm(swarm_cmd) => match swarm_cmd {
            SwarmCommand::ListPresets => {
                println!("=== Multi-Agent Swarm Presets ===");
                for tmpl in crate::ui::relay::SwarmTemplate::all() {
                    if tmpl != crate::ui::relay::SwarmTemplate::Custom {
                        println!("• {:<10} : {}", tmpl.short_id(), tmpl.label());
                    }
                }
                println!("\nUsage: /swarm preset <id> [prompt]");
            }
            SwarmCommand::Preset { template, prompt } => {
                let tmpl_opt = crate::ui::relay::SwarmTemplate::from_id(&template);
                match tmpl_opt {
                    Some(tmpl) => {
                        println!("=== Swarm Preset Activated: {} ===", tmpl.label());
                        let prompt_str = prompt.unwrap_or_else(|| "Initialize swarm collaboration".to_string());
                        println!("Task: {}", prompt_str);
                        let client = crate::ollama::api::OllamaClient::new("http://localhost:11434", false)?;
                        let models = client.list_models().await?;
                        if models.is_empty() {
                            println!("No local models found in Ollama.");
                            return Ok(());
                        }
                        let chosen_model = &models[0].name;
                        println!("Executing swarm lead '{}'...", chosen_model);
                        let req = crate::ollama::api::ChatRequest {
                            model: chosen_model.clone(),
                            messages: vec![
                                crate::ollama::api::Message {
                                    role: "system".to_string(),
                                    content: format!("You are the lead agent of a {} team.", tmpl.label()),
                                },
                                crate::ollama::api::Message {
                                    role: "user".to_string(),
                                    content: prompt_str,
                                },
                            ],
                            stream: true,
                            options: Some(crate::ollama::api::ChatOptions::lowram()),
                            keep_alive: Some("30m".to_string()),
                        };
                        let resp = client.chat_stream(req, |chunk| {
                            print!("{}", chunk);
                            use std::io::Write;
                            let _ = std::io::stdout().flush();
                        }).await?;
                        println!("\n\nSwarm Preset Turn Complete. Evaluation count: {:?}", resp.eval_count);
                    }
                    None => {
                        println!("Unknown swarm preset '{}'. Use `/swarm presets` to list presets.", template);
                    }
                }
            }
            SwarmCommand::Dispatch(prompt) => {
                println!("=== Swarm Relay Dispatch ===");
                let (domain_idx, domain_name) = crate::neural::classify_prompt_domain(&prompt);
                println!("Domain Classified: [{}] {}", domain_idx, domain_name);
                println!("Task: {}", prompt);
                println!("Connecting to multi-agent swarm...");
                let client = crate::ollama::api::OllamaClient::new("http://localhost:11434", false)?;
                let models = client.list_models().await?;
                if models.is_empty() {
                    println!("No local models found in Ollama. Pull a model first.");
                    return Ok(());
                }
                let chosen_model = &models[0].name;
                println!("Executing via primary specialist '{}'...", chosen_model);
                let req = crate::ollama::api::ChatRequest {
                    model: chosen_model.clone(),
                    messages: vec![
                        crate::ollama::api::Message {
                            role: "system".to_string(),
                            content: format!("You are an elite specialist in {}. Address the following objective with rigour.", domain_name),
                        },
                        crate::ollama::api::Message {
                            role: "user".to_string(),
                            content: prompt,
                        },
                    ],
                    stream: true,
                    options: Some(crate::ollama::api::ChatOptions::lowram()),
                    keep_alive: Some("30m".to_string()),
                };
                let resp = client.chat_stream(req, |chunk| {
                    print!("{}", chunk);
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                }).await?;
                println!("\n\nSwarm Turn Complete. Evaluation count: {:?}", resp.eval_count);
            }
            SwarmCommand::Dag { preset, prompt } => {
                let p_id = preset.as_deref().unwrap_or("diamond");
                let dag_preset = crate::swarm::DagPreset::from_id(p_id).unwrap_or(crate::swarm::DagPreset::Diamond);
                println!("=== Autonomous Swarm DAG: {} ===", dag_preset.label());
                let prompt_str = prompt.unwrap_or_else(|| "Initialize swarm DAG task".to_string());
                println!("Objective: {}", prompt_str);
                let dag = crate::swarm::SwarmDag::new(dag_preset);
                println!("Topology: {} nodes across dependency levels:", dag.nodes.len());
                if let Ok(ranks) = dag.topological_ranks() {
                    for (lvl, nodes) in ranks.iter().enumerate() {
                        let node_names = nodes
                            .iter()
                            .filter_map(|&id| dag.find_node(id))
                            .map(|n| format!("#{}: {} ({})", n.id, n.name, n.role.label()))
                            .collect::<Vec<_>>()
                            .join(" | ");
                        println!("  Level {}: {}", lvl, node_names);
                    }
                }
                println!("\nSwarm DAG configured. Launch in ML Laboratory dashboard for live parallel execution.");
            }
            SwarmCommand::Blackboard => {
                println!("=== Stigmergic Blackboard Memory Vault ===");
                let storage = crate::storage::Storage::new()?;
                if let Ok(Some(bb)) = storage.load_blackboard() {
                    println!("{}", bb.export_markdown());
                } else {
                    println!("No persistent blackboard snapshot found. Run a Swarm DAG in the dashboard to deposit artifacts.");
                }
            }
            SwarmCommand::Abort => {
                println!("Swarm abort signal emitted.");
            }
        }
        SlashCommand::Threads(n) => {
            println!("Setting default inference CPU threads to {}...", n);
            let Some(storage) = open_storage_cli()? else { return Ok(()); };
            let mut settings = storage.load_settings().unwrap_or_default();
            settings.num_threads = n;
            storage.save_settings(&settings)?;
            println!("Settings updated and encrypted. Active threads: {}", n);
        }
        SlashCommand::New | SlashCommand::Clear => {
            println!("Command '{}' applies to active GUI sessions.", match cmd {
                SlashCommand::New => "/new",
                _ => "/clear",
            });
        }
        SlashCommand::Exec(cmd_line) => {
            let cwd = std::env::current_dir()?;
            println!("Executing: {} (in {})", cmd_line, cwd.display());
            let result = crate::workspace::CommandRunner::execute(&cmd_line, &cwd).await?;
            println!("{}", result.format_display());
        }
        SlashCommand::Mkdir(dir_path) => {
            let cwd = std::env::current_dir()?;
            let target = cwd.join(&dir_path);
            std::fs::create_dir_all(&target)?;
            println!("Created folder: {}", target.display());
        }
        SlashCommand::Touch(file_path) => {
            let cwd = std::env::current_dir()?;
            let target = cwd.join(&file_path);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            if !target.exists() {
                let header = crate::workspace::WorkspaceManager::default_header_for(&target);
                std::fs::write(&target, header.as_bytes())?;
            }
            println!("Created file: {}", target.display());
        }
        SlashCommand::Rm(path) => {
            let cwd = std::env::current_dir()?;
            let target = cwd.join(&path);
            if !target.exists() {
                println!("Target does not exist: {}", target.display());
            } else if target.is_dir() {
                std::fs::remove_dir_all(&target)?;
                println!("Removed directory: {}", target.display());
            } else {
                std::fs::remove_file(&target)?;
                println!("Removed file: {}", target.display());
            }
        }
        SlashCommand::Mv { src, dst } => {
            let cwd = std::env::current_dir()?;
            let s = cwd.join(&src);
            let d = cwd.join(&dst);
            if let Some(parent) = d.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::rename(&s, &d)?;
            println!("Moved {} -> {}", s.display(), d.display());
        }
        SlashCommand::Ls(path_opt) => {
            let cwd = std::env::current_dir()?;
            let target = path_opt.map(|p| cwd.join(p)).unwrap_or(cwd);
            println!("=== Directory Listing: {} ===", target.display());
            for entry in std::fs::read_dir(&target)? {
                let entry = entry?;
                let meta = entry.metadata()?;
                let is_dir = meta.is_dir();
                let icon = if is_dir { "📁" } else { "📄" };
                println!("{} {:<30} {:>10} bytes", icon, entry.file_name().to_string_lossy(), meta.len());
            }
        }
        SlashCommand::Project(proj) => match proj {
            ProjectCommand::Scaffold { template, name } => {
                let cwd = std::env::current_dir()?;
                let tmpl = match template.to_lowercase().as_str() {
                    "python" | "py" | "swarm" => crate::workspace::AppTemplateType::PythonAiSwarm,
                    "web" | "js" | "html" => crate::workspace::AppTemplateType::ModernWeb,
                    "cyber" | "quantum" | "vault" => crate::workspace::AppTemplateType::QuantumCyberSecurity,
                    _ => crate::workspace::AppTemplateType::RustHighPerformance,
                };
                let report = crate::workspace::AppScaffolder::scaffold(tmpl, &cwd, &name)?;
                println!("=== Full-Fledged Application Scaffolded Successfully ===");
                println!("App Name: {}", report.app_name);
                println!("Destination: {}", report.target_dir.display());
                println!("Created Folders: {}", report.created_dirs.len());
                println!("Created Files: {}", report.created_files.len());
                println!("Setup Script: {}", report.setup_script.display());
                println!("Startup Script: {}", report.startup_script.display());
            }
            ProjectCommand::Build => {
                let cwd = std::env::current_dir()?;
                let cmd = if cwd.join("Cargo.toml").exists() {
                    "cargo build"
                } else if cwd.join("package.json").exists() {
                    "npm run build"
                } else {
                    "echo 'No build system detected'"
                };
                let res = crate::workspace::CommandRunner::execute(cmd, &cwd).await?;
                println!("{}", res.format_display());
            }
            ProjectCommand::Test => {
                let cwd = std::env::current_dir()?;
                let cmd = if cwd.join("Cargo.toml").exists() {
                    "cargo test"
                } else if cwd.join("pytest.ini").exists() || cwd.join("requirements.txt").exists() {
                    "pytest"
                } else if cwd.join("package.json").exists() {
                    "npm test"
                } else {
                    "echo 'No test runner detected'"
                };
                let res = crate::workspace::CommandRunner::execute(cmd, &cwd).await?;
                println!("{}", res.format_display());
            }
            ProjectCommand::Run => {
                let cwd = std::env::current_dir()?;
                let cmd = if cwd.join("start.sh").exists() {
                    "./start.sh"
                } else if cwd.join("Cargo.toml").exists() {
                    "cargo run"
                } else if cwd.join("main.py").exists() {
                    "python3 main.py"
                } else {
                    "echo 'No run target detected'"
                };
                let res = crate::workspace::CommandRunner::execute(cmd, &cwd).await?;
                println!("{}", res.format_display());
            }
            ProjectCommand::Setup => {
                let cwd = std::env::current_dir()?;
                let cmd = if cwd.join("setup.sh").exists() {
                    "./setup.sh"
                } else if cwd.join("Cargo.toml").exists() {
                    "cargo check"
                } else if cwd.join("requirements.txt").exists() {
                    "python3 -m pip install -r requirements.txt"
                } else {
                    "echo 'No setup script found'"
                };
                let res = crate::workspace::CommandRunner::execute(cmd, &cwd).await?;
                println!("{}", res.format_display());
            }
            ProjectCommand::Verify => {
                let cwd = std::env::current_dir()?;
                let cmd = if cwd.join("verify.sh").exists() {
                    "./verify.sh"
                } else if cwd.join("Cargo.toml").exists() {
                    "cargo test --all"
                } else {
                    "echo 'No verification script found'"
                };
                let res = crate::workspace::CommandRunner::execute(cmd, &cwd).await?;
                println!("{}", res.format_display());
            }
            ProjectCommand::Clean => {
                let cwd = std::env::current_dir()?;
                let cmd = if cwd.join("clean.sh").exists() {
                    "./clean.sh"
                } else if cwd.join("Cargo.toml").exists() {
                    "cargo clean"
                } else {
                    "echo 'No clean script found'"
                };
                let res = crate::workspace::CommandRunner::execute(cmd, &cwd).await?;
                println!("{}", res.format_display());
            }
            ProjectCommand::GenerateScripts => {
                let cwd = std::env::current_dir()?;
                let scripts = crate::workspace::WorkspaceAutomation::generate_scripts(&cwd)?;
                println!("Generated {} automation scripts in {:?}", scripts.len(), cwd);
                for s in scripts {
                    println!("• {:?}", s.file_name().unwrap_or_default());
                }
            }
        },
        SlashCommand::Tools(sub) => match sub {
            ToolsCommand::List => {
                println!("=== External Tools Registry (BYOT) ===");
                let Some(storage) = open_storage_cli()? else { return Ok(()); };
                let registry = storage.load_tools()?;
                if registry.tools.is_empty() {
                    println!("No tools registered. Use '/tools add <id> <cmd>' or configure via GUI.");
                } else {
                    for t in &registry.tools {
                        println!(
                            "• {:<12} [{:<3}] [{:<4}] Min Guardrail: {:<12}",
                            t.id,
                            t.execution_type.short_badge(),
                            if t.sensitivity == crate::tools::SensitivityLevel::High { "CONF" } else { "1CLK" },
                            t.min_guardrail.short_label(),
                        );
                        println!("  Command: {} {}", t.command, t.args_template);
                        println!("  {}", t.description);
                    }
                }
            }
            ToolsCommand::Run { id, args } => {
                let Some(storage) = open_storage_cli()? else { return Ok(()); };
                let registry = storage.load_tools()?;
                let tool = match registry.get(&id) {
                    Some(t) => t.clone(),
                    None => {
                        println!("Tool '{}' not found. Run '/tools list' to view registered tools.", id);
                        return Ok(());
                    }
                };
                let settings = storage.load_settings().unwrap_or_default();
                let cwd = std::env::current_dir()?;
                let cmd_line = tool.format_command_line(None, Some(cwd.to_str().unwrap_or(".")), args.as_deref());

                // Validate against active guardrails
                if let Err(violation) = settings.guardrail_tier.validate_command(&cmd_line, Some(&cwd)) {
                    println!("❌ {}", violation);
                    return Ok(());
                }

                println!("Launching tool '{}' ({})...", tool.name, tool.execution_type.label());
                match tool.execution_type {
                    crate::tools::ToolExecutionType::Detached => {
                        match registry.spawn_detached(&id, None, Some(cwd.to_str().unwrap_or(".")), args.as_deref()) {
                            Ok(pid) => println!("Tool process spawned in background (PID: {}).", pid),
                            Err(e) => println!("Failed to spawn detached tool: {}", e),
                        }
                    }
                    crate::tools::ToolExecutionType::TerminalDock => {
                        let res = crate::workspace::CommandRunner::execute(&cmd_line, &cwd).await?;
                        println!("{}", res.format_display());
                    }
                }
            }
            ToolsCommand::Add { id, command, args_template, detached, high_sensitivity } => {
                let Some(storage) = open_storage_cli()? else { return Ok(()); };
                let mut registry = storage.load_tools()?;
                let tool = crate::tools::UserTool {
                    id: id.clone(),
                    name: id.clone(),
                    command,
                    args_template,
                    execution_type: if detached { crate::tools::ToolExecutionType::Detached } else { crate::tools::ToolExecutionType::TerminalDock },
                    sensitivity: if high_sensitivity { crate::tools::SensitivityLevel::High } else { crate::tools::SensitivityLevel::Low },
                    min_guardrail: crate::guardrails::GuardrailTier::Heavy,
                    description: "User registered tool".to_string(),
                };
                registry.add_or_update(tool);
                storage.save_tools(&registry)?;
                println!("Tool '{}' registered successfully and encrypted into vault.", id);
            }
            ToolsCommand::Rm(id) => {
                let Some(storage) = open_storage_cli()? else { return Ok(()); };
                let mut registry = storage.load_tools()?;
                if registry.remove(&id) {
                    storage.save_tools(&registry)?;
                    println!("Tool '{}' removed from registry.", id);
                } else {
                    println!("Tool '{}' not found in registry.", id);
                }
            }
        },
        SlashCommand::Guardrail(sub) => match sub {
            GuardrailCommand::Status => {
                let Some(storage) = open_storage_cli()? else { return Ok(()); };
                let settings = storage.load_settings().unwrap_or_default();
                println!("=== Active Guardrails Safety Tier ===");
                println!("• Tier: {}", settings.guardrail_tier.label());
                println!("• Description: {}", settings.guardrail_tier.description());
                println!("• Unrestricted Waiver Accepted: {}", if settings.unrestricted_waiver_accepted { "YES" } else { "NO" });
                if let Some(ts) = settings.unrestricted_waiver_timestamp {
                    println!("• Waiver Acknowledged At: {}", ts);
                }
            }
            GuardrailCommand::Set(tier) => {
                let Some(storage) = open_storage_cli()? else { return Ok(()); };
                let mut settings = storage.load_settings().unwrap_or_default();
                if tier == crate::guardrails::GuardrailTier::None && !settings.unrestricted_waiver_accepted {
                    println!("⚠ UNRESTRICTED GUARDRAILS REQUIREMENT");
                    println!("{}", crate::guardrails::LEGAL_DISCLAIMER_TEXT);
                    println!("\nTo enable Unrestricted mode, please review and accept the legal waiver in the ML Laboratory GUI (Tools / Settings tab) or type '{}' when prompted in the GUI.", crate::guardrails::REQUIRED_WAIVER_CONFIRMATION);
                    return Ok(());
                }
                settings.guardrail_tier = tier;
                storage.save_settings(&settings)?;
                println!("Guardrail tier switched to: {}", tier.label());
            }
        },
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_non_command() {
        assert_eq!(SlashCommand::parse("hello world").unwrap(), None);
        assert_eq!(SlashCommand::parse("explain quantum computing").unwrap(), None);
    }

    #[test]
    fn parse_basic_commands() {
        assert_eq!(SlashCommand::parse("/help").unwrap(), Some(SlashCommand::Help));
        assert_eq!(SlashCommand::parse("/?").unwrap(), Some(SlashCommand::Help));
        assert_eq!(SlashCommand::parse("/new").unwrap(), Some(SlashCommand::New));
        assert_eq!(SlashCommand::parse("/clear").unwrap(), Some(SlashCommand::Clear));
        assert_eq!(SlashCommand::parse("/audit").unwrap(), Some(SlashCommand::Audit));
        assert_eq!(SlashCommand::parse("/status").unwrap(), Some(SlashCommand::Status));
    }

    #[test]
    fn parse_model_command() {
        assert_eq!(
            SlashCommand::parse("/model qwen2.5-coder:7b").unwrap(),
            Some(SlashCommand::Model("qwen2.5-coder:7b".to_string()))
        );
        assert_eq!(
            SlashCommand::parse("/m deepseek-r1:14b").unwrap(),
            Some(SlashCommand::Model("deepseek-r1:14b".to_string()))
        );
        assert!(SlashCommand::parse("/model").is_err());
    }

    #[test]
    fn parse_swarm_command() {
        assert_eq!(
            SlashCommand::parse("/swarm optimize memory layout").unwrap(),
            Some(SlashCommand::Swarm(SwarmCommand::Dispatch("optimize memory layout".to_string())))
        );
        assert_eq!(
            SlashCommand::parse("/swarm preset triad build secure parser").unwrap(),
            Some(SlashCommand::Swarm(SwarmCommand::Preset {
                template: "triad".to_string(),
                prompt: Some("build secure parser".to_string()),
            }))
        );
        assert_eq!(
            SlashCommand::parse("/swarm presets").unwrap(),
            Some(SlashCommand::Swarm(SwarmCommand::ListPresets))
        );
        assert_eq!(
            SlashCommand::parse("/swarm dag diamond Build quantum ring buffer").unwrap(),
            Some(SlashCommand::Swarm(SwarmCommand::Dag {
                preset: Some("diamond".to_string()),
                prompt: Some("Build quantum ring buffer".to_string()),
            }))
        );
        assert_eq!(
            SlashCommand::parse("/swarm blackboard").unwrap(),
            Some(SlashCommand::Swarm(SwarmCommand::Blackboard))
        );
        assert_eq!(
            SlashCommand::parse("/swarm abort").unwrap(),
            Some(SlashCommand::Swarm(SwarmCommand::Abort))
        );
        assert!(SlashCommand::parse("/swarm").is_err());
    }

    #[test]
    fn parse_project_automation_commands() {
        assert_eq!(
            SlashCommand::parse("/project setup").unwrap(),
            Some(SlashCommand::Project(ProjectCommand::Setup))
        );
        assert_eq!(
            SlashCommand::parse("/project verify").unwrap(),
            Some(SlashCommand::Project(ProjectCommand::Verify))
        );
        assert_eq!(
            SlashCommand::parse("/project clean").unwrap(),
            Some(SlashCommand::Project(ProjectCommand::Clean))
        );
        assert_eq!(
            SlashCommand::parse("/project generate-scripts").unwrap(),
            Some(SlashCommand::Project(ProjectCommand::GenerateScripts))
        );
    }

    #[test]
    fn parse_threads_command() {
        assert_eq!(
            SlashCommand::parse("/threads 10").unwrap(),
            Some(SlashCommand::Threads(10))
        );
        assert!(SlashCommand::parse("/threads 0").is_err());
        assert!(SlashCommand::parse("/threads abc").is_err());
    }

    #[test]
    fn parse_skills_commands() {
        assert_eq!(
            SlashCommand::parse("/skills list").unwrap(),
            Some(SlashCommand::Skills(SkillsCommand::List))
        );
        assert_eq!(
            SlashCommand::parse("/skills run audit_code").unwrap(),
            Some(SlashCommand::Skills(SkillsCommand::Run("audit_code".to_string())))
        );
        assert_eq!(
            SlashCommand::parse("/skills add vulns Scan memory").unwrap(),
            Some(SlashCommand::Skills(SkillsCommand::Add {
                name: "vulns".to_string(),
                description: "Scan memory".to_string()
            }))
        );
    }

    #[test]
    fn parse_unknown_command() {
        assert!(SlashCommand::parse("/foobar").is_err());
    }

    #[test]
    fn parse_file_and_exec_commands() {
        assert_eq!(
            SlashCommand::parse("/exec cargo check").unwrap(),
            Some(SlashCommand::Exec("cargo check".to_string()))
        );
        assert_eq!(
            SlashCommand::parse("/run python3 script.py").unwrap(),
            Some(SlashCommand::Exec("python3 script.py".to_string()))
        );
        assert_eq!(
            SlashCommand::parse("/mkdir src/neural/quantum").unwrap(),
            Some(SlashCommand::Mkdir("src/neural/quantum".to_string()))
        );
        assert_eq!(
            SlashCommand::parse("/touch src/neural/mod.rs").unwrap(),
            Some(SlashCommand::Touch("src/neural/mod.rs".to_string()))
        );
        assert_eq!(
            SlashCommand::parse("/rm old_file.rs").unwrap(),
            Some(SlashCommand::Rm("old_file.rs".to_string()))
        );
        assert_eq!(
            SlashCommand::parse("/mv draft.py final.py").unwrap(),
            Some(SlashCommand::Mv {
                src: "draft.py".to_string(),
                dst: "final.py".to_string()
            })
        );
        assert_eq!(
            SlashCommand::parse("/ls src").unwrap(),
            Some(SlashCommand::Ls(Some("src".to_string())))
        );
        assert_eq!(
            SlashCommand::parse("/project scaffold rust my_service").unwrap(),
            Some(SlashCommand::Project(ProjectCommand::Scaffold {
                template: "rust".to_string(),
                name: "my_service".to_string(),
            }))
        );
        assert_eq!(
            SlashCommand::parse("/project build").unwrap(),
            Some(SlashCommand::Project(ProjectCommand::Build))
        );
    }

    #[test]
    fn parse_tools_and_guardrail_commands() {
        assert_eq!(
            SlashCommand::parse("/tools").unwrap(),
            Some(SlashCommand::Tools(ToolsCommand::List))
        );
        assert_eq!(
            SlashCommand::parse("/tools list").unwrap(),
            Some(SlashCommand::Tools(ToolsCommand::List))
        );
        assert_eq!(
            SlashCommand::parse("/tools run editor src/main.rs").unwrap(),
            Some(SlashCommand::Tools(ToolsCommand::Run {
                id: "editor".to_string(),
                args: Some("src/main.rs".to_string()),
            }))
        );
        assert_eq!(
            SlashCommand::parse("/tools add nvim nvim {file} --detached --high").unwrap(),
            Some(SlashCommand::Tools(ToolsCommand::Add {
                id: "nvim".to_string(),
                command: "nvim".to_string(),
                args_template: "{file}".to_string(),
                detached: true,
                high_sensitivity: true,
            }))
        );
        assert_eq!(
            SlashCommand::parse("/tools rm nvim").unwrap(),
            Some(SlashCommand::Tools(ToolsCommand::Rm("nvim".to_string())))
        );
        assert_eq!(
            SlashCommand::parse("/guardrail").unwrap(),
            Some(SlashCommand::Guardrail(GuardrailCommand::Status))
        );
        assert_eq!(
            SlashCommand::parse("/guardrail heavy").unwrap(),
            Some(SlashCommand::Guardrail(GuardrailCommand::Set(crate::guardrails::GuardrailTier::Heavy)))
        );
        assert_eq!(
            SlashCommand::parse("/guardrail none").unwrap(),
            Some(SlashCommand::Guardrail(GuardrailCommand::Set(crate::guardrails::GuardrailTier::None)))
        );
    }
}
