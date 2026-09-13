// Copyright 2026 Sean M. Stow. All rights reserved.
//! Autonomous Multi-Agent Directed Acyclic Graph (DAG) Swarm Engine.
//! Provides asynchronous dependency graph resolution, topological level scheduling,
//! parallel branch execution queues, and multi-agent presets.

use crate::ollama::api::Model;
use crate::ui::app::ModelRole;
use crate::ui::relay::RelayPanel;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

pub type NodeId = usize;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeStatus {
    Pending,
    Ready,
    Running,
    Completed(f32), // duration in seconds
    Recovered { attempts: u32, duration: f32 },
    Failed(String),
    Skipped,
}

impl NodeStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            NodeStatus::Completed(_) | NodeStatus::Recovered { .. } | NodeStatus::Failed(_) | NodeStatus::Skipped
        )
    }

    pub fn is_completed(&self) -> bool {
        matches!(self, NodeStatus::Completed(_) | NodeStatus::Recovered { .. })
    }

    pub fn is_failed(&self) -> bool {
        matches!(self, NodeStatus::Failed(_))
    }

    pub fn is_running(&self) -> bool {
        matches!(self, NodeStatus::Running)
    }

    pub fn label(&self) -> String {
        match self {
            NodeStatus::Pending => "⏳ Pending".to_string(),
            NodeStatus::Ready => "⚡ Ready".to_string(),
            NodeStatus::Running => "▶ Running".to_string(),
            NodeStatus::Completed(dur) => format!("✔ Completed ({:.1}s)", dur),
            NodeStatus::Recovered { attempts, duration } => {
                format!("✔ Recovered (try #{}, {:.1}s)", attempts, duration)
            }
            NodeStatus::Failed(err) => format!("✖ Failed: {}", err),
            NodeStatus::Skipped => "⏭ Skipped".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum EdgeCondition {
    #[default]
    Always,
    OnSuccess,
    OnFailure,
}

impl EdgeCondition {
    pub fn label(&self) -> &'static str {
        match self {
            EdgeCondition::Always => "Always",
            EdgeCondition::OnSuccess => "On Success",
            EdgeCondition::OnFailure => "On Failure (Remediation)",
        }
    }

    pub fn badge(&self) -> &'static str {
        match self {
            EdgeCondition::Always => "➜ Always",
            EdgeCondition::OnSuccess => "✔ If Pass",
            EdgeCondition::OnFailure => "✖ If Fail",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmTaskNode {
    pub id: NodeId,
    pub name: String,
    pub role: ModelRole,
    pub model: Option<String>,
    pub directive: String,
    pub dependencies: Vec<NodeId>,
    pub status: NodeStatus,
    pub output: String,
    pub duration: Option<f32>,
    pub collapsed: bool,
    pub domain: String,
    pub tags: Vec<String>,
    pub retries: u32,
    pub max_retries: u32,
    pub fallback_model: Option<String>,
    pub last_error: Option<String>,
    pub tool_id: Option<String>,
    pub auto_exec_tool: bool,
    pub tool_output: Option<String>,
    pub edge_condition: EdgeCondition,
}

impl SwarmTaskNode {
    pub fn new(
        id: NodeId,
        name: impl Into<String>,
        role: ModelRole,
        directive: impl Into<String>,
        dependencies: Vec<NodeId>,
        domain: impl Into<String>,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            role,
            model: None,
            directive: directive.into(),
            dependencies,
            status: NodeStatus::Pending,
            output: String::new(),
            duration: None,
            collapsed: false,
            domain: domain.into(),
            tags: Vec::new(),
            retries: 0,
            max_retries: 2,
            fallback_model: None,
            last_error: None,
            tool_id: None,
            auto_exec_tool: false,
            tool_output: None,
            edge_condition: EdgeCondition::Always,
        }
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    pub fn with_tool(mut self, tool_id: impl Into<String>, auto_exec: bool) -> Self {
        self.tool_id = Some(tool_id.into());
        self.auto_exec_tool = auto_exec;
        self
    }

    pub fn with_condition(mut self, condition: EdgeCondition) -> Self {
        self.edge_condition = condition;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DagPreset {
    Diamond,
    CyberSocGrid,
    QuantumScientific,
    FullStackForge,
    Custom,
}

impl DagPreset {
    pub fn all() -> Vec<DagPreset> {
        vec![
            DagPreset::Diamond,
            DagPreset::CyberSocGrid,
            DagPreset::QuantumScientific,
            DagPreset::FullStackForge,
            DagPreset::Custom,
        ]
    }

    pub fn label(self) -> &'static str {
        match self {
            DagPreset::Diamond => "💎 Diamond Swarm (Architect → [Backend || Frontend || Security] → Queen Integrator)",
            DagPreset::CyberSocGrid => "🛡️ Cyber / SOC Grid (Triage → [Forensics || Threat Hunter] → Remediation → Commander)",
            DagPreset::QuantumScientific => "⚛️ Quantum Algorithm Swarm (Hypothesis → [State Vector || Noise Critic] → Compiler → Paper)",
            DagPreset::FullStackForge => "🌐 Full-Stack Dev Forge (Architect → [DB Core || Reactive UI] → Middlewares → QA → Release)",
            DagPreset::Custom => "⚙️ Custom Swarm DAG Topology",
        }
    }

    pub fn short_id(self) -> &'static str {
        match self {
            DagPreset::Diamond => "diamond",
            DagPreset::CyberSocGrid => "soc",
            DagPreset::QuantumScientific => "quantum",
            DagPreset::FullStackForge => "fullstack",
            DagPreset::Custom => "custom",
        }
    }

    pub fn from_id(id: &str) -> Option<DagPreset> {
        match id.to_lowercase().trim() {
            "diamond" | "parallel" | "triad" => Some(DagPreset::Diamond),
            "soc" | "cyber" | "cybersoc" => Some(DagPreset::CyberSocGrid),
            "quantum" | "scientific" => Some(DagPreset::QuantumScientific),
            "fullstack" | "forge" | "dev" => Some(DagPreset::FullStackForge),
            "custom" => Some(DagPreset::Custom),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmDag {
    pub title: String,
    pub objective: String,
    pub nodes: Vec<SwarmTaskNode>,
    pub preset: DagPreset,
    pub consensus_score: Option<f32>,
    pub final_synthesis: String,
    pub is_running: bool,
}

impl Default for SwarmDag {
    fn default() -> Self {
        Self::new(DagPreset::Diamond)
    }
}

impl SwarmDag {
    pub fn new(preset: DagPreset) -> Self {
        let mut dag = Self {
            title: preset.label().to_string(),
            objective: String::new(),
            nodes: Vec::new(),
            preset,
            consensus_score: None,
            final_synthesis: String::new(),
            is_running: false,
        };
        dag.apply_preset(preset);
        dag
    }

    pub fn add_node(&mut self, node: SwarmTaskNode) {
        self.nodes.push(node);
    }

    pub fn add_dependency(&mut self, child_id: NodeId, parent_id: NodeId) -> Result<(), String> {
        if child_id == parent_id {
            return Err("Self-dependency detected: node cannot depend on itself".to_string());
        }
        if let Some(node) = self.nodes.iter_mut().find(|n| n.id == child_id) {
            if !node.dependencies.contains(&parent_id) {
                node.dependencies.push(parent_id);
            }
        } else {
            return Err(format!("Node {} not found", child_id));
        }
        self.validate_acyclic()
    }

    pub fn find_node(&self, id: NodeId) -> Option<&SwarmTaskNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn find_node_mut(&mut self, id: NodeId) -> Option<&mut SwarmTaskNode> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    /// Validates that the graph has no directed cycles (is a strict DAG).
    pub fn validate_acyclic(&self) -> Result<(), String> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for n in &self.nodes {
            in_degree.insert(n.id, n.dependencies.len());
            for &dep in &n.dependencies {
                adj.entry(dep).or_default().push(n.id);
            }
        }

        let mut queue: VecDeque<NodeId> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut visited = 0;
        while let Some(u) = queue.pop_front() {
            visited += 1;
            if let Some(children) = adj.get(&u) {
                for &v in children {
                    if let Some(deg) = in_degree.get_mut(&v) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(v);
                        }
                    }
                }
            }
        }

        if visited == self.nodes.len() {
            Ok(())
        } else {
            Err("Cyclic dependency loop detected in Swarm DAG graph!".to_string())
        }
    }

    /// Partitions the nodes into topological execution ranks (levels).
    /// Nodes in the same level have all dependencies satisfied by prior levels
    /// and can execute completely in parallel!
    pub fn topological_ranks(&self) -> Result<Vec<Vec<NodeId>>, String> {
        self.validate_acyclic()?;

        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }

        let mut node_levels: HashMap<NodeId, usize> = HashMap::new();

        // Repeatedly compute longest path from roots
        let mut changed = true;
        let mut iterations = 0;
        let max_iter = self.nodes.len() + 2;

        for n in &self.nodes {
            node_levels.insert(n.id, 0);
        }

        while changed && iterations < max_iter {
            changed = false;
            iterations += 1;
            for n in &self.nodes {
                let current_level = node_levels[&n.id];
                let mut max_parent_level = 0;
                let has_parents = !n.dependencies.is_empty();
                for &parent_id in &n.dependencies {
                    if let Some(&p_lvl) = node_levels.get(&parent_id) {
                        if p_lvl + 1 > max_parent_level {
                            max_parent_level = p_lvl + 1;
                        }
                    }
                }
                if has_parents && max_parent_level > current_level {
                    node_levels.insert(n.id, max_parent_level);
                    changed = true;
                }
            }
        }

        if iterations >= max_iter {
            return Err("Cycle detected while computing topological ranks".to_string());
        }

        let max_level = node_levels.values().copied().max().unwrap_or(0);
        let mut ranks: Vec<Vec<NodeId>> = vec![Vec::new(); max_level + 1];

        for n in &self.nodes {
            let lvl = node_levels[&n.id];
            ranks[lvl].push(n.id);
        }

        // Sort each rank by node ID for deterministic execution order
        for rank in &mut ranks {
            rank.sort();
        }

        Ok(ranks)
    }

    /// Evaluates which nodes are currently ready to execute:
    /// Status is `Pending` or `Ready`, all dependencies are terminal,
    /// and `edge_condition` is satisfied.
    pub fn get_ready_nodes(&mut self) -> Vec<NodeId> {
        let terminal_statuses: HashMap<NodeId, NodeStatus> = self
            .nodes
            .iter()
            .filter(|n| n.status.is_terminal())
            .map(|n| (n.id, n.status.clone()))
            .collect();

        let mut ready = Vec::new();
        for n in &mut self.nodes {
            if matches!(n.status, NodeStatus::Pending | NodeStatus::Ready) {
                let all_deps_terminal = n.dependencies.iter().all(|dep| terminal_statuses.contains_key(dep));
                if all_deps_terminal {
                    let should_run = match n.edge_condition {
                        EdgeCondition::Always => true,
                        EdgeCondition::OnSuccess => n.dependencies.iter().all(|dep| {
                            terminal_statuses.get(dep).map_or(false, |st| st.is_completed())
                        }),
                        EdgeCondition::OnFailure => n.dependencies.iter().any(|dep| {
                            terminal_statuses.get(dep).map_or(false, |st| st.is_failed())
                        }),
                    };

                    if should_run {
                        n.status = NodeStatus::Ready;
                        ready.push(n.id);
                    } else {
                        n.status = NodeStatus::Skipped;
                    }
                }
            }
        }
        ready
    }

    pub fn mark_running(&mut self, id: NodeId) {
        if let Some(n) = self.find_node_mut(id) {
            n.status = NodeStatus::Running;
        }
    }

    pub fn mark_completed(&mut self, id: NodeId, output: String, duration: f32) {
        if let Some(n) = self.find_node_mut(id) {
            n.output = output;
            n.duration = Some(duration);
            if n.retries > 0 {
                n.status = NodeStatus::Recovered {
                    attempts: n.retries + 1,
                    duration,
                };
            } else {
                n.status = NodeStatus::Completed(duration);
            }
            n.collapsed = true;
        }
        self.update_finished_state();
    }

    /// Check if a node can attempt automatic retry
    pub fn can_retry(&self, id: NodeId) -> bool {
        if let Some(n) = self.find_node(id) {
            n.retries < n.max_retries
        } else {
            false
        }
    }

    /// Records an execution failure. If node has retries remaining, increments retry count,
    /// sets status to `Ready`, records `last_error`, and returns `true` (indicating retry).
    /// Otherwise, marks node `Failed`, cascades skip downstream, updates finished state,
    /// and returns `false`.
    pub fn record_failure(&mut self, id: NodeId, err: String) -> bool {
        let will_retry = if let Some(n) = self.find_node_mut(id) {
            n.retries += 1;
            n.last_error = Some(err.clone());
            let retry_allowed = n.retries <= n.max_retries;
            if retry_allowed {
                n.status = NodeStatus::Ready;
            } else {
                n.status = NodeStatus::Failed(err.clone());
            }
            retry_allowed
        } else {
            return false;
        };

        if !will_retry {
            self.cascade_skip(id);
            self.update_finished_state();
        }
        will_retry
    }

    pub fn mark_failed(&mut self, id: NodeId, err: String) {
        if let Some(n) = self.find_node_mut(id) {
            n.last_error = Some(err.clone());
            n.status = NodeStatus::Failed(err);
        }
        // Cascade skip to dependent children
        self.cascade_skip(id);
        self.update_finished_state();
    }

    /// Reset a node for manual or automated retry.
    /// Resets retry counter, last error, clears output, sets status to Ready,
    /// and un-skips downstream nodes that were skipped due to this node.
    pub fn reset_node_for_retry(&mut self, id: NodeId, model_override: Option<String>) {
        if let Some(n) = self.find_node_mut(id) {
            n.retries = 0;
            n.last_error = None;
            n.output.clear();
            n.duration = None;
            n.collapsed = false;
            n.status = NodeStatus::Ready;
            if let Some(m) = model_override {
                n.model = Some(m);
            }
        }
        self.unskip_downstream(id);
    }

    /// Recursively un-skips downstream nodes if their upstream failed dependency was recovered or reset.
    pub fn unskip_downstream(&mut self, parent_id: NodeId) {
        let mut to_check = vec![parent_id];
        while let Some(pid) = to_check.pop() {
            for n in &mut self.nodes {
                if n.dependencies.contains(&pid) && matches!(n.status, NodeStatus::Skipped) {
                    n.status = NodeStatus::Pending;
                    to_check.push(n.id);
                }
            }
        }
    }

    fn cascade_skip(&mut self, failed_id: NodeId) {
        let mut to_skip = vec![failed_id];
        while let Some(parent) = to_skip.pop() {
            for n in &mut self.nodes {
                // If child node has OnFailure condition, parent failure is its trigger so do not skip it!
                if n.dependencies.contains(&parent) && !n.status.is_terminal() && n.edge_condition != EdgeCondition::OnFailure {
                    n.status = NodeStatus::Skipped;
                    to_skip.push(n.id);
                }
            }
        }
    }

    pub fn reset(&mut self) {
        self.is_running = false;
        self.consensus_score = None;
        self.final_synthesis.clear();
        for n in &mut self.nodes {
            n.status = NodeStatus::Pending;
            n.output.clear();
            n.duration = None;
            n.collapsed = false;
            n.retries = 0;
            n.last_error = None;
            n.tool_output = None;
        }
    }

    pub fn abort(&mut self) {
        self.is_running = false;
        for n in &mut self.nodes {
            if n.status.is_running() || matches!(n.status, NodeStatus::Ready) {
                n.status = NodeStatus::Failed("Aborted by operator".to_string());
            }
        }
    }

    pub fn is_finished(&self) -> bool {
        self.nodes.iter().all(|n| n.status.is_terminal())
    }

    pub fn all_completed_successfully(&self) -> bool {
        !self.nodes.is_empty() && self.nodes.iter().all(|n| n.status.is_completed())
    }

    pub fn progress(&self) -> (usize, usize) {
        let completed = self.nodes.iter().filter(|n| n.status.is_completed()).count();
        (completed, self.nodes.len())
    }

    fn update_finished_state(&mut self) {
        if self.is_finished() {
            self.is_running = false;
            // Compute consensus score across Coder and Critic nodes
            let worker_out = self
                .nodes
                .iter()
                .filter(|n| n.role == ModelRole::Coder)
                .map(|n| n.output.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            let critic_out = self
                .nodes
                .iter()
                .filter(|n| n.role == ModelRole::Critic)
                .map(|n| n.output.as_str())
                .collect::<Vec<_>>()
                .join("\n");

            if !worker_out.is_empty() && !critic_out.is_empty() {
                let lower = critic_out.to_lowercase();
                let mut score: f32 = 0.88;
                for p in &["verified", "secure", "clean", "passed", "optimal", "robust", "valid"] {
                    if lower.contains(p) {
                        score += 0.03;
                    }
                }
                for d in &["vulnerability", "risk", "flaw", "insecure", "overflow", "exploit"] {
                    if lower.contains(d) {
                        score -= 0.05;
                    }
                }
                self.consensus_score = Some(score.clamp(0.20, 0.99));
            } else {
                self.consensus_score = Some(0.92);
            }

            // If the last node or a synthesis node produced output, set final_synthesis
            if let Some(last) = self.nodes.last() {
                if last.status.is_completed() && !last.output.is_empty() {
                    self.final_synthesis = last.output.clone();
                }
            }
        }
    }

    /// Auto-matches models for nodes that have no model assigned
    pub fn auto_assign_models(&mut self, available_models: &[Model]) {
        if available_models.is_empty() {
            return;
        }
        for (idx, node) in self.nodes.iter_mut().enumerate() {
            if node.model.is_none() {
                if let Some(matched) = RelayPanel::find_best_model_for_role(&node.role, available_models) {
                    node.model = Some(matched);
                } else {
                    let m = &available_models[idx % available_models.len()];
                    node.model = Some(m.name.clone());
                }
            }
        }
    }

    /// Extract chat slots mapping for 1-click loading into Chat split view
    pub fn extract_chat_slots(&self, available_models: &[Model]) -> Vec<(ModelRole, Option<String>)> {
        self.nodes
            .iter()
            .map(|node| {
                let model = node
                    .model
                    .clone()
                    .or_else(|| RelayPanel::find_best_model_for_role(&node.role, available_models));
                (node.role.clone(), model)
            })
            .collect()
    }

    pub fn apply_preset(&mut self, preset: DagPreset) {
        self.preset = preset;
        self.title = preset.label().to_string();
        self.reset();
        match preset {
            DagPreset::Diamond => {
                // Diamond Topology:
                // Node 0: Systems Architect (Root)
                // Nodes 1, 2, 3: Backend, Frontend, Security Critic (Parallel Level 1)
                // Node 4: Queen Integrator (Convergence Level 2)
                self.nodes = vec![
                    SwarmTaskNode::new(
                        0,
                        "Lead Systems Architect",
                        ModelRole::Planner,
                        "Analyze the core objective and specify the modular architecture, interface contracts, and core data flow.",
                        vec![],
                        "architecture",
                    ).with_tags(vec!["blueprint".into(), "contracts".into()]),
                    SwarmTaskNode::new(
                        1,
                        "Backend & Systems Engineer",
                        ModelRole::Coder,
                        "Implement the high-performance backend, database schemas, and cryptographic core according to the architecture.",
                        vec![0],
                        "backend",
                    ).with_tags(vec!["systems".into(), "backend".into()]),
                    SwarmTaskNode::new(
                        2,
                        "Frontend & Reactive UI Specialist",
                        ModelRole::Coder,
                        "Design and implement the high-contrast client UI, user flows, and interactive state management.",
                        vec![0],
                        "frontend",
                    ).with_tags(vec!["ui".into(), "client".into()]),
                    SwarmTaskNode::new(
                        3,
                        "Adversarial Security Auditor",
                        ModelRole::Critic,
                        "Perform an adversarial security scan on the architectural plan and interfaces, identifying threat surfaces.",
                        vec![0],
                        "security",
                    ).with_tags(vec!["audit".into(), "redteam".into()]),
                    SwarmTaskNode::new(
                        4,
                        "Queen Integration Synthesizer",
                        ModelRole::General,
                        "Harmonize backend, frontend, and security audit outputs into an authoritative, complete, production-ready system deliverable.",
                        vec![1, 2, 3],
                        "synthesis",
                    ).with_tags(vec!["synthesis".into(), "deliverable".into()]),
                ];
            }
            DagPreset::CyberSocGrid => {
                // Cyber SOC Defense Grid:
                // Node 0: Threat Intel & Triage (Root)
                // Nodes 1, 2: Forensics & Threat Hunter (Parallel Level 1)
                // Node 3: Hardening & Remediation Coder (Level 2)
                // Node 4: Incident Commander Briefing (Level 3)
                self.nodes = vec![
                    SwarmTaskNode::new(
                        0,
                        "SOC Alert Triage & Threat Intel",
                        ModelRole::Researcher,
                        "Triage the incident telemetry, identify initial vectors, IOC signatures, and adversarial ATT&CK techniques.",
                        vec![],
                        "threat_intel",
                    ).with_tags(vec!["triage".into(), "ioc".into()]),
                    SwarmTaskNode::new(
                        1,
                        "Packet & Memory Forensics Analyst",
                        ModelRole::Critic,
                        "Deep dive into network pcap patterns, memory dump anomalies, and protocol dissection for lateral movement.",
                        vec![0],
                        "forensics",
                    ).with_tags(vec!["pcap".into(), "memory".into()]).with_tool("hexdump", false),
                    SwarmTaskNode::new(
                        2,
                        "Threat Hunter & ATT&CK Matrix Specialist",
                        ModelRole::Critic,
                        "Map adversarial behavior against MITRE ATT&CK, trace persistence mechanisms, and identify blind spots.",
                        vec![0],
                        "threat_hunting",
                    ).with_tags(vec!["mitre".into(), "hunting".into()]),
                    SwarmTaskNode::new(
                        3,
                        "Defensive Hardening & Remediation Engineer",
                        ModelRole::Coder,
                        "Produce concrete defensive mitigation scripts, firewall ACL rules, YARA/Sigma signatures, and patch code.",
                        vec![1, 2],
                        "remediation",
                    ).with_tags(vec!["patch".into(), "sigma".into(), "hardening".into()]),
                    SwarmTaskNode::new(
                        4,
                        "Incident Commander Briefing Officer",
                        ModelRole::Planner,
                        "Synthesize forensic findings, hunting metrics, and remediation actions into an executive incident briefing.",
                        vec![3],
                        "executive_briefing",
                    ).with_tags(vec!["incident_report".into(), "executive".into()]),
                ];
            }
            DagPreset::QuantumScientific => {
                // Quantum Algorithm & Simulation Swarm:
                // Node 0: Hamiltonian Formulation (Root)
                // Nodes 1, 2: Circuit Synthesizer & Noise Critic (Parallel Level 1)
                // Node 3: Numerical Simulation Compiler (Level 2)
                // Node 4: Peer Review & Scientific Paper (Level 3)
                self.nodes = vec![
                    SwarmTaskNode::new(
                        0,
                        "Quantum Theoretical Physicist",
                        ModelRole::Researcher,
                        "Formulate the Hamiltonian operator, state space boundaries, and mathematical proofs for the quantum problem.",
                        vec![],
                        "quantum_theory",
                    ).with_tags(vec!["hamiltonian".into(), "math".into()]),
                    SwarmTaskNode::new(
                        1,
                        "Quantum Circuit Synthesizer",
                        ModelRole::Planner,
                        "Decompose the unitary evolution into parameterized quantum gates (Hadamard, CNOT, Phase, Rz) and circuit diagrams.",
                        vec![0],
                        "circuit_design",
                    ).with_tags(vec!["gates".into(), "circuit".into()]),
                    SwarmTaskNode::new(
                        2,
                        "Decoherence & Noise Auditor",
                        ModelRole::Critic,
                        "Analyze environmental decoherence, gate infidelities, T1/T2 relaxation times, and error mitigation strategies.",
                        vec![0],
                        "error_mitigation",
                    ).with_tags(vec!["decoherence".into(), "noise".into()]),
                    SwarmTaskNode::new(
                        3,
                        "Simulation & Numerical Compiler",
                        ModelRole::Coder,
                        "Implement the state vector simulation code or OpenQASM export with observable expectation value measurements.",
                        vec![1, 2],
                        "simulation_code",
                    ).with_tags(vec!["simulation".into(), "openqasm".into()]),
                    SwarmTaskNode::new(
                        4,
                        "Scientific Paper Synthesizer",
                        ModelRole::General,
                        "Compose an authoritative, publication-grade scientific treatise summarizing the Hamiltonian, circuits, and benchmarks.",
                        vec![3],
                        "scientific_paper",
                    ).with_tags(vec!["paper".into(), "benchmarks".into()]),
                ];
            }
            DagPreset::FullStackForge => {
                // Full-Stack Dev Forge:
                // Node 0: Product & Architectural Blueprint (Root)
                // Nodes 1, 2: DB Core & Client UI (Parallel Level 1)
                // Node 3: Business Logic & API Middlewares (Level 2)
                // Node 4: End-to-End Test & Security Auditor (Level 3)
                // Node 5: Release Consolidator (Level 4)
                self.nodes = vec![
                    SwarmTaskNode::new(
                        0,
                        "Lead Product & System Architect",
                        ModelRole::Planner,
                        "Specify the end-to-end user stories, data schemas, API contracts, and technology stack invariants.",
                        vec![],
                        "architecture",
                    ).with_tags(vec!["spec".into(), "architecture".into()]),
                    SwarmTaskNode::new(
                        1,
                        "Database & Storage Engineer",
                        ModelRole::Coder,
                        "Implement the relational or document persistence layer, migration scripts, and indexing strategies.",
                        vec![0],
                        "database",
                    ).with_tags(vec!["db".into(), "storage".into()]),
                    SwarmTaskNode::new(
                        2,
                        "UI/UX Client Engineer",
                        ModelRole::Coder,
                        "Implement the client interface, components, layouts, and responsive state synchronization.",
                        vec![0],
                        "ui",
                    ).with_tags(vec!["ui".into(), "components".into()]),
                    SwarmTaskNode::new(
                        3,
                        "Business Logic & API Services",
                        ModelRole::Coder,
                        "Build the backend business logic services connecting the storage models to the client UI endpoints.",
                        vec![1, 2],
                        "api",
                    ).with_tags(vec!["services".into(), "api".into()]),
                    SwarmTaskNode::new(
                        4,
                        "End-to-End QA & Security Auditor",
                        ModelRole::Critic,
                        "Generate regression test suites, verify API contract boundaries, and audit for edge case vulnerabilities.",
                        vec![3],
                        "audit",
                    ).with_tags(vec!["tests".into(), "audit".into()]).with_tool("cargo-check", false),
                    SwarmTaskNode::new(
                        5,
                        "Master Release Consolidator",
                        ModelRole::General,
                        "Assemble all files, scripts, and documentation into a turnkey production release archive.",
                        vec![4],
                        "release",
                    ).with_tags(vec!["release".into(), "production".into()]),
                ];
            }
            DagPreset::Custom => {
                self.nodes = vec![
                    SwarmTaskNode::new(
                        0,
                        "Swarm Node 1",
                        ModelRole::Planner,
                        "Define task specification and plan.",
                        vec![],
                        "planning",
                    ),
                    SwarmTaskNode::new(
                        1,
                        "Swarm Node 2",
                        ModelRole::Coder,
                        "Implement solution.",
                        vec![0],
                        "code",
                    ),
                ];
            }
        }
    }

    pub fn export_markdown(&self) -> String {
        let mut md = format!(
            "# 🕸 Swarm DAG Execution Report: {}\n\n**Objective:**\n{}\n\n**Preset:** `{}` | **Status:** {}\n\n---\n\n",
            self.title,
            if self.objective.is_empty() { "[No objective set]" } else { &self.objective },
            self.preset.short_id(),
            if self.is_running { "Running" } else if self.is_finished() { "Finished" } else { "Idle" }
        );

        if let Ok(ranks) = self.topological_ranks() {
            for (lvl, rank_nodes) in ranks.iter().enumerate() {
                md.push_str(&format!("## Topological Rank Level {}\n\n", lvl));
                for &id in rank_nodes {
                    if let Some(node) = self.find_node(id) {
                        let dur = node.duration.map(|d| format!("{:.1}s", d)).unwrap_or_else(|| "N/A".to_string());
                        let deps_str = if node.dependencies.is_empty() {
                            "None (Root)".to_string()
                        } else {
                            node.dependencies.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
                        };
                        md.push_str(&format!(
                            "### Node {} · {} (`{}`)\n*Dependencies:* [{}] | *Model:* `{}` | *Duration:* {}\n*Domain:* `{}`\n\n{}\n\n",
                            node.id,
                            node.name,
                            node.role.label(),
                            deps_str,
                            node.model.as_deref().unwrap_or("unassigned"),
                            dur,
                            node.domain,
                            if node.output.is_empty() { "[No output]" } else { &node.output }
                        ));
                    }
                }
                md.push_str("---\n\n");
            }
        }

        if !self.final_synthesis.is_empty() {
            md.push_str(&format!("## 👑 Final Synthesis\n\n{}\n", self.final_synthesis));
        }

        md
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dag_cycle_detection_and_topological_ranks() {
        let mut dag = SwarmDag::new(DagPreset::Diamond);
        assert!(dag.validate_acyclic().is_ok());

        let ranks = dag.topological_ranks().expect("Should compute ranks");
        assert_eq!(ranks.len(), 3);
        assert_eq!(ranks[0], vec![0]); // Root Architect
        assert_eq!(ranks[1], vec![1, 2, 3]); // Parallel Backend, Frontend, Security
        assert_eq!(ranks[2], vec![4]); // Queen Integrator

        // Introduce an intentional cycle: Node 0 depends on Node 4
        dag.nodes[0].dependencies.push(4);
        assert!(dag.validate_acyclic().is_err());
        assert!(dag.topological_ranks().is_err());
    }

    #[test]
    fn test_dag_ready_queue_and_dependency_resolution() {
        let mut dag = SwarmDag::new(DagPreset::Diamond);

        // Initially only Node 0 (Root) is ready
        let ready = dag.get_ready_nodes();
        assert_eq!(ready, vec![0]);

        // Mark Node 0 completed
        dag.mark_completed(0, "Architecture Blueprint complete".to_string(), 2.5);

        // Now nodes 1, 2, 3 should all be ready in parallel
        let mut ready2 = dag.get_ready_nodes();
        ready2.sort();
        assert_eq!(ready2, vec![1, 2, 3]);

        // Complete 1 and 2, node 4 should NOT be ready yet because 3 is still pending
        dag.mark_completed(1, "Backend built".to_string(), 3.0);
        dag.mark_completed(2, "Frontend built".to_string(), 2.8);
        let ready3 = dag.get_ready_nodes();
        assert!(ready3.contains(&3));
        assert!(!ready3.contains(&4));

        // Complete node 3, now node 4 must become ready
        dag.mark_completed(3, "Audit clean".to_string(), 1.5);
        let ready4 = dag.get_ready_nodes();
        assert_eq!(ready4, vec![4]);

        // Complete node 4 -> graph finished
        dag.mark_completed(4, "Consolidated deliverable".to_string(), 2.0);
        assert!(dag.is_finished());
        assert!(dag.all_completed_successfully());
        assert_eq!(dag.progress(), (5, 5));
    }

    #[test]
    fn test_all_dag_presets_are_acyclic() {
        for preset in DagPreset::all() {
            let dag = SwarmDag::new(preset);
            assert!(dag.validate_acyclic().is_ok(), "Preset {:?} must be acyclic", preset);
            let ranks = dag.topological_ranks().expect("Ranks must be valid");
            assert!(!ranks.is_empty());
        }
    }

    #[test]
    fn test_node_retry_and_recovery_cycle() {
        let mut dag = SwarmDag::new(DagPreset::Diamond);
        dag.get_ready_nodes(); // node 0 ready
        dag.mark_completed(0, "Arch done".to_string(), 1.0);

        let ready = dag.get_ready_nodes();
        assert_eq!(ready, vec![1, 2, 3]);

        // Fail node 1 (attempt 1) -> retries allowed (max_retries is 2)
        assert!(dag.can_retry(1));
        let will_retry_1 = dag.record_failure(1, "Connection reset by peer".to_string());
        assert!(will_retry_1);
        assert_eq!(dag.find_node(1).unwrap().retries, 1);
        assert_eq!(dag.find_node(1).unwrap().status, NodeStatus::Ready);
        assert!(!dag.find_node(4).unwrap().status.is_terminal(), "Child must not be skipped on retryable failure");

        // Fail node 1 (attempt 2) -> still allows 2nd retry
        let will_retry_2 = dag.record_failure(1, "Out of memory error".to_string());
        assert!(will_retry_2);
        assert_eq!(dag.find_node(1).unwrap().retries, 2);
        assert_eq!(dag.find_node(1).unwrap().status, NodeStatus::Ready);

        // Node 1 succeeds on retry -> marks status Recovered!
        dag.mark_completed(1, "Backend recovered".to_string(), 3.2);
        let n1 = dag.find_node(1).unwrap();
        assert!(matches!(n1.status, NodeStatus::Recovered { attempts: 3, .. }));
        assert!(n1.status.is_completed());

        // Complete 2 and 3
        dag.mark_completed(2, "Frontend ok".to_string(), 2.0);
        dag.mark_completed(3, "Sec ok".to_string(), 1.8);

        // Child node 4 now becomes ready!
        let ready_final = dag.get_ready_nodes();
        assert_eq!(ready_final, vec![4]);
    }

    #[test]
    fn test_terminal_failure_and_manual_unskip() {
        let mut dag = SwarmDag::new(DagPreset::Diamond);
        dag.get_ready_nodes();
        dag.mark_completed(0, "Arch done".to_string(), 1.0);

        // Set max retries to 1 for node 1
        dag.find_node_mut(1).unwrap().max_retries = 1;

        // 1st failure: retries allowed
        assert!(dag.record_failure(1, "Ollama timeout".to_string()));

        // 2nd failure: exceeds max_retries -> terminal failure!
        let will_retry = dag.record_failure(1, "Crash".to_string());
        assert!(!will_retry);
        assert!(dag.find_node(1).unwrap().status.is_failed());

        // Child node 4 should now be skipped
        assert_eq!(dag.find_node(4).unwrap().status, NodeStatus::Skipped);

        // Manual operator unblock / retry:
        dag.reset_node_for_retry(1, Some("qwen2.5-coder:7b".to_string()));
        assert_eq!(dag.find_node(1).unwrap().status, NodeStatus::Ready);
        assert_eq!(dag.find_node(1).unwrap().model.as_deref(), Some("qwen2.5-coder:7b"));
        assert_eq!(dag.find_node(1).unwrap().retries, 0);

        // Child node 4 is un-skipped back to Pending!
        assert_eq!(dag.find_node(4).unwrap().status, NodeStatus::Pending);
    }

    #[test]
    fn test_node_tool_binding_and_reset() {
        let mut node = SwarmTaskNode::new(
            10,
            "Linter Node",
            crate::ui::app::ModelRole::Coder,
            "Lint source code",
            vec![],
            "qa",
        ).with_tool("cargo-check", true);

        assert_eq!(node.tool_id.as_deref(), Some("cargo-check"));
        assert!(node.auto_exec_tool);
        assert!(node.tool_output.is_none());

        node.tool_output = Some("error[E0308]: mismatched types".to_string());
        assert!(node.tool_output.is_some());

        // Test presets binding
        let soc_dag = SwarmDag::new(DagPreset::CyberSocGrid);
        let forensics_node = soc_dag.find_node(1).expect("Forensics node exists");
        assert_eq!(forensics_node.tool_id.as_deref(), Some("hexdump"));

        let forge_dag = SwarmDag::new(DagPreset::FullStackForge);
        let qa_node = forge_dag.find_node(4).expect("QA node exists");
        assert_eq!(qa_node.tool_id.as_deref(), Some("cargo-check"));

        // Test dag.reset() clears tool_output
        let mut test_dag = SwarmDag::new(DagPreset::Diamond);
        test_dag.find_node_mut(0).unwrap().tool_output = Some("previous tool output".to_string());
        test_dag.reset();
        assert!(test_dag.find_node(0).unwrap().tool_output.is_none());
    }

    #[test]
    fn test_dynamic_edge_conditions_and_remediation_branching() {
        let mut dag = SwarmDag::new(DagPreset::Custom);
        // Node 0: Build (root)
        let n0 = SwarmTaskNode::new(0, "Build", crate::ui::app::ModelRole::Coder, "Build", vec![], "dev");
        // Node 1: Deploy (triggers OnSuccess of Node 0)
        let n1 = SwarmTaskNode::new(1, "Deploy", crate::ui::app::ModelRole::Planner, "Deploy", vec![0], "ops")
            .with_condition(EdgeCondition::OnSuccess);
        // Node 2: Remediate (triggers OnFailure of Node 0)
        let n2 = SwarmTaskNode::new(2, "Remediate", crate::ui::app::ModelRole::Critic, "Fix", vec![0], "cyber")
            .with_condition(EdgeCondition::OnFailure);

        dag.nodes = vec![n0, n1, n2];

        // Initially Node 0 is ready
        let ready = dag.get_ready_nodes();
        assert_eq!(ready, vec![0]);

        // Case A: Node 0 succeeds -> Node 1 is Ready, Node 2 is Skipped!
        dag.mark_completed(0, "Build ok".to_string(), 1.2);
        let ready = dag.get_ready_nodes();
        assert_eq!(ready, vec![1]);
        assert_eq!(dag.find_node(1).unwrap().status, NodeStatus::Ready);
        assert_eq!(dag.find_node(2).unwrap().status, NodeStatus::Skipped);

        // Case B: Reset and Node 0 fails -> Node 1 is Skipped, Node 2 (Remediation) is Ready!
        dag.reset();
        dag.find_node_mut(0).unwrap().max_retries = 0; // force terminal fail
        dag.record_failure(0, "Compilation syntax error".to_string());
        assert!(dag.find_node(0).unwrap().status.is_failed());

        let ready = dag.get_ready_nodes();
        assert_eq!(ready, vec![2]);
        assert_eq!(dag.find_node(2).unwrap().status, NodeStatus::Ready);
        assert_eq!(dag.find_node(1).unwrap().status, NodeStatus::Skipped);
    }
}
