// Copyright 2026 Sean M. Stow. All rights reserved.
//! Stigmergic Blackboard Memory for Multi-Agent Swarm Collaboration.
//! Inspired by biological swarm stigmergy (indirect communication via environmental modification).
//! Agents deposit artifacts tagged with domains and pheromone intensity scores.
//! Artifacts undergo temporal decay (evaporation) or reinforcement based on consensus.

use crate::swarm::dag::{SwarmDag, SwarmTaskNode};
use crate::ui::app::ModelRole;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackboardArtifact {
    pub id: String,
    pub author_node_id: usize,
    pub author_name: String,
    pub author_role: ModelRole,
    pub domain: String,
    pub title: String,
    pub content: String,
    pub pheromone_score: f32,
    pub tags: Vec<String>,
    pub created_at_rfc3339: String,
}

impl BlackboardArtifact {
    pub fn new(
        author_node_id: usize,
        author_name: impl Into<String>,
        author_role: ModelRole,
        domain: impl Into<String>,
        title: impl Into<String>,
        content: impl Into<String>,
        initial_pheromone: f32,
        tags: Vec<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            author_node_id,
            author_name: author_name.into(),
            author_role,
            domain: domain.into(),
            title: title.into(),
            content: content.into(),
            pheromone_score: initial_pheromone.clamp(0.1, 20.0),
            tags,
            created_at_rfc3339: Utc::now().to_rfc3339(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StigmergicBlackboard {
    pub artifacts: Vec<BlackboardArtifact>,
    pub evaporation_rate: f32,
}

impl Default for StigmergicBlackboard {
    fn default() -> Self {
        Self::new()
    }
}

impl StigmergicBlackboard {
    pub fn new() -> Self {
        Self {
            artifacts: Vec::new(),
            evaporation_rate: 0.05,
        }
    }

    /// Deposit a new artifact into the stigmergic blackboard
    pub fn deposit(&mut self, artifact: BlackboardArtifact) {
        self.artifacts.push(artifact);
    }

    /// Reinforce an artifact's pheromone score upon verification, audit pass, or user satisfaction
    pub fn reinforce(&mut self, artifact_id: &str, delta: f32) {
        if let Some(art) = self.artifacts.iter_mut().find(|a| a.id == artifact_id) {
            art.pheromone_score = (art.pheromone_score + delta).clamp(0.1, 20.0);
        }
    }

    /// Bio-inspired temporal stigmergic evaporation:
    /// Pheromone trails decay toward baseline 1.0 over time:
    /// tau <- (1 - rho)*tau + rho * 1.0
    pub fn evaporate(&mut self) {
        let rho = self.evaporation_rate.clamp(0.001, 0.5);
        for art in &mut self.artifacts {
            art.pheromone_score = ((1.0 - rho) * art.pheromone_score + rho * 1.0).clamp(0.1, 20.0);
        }
    }

    pub fn clear(&mut self) {
        self.artifacts.clear();
    }

    /// Retrieve artifacts belonging to a specific domain, ordered by pheromone score descending
    pub fn get_domain_artifacts(&self, domain: &str) -> Vec<&BlackboardArtifact> {
        let mut matching: Vec<&BlackboardArtifact> = self
            .artifacts
            .iter()
            .filter(|a| a.domain.eq_ignore_ascii_case(domain))
            .collect();
        matching.sort_by(|a, b| b.pheromone_score.partial_cmp(&a.pheromone_score).unwrap_or(std::cmp::Ordering::Equal));
        matching
    }

    /// Retrieve the highest-pheromone artifacts across all domains
    pub fn get_top_artifacts(&self, limit: usize) -> Vec<&BlackboardArtifact> {
        let mut sorted: Vec<&BlackboardArtifact> = self.artifacts.iter().collect();
        sorted.sort_by(|a, b| b.pheromone_score.partial_cmp(&a.pheromone_score).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(limit);
        sorted
    }

    /// Intelligent context assembly for a node about to execute in the DAG:
    /// Combines the initial user objective, outputs from direct dependency nodes,
    /// and high-pheromone blackboard artifacts from related domains.
    pub fn assemble_context_for_node(&self, node: &SwarmTaskNode, dag: &SwarmDag) -> String {
        let mut ctx = format!("=== INITIAL SWARM OBJECTIVE ===\n{}\n\n", dag.objective);

        // Direct dependency predecessor outputs
        if !node.dependencies.is_empty() {
            ctx.push_str("=== UPSTREAM DEPENDENCY OUTPUTS ===\n");
            for &dep_id in &node.dependencies {
                if let Some(dep_node) = dag.find_node(dep_id) {
                    ctx.push_str(&format!(
                        "--- Predecessor Node {}: {} (`{}`) ---\n{}\n\n",
                        dep_node.id,
                        dep_node.name,
                        dep_node.role.label(),
                        if dep_node.output.is_empty() { "[No output]" } else { &dep_node.output }
                    ));
                }
            }
        }

        // Relevant blackboard artifacts (by domain or top pheromones)
        let relevant_artifacts: Vec<&BlackboardArtifact> = self
            .artifacts
            .iter()
            .filter(|a| {
                // Exclude outputs authored by this node itself
                a.author_node_id != node.id
                    && (a.domain.eq_ignore_ascii_case(&node.domain)
                        || a.tags.iter().any(|t| node.tags.contains(t))
                        || a.pheromone_score >= 1.5)
            })
            .collect();

        if !relevant_artifacts.is_empty() {
            ctx.push_str("=== STIGMERGIC BLACKBOARD MEMORY (High-Pheromone Insights) ===\n");
            for art in relevant_artifacts.iter().take(3) {
                ctx.push_str(&format!(
                    "--- [Artifact: {} | Domain: {} | Pheromone: {:.1}] ---\n{}\n\n",
                    art.title,
                    art.domain,
                    art.pheromone_score,
                    art.content
                ));
            }
        }

        ctx.push_str(&format!(
            "=== YOUR SPECIALIZED ROLE DIRECTIVE ===\nRole: {}\nDomain: {}\nInstructions:\n{}\n",
            node.role.label(),
            node.domain,
            node.directive
        ));

        ctx
    }

    pub fn export_markdown(&self) -> String {
        let mut md = format!(
            "# 🐝 Stigmergic Blackboard Memory Vault\n\nTotal Artifacts: {} | Evaporation Rate: {:.2}\n\n---\n\n",
            self.artifacts.len(),
            self.evaporation_rate
        );

        if self.artifacts.is_empty() {
            md.push_str("*Blackboard is currently empty. Run a Swarm DAG to deposit artifacts.*\n");
            return md;
        }

        for (i, art) in self.artifacts.iter().enumerate() {
            md.push_str(&format!(
                "### {}. {} (🔥 Pheromone: {:.2})\n- **Author:** Node {} · {} (`{}`)\n- **Domain:** `{}` | **Tags:** `{}`\n- **Timestamp:** {}\n\n```\n{}\n```\n\n---\n\n",
                i + 1,
                art.title,
                art.pheromone_score,
                art.author_node_id,
                art.author_name,
                art.author_role.label(),
                art.domain,
                art.tags.join(", "),
                art.created_at_rfc3339,
                art.content
            ));
        }

        md
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stigmergic_blackboard_deposit_and_query() {
        let mut bb = StigmergicBlackboard::new();
        assert_eq!(bb.artifacts.len(), 0);

        let art1 = BlackboardArtifact::new(
            0,
            "Architect",
            ModelRole::Planner,
            "architecture",
            "Core Interfaces",
            "pub trait Engine { fn run(&self); }",
            1.0,
            vec!["spec".into()],
        );
        bb.deposit(art1);

        let art2 = BlackboardArtifact::new(
            1,
            "Coder",
            ModelRole::Coder,
            "backend",
            "Storage Engine",
            "impl Engine for SledEngine { ... }",
            2.5,
            vec!["sled".into()],
        );
        bb.deposit(art2);

        assert_eq!(bb.artifacts.len(), 2);
        let top = bb.get_top_artifacts(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].title, "Storage Engine");

        let arch_artifacts = bb.get_domain_artifacts("architecture");
        assert_eq!(arch_artifacts.len(), 1);
        assert_eq!(arch_artifacts[0].title, "Core Interfaces");
    }

    #[test]
    fn test_stigmergic_reinforce_and_evaporate() {
        let mut bb = StigmergicBlackboard::new();
        let art = BlackboardArtifact::new(
            0,
            "Critic",
            ModelRole::Critic,
            "security",
            "Audit Findings",
            "All buffer boundaries verified.",
            1.0,
            vec!["sec".into()],
        );
        let art_id = art.id.clone();
        bb.deposit(art);

        // Reinforce by 2.0 -> score becomes 3.0
        bb.reinforce(&art_id, 2.0);
        assert!((bb.artifacts[0].pheromone_score - 3.0).abs() < 1e-4);

        // Evaporate: score decays towards 1.0
        bb.evaporation_rate = 0.10;
        bb.evaporate();
        // (1.0 - 0.1)*3.0 + 0.1*1.0 = 2.7 + 0.1 = 2.8
        assert!((bb.artifacts[0].pheromone_score - 2.8).abs() < 1e-3);
    }

    #[test]
    fn test_blackboard_context_assembly() {
        let bb = StigmergicBlackboard::new();
        let mut dag = SwarmDag::new(crate::swarm::dag::DagPreset::Diamond);
        dag.objective = "Build a fast lock-free queue".to_string();
        dag.mark_completed(0, "Architecture: use AtomicPtr and CAS".to_string(), 1.2);

        let node1 = dag.find_node(1).unwrap();
        let ctx = bb.assemble_context_for_node(node1, &dag);

        assert!(ctx.contains("INITIAL SWARM OBJECTIVE"));
        assert!(ctx.contains("Build a fast lock-free queue"));
        assert!(ctx.contains("UPSTREAM DEPENDENCY OUTPUTS"));
        assert!(ctx.contains("Architecture: use AtomicPtr and CAS"));
        assert!(ctx.contains("YOUR SPECIALIZED ROLE DIRECTIVE"));
    }
}
