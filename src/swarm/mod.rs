// Copyright 2026 Sean M. Stow. All rights reserved.
//! Multi-Agent Swarm Intelligence Module.
//! Integrates Directed Acyclic Graph (DAG) task scheduling, parallel asynchronous execution,
//! and bio-inspired Stigmergic Blackboard memory.

pub mod blackboard;
pub mod dag;

pub use blackboard::{BlackboardArtifact, StigmergicBlackboard};
pub use dag::{DagPreset, NodeId, NodeStatus, SwarmDag, SwarmTaskNode};
