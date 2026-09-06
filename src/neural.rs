use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// Serializable wrappers for ndarray types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableArray2 {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl From<Array2<f32>> for SerializableArray2 {
    fn from(arr: Array2<f32>) -> Self {
        let (rows, cols) = arr.dim();
        Self {
            data: arr.into_raw_vec(),
            rows,
            cols,
        }
    }
}

impl From<SerializableArray2> for Array2<f32> {
    fn from(s: SerializableArray2) -> Self {
        Array2::from_shape_vec((s.rows, s.cols), s.data).unwrap()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableArray1 {
    pub data: Vec<f32>,
}

impl From<Array1<f32>> for SerializableArray1 {
    fn from(arr: Array1<f32>) -> Self {
        Self {
            data: arr.into_raw_vec(),
        }
    }
}

impl From<SerializableArray1> for Array1<f32> {
    fn from(s: SerializableArray1) -> Self {
        Array1::from_vec(s.data)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfileNetwork {
    pub id: Uuid,
    pub name: String,
    pub input_dim: usize,
    pub hidden_dims: Vec<usize>,
    pub output_dim: usize,
    pub weights: Vec<SerializableArray2>,
    pub biases: Vec<SerializableArray1>,
    pub learning_rate: f32,
    pub experience_buffer: Vec<Experience>,
    pub performance_history: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub state: SerializableArray1,
    pub action: usize,
    pub reward: f32,
    pub next_state: SerializableArray1,
    pub done: bool,
}

impl ModelProfileNetwork {
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>, output_dim: usize) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut prev_dim = input_dim;
        
        let mut rng = rand::thread_rng();
        
        for &hidden_dim in &hidden_dims {
            let w = Array2::from_shape_fn((hidden_dim, prev_dim), |_| {
                rng.gen_range(-0.1..0.1)
            });
            let b = Array1::zeros(hidden_dim);
            weights.push(w.into());
            biases.push(b.into());
            prev_dim = hidden_dim;
        }
        
        let w = Array2::from_shape_fn((output_dim, prev_dim), |_| {
            rng.gen_range(-0.1..0.1)
        });
        let b = Array1::zeros(output_dim);
        weights.push(w.into());
        biases.push(b.into());
        
        Self {
            id: Uuid::new_v4(),
            name: "ModelProfileNN".to_string(),
            input_dim,
            hidden_dims,
            output_dim,
            weights,
            biases,
            learning_rate: 0.01,
            experience_buffer: Vec::new(),
            performance_history: Vec::new(),
        }
    }
    
    fn weights_as_arrays(&self) -> Vec<Array2<f32>> {
        self.weights.iter().cloned().map(|w| w.into()).collect()
    }
    
    fn biases_as_arrays(&self) -> Vec<Array1<f32>> {
        self.biases.iter().cloned().map(|b| b.into()).collect()
    }
    
    fn update_weights(&mut self, weights: Vec<Array2<f32>>, biases: Vec<Array1<f32>>) {
        self.weights = weights.into_iter().map(|w| w.into()).collect();
        self.biases = biases.into_iter().map(|b| b.into()).collect();
    }

    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let weights = self.weights_as_arrays();
        let biases = self.biases_as_arrays();
        let mut x = input.clone();
        
        for i in 0..weights.len() {
            x = weights[i].dot(&x) + &biases[i];
            
            if i < weights.len() - 1 {
                x.mapv_inplace(|v| v.max(0.0));
            }
        }
        
        let max_val = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_x = x.mapv(|v| (v - max_val).exp());
        let sum = exp_x.sum();
        exp_x / sum
    }

    pub fn select_model_profile(&self, context: &Array1<f32>) -> usize {
        let output = self.forward(context);
        output.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn add_experience(&mut self, experience: Experience) {
        self.experience_buffer.push(experience);
        
        if self.experience_buffer.len() > 10000 {
            self.experience_buffer.remove(0);
        }
    }

    pub fn train_step(&mut self) -> Result<f32> {
        if self.experience_buffer.len() < 32 {
            return Ok(0.0);
        }
        
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let batch: Vec<_> = self.experience_buffer.choose_multiple(&mut rng, 32).cloned().collect();
        
        let mut weights = self.weights_as_arrays();
        let mut biases = self.biases_as_arrays();
        let mut total_loss = 0.0;
        
        for exp in &batch {
            let prediction = self.forward(&exp.state.clone().into());
            let mut target = prediction.clone();
            target[exp.action] = exp.reward;
            
            let loss = (&prediction - &target).mapv(|v| v * v).sum();
            total_loss += loss;
            
            let grad = &prediction - &target;
            let lr = self.learning_rate;
            
            let last_idx = weights.len() - 1;
            let hidden_dim = self.hidden_dims.last().copied().unwrap_or(self.input_dim);
            for i in 0..self.output_dim {
                for j in 0..hidden_dim {
                    if j < exp.state.data.len() {
                        weights[last_idx][[i, j]] -= lr * grad[i] * exp.state.data[j].min(1.0).max(-1.0);
                    }
                }
                biases[last_idx][i] -= lr * grad[i];
            }
        }
        
        self.update_weights(weights, biases);
        
        Ok(total_loss / batch.len() as f32)
    }

    pub fn update_performance(&mut self, reward: f32) {
        self.performance_history.push(reward);
        if self.performance_history.len() > 1000 {
            self.performance_history.remove(0);
        }
    }

    pub fn avg_performance(&self) -> f32 {
        if self.performance_history.is_empty() {
            0.0
        } else {
            self.performance_history.iter().sum::<f32>() / self.performance_history.len() as f32
        }
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let data = bincode::serialize(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        let data = std::fs::read(path)?;
        let network: Self = bincode::deserialize(&data)?;
        Ok(network)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveModelSelector {
    pub network: ModelProfileNetwork,
    pub profiles: Vec<ModelProfileInfo>,
    pub context_encoder: ContextEncoder,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfileInfo {
    pub id: Uuid,
    pub name: String,
    pub model_name: String,
    pub system_prompt: String,
    pub temperature: f32,
    pub top_p: f32,
    pub skills: Vec<SkillInfo>,
    pub success_rate: f32,
    pub avg_response_time: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillInfo {
    pub name: String,
    pub proficiency: f32,
    pub category: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEncoder {
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub embeddings: SerializableArray2,
}

impl ContextEncoder {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let embeddings = Array2::from_shape_fn((vocab_size, embedding_dim), |_| {
            rng.gen_range(-0.1..0.1)
        });
        
        Self {
            vocab_size,
            embedding_dim,
            embeddings: embeddings.into(),
        }
    }

    pub fn encode(&self, text: &str) -> Array1<f32> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let embeddings: Array2<f32> = self.embeddings.clone().into();
        let mut vec = Array1::zeros(self.embedding_dim);
        
        for word in &words {
            let hash = Self::hash_word(word) % self.vocab_size;
            vec += &embeddings.row(hash);
        }
        
        if !words.is_empty() {
            vec /= words.len() as f32;
        }
        
        vec
    }

    fn hash_word(word: &str) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        word.hash(&mut hasher);
        hasher.finish() as usize
    }
}

impl AdaptiveModelSelector {
    pub fn new(num_profiles: usize, context_dim: usize) -> Self {
        Self {
            network: ModelProfileNetwork::new(context_dim, vec![128, 64], num_profiles),
            profiles: Vec::new(),
            context_encoder: ContextEncoder::new(10000, context_dim),
        }
    }

    pub fn add_profile(&mut self, profile: ModelProfileInfo) {
        self.profiles.push(profile);
    }

    pub fn select_best_profile(&self, user_input: &str, task_type: &str) -> Option<&ModelProfileInfo> {
        let context = self.build_context(user_input, task_type);
        let profile_idx = self.network.select_model_profile(&context);
        
        self.profiles.get(profile_idx)
    }

    fn build_context(&self, user_input: &str, task_type: &str) -> Array1<f32> {
        let text_embedding = self.context_encoder.encode(user_input);
        let task_embedding = self.context_encoder.encode(task_type);
        
        let mut context = Array1::zeros(self.network.input_dim);
        let half = self.network.input_dim / 2;
        
        for i in 0..half.min(text_embedding.len()) {
            context[i] = text_embedding[i];
        }
        for i in 0..(self.network.input_dim - half).min(task_embedding.len()) {
            context[half + i] = task_embedding[i];
        }
        
        context
    }

    pub fn record_interaction(&mut self, user_input: &str, task_type: &str, profile_idx: usize, success: bool, response_time: f32) {
        let context = self.build_context(user_input, task_type);
        let reward = if success { 1.0 } else { -0.5 };
        
        let time_reward = (1.0 - (response_time / 30.0).min(1.0)) * 0.5;
        let total_reward = reward + time_reward;
        
        let experience = Experience {
            state: context.clone().into(),
            action: profile_idx,
            reward: total_reward,
            next_state: context.into(),
            done: true,
        };
        
        self.network.add_experience(experience);
        self.network.update_performance(total_reward);
        
        if let Some(profile) = self.profiles.get_mut(profile_idx) {
            profile.success_rate = 0.9 * profile.success_rate + 0.1 * if success { 1.0 } else { 0.0 };
            profile.avg_response_time = 0.9 * profile.avg_response_time + 0.1 * response_time;
        }
        
        if self.network.experience_buffer.len() % 50 == 0 {
            let _ = self.network.train_step();
        }
    }

    pub fn learn_skill(&mut self, profile_idx: usize, skill_name: String, skill_category: String) {
        if let Some(profile) = self.profiles.get_mut(profile_idx) {
            if !profile.skills.iter().any(|s| s.name == skill_name) {
                profile.skills.push(SkillInfo {
                    name: skill_name.clone(),
                    proficiency: 0.1,
                    category: skill_category,
                });
            } else if let Some(skill) = profile.skills.iter_mut().find(|s| s.name == skill_name) {
                skill.proficiency = (skill.proficiency + 0.1).min(1.0);
            }
        }
    }
}
