use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryType {
    Episodic,
    Semantic,
    Procedural,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryScope {
    Private,
    Team,
    Org,
    Session,
}

impl std::fmt::Display for MemoryScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryScope::Private => write!(f, "private"),
            MemoryScope::Team => write!(f, "team"),
            MemoryScope::Org => write!(f, "org"),
            MemoryScope::Session => write!(f, "session"),
        }
    }
}

impl std::str::FromStr for MemoryScope {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "private" => Ok(MemoryScope::Private),
            "team" => Ok(MemoryScope::Team),
            "org" => Ok(MemoryScope::Org),
            "session" => Ok(MemoryScope::Session),
            _ => Err(format!("unknown scope: {s}")),
        }
    }
}

impl std::fmt::Display for MemoryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryType::Episodic => write!(f, "episodic"),
            MemoryType::Semantic => write!(f, "semantic"),
            MemoryType::Procedural => write!(f, "procedural"),
        }
    }
}

impl std::str::FromStr for MemoryType {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "episodic" => Ok(MemoryType::Episodic),
            "semantic" => Ok(MemoryType::Semantic),
            "procedural" => Ok(MemoryType::Procedural),
            _ => Err(format!("unknown memory type: {s}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUnit {
    pub id: Uuid,
    pub content: String,
    pub memory_type: MemoryType,
    pub agent_id: String,
    pub scope: MemoryScope,
    pub importance: f64,
    pub embedding: Option<Vec<f32>>,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub accessed_at: DateTime<Utc>,
    pub access_count: i64,
    pub decay_score: f64,
    pub team_id: Option<String>,
}

impl MemoryUnit {
    pub fn effective_importance(&self) -> f64 {
        self.importance * self.decay_score
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticTriple {
    pub id: Uuid,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub agent_id: String,
    pub confidence: f64,
    pub valid_from: DateTime<Utc>,
    pub valid_until: Option<DateTime<Utc>>,
    pub source_episodes: Vec<String>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureStep {
    pub action: String,
    pub tool: Option<String>,
    pub parameters: serde_json::Value,
    pub expected_output: Option<String>,
    pub fallback_action: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureTemplate {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub steps: Vec<ProcedureStep>,
    pub agent_id: String,
    pub success_rate: f64,
    pub execution_count: i64,
    pub source_episodes: Vec<String>,
    pub applicable_contexts: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Lightweight row returned by EpisodicStore::scan_batch for decay processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayScanRow {
    pub id: Uuid,
    pub accessed_at: DateTime<Utc>,
    pub decay_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn sample_memory_unit() -> MemoryUnit {
        MemoryUnit {
            id: Uuid::new_v4(),
            content: "Test memory content".to_string(),
            memory_type: MemoryType::Episodic,
            agent_id: "agent-1".to_string(),
            scope: MemoryScope::Private,
            importance: 0.75,
            embedding: Some(vec![0.1, 0.2, 0.3]),
            metadata: serde_json::json!({"key": "value"}),
            created_at: Utc::now(),
            accessed_at: Utc::now(),
            access_count: 3,
            decay_score: 0.9,
            team_id: None,
        }
    }

    fn sample_triple() -> SemanticTriple {
        SemanticTriple {
            id: Uuid::new_v4(),
            subject: "Alice".to_string(),
            predicate: "works_at".to_string(),
            object: "Acme Corp".to_string(),
            agent_id: "agent-1".to_string(),
            confidence: 0.95,
            valid_from: Utc::now(),
            valid_until: None,
            source_episodes: vec!["ep-1".to_string()],
            metadata: serde_json::json!({}),
        }
    }

    fn sample_procedure() -> ProcedureTemplate {
        ProcedureTemplate {
            id: Uuid::new_v4(),
            name: "data_ingestion".to_string(),
            description: "Ingest data from source".to_string(),
            steps: vec![ProcedureStep {
                action: "fetch".to_string(),
                tool: Some("http_client".to_string()),
                parameters: serde_json::json!({"url": "https://example.com"}),
                expected_output: Some("json_data".to_string()),
                fallback_action: Some("retry".to_string()),
            }],
            agent_id: "agent-1".to_string(),
            success_rate: 0.85,
            execution_count: 10,
            source_episodes: vec![],
            applicable_contexts: vec!["data_pipeline".to_string()],
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    #[test]
    fn memory_unit_serde_roundtrip() {
        let unit = sample_memory_unit();
        let json = serde_json::to_string(&unit).expect("serialize");
        let back: MemoryUnit = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(unit.id, back.id);
        assert_eq!(unit.content, back.content);
        assert_eq!(unit.memory_type, back.memory_type);
        assert_eq!(unit.scope, back.scope);
        assert_eq!(unit.importance, back.importance);
        assert_eq!(unit.embedding, back.embedding);
        assert_eq!(unit.access_count, back.access_count);
    }

    #[test]
    fn semantic_triple_serde_roundtrip() {
        let triple = sample_triple();
        let json = serde_json::to_string(&triple).expect("serialize");
        let back: SemanticTriple = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(triple.id, back.id);
        assert_eq!(triple.subject, back.subject);
        assert_eq!(triple.predicate, back.predicate);
        assert_eq!(triple.object, back.object);
        assert_eq!(triple.confidence, back.confidence);
        assert!(back.valid_until.is_none());
    }

    #[test]
    fn procedure_template_serde_roundtrip() {
        let proc = sample_procedure();
        let json = serde_json::to_string(&proc).expect("serialize");
        let back: ProcedureTemplate = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(proc.id, back.id);
        assert_eq!(proc.name, back.name);
        assert_eq!(proc.steps.len(), back.steps.len());
        assert_eq!(proc.steps[0].action, back.steps[0].action);
        assert_eq!(proc.success_rate, back.success_rate);
    }

    #[test]
    fn memory_scope_display_and_parse() {
        for scope in [
            MemoryScope::Private,
            MemoryScope::Team,
            MemoryScope::Org,
            MemoryScope::Session,
        ] {
            let s = scope.to_string();
            let parsed: MemoryScope = s.parse().expect("parse scope");
            assert_eq!(scope, parsed);
        }
    }

    #[test]
    fn memory_type_display_and_parse() {
        for mt in [
            MemoryType::Episodic,
            MemoryType::Semantic,
            MemoryType::Procedural,
        ] {
            let s = mt.to_string();
            let parsed: MemoryType = s.parse().expect("parse type");
            assert_eq!(mt, parsed);
        }
    }

    #[test]
    fn effective_importance_product() {
        let mut unit = sample_memory_unit();
        unit.importance = 0.8;
        unit.decay_score = 0.5;
        assert!((unit.effective_importance() - 0.4).abs() < 1e-10);
    }

    #[test]
    fn decay_scan_row_serde_roundtrip() {
        let row = DecayScanRow {
            id: Uuid::new_v4(),
            accessed_at: Utc::now(),
            decay_score: 0.7,
        };
        let json = serde_json::to_string(&row).expect("serialize");
        let back: DecayScanRow = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(row.id, back.id);
        assert_eq!(row.decay_score, back.decay_score);
    }
}
