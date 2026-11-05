use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EdgeConfidence {
    Extracted,                  // 1.0 - Direct from code/AST/SQL
    Inferred(f32),              // 0.0 - 1.0 - Derived from LLM/Docs
    Statistical { p_val: f32 }, // From causal discovery algorithms
    Ambiguous,                  // Conflicting sources
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    Treatment,
    Outcome,
    Confounder,
    Mediator,
    Instrumental,
    General,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalNode {
    pub name: String,
    pub node_type: NodeType,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEdge {
    pub confidence: EdgeConfidence,
    pub provenance: Option<String>,
    pub expression: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalGraphExport {
    pub nodes: Vec<CausalNode>,
    pub edges: Vec<(String, String, CausalEdge)>,
}

pub struct CausalGraph {
    pub graph: DiGraph<CausalNode, CausalEdge>,
    pub node_indices: HashMap<String, NodeIndex>,
}

impl Default for CausalGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_indices: HashMap::new(),
        }
    }

    pub fn add_or_get_node(&mut self, name: &str, node_type: NodeType) -> NodeIndex {
        if let Some(&idx) = self.node_indices.get(name) {
            return idx;
        }
        let node = CausalNode {
            name: name.to_string(),
            node_type,
            description: None,
        };
        let idx = self.graph.add_node(node);
        self.node_indices.insert(name.to_string(), idx);
        idx
    }

    pub fn add_edge(
        &mut self,
        src: &str,
        dst: &str,
        confidence: EdgeConfidence,
        provenance: Option<String>,
        expression: Option<String>,
    ) {
        let u = self.add_or_get_node(src, NodeType::General);
        let v = self.add_or_get_node(dst, NodeType::General);
        self.graph.add_edge(
            u,
            v,
            CausalEdge {
                confidence,
                provenance,
                expression,
            },
        );
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn export_structure(&self) -> CausalGraphExport {
        let nodes: Vec<CausalNode> = self.graph.node_weights().cloned().collect();
        let mut edges = Vec::new();

        for edge_ref in self.graph.edge_references() {
            let src = &self.graph[edge_ref.source()].name;
            let dst = &self.graph[edge_ref.target()].name;
            edges.push((src.clone(), dst.clone(), edge_ref.weight().clone()));
        }

        CausalGraphExport { nodes, edges }
    }
}
