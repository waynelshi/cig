use crate::dsep::DSeparationEngine;
use crate::graph::CausalGraph;
use petgraph::graph::NodeIndex;
use petgraph::visit::{Dfs, EdgeRef};
use petgraph::Direction;
use std::collections::HashSet;

pub struct CausalCriteriaEngine<'a> {
    causal_graph: &'a CausalGraph,
}

impl<'a> CausalCriteriaEngine<'a> {
    pub fn new(causal_graph: &'a CausalGraph) -> Self {
        Self { causal_graph }
    }

    /// Computes all descendants of a given node in the graph
    fn get_descendants(&self, node: NodeIndex) -> HashSet<NodeIndex> {
        let mut descendants = HashSet::new();
        let mut dfs = Dfs::new(&self.causal_graph.graph, node);
        while let Some(nx) = dfs.next(&self.causal_graph.graph) {
            if nx != node {
                descendants.insert(nx);
            }
        }
        descendants
    }

    /// Checks if a candidate set Z satisfies the Backdoor Criterion for (X, Y)
    pub fn is_valid_backdoor_set(&self, x: &str, y: &str, z_set: &[String]) -> bool {
        let x_idx = match self.causal_graph.node_indices.get(x) {
            Some(&i) => i,
            None => return false,
        };
        let y_idx = match self.causal_graph.node_indices.get(y) {
            Some(&i) => i,
            None => return false,
        };

        let z_indices: HashSet<NodeIndex> = z_set
            .iter()
            .filter_map(|n| self.causal_graph.node_indices.get(n).copied())
            .collect();

        // 1. No node in Z can be X, Y, or a descendant of X
        if z_indices.contains(&x_idx) || z_indices.contains(&y_idx) {
            return false;
        }
        let x_descendants = self.get_descendants(x_idx);
        if z_indices.iter().any(|z| x_descendants.contains(z)) {
            return false;
        }

        // 2. Construct G_underline_X (remove outgoing edges from X)
        let mut g_underline_x = self.causal_graph.graph.clone();
        let outgoing: Vec<_> = g_underline_x
            .edges_directed(x_idx, Direction::Outgoing)
            .map(|e| e.id())
            .collect();
        for e in outgoing {
            g_underline_x.remove_edge(e);
        }

        let temp_subgraph = CausalGraph {
            graph: g_underline_x,
            node_indices: self.causal_graph.node_indices.clone(),
        };

        // 3. Test d-separation in G_underline_X
        let dsep = DSeparationEngine::new(&temp_subgraph);
        dsep.is_d_separated(&[x.to_string()], &[y.to_string()], z_set)
    }

    /// Finds all valid backdoor adjustment sets (sorted by size: minimal first)
    pub fn find_backdoor_adjustment_sets(&self, x: &str, y: &str) -> Vec<Vec<String>> {
        let x_idx = match self.causal_graph.node_indices.get(x) {
            Some(&i) => i,
            None => return vec![],
        };
        let y_idx = match self.causal_graph.node_indices.get(y) {
            Some(&i) => i,
            None => return vec![],
        };

        let x_descendants = self.get_descendants(x_idx);

        // Candidate nodes cannot be X, Y, or descendants of X
        let candidate_nodes: Vec<String> = self
            .causal_graph
            .node_indices
            .iter()
            .filter(|(&ref _name, &idx)| {
                idx != x_idx && idx != y_idx && !x_descendants.contains(&idx)
            })
            .map(|(name, _)| name.clone())
            .collect();

        let mut valid_sets = Vec::new();
        let n = candidate_nodes.len();

        // Search power set (combinations) by increasing size
        for k in 0..=n {
            let mut combinations = Vec::new();
            Self::generate_combinations(&candidate_nodes, k, 0, &mut Vec::new(), &mut combinations);

            for subset in combinations {
                if self.is_valid_backdoor_set(x, y, &subset) {
                    valid_sets.push(subset);
                }
            }
        }

        valid_sets
    }

    /// Checks if a candidate mediator set M satisfies the Frontdoor Criterion for (X, Y)
    pub fn is_valid_frontdoor_set(&self, x: &str, y: &str, m_set: &[String]) -> bool {
        let x_idx = match self.causal_graph.node_indices.get(x) {
            Some(&i) => i,
            None => return false,
        };
        let y_idx = match self.causal_graph.node_indices.get(y) {
            Some(&i) => i,
            None => return false,
        };

        if m_set.is_empty() {
            return false;
        }

        let m_indices: HashSet<NodeIndex> = m_set
            .iter()
            .filter_map(|n| self.causal_graph.node_indices.get(n).copied())
            .collect();

        if m_indices.contains(&x_idx) || m_indices.contains(&y_idx) {
            return false;
        }

        // Condition 1: M intercepts all directed paths from X to Y
        // (In G \ M, no directed path exists from X to Y)
        let mut g_no_m = self.causal_graph.graph.clone();
        for &m_node in &m_indices {
            // Remove incoming & outgoing edges of M
            let edges: Vec<_> = g_no_m.edges(m_node).map(|e| e.id()).collect();
            for e in edges {
                g_no_m.remove_edge(e);
            }
        }

        let mut dfs_path = Dfs::new(&g_no_m, x_idx);
        let mut path_found = false;
        while let Some(nx) = dfs_path.next(&g_no_m) {
            if nx == y_idx {
                path_found = true;
                break;
            }
        }
        if path_found {
            return false; // A path exists that bypasses M
        }

        // Condition 2: No unblocked backdoor paths from X to M
        // (Empty set d-separates X and M in G_underline_X)
        let mut g_underline_x = self.causal_graph.graph.clone();
        let outgoing_x: Vec<_> = g_underline_x
            .edges_directed(x_idx, Direction::Outgoing)
            .map(|e| e.id())
            .collect();
        for e in outgoing_x {
            g_underline_x.remove_edge(e);
        }

        let temp_gx = CausalGraph {
            graph: g_underline_x,
            node_indices: self.causal_graph.node_indices.clone(),
        };
        let dsep_x = DSeparationEngine::new(&temp_gx);
        if !dsep_x.is_d_separated(&[x.to_string()], m_set, &[]) {
            return false;
        }

        // Condition 3: All backdoor paths from M to Y are blocked by {X}
        // 3a. X cannot be a descendant of any node in M
        for &m_node in &m_indices {
            if self.get_descendants(m_node).contains(&x_idx) {
                return false;
            }
        }

        // 3b. {X} d-separates M and Y in G_underline_M
        let mut g_underline_m = self.causal_graph.graph.clone();
        for &m_node in &m_indices {
            let outgoing_m: Vec<_> = g_underline_m
                .edges_directed(m_node, Direction::Outgoing)
                .map(|e| e.id())
                .collect();
            for e in outgoing_m {
                g_underline_m.remove_edge(e);
            }
        }

        let temp_gm = CausalGraph {
            graph: g_underline_m,
            node_indices: self.causal_graph.node_indices.clone(),
        };
        let dsep_m = DSeparationEngine::new(&temp_gm);
        dsep_m.is_d_separated(m_set, &[y.to_string()], &[x.to_string()])
    }

    /// Finds all valid frontdoor adjustment sets
    pub fn find_frontdoor_adjustment_sets(&self, x: &str, y: &str) -> Vec<Vec<String>> {
        let x_idx = match self.causal_graph.node_indices.get(x) {
            Some(&i) => i,
            None => return vec![],
        };
        let y_idx = match self.causal_graph.node_indices.get(y) {
            Some(&i) => i,
            None => return vec![],
        };

        let candidate_nodes: Vec<String> = self
            .causal_graph
            .node_indices
            .iter()
            .filter(|(_, &idx)| idx != x_idx && idx != y_idx)
            .map(|(name, _)| name.clone())
            .collect();

        let mut valid_sets = Vec::new();
        let n = candidate_nodes.len();

        for k in 1..=n {
            let mut combinations = Vec::new();
            Self::generate_combinations(&candidate_nodes, k, 0, &mut Vec::new(), &mut combinations);

            for subset in combinations {
                if self.is_valid_frontdoor_set(x, y, &subset) {
                    valid_sets.push(subset);
                }
            }
        }

        valid_sets
    }

    /// Checks if candidate node Z is a valid Instrumental Variable for (X -> Y)
    pub fn is_valid_instrument(&self, z: &str, x: &str, y: &str) -> bool {
        let z_idx = match self.causal_graph.node_indices.get(z) {
            Some(&i) => i,
            None => return false,
        };
        let x_idx = match self.causal_graph.node_indices.get(x) {
            Some(&i) => i,
            None => return false,
        };
        let y_idx = match self.causal_graph.node_indices.get(y) {
            Some(&i) => i,
            None => return false,
        };

        if z_idx == x_idx || z_idx == y_idx {
            return false;
        }

        // Condition 1: Relevance - Directed path exists from Z to X
        let mut dfs_relevance = Dfs::new(&self.causal_graph.graph, z_idx);
        let mut reaches_x = false;
        while let Some(nx) = dfs_relevance.next(&self.causal_graph.graph) {
            if nx == x_idx {
                reaches_x = true;
                break;
            }
        }
        if !reaches_x {
            return false;
        }

        // Condition 2: Exclusion Restriction - In G_underline_X (intervening on X),
        // Z and Y are d-separated by empty set (no paths between Z and Y that bypass X).
        let mut g_underline_x = self.causal_graph.graph.clone();
        let outgoing_x: Vec<_> = g_underline_x
            .edges_directed(x_idx, Direction::Outgoing)
            .map(|e| e.id())
            .collect();
        for e in outgoing_x {
            g_underline_x.remove_edge(e);
        }

        let temp_gx = CausalGraph {
            graph: g_underline_x,
            node_indices: self.causal_graph.node_indices.clone(),
        };
        let dsep_x = DSeparationEngine::new(&temp_gx);
        if !dsep_x.is_d_separated(&[z.to_string()], &[y.to_string()], &[]) {
            return false;
        }

        // Condition 3: Exogeneity - In G_overline_Z (removing outgoing edges from Z),
        // Z and Y are d-separated by empty set (no confounding backdoor into Z).
        let mut g_overline_z = self.causal_graph.graph.clone();
        let outgoing_z: Vec<_> = g_overline_z
            .edges_directed(z_idx, Direction::Outgoing)
            .map(|e| e.id())
            .collect();
        for e in outgoing_z {
            g_overline_z.remove_edge(e);
        }

        let temp_gz = CausalGraph {
            graph: g_overline_z,
            node_indices: self.causal_graph.node_indices.clone(),
        };
        let dsep_z = DSeparationEngine::new(&temp_gz);
        dsep_z.is_d_separated(&[z.to_string()], &[y.to_string()], &[])
    }

    /// Finds all valid single instrumental variables for (X -> Y)
    pub fn find_instrumental_variables(&self, x: &str, y: &str) -> Vec<String> {
        let mut valid_instruments = Vec::new();
        for (name, _) in &self.causal_graph.node_indices {
            if self.is_valid_instrument(name, x, y) {
                valid_instruments.push(name.clone());
            }
        }
        valid_instruments.sort();
        valid_instruments
    }

    fn generate_combinations(
        items: &[String],
        k: usize,
        start: usize,
        current: &mut Vec<String>,
        results: &mut Vec<Vec<String>>,
    ) {
        if current.len() == k {
            results.push(current.clone());
            return;
        }
        for i in start..items.len() {
            current.push(items[i].clone());
            Self::generate_combinations(items, k, i + 1, current, results);
            current.pop();
        }
    }
}