use crate::graph::CausalGraph;
use petgraph::graph::NodeIndex;
use petgraph::visit::{Dfs, EdgeRef};
use petgraph::Direction;
use std::collections::{HashSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TraversalDir {
    Incoming, // Arrived via -> Node
    Outgoing, // Arrived via <- Node
}

pub struct DSeparationEngine<'a> {
    causal_graph: &'a CausalGraph,
}

impl<'a> DSeparationEngine<'a> {
    pub fn new(causal_graph: &'a CausalGraph) -> Self {
        Self { causal_graph }
    }

    /// Computes the set of all descendants for every node in the graph.
    fn compute_descendants(&self) -> Vec<HashSet<NodeIndex>> {
        let n = self.causal_graph.graph.node_count();
        let mut descendants = vec![HashSet::new(); n];

        for i in 0..n {
            let start = NodeIndex::new(i);
            let mut dfs = Dfs::new(&self.causal_graph.graph, start);
            while let Some(nx) = dfs.next(&self.causal_graph.graph) {
                if nx != start {
                    descendants[i].insert(nx);
                }
            }
        }
        descendants
    }

    /// Checks whether set X and set Y are d-separated by conditioning set Z.
    pub fn is_d_separated(
        &self,
        x_names: &[String],
        y_names: &[String],
        z_names: &[String],
    ) -> bool {
        let x_set: HashSet<NodeIndex> = x_names
            .iter()
            .filter_map(|n| self.causal_graph.node_indices.get(n).copied())
            .collect();
        let y_set: HashSet<NodeIndex> = y_names
            .iter()
            .filter_map(|n| self.causal_graph.node_indices.get(n).copied())
            .collect();
        let z_set: HashSet<NodeIndex> = z_names
            .iter()
            .filter_map(|n| self.causal_graph.node_indices.get(n).copied())
            .collect();

        if x_set.is_empty() || y_set.is_empty() {
            return true;
        }

        // Precompute descendant sets to evaluate colliders with conditioned descendants
        let all_descendants = self.compute_descendants();

        // A node activates a collider if the node itself is in Z, or any of its descendants is in Z
        let is_collider_active = |node: NodeIndex| -> bool {
            if z_set.contains(&node) {
                return true;
            }
            all_descendants[node.index()]
                .iter()
                .any(|desc| z_set.contains(desc))
        };

        // Queue holds: (current_node, arrival_direction)
        let mut queue: VecDeque<(NodeIndex, TraversalDir)> = VecDeque::new();
        let mut visited: HashSet<(NodeIndex, TraversalDir)> = HashSet::new();

        // Initialize queue from all nodes in X
        for &x in &x_set {
            queue.push_back((x, TraversalDir::Outgoing));
            visited.insert((x, TraversalDir::Outgoing));

            queue.push_back((x, TraversalDir::Incoming));
            visited.insert((x, TraversalDir::Incoming));
        }

        while let Some((curr, dir)) = queue.pop_front() {
            if y_set.contains(&curr) && !x_set.contains(&curr) {
                // Found an active path to Y -> Not d-separated
                return false;
            }

            let g = &self.causal_graph.graph;

            match dir {
                TraversalDir::Incoming => {
                    // Case 1: Traversal continues as non-collider (curr -> next)
                    // Allowed ONLY IF curr is NOT in Z
                    if !z_set.contains(&curr) {
                        for edge in g.edges_directed(curr, Direction::Outgoing) {
                            let next = edge.target();
                            if !visited.contains(&(next, TraversalDir::Incoming)) {
                                visited.insert((next, TraversalDir::Incoming));
                                queue.push_back((next, TraversalDir::Incoming));
                            }
                        }
                    }

                    // Case 2: Traversal continues as collider (curr <- next)
                    // Allowed ONLY IF curr or a descendant of curr is in Z
                    if is_collider_active(curr) {
                        for edge in g.edges_directed(curr, Direction::Incoming) {
                            let next = edge.source();
                            if !visited.contains(&(next, TraversalDir::Outgoing)) {
                                visited.insert((next, TraversalDir::Outgoing));
                                queue.push_back((next, TraversalDir::Outgoing));
                            }
                        }
                    }
                }
                TraversalDir::Outgoing => {
                    // Arrived at curr via `prev <- curr` (or starting at X)
                    // Allowed ONLY IF curr is NOT in Z
                    if !z_set.contains(&curr) {
                        // curr -> next
                        for edge in g.edges_directed(curr, Direction::Outgoing) {
                            let next = edge.target();
                            if !visited.contains(&(next, TraversalDir::Incoming)) {
                                visited.insert((next, TraversalDir::Incoming));
                                queue.push_back((next, TraversalDir::Incoming));
                            }
                        }
                        // curr <- next
                        for edge in g.edges_directed(curr, Direction::Incoming) {
                            let next = edge.source();
                            if !visited.contains(&(next, TraversalDir::Outgoing)) {
                                visited.insert((next, TraversalDir::Outgoing));
                                queue.push_back((next, TraversalDir::Outgoing));
                            }
                        }
                    }
                }
            }
        }

        true
    }
}
