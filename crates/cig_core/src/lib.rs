pub mod criteria;
pub mod dsep;
pub mod graph;

use criteria::CausalCriteriaEngine;
use dsep::DSeparationEngine;
use graph::{CausalGraph, EdgeConfidence};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyclass]
pub struct PyCausalEngine {
    inner: CausalGraph,
}

#[pymethods]
impl PyCausalEngine {
    #[new]
    fn new() -> Self {
        PyCausalEngine {
            inner: CausalGraph::new(),
        }
    }

    #[pyo3(signature = (source, target, provenance=None, expression=None))]
    fn add_extracted_edge(
        &mut self,
        source: &str,
        target: &str,
        provenance: Option<String>,
        expression: Option<String>,
    ) {
        self.inner.add_edge(
            source,
            target,
            EdgeConfidence::Extracted,
            provenance,
            expression,
        );
    }

    #[pyo3(signature = (source, target, confidence, provenance=None, expression=None))]
    fn add_inferred_edge(
        &mut self,
        source: &str,
        target: &str,
        confidence: f32,
        provenance: Option<String>,
        expression: Option<String>,
    ) {
        self.inner.add_edge(
            source,
            target,
            EdgeConfidence::Inferred(confidence),
            provenance,
            expression,
        );
    }

    /// Ingest a list of Pydantic ExtractedEdge objects or dicts directly using PyO3 0.22 Bound API
    fn insert_extracted_edges(&mut self, edges: &Bound<'_, PyList>) -> PyResult<usize> {
        let mut count = 0;
        for item in edges.iter() {
            let (source, target, provenance, expression): (
                String,
                String,
                Option<String>,
                Option<String>,
            ) = if let Ok(dict) = item.downcast::<PyDict>() {
                let src = dict
                    .get_item("source")?
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing source"))?
                    .extract::<String>()?;
                let tgt = dict
                    .get_item("target")?
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing target"))?
                    .extract::<String>()?;
                let prov = dict
                    .get_item("provenance")?
                    .map(|v| v.extract::<String>().ok())
                    .flatten();
                let expr = dict
                    .get_item("expression")?
                    .map(|v| v.extract::<String>().ok())
                    .flatten();
                (src, tgt, prov, expr)
            } else {
                let src = item.getattr("source")?.extract::<String>()?;
                let tgt = item.getattr("target")?.extract::<String>()?;
                let prov = item.getattr("provenance")?.extract::<Option<String>>()?;
                let expr = item.getattr("expression")?.extract::<Option<String>>()?;
                (src, tgt, prov, expr)
            };

            self.inner.add_edge(
                &source,
                &target,
                EdgeConfidence::Extracted,
                provenance,
                expression,
            );
            count += 1;
        }
        Ok(count)
    }

    /// Ingest a list of Pydantic InferredEdge objects or dicts directly
    fn insert_inferred_edges(&mut self, edges: &Bound<'_, PyList>) -> PyResult<usize> {
        let mut count = 0;
        for item in edges.iter() {
            let (source, target, confidence, provenance, expression): (
                String,
                String,
                f32,
                Option<String>,
                Option<String>,
            ) = if let Ok(dict) = item.downcast::<PyDict>() {
                let src = dict
                    .get_item("source")?
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing source"))?
                    .extract::<String>()?;
                let tgt = dict
                    .get_item("target")?
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing target"))?
                    .extract::<String>()?;
                let conf = dict
                    .get_item("confidence")?
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing confidence"))?
                    .extract::<f32>()?;
                let prov = dict
                    .get_item("provenance")?
                    .map(|v| v.extract::<String>().ok())
                    .flatten();
                let expr = dict
                    .get_item("rationale")?
                    .map(|v| v.extract::<String>().ok())
                    .flatten();
                (src, tgt, conf, prov, expr)
            } else {
                let src = item.getattr("source")?.extract::<String>()?;
                let tgt = item.getattr("target")?.extract::<String>()?;
                let conf = item.getattr("confidence")?.extract::<f32>()?;
                let prov = item.getattr("provenance")?.extract::<Option<String>>()?;
                let expr = item.getattr("rationale")?.extract::<Option<String>>()?;
                (src, tgt, conf, prov, expr)
            };

            self.inner.add_edge(
                &source,
                &target,
                EdgeConfidence::Inferred(confidence),
                provenance,
                expression,
            );
            count += 1;
        }
        Ok(count)
    }

    fn is_d_separated(&self, x: Vec<String>, y: Vec<String>, z: Vec<String>) -> bool {
        let engine = DSeparationEngine::new(&self.inner);
        engine.is_d_separated(&x, &y, &z)
    }

    fn is_valid_backdoor_set(&self, treatment: &str, outcome: &str, z_set: Vec<String>) -> bool {
        let engine = CausalCriteriaEngine::new(&self.inner);
        engine.is_valid_backdoor_set(treatment, outcome, &z_set)
    }

    fn find_backdoor_adjustment_sets(&self, treatment: &str, outcome: &str) -> Vec<Vec<String>> {
        let engine = CausalCriteriaEngine::new(&self.inner);
        engine.find_backdoor_adjustment_sets(treatment, outcome)
    }

    fn is_valid_frontdoor_set(&self, treatment: &str, outcome: &str, m_set: Vec<String>) -> bool {
        let engine = CausalCriteriaEngine::new(&self.inner);
        engine.is_valid_frontdoor_set(treatment, outcome, &m_set)
    }

    fn find_frontdoor_adjustment_sets(&self, treatment: &str, outcome: &str) -> Vec<Vec<String>> {
        let engine = CausalCriteriaEngine::new(&self.inner);
        engine.find_frontdoor_adjustment_sets(treatment, outcome)
    }

    fn is_valid_instrument(&self, instrument: &str, treatment: &str, outcome: &str) -> bool {
        let engine = CausalCriteriaEngine::new(&self.inner);
        engine.is_valid_instrument(instrument, treatment, outcome)
    }

    fn find_instrumental_variables(&self, treatment: &str, outcome: &str) -> Vec<String> {
        let engine = CausalCriteriaEngine::new(&self.inner);
        engine.find_instrumental_variables(treatment, outcome)
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.node_count()
    }

    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.edge_count()
    }

    fn export_json(&self) -> PyResult<String> {
        let export = self.inner.export_structure();
        serde_json::to_string_pretty(&export)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
}

#[pymodule]
fn _cig_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCausalEngine>()?;
    Ok(())
}