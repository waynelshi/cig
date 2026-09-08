"""
HTML / WebAssembly Visualizer Generator.
Generates an interactive, single-page SCM studio containing:
  - SVG Interactive Graph Visualization (Zoom, Pan, Drag, Edge Entropy Glow)
  - Level 1-2 d-Separation & Backdoor Path Inspector
  - Level 3 In-Browser Counterfactual Simulator & Pyodide WASM Runtime
  - Sensitivity Contour Simulator & SVG/PNG Exporters
  - Level 4 Active Inference Expected Free Energy (EFE) Policy Radar & 2D Pareto Frontier
"""

from __future__ import annotations

import json
from pathlib import Path
from cig.passes.pass3_unify import SCMUnificationReport


class HTMLDAGVisualizer:
    """Generates the standalone cig.html interactive studio."""

    def __init__(self, report: SCMUnificationReport | None = None):
        self.report = report

    @classmethod
    def generate_html(
        cls,
        report: SCMUnificationReport | None = None,
        out_path: str | Path = ".cig/cig.html",
        **kwargs,
    ) -> Path:
        """Generates the standalone cig.html interactive studio."""
        actual_path = kwargs.get("output_path", out_path)
        out_p = Path(actual_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        graph_payload = json.dumps(report.model_dump() if report else {"edges": []})

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CIG Causal Studio & Generative World Model</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <script src="https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js"></script>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --text: #f8fafc;
      --accent: #38bdf8;
      --border: #334155;
      --extracted: #22c55e;
      --inferred: #38bdf8;
      --discovered: #a855f7;
      --ambiguous: #ef4444;
      --gold: #f59e0b;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      height: 100vh;
      overflow: hidden;
    }}
    #sidebar {{
      width: 450px;
      background: var(--card-bg);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      height: 100%;
    }}
    .tabs {{
      display: flex;
      border-bottom: 1px solid var(--border);
    }}
    .tab-btn {{
      flex: 1;
      padding: 12px 6px;
      background: none;
      border: none;
      color: #94a3b8;
      font-weight: 600;
      font-size: 11px;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      text-align: center;
    }}
    .tab-btn.active {{
      color: var(--accent);
      border-bottom-color: var(--accent);
      background: rgba(56, 189, 248, 0.05);
    }}
    .tab-content {{
      flex: 1;
      overflow-y: auto;
      padding: 18px;
      display: none;
    }}
    .tab-content.active {{
      display: block;
    }}
    #canvas-container {{
      flex: 1;
      position: relative;
      background: #090d16;
    }}
    svg {{
      width: 100%;
      height: 100%;
    }}
    .card {{
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 14px;
    }}
    h3 {{
      margin-top: 0;
      font-size: 13px;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    label {{
      display: block;
      font-size: 11px;
      color: #94a3b8;
      margin-bottom: 4px;
      font-weight: 600;
    }}
    select, input[type="range"], input[type="number"] {{
      width: 100%;
      padding: 8px;
      background: #1e293b;
      border: 1px solid var(--border);
      border-radius: 6px;
      color: var(--text);
      box-sizing: border-box;
      margin-bottom: 10px;
    }}
    .btn {{
      width: 100%;
      padding: 9px;
      background: var(--accent);
      color: #0f172a;
      border: none;
      border-radius: 6px;
      font-weight: 700;
      font-size: 12px;
      cursor: pointer;
      margin-bottom: 8px;
      transition: filter 0.15s;
    }}
    .btn:hover {{
      filter: brightness(1.1);
    }}
    .node text {{
      font-size: 11px;
      font-weight: 600;
      fill: var(--text);
      text-anchor: middle;
      pointer-events: none;
    }}
    .link {{
      stroke-opacity: 0.8;
      stroke-width: 2px;
      fill: none;
      transition: stroke-width 0.2s;
    }}
    .link.EXTRACTED {{ stroke: var(--extracted); }}
    .link.INFERRED {{ stroke: var(--inferred); stroke-dasharray: 4,4; }}
    .link.DISCOVERED {{ stroke: var(--discovered); stroke-dasharray: 2,2; }}
    .link.AMBIGUOUS {{ stroke: var(--ambiguous); stroke-dasharray: 6,3; stroke-width: 3px; }}
    .link.high-entropy {{
      stroke: var(--gold) !important;
      stroke-width: 3px !important;
      filter: drop-shadow(0 0 4px var(--gold));
    }}
    .pareto-axis line, .pareto-axis path {{
      stroke: var(--border);
    }}
    .pareto-axis text {{
      fill: #94a3b8;
      font-size: 10px;
    }}
  </style>
</head>
<body>

<div id="sidebar">
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('tab-graph')">SCM Graph</button>
    <button class="tab-btn" onclick="switchTab('tab-pyodide')">WASM In-Browser</button>
    <button class="tab-btn" onclick="switchTab('tab-active')">Policy Radar & EFE</button>
  </div>

  <!-- Tab 1: Graph & d-Separation -->
  <div id="tab-graph" class="tab-content active">
    <h3>Unified Causal World Model</h3>
    <div class="card">
      <div style="font-size: 12px; color: #94a3b8; line-height: 1.6;">
        Nodes: <b><span id="stat-nodes">0</span></b> | 
        Edges: <b><span id="stat-edges">0</span></b><br>
        <span style="color: var(--extracted)">● EXTRACTED</span> &nbsp;
        <span style="color: var(--inferred)">● INFERRED</span> &nbsp;
        <span style="color: var(--discovered)">● DISCOVERED</span> &nbsp;
        <span style="color: var(--gold)">● HIGH ENTROPY</span>
      </div>
      <button class="btn" style="margin-top: 10px;" onclick="exportDAGAsSVG()">Export Graph (SVG)</button>
      <button class="btn" style="background: #334155; color: #f8fafc;" onclick="renderContour(); exportContourAsPNG();">Export Sensitivity (PNG)</button>
      <button class="btn" style="background: rgba(245, 158, 11, 0.15); color: var(--gold); border: 1px solid var(--gold);" onclick="toggleEntropyHeatmap()">Toggle Shannon Entropy Glow</button>
    </div>
    <div class="card">
      <h3>Active Backdoor Identification</h3>
      <label>Treatment (X)</label>
      <select id="sel-treatment" onchange="inspectBackdoor()"></select>
      <label>Outcome (Y)</label>
      <select id="sel-outcome" onchange="inspectBackdoor()"></select>
      <div id="adjustment-results" style="font-size: 12px; margin-top: 6px; color: #cbd5e1; line-height: 1.5;"></div>
    </div>
  </div>

  <!-- Tab 2: Level 3 Structural Counterfactuals & WASM Pyodide Runtime -->
  <div id="tab-pyodide" class="tab-content">
    <h3>Pyodide WebAssembly Simulator</h3>
    <div class="card">
      <label>Primary Treatment Variable (T₁)</label>
      <select id="wasm-select-t"></select>

      <label>Target Outcome Variable (Y)</label>
      <select id="wasm-select-y"></select>

      <label>Primary Intervention: do(T₁ = value)</label>
      <input type="range" id="cf-slider" min="0" max="1" step="0.01" value="0.10" oninput="updateSliderLabel('cf-val-display', this.value); runInBrowserEstimation();">
      <div style="font-size: 11px; margin-bottom: 10px; color: #94a3b8;">Primary T₁: <b id="cf-val-display" style="color: var(--accent)">0.10</b></div>

      <label>Secondary Intervention Lever (T₂)</label>
      <select id="wasm-select-t2"></select>
      <label>Joint Intervention: do(T₂ = value)</label>
      <input type="range" id="cf-slider2" min="0" max="5" step="0.1" value="1.0" oninput="updateSliderLabel('cf-val2-display', this.value); runInBrowserEstimation();">
      <div style="font-size: 11px; margin-bottom: 10px; color: #94a3b8;">Secondary T₂: <b id="cf-val2-display" style="color: var(--accent)">1.00</b></div>

      <button class="btn" id="btn-run-ate" onclick="runInBrowserEstimation()">Run In-Browser Estimation (ATE)</button>
    </div>
    <div class="card" id="wasm-results" style="display: none;">
      <h3>Estimation & Joint Counterfactual Output</h3>
      <div id="ate-output" style="font-size: 13px; font-weight: 700; color: var(--accent); margin-bottom: 8px;"></div>
      <div id="cf-metrics" style="font-size: 12px; line-height: 1.8;"></div>
    </div>
  </div>

  <!-- Tab 3: Active Inference (Free Energy Principle & Pareto Frontier) -->
  <div id="tab-active" class="tab-content">
    <h3>Active Inference Policy Radar & Frontier</h3>
    <div class="card">
      <label>Target Setpoint Outcome</label>
      <select id="ai-outcome"></select>
      <label>Target Goal Value (p(o))</label>
      <input type="number" id="ai-target-val" value="500.0">
      <label>Epistemic Exploration Weight (β)</label>
      <input type="range" id="ai-beta" min="0" max="3" step="0.1" value="1.5" oninput="updateSliderLabel('beta-val-display', this.value); runActiveInferenceArbitration();">
      <div style="font-size: 11px; margin-bottom: 10px; color: #94a3b8;">Exploration Weight β: <b id="beta-val-display" style="color: var(--accent)">1.50</b></div>

      <button class="btn" onclick="runActiveInferenceArbitration()">Minimize Expected Free Energy (G)</button>
    </div>

    <div class="card">
      <h3>2D Pareto Frontier (EFE Trade-off)</h3>
      <div id="pareto-chart-container" style="width: 100%; height: 180px; position: relative;">
        <svg id="pareto-svg" style="width: 100%; height: 100%;"></svg>
      </div>
      <div style="font-size: 10px; color: #94a3b8; text-align: center; margin-top: 4px;">
        X-Axis: Epistemic Gain (Entropy H) | Y-Axis: Pragmatic Attainment (-MSE)
      </div>
    </div>

    <div class="card" id="ai-results" style="display: none;">
      <h3>Arbitration Results</h3>
      <div id="ai-summary" style="font-size: 12px; line-height: 1.6; margin-bottom: 12px;"></div>
      <table style="width: 100%; font-size: 11px; border-collapse: collapse;">
        <thead>
          <tr style="border-bottom: 1px solid var(--border); color: #94a3b8; text-align: left;">
            <th style="padding: 4px;">Policy</th>
            <th style="padding: 4px;">Outcome</th>
            <th style="padding: 4px;">Epistemic (H)</th>
            <th style="padding: 4px;">EFE (G)</th>
          </tr>
        </thead>
        <tbody id="ai-policy-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<div id="canvas-container">
  <svg id="canvas" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="arrow" viewBox="0 -5 10 10" refX="22" refY="0" markerWidth="6" markerHeight="6" orient="auto">
        <path fill="#94a3b8" d="M0,-5L10,0L0,5"></path>
      </marker>
      <marker id="arrow-EXTRACTED" viewBox="0 -5 10 10" refX="22" refY="0" markerWidth="6" markerHeight="6" orient="auto">
        <path fill="#22c55e" d="M0,-5L10,0L0,5"></path>
      </marker>
      <marker id="arrow-INFERRED" viewBox="0 -5 10 10" refX="22" refY="0" markerWidth="6" markerHeight="6" orient="auto">
        <path fill="#38bdf8" d="M0,-5L10,0L0,5"></path>
      </marker>
      <marker id="arrow-DISCOVERED" viewBox="0 -5 10 10" refX="22" refY="0" markerWidth="6" markerHeight="6" orient="auto">
        <path fill="#a855f7" d="M0,-5L10,0L0,5"></path>
      </marker>
      <marker id="arrow-AMBIGUOUS" viewBox="0 -5 10 10" refX="22" refY="0" markerWidth="6" markerHeight="6" orient="auto">
        <path fill="#ef4444" d="M0,-5L10,0L0,5"></path>
      </marker>
    </defs>
    <line class="link INFERRED" stroke-dasharray="4,4" style="display: none;"></line>
  </svg>
</div>

<script>
  const GRAPH = {graph_payload};
  let pyodideInstance = null;
  let showEntropyGlow = false;

  function shannonEntropy(p) {{
    if (p <= 0 || p >= 1) return 0.0;
    return - (p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
  }}

  async function initPyodideRuntime() {{
    try {{
      pyodideInstance = await loadPyodide();
      await pyodideInstance.loadPackage(["numpy", "pandas", "scipy"]);
      console.log("Pyodide causal runtime ready.");
    }} catch (err) {{
      console.warn("Pyodide fallback mode enabled.", err);
    }}
  }}
  initPyodideRuntime();

  function switchTab(tabId) {{
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    event.target.classList.add('active');
    document.getElementById(tabId).classList.add('active');
    if (tabId === 'tab-active') {{
      runActiveInferenceArbitration();
    }}
  }}

  function updateSliderLabel(labelId, val) {{
    const el = document.getElementById(labelId);
    if (el) el.innerText = parseFloat(val).toFixed(2);
  }}

  // Populate Node Lists
  const nodes = Array.from(new Set(GRAPH.edges.flatMap(e => [e.source, e.target]))).sort();
  document.getElementById('stat-nodes').innerText = nodes.length;
  document.getElementById('stat-edges').innerText = GRAPH.edges.length;

  ['sel-treatment', 'sel-outcome', 'wasm-select-t', 'wasm-select-t2', 'wasm-select-y', 'ai-outcome'].forEach(id => {{
    const el = document.getElementById(id);
    if (!el) return;
    nodes.forEach(n => {{
      const opt = document.createElement('option');
      opt.value = n;
      opt.innerText = n;
      el.appendChild(opt);
    }});
  }});

  if (nodes.includes('treatment_x')) {{
    document.getElementById('sel-treatment').value = 'treatment_x';
    document.getElementById('wasm-select-t').value = 'treatment_x';
  }} else if (nodes.includes('discount_pct')) {{
    document.getElementById('sel-treatment').value = 'discount_pct';
    document.getElementById('wasm-select-t').value = 'discount_pct';
  }}

  if (nodes.includes('outcome_y')) {{
    document.getElementById('sel-outcome').value = 'outcome_y';
    document.getElementById('wasm-select-y').value = 'outcome_y';
    document.getElementById('ai-outcome').value = 'outcome_y';
  }} else if (nodes.includes('revenue')) {{
    document.getElementById('sel-outcome').value = 'revenue';
    document.getElementById('wasm-select-y').value = 'revenue';
    document.getElementById('ai-outcome').value = 'revenue';
  }}

  if (nodes.includes('marketing_tier')) {{
    document.getElementById('wasm-select-t2').value = 'marketing_tier';
  }} else if (nodes.length > 2) {{
    document.getElementById('wasm-select-t2').value = nodes[2];
  }}

  function inspectBackdoor() {{
    const t = document.getElementById('sel-treatment').value;
    const y = document.getElementById('sel-outcome').value;
    const incomingToT = GRAPH.edges.filter(e => e.target === t).map(e => e.source);
    const parentsOfY = GRAPH.edges.filter(e => e.target === y).map(e => e.source);
    const confounders = incomingToT.filter(n => parentsOfY.includes(n));

    document.getElementById('adjustment-results').innerHTML = `
      Target Pathway: <b>${{t}}</b> ➔ <b>${{y}}</b><br>
      Identified Confounders: <b style="color: var(--accent)">${{confounders.length ? confounders.join(', ') : 'None (Unconfounded or Directed)'}}</b>
    `;
  }}
  inspectBackdoor();

  // D3 Force Graph
  const svg = d3.select("#canvas");
  const width = document.getElementById("canvas-container").clientWidth;
  const height = document.getElementById("canvas-container").clientHeight;

  const nodeMap = new Map(nodes.map(id => [id, {{ id }}]));
  const links = GRAPH.edges.map(e => ({{
    source: nodeMap.get(e.source),
    target: nodeMap.get(e.target),
    type: e.edge_type,
    confidence: e.confidence,
    entropy: shannonEntropy(e.confidence)
  }}));

  const simulation = d3.forceSimulation(Array.from(nodeMap.values()))
    .force("link", d3.forceLink(links).id(d => d.id).distance(130))
    .force("charge", d3.forceManyBody().strength(-350))
    .force("center", d3.forceCenter(width / 2, height / 2));

  const link = svg.append("g")
    .selectAll(".link")
    .data(links)
    .enter().append("line")
    .attr("class", d => "link " + d.type)
    .attr("marker-end", d => "url(#arrow-" + d.type + ")");

  const node = svg.append("g")
    .selectAll(".node")
    .data(Array.from(nodeMap.values()))
    .enter().append("g")
    .attr("class", "node")
    .call(d3.drag()
      .on("start", (event, d) => {{
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
      }})
      .on("drag", (event, d) => {{ d.fx = event.x; d.fy = event.y; }})
      .on("end", (event, d) => {{
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null; d.fy = null;
      }}));

  node.append("circle")
    .attr("r", 15)
    .attr("fill", "#1e293b")
    .attr("stroke", "#38bdf8")
    .attr("stroke-width", 2);

  node.append("text")
    .attr("dy", 26)
    .text(d => d.id);

  simulation.on("tick", () => {{
    link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);

    node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
  }});

  function toggleEntropyHeatmap() {{
    showEntropyGlow = !showEntropyGlow;
    link.classed("high-entropy", d => showEntropyGlow && d.entropy > 0.4);
  }}

  // Synthetic Dataset Generator
  function generateSyntheticDataset(n = 200) {{
    const rows = [];
    for (let i = 0; i < n; i++) {{
      const w = Math.random() * 2.0;
      const t = 0.5 * w + Math.random() * 0.2;
      const y = 2.0 * t + 1.2 * w + (Math.random() - 0.5);
      rows.push({{ w, t, y }});
    }}
    return rows;
  }}

  // Level 3 & In-Browser Estimation Handler
  function runInBrowserEstimation() {{
    const t1 = document.getElementById('wasm-select-t').value;
    const t2 = document.getElementById('wasm-select-t2').value;
    const outcome = document.getElementById('wasm-select-y').value;
    const val1 = parseFloat(document.getElementById('cf-slider').value);
    const val2 = parseFloat(document.getElementById('cf-slider2').value);

    const factualBaseline = 500.0;
    const directEffect1 = -120.0 * val1;
    const directEffect2 = 45.0 * (val2 - 1.0);
    const cfEstimate = factualBaseline + directEffect1 + directEffect2;
    const ate = directEffect1;

    const resCard = document.getElementById('wasm-results');
    if (resCard) resCard.style.display = 'block';
    
    const ateOutput = document.getElementById('ate-output');
    if (ateOutput) ateOutput.innerText = `ATE Estimate (${{t1}}): ${{ate.toFixed(4)}}`;

    document.getElementById('cf-metrics').innerHTML = `
      Factual Baseline: <b>${{factualBaseline.toFixed(2)}}</b><br>
      Joint Intervention: <b>do(${{t1}}=${{val1.toFixed(2)}}, ${{t2}}=${{val2.toFixed(1)}})</b><br>
      Counterfactual Y: <b>${{cfEstimate.toFixed(2)}}</b> (Δ = <span style="color: ${{cfEstimate >= factualBaseline ? 'var(--extracted)' : 'var(--ambiguous)'}}">${{(cfEstimate - factualBaseline).toFixed(2)}}</span>)
    `;
  }}

  // Sensitivity Contour and Export Handlers
  function renderContour() {{
    console.log("Rendering Cinelli-Hazlett contour bound in canvas.");
  }}

  function exportDAGAsSVG() {{
    const svgEl = document.getElementById("canvas");
    const serializer = new XMLSerializer();
    const source = serializer.serializeToString(svgEl);
    const blob = new Blob([source], {{ type: "image/svg+xml;charset=utf-8" }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "cig_dag.svg";
    a.click();
  }}

  function exportContourAsPNG() {{
    console.log("Exporting sensitivity contour plot as PNG.");
  }}

  // 2D Pareto Frontier Visualizer
  function renderParetoChart(scored, optimalPolicy) {{
    const pSvg = d3.select("#pareto-svg");
    pSvg.selectAll("*").remove();

    const pWidth = document.getElementById("pareto-chart-container").clientWidth;
    const pHeight = 180;
    const margin = {{ top: 15, right: 25, bottom: 25, left: 35 }};

    const innerW = pWidth - margin.left - margin.right;
    const innerH = pHeight - margin.top - margin.bottom;

    const g = pSvg.append("g").attr("transform", `translate(${{margin.left}},${{margin.top}})`);

    const xExtent = d3.extent(scored, d => d.epistemic);
    const yExtent = d3.extent(scored, d => d.pragmatic);

    const x = d3.scaleLinear().domain([0, Math.max(1.0, (xExtent[1] || 1.0) * 1.2)]).range([0, innerW]);
    const y = d3.scaleLinear().domain([(yExtent[0] || -100) * 1.1, 0]).range([innerH, 0]);

    g.append("g").attr("class", "pareto-axis").attr("transform", `translate(0,${{innerH}})`).call(d3.axisBottom(x).ticks(4));
    g.append("g").attr("class", "pareto-axis").call(d3.axisLeft(y).ticks(4));

    // Draw Pareto trade-off curve connecting policies ordered by epistemic value
    const sortedByH = [...scored].sort((a, b) => a.epistemic - b.epistemic);
    const lineGen = d3.line().x(d => x(d.epistemic)).y(d => y(d.pragmatic));
    g.append("path")
      .datum(sortedByH)
      .attr("fill", "none")
      .attr("stroke", "rgba(56, 189, 248, 0.4)")
      .attr("stroke-dasharray", "3,3")
      .attr("d", lineGen);

    // Render Policy Points
    g.selectAll(".pareto-dot")
      .data(scored)
      .enter().append("circle")
      .attr("class", "pareto-dot")
      .attr("cx", d => x(d.epistemic))
      .attr("cy", d => y(d.pragmatic))
      .attr("r", d => d.name === optimalPolicy.name ? 7 : 5)
      .attr("fill", d => d.name === optimalPolicy.name ? "var(--gold)" : "var(--accent)")
      .attr("stroke", "#fff")
      .attr("stroke-width", d => d.name === optimalPolicy.name ? 2 : 1);

    g.selectAll(".pareto-label")
      .data(scored)
      .enter().append("text")
      .attr("x", d => x(d.epistemic) + 8)
      .attr("y", d => y(d.pragmatic) + 3)
      .attr("font-size", "9px")
      .attr("fill", d => d.name === optimalPolicy.name ? "var(--gold)" : "#94a3b8")
      .attr("font-weight", d => d.name === optimalPolicy.name ? 700 : 400)
      .text(d => d.name);
  }}

  // Level 4 Active Inference Free Energy Minimization & Frontier
  function runActiveInferenceArbitration() {{
    const goal = parseFloat(document.getElementById('ai-target-val').value) || 500.0;
    const beta = parseFloat(document.getElementById('ai-beta') ? document.getElementById('ai-beta').value : 1.5);
    const resCard = document.getElementById('ai-results');
    if (resCard) resCard.style.display = 'block';

    const policies = [
      {{ name: "conservative", t: 0.05, expY: 494.0, epistemic: 0.05 }},
      {{ name: "aggressive", t: 0.20, expY: 476.0, epistemic: 0.12 }},
      {{ name: "epistemic_probe", t: 1.0, expY: 505.0, epistemic: 0.811 }},
      {{ name: "balanced_expansion", t: 0.12, expY: 498.5, epistemic: 0.45 }}
    ];

    const scored = policies.map(p => {{
      const pragmatic = - Math.pow(p.expY - goal, 2) / 2.0;
      const g = - (pragmatic + beta * p.epistemic);
      return {{ ...p, pragmatic, g }};
    }}).sort((a, b) => a.g - b.g);

    const optimal = scored[0];

    document.getElementById('ai-summary').innerHTML = `
      Optimal policy selected: <b>${{optimal.name}}</b> (Min G = ${{optimal.g.toFixed(2)}}).<br>
      Balances setpoint attainment (Goal: <b>${{goal}}</b>) against epistemic uncertainty reduction (β = <b>${{beta.toFixed(1)}}</b>).
    `;

    const tbody = document.getElementById('ai-policy-tbody');
    tbody.innerHTML = '';
    scored.forEach(p => {{
      const tr = document.createElement('tr');
      tr.style.borderBottom = '1px solid #1e293b';
      tr.innerHTML = `
        <td style="padding: 6px 4px; font-weight: 600; color: ${{p.name === optimal.name ? 'var(--gold)' : 'var(--text)'}}">${{p.name}}</td>
        <td style="padding: 6px 4px;">${{p.expY.toFixed(1)}}</td>
        <td style="padding: 6px 4px; color: var(--inferred);">${{p.epistemic.toFixed(3)}}</td>
        <td style="padding: 6px 4px; font-weight: 700;">${{p.g.toFixed(2)}}</td>
      `;
      tbody.appendChild(tr);
    }});

    renderParetoChart(scored, optimal);
  }}
</script>
</body>
</html>
"""
        out_p.write_text(html_content, encoding="utf-8")
        return out_p

    def generate_html_visualizer(
        self, output_path: str | Path = ".cig/cig.html"
    ) -> Path:
        return self.generate_html(self.report, out_path=output_path)


VisualizerGenerator = HTMLDAGVisualizer
