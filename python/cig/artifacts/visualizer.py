"""
Interactive HTML Visualizer for cig Structural Causal Models, Sensitivity Analysis,
and Client-Side Causal Estimation via Pyodide WebAssembly.
"""

from __future__ import annotations

from pathlib import Path
import json
from cig.passes.pass3_unify import SCMUnificationReport


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>cig — Causal Inference Graph & In-Browser SCM Engine</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <script src="https://cdn.jsdelivr.net/pyodide/v0.25.1/full/pyodide.js"></script>
  <style>
    :root {
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --text-muted: #8b949e;
      --accent-blue: #58a6ff;
      --green: #3fb950;
      --purple: #bc8cff;
      --amber: #d29922;
      --red: #f85149;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      display: flex;
      height: 100vh;
      overflow: hidden;
    }

    #sidebar {
      width: 320px;
      background: var(--card-bg);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      z-index: 10;
    }

    .header {
      padding: 16px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .header h1 {
      font-size: 1.1rem;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .badge-pill {
      font-size: 0.75rem;
      padding: 2px 8px;
      border-radius: 12px;
      background: #21262d;
      color: var(--accent-blue);
      border: 1px solid var(--border);
    }

    .nav-tabs {
      display: flex;
      border-bottom: 1px solid var(--border);
      background: #0d1117;
    }

    .nav-tab {
      flex: 1;
      padding: 10px;
      text-align: center;
      font-size: 0.75rem;
      font-weight: 600;
      cursor: pointer;
      color: var(--text-muted);
      border-bottom: 2px solid transparent;
    }

    .nav-tab.active {
      color: var(--accent-blue);
      border-bottom-color: var(--accent-blue);
      background: var(--card-bg);
    }

    .stats-panel {
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    .stat-box {
      background: #0d1117;
      padding: 8px;
      border-radius: 6px;
      border: 1px solid var(--border);
    }

    .stat-label { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; }
    .stat-value { font-size: 1.1rem; font-weight: 600; color: #fff; margin-top: 2px; }

    .controls-panel {
      padding: 14px 16px;
      flex: 1;
      overflow-y: auto;
    }

    .filter-group { margin-bottom: 14px; }
    .filter-group label {
      display: block;
      font-size: 0.8rem;
      font-weight: 600;
      margin-bottom: 6px;
      color: var(--text-muted);
    }

    .search-input, .select-input {
      width: 100%;
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 12px;
      color: #fff;
      font-size: 0.85rem;
      outline: none;
    }
    .search-input:focus, .select-input:focus { border-color: var(--accent-blue); }

    .btn-group {
      display: flex;
      gap: 8px;
      margin-top: 10px;
    }

    .btn-action {
      flex: 1;
      background: #21262d;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 12px;
      color: #c9d1d9;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      transition: all 0.15s ease-in-out;
    }
    .btn-action:hover {
      background: #30363d;
      border-color: var(--accent-blue);
      color: #fff;
    }
    .btn-action.primary {
      background: #238636;
      border-color: #2ea043;
      color: #fff;
    }
    .btn-action.primary:hover { background: #2ea043; }

    .legend {
      margin-top: 14px;
      background: #0d1117;
      padding: 10px;
      border-radius: 6px;
      border: 1px solid var(--border);
    }

    .legend-item {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.75rem;
      margin-bottom: 6px;
    }
    .legend-item:last-child { margin-bottom: 0; }
    .legend-dot { width: 10px; height: 10px; border-radius: 50%; }

    /* Main Viewport */
    #canvas-container {
      flex: 1;
      height: 100%;
      position: relative;
    }
    #network-canvas { width: 100%; height: 100%; }

    /* Views */
    .view-card-container {
      display: none;
      width: 100%;
      height: 100%;
      padding: 24px;
      overflow-y: auto;
      background: #0d1117;
    }

    .view-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      max-width: 860px;
      margin: 0 auto;
    }

    .grid-2col {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 16px;
    }

    .grid-4col {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 16px;
    }

    .form-group {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .form-group label { font-size: 0.75rem; color: var(--text-muted); font-weight: 600; }
    .form-group input, .form-group select {
      background: #0d1117;
      border: 1px solid var(--border);
      color: #fff;
      padding: 6px 8px;
      border-radius: 4px;
      font-size: 0.85rem;
    }

    .metrics-summary-bar {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px 16px;
      margin-bottom: 16px;
    }

    .metric-item-label { font-size: 0.7rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; }
    .metric-item-value { font-size: 1.15rem; font-weight: 700; color: var(--accent-blue); margin-top: 2px; }

    #contour-canvas {
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
      width: 100%;
      height: 480px;
    }

    .pyodide-status {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.75rem;
      padding: 4px 10px;
      border-radius: 12px;
      background: #21262d;
      border: 1px solid var(--border);
    }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--amber); }
    .status-dot.ready { background: var(--green); }

    .result-console {
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px;
      font-family: monospace;
      font-size: 0.8rem;
      color: var(--text);
      white-space: pre-wrap;
      max-height: 280px;
      overflow-y: auto;
    }

    #inspector {
      position: absolute;
      right: 16px;
      top: 16px;
      width: 320px;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      display: none;
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
      z-index: 100;
    }

    #inspector h3 { font-size: 0.95rem; color: #fff; margin-bottom: 8px; }
    .inspector-section { margin-top: 10px; font-size: 0.8rem; line-height: 1.4; }
    .inspector-section strong { color: var(--text-muted); display: block; margin-bottom: 2px; }
    .code-block {
      background: #0d1117;
      padding: 6px 8px;
      border-radius: 4px;
      font-family: monospace;
      font-size: 0.75rem;
      color: var(--accent-blue);
      border: 1px solid var(--border);
      overflow-x: auto;
      white-space: pre-wrap;
    }
    .close-btn {
      float: right;
      background: none;
      border: none;
      color: var(--text-muted);
      cursor: pointer;
      font-size: 1rem;
    }
  </style>
</head>
<body>

  <aside id="sidebar">
    <div class="header">
      <h1><span>⚡</span> cig causal graph</h1>
      <span class="badge-pill">v0.1.0</span>
    </div>

    <div class="nav-tabs">
      <div class="nav-tab active" id="tab-graph" onclick="switchTab('graph')">Causal DAG</div>
      <div class="nav-tab" id="tab-sensitivity" onclick="switchTab('sensitivity')">Sensitivity</div>
      <div class="nav-tab" id="tab-pyodide" onclick="switchTab('pyodide')">WASM ATE</div>
    </div>

    <div class="stats-panel">
      <div class="stat-box">
        <div class="stat-label">Variables</div>
        <div class="stat-value" id="stat-nodes">0</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Causal Edges</div>
        <div class="stat-value" id="stat-edges">0</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Deterministic</div>
        <div class="stat-value" id="stat-extracted" style="color: var(--green);">0</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Hypothesized</div>
        <div class="stat-value" id="stat-inferred" style="color: var(--purple);">0</div>
      </div>
    </div>

    <div class="controls-panel">
      <div class="filter-group">
        <label>SEARCH NODE</label>
        <input type="text" id="node-search" class="search-input" placeholder="e.g. revenue, price..." oninput="filterGraphNodes(this.value)" />
      </div>

      <div class="filter-group">
        <label>EXPORT DAG</label>
        <div class="btn-group">
          <button class="btn-action" id="btn-export-dag-png" onclick="exportDAGAsPNG()">📷 Export PNG</button>
          <button class="btn-action" id="btn-export-dag-svg" onclick="exportDAGAsSVG()">📐 Export SVG</button>
        </div>
      </div>

      <div class="legend">
        <div class="legend-item">
          <div class="legend-dot" style="background: var(--green);"></div>
          <span><strong>EXTRACTED (1.0)</strong> — SQL / Code AST</span>
        </div>
        <div class="legend-item">
          <div class="legend-dot" style="background: var(--purple);"></div>
          <span><strong>INFERRED</strong> — Domain Docs / LLM</span>
        </div>
        <div class="legend-item">
          <div class="legend-dot" style="background: var(--red);"></div>
          <span><strong>AMBIGUOUS</strong> — Contradiction</span>
        </div>
        <div class="legend-item">
          <div class="legend-dot" style="background: var(--amber);"></div>
          <span><strong>GOD CONFOUNDER</strong> — Hub Node</span>
        </div>
      </div>
    </div>
  </aside>

  <main id="canvas-container">
    <div id="network-canvas"></div>

    <!-- Sensitivity View -->
    <div id="sensitivity-view" class="view-card-container">
      <div class="view-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <h2 style="font-size: 1.15rem; color: #fff;">Cinelli-Hazlett Sensitivity Simulator</h2>
          <div class="btn-group" style="margin-top: 0;">
            <button class="btn-action" id="btn-export-contour-png" onclick="exportContourAsPNG()">📷 Export PNG</button>
            <button class="btn-action" id="btn-export-contour-svg" onclick="exportContourAsSVG()">📐 Export SVG</button>
          </div>
        </div>
        <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 14px;">
          Simulate unobserved confounder strength thresholds ($R^2_{Y \\sim Z \\mid X,D}$, $R^2_{D \\sim Z \\mid X}$) required to eliminate treatment effect.
        </p>

        <div class="grid-2col" style="padding-bottom: 12px; border-bottom: 1px solid var(--border);">
          <div class="form-group">
            <label>TREATMENT (D)</label>
            <select id="select-treatment" class="select-input" onchange="renderContour()"></select>
          </div>
          <div class="form-group">
            <label>OUTCOME (Y)</label>
            <select id="select-outcome" class="select-input" onchange="renderContour()"></select>
          </div>
        </div>

        <div class="grid-4col">
          <div class="form-group">
            <label>POINT ESTIMATE (τ̂)</label>
            <input type="number" id="sens-tau" value="2.5" step="0.1" oninput="renderContour()" />
          </div>
          <div class="form-group">
            <label>STANDARD ERROR (SE)</label>
            <input type="number" id="sens-se" value="0.4" step="0.05" oninput="renderContour()" />
          </div>
          <div class="form-group">
            <label>DEGREES OF FREEDOM (ν)</label>
            <input type="number" id="sens-dof" value="500" step="10" oninput="renderContour()" />
          </div>
          <div class="form-group">
            <label>BIAS REDUCTION (q)</label>
            <input type="number" id="sens-q" value="1.0" min="0.1" max="1.0" step="0.1" oninput="renderContour()" />
          </div>
        </div>

        <div class="metrics-summary-bar">
          <div>
            <div class="metric-item-label">Robustness Value (RV_q)</div>
            <div class="metric-item-value" id="val-rv-q">0.0%</div>
          </div>
          <div>
            <div class="metric-item-label">Significance (RV_α=0.05)</div>
            <div class="metric-item-value" id="val-rv-alpha" style="color: var(--purple);">0.0%</div>
          </div>
          <div>
            <div class="metric-item-label">VanderWeele E-Value</div>
            <div class="metric-item-value" id="val-e-value" style="color: var(--green);">0.00</div>
          </div>
        </div>

        <canvas id="contour-canvas" width="820" height="480"></canvas>
      </div>
    </div>

    <!-- Pyodide In-Browser ATE Estimation View -->
    <div id="pyodide-view" class="view-card-container">
      <div class="view-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <h2 style="font-size: 1.15rem; color: #fff;">In-Browser Causal ATE Estimation (WASM)</h2>
          <div class="pyodide-status">
            <div class="status-dot" id="wasm-status-dot"></div>
            <span id="wasm-status-text">Initializing Pyodide...</span>
          </div>
        </div>
        <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 14px;">
          Execute client-side Backdoor Adjustment & OLS estimation in WebAssembly without sending raw data to a server.
        </p>

        <div class="grid-2col">
          <div class="form-group">
            <label>TREATMENT VARIABLE (T)</label>
            <select id="wasm-select-t" class="select-input"></select>
          </div>
          <div class="form-group">
            <label>OUTCOME VARIABLE (Y)</label>
            <select id="wasm-select-y" class="select-input"></select>
          </div>
        </div>

        <div class="form-group" style="margin-bottom: 14px;">
          <label>ADJUSTMENT SET (W)</label>
          <input type="text" id="wasm-input-w" class="search-input" placeholder="e.g. confounder1, confounder2 (auto-detected if blank)" />
        </div>

        <div class="grid-2col" style="margin-bottom: 14px;">
          <div class="form-group">
            <label>UPLOAD CSV DATASET</label>
            <input type="file" id="wasm-file-input" accept=".csv" style="color: var(--text-muted); font-size: 0.8rem;" onchange="handleCSVUpload(event)" />
          </div>
          <div class="form-group" style="justify-content: flex-end;">
            <button class="btn-action" onclick="generateSyntheticDataset()">🎲 Generate Synthetic Sample (N=1000)</button>
          </div>
        </div>

        <button class="btn-action primary" id="btn-run-ate" onclick="runInBrowserEstimation()" style="width: 100%; margin-bottom: 16px;">
          🚀 Run Causal ATE Estimation
        </button>

        <div class="metrics-summary-bar">
          <div>
            <div class="metric-item-label">Estimated ATE (τ̂)</div>
            <div class="metric-item-value" id="wasm-ate-val">—</div>
          </div>
          <div>
            <div class="metric-item-label">95% Confidence Interval</div>
            <div class="metric-item-value" id="wasm-ci-val" style="color: var(--purple); font-size: 0.95rem;">—</div>
          </div>
          <div>
            <div class="metric-item-label">p-value</div>
            <div class="metric-item-value" id="wasm-p-val" style="color: var(--green);">—</div>
          </div>
        </div>

        <div class="result-console" id="wasm-console">Engine output will appear here after execution...</div>
      </div>
    </div>

    <!-- Inspector -->
    <div id="inspector">
      <button class="close-btn" onclick="closeInspector()">✕</button>
      <h3 id="inspect-title">Edge Inspector</h3>
      <div class="inspector-section">
        <strong>TYPE & CONFIDENCE</strong>
        <span id="inspect-type">-</span>
      </div>
      <div class="inspector-section">
        <strong>PROVENANCE</strong>
        <div id="inspect-provenance" class="code-block">-</div>
      </div>
      <div class="inspector-section">
        <strong>MECHANISM / EXPRESSION</strong>
        <div id="inspect-expression" class="code-block">-</div>
      </div>
    </div>
  </main>

  <script>
    const graphData = __CIG_DATA_PLACEHOLDER__;

    document.getElementById('stat-nodes').innerText = graphData.total_nodes;
    document.getElementById('stat-edges').innerText = graphData.total_edges;
    document.getElementById('stat-extracted').innerText = graphData.extracted_edges;
    document.getElementById('stat-inferred').innerText = graphData.inferred_edges;

    const godSet = new Set(graphData.god_confounders || []);
    const allNodeNames = new Set();
    const visNodes = [];
    const visEdges = [];

    graphData.edges.forEach(e => {
      [e.source, e.target].forEach(name => {
        if (!allNodeNames.has(name)) {
          allNodeNames.add(name);
          const isGod = godSet.has(name);
          visNodes.push({
            id: name,
            label: name,
            shape: 'box',
            margin: 10,
            color: {
              background: isGod ? '#382800' : '#161b22',
              border: isGod ? '#d29922' : '#58a6ff',
              highlight: { background: '#21262d', border: '#79c0ff' }
            },
            font: { color: isGod ? '#f1e05a' : '#c9d1d9', face: 'monospace', size: 13 },
            borderWidth: isGod ? 2 : 1
          });
        }
      });

      let edgeColor = '#3fb950';
      let dashes = false;
      if (e.edge_type === 'INFERRED') {
        edgeColor = '#bc8cff';
        dashes = true;
      } else if (e.edge_type === 'AMBIGUOUS') {
        edgeColor = '#f85149';
        dashes = [4, 4];
      }

      visEdges.push({
        id: `${e.source}-->${e.target}`,
        from: e.source,
        to: e.target,
        arrows: 'to',
        color: { color: edgeColor, highlight: '#ffffff' },
        dashes: dashes,
        width: e.confidence * 2.5,
        title: `${e.edge_type} (${e.confidence})`,
        data: e
      });
    });

    // Populate Selectors
    const sortedNodes = Array.from(allNodeNames).sort();
    ['select-treatment', 'select-outcome', 'wasm-select-t', 'wasm-select-y'].forEach((id, idx) => {
      const select = document.getElementById(id);
      sortedNodes.forEach(node => {
        const opt = document.createElement('option');
        opt.value = node;
        opt.text = node;
        select.appendChild(opt);
      });
      if (sortedNodes.length > 1) {
        select.selectedIndex = idx % 2 === 0 ? 0 : 1;
      }
    });

    // Vis Network
    const container = document.getElementById('network-canvas');
    const network = new vis.Network(container, { nodes: new vis.DataSet(visNodes), edges: new vis.DataSet(visEdges) }, {
      physics: {
        solver: 'forceAtlas2Based',
        forceAtlas2Based: { gravitationalConstant: -70, centralGravity: 0.008, springLength: 120, springConstant: 0.08 }
      },
      interaction: { hover: true, tooltipDelay: 200 }
    });

    network.on('click', function(params) {
      if (params.edges.length > 0 && params.nodes.length === 0) {
        const edge = visEdges.find(e => e.id === params.edges[0]);
        if (edge && edge.data) openInspector(edge.data);
      } else {
        closeInspector();
      }
    });

    function filterGraphNodes(query) {
      const q = query.trim().toLowerCase();
      if (!q) { network.selectNodes([]); return; }
      network.selectNodes(visNodes.filter(n => n.id.toLowerCase().includes(q)).map(n => n.id));
    }

    function openInspector(edgeData) {
      const panel = document.getElementById('inspector');
      document.getElementById('inspect-title').innerText = `${edgeData.source} ➔ ${edgeData.target}`;
      document.getElementById('inspect-type').innerText = `${edgeData.edge_type} (Confidence: ${edgeData.confidence})`;
      document.getElementById('inspect-provenance').innerText = edgeData.provenance.join('\\n') || 'None';
      document.getElementById('inspect-expression').innerText = edgeData.mechanisms.join('\\n') || edgeData.conflict_notes || 'None';
      panel.style.display = 'block';
    }

    function closeInspector() {
      document.getElementById('inspector').style.display = 'none';
    }

    function switchTab(tab) {
      document.getElementById('tab-graph').classList.toggle('active', tab === 'graph');
      document.getElementById('tab-sensitivity').classList.toggle('active', tab === 'sensitivity');
      document.getElementById('tab-pyodide').classList.toggle('active', tab === 'pyodide');

      document.getElementById('network-canvas').style.display = tab === 'graph' ? 'block' : 'none';
      document.getElementById('sensitivity-view').style.display = tab === 'sensitivity' ? 'block' : 'none';
      document.getElementById('pyodide-view').style.display = tab === 'pyodide' ? 'block' : 'none';

      if (tab === 'sensitivity') renderContour();
      if (tab === 'pyodide') ensurePyodideLoaded();
    }

    // Export Utilities
    function downloadBlob(blob, filename) {
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(link.href);
    }

    function exportDAGAsPNG() {
      const visCanvas = container.querySelector('canvas');
      if (visCanvas) visCanvas.toBlob(blob => downloadBlob(blob, 'cig-causal-dag.png'));
    }

    function exportDAGAsSVG() {
      const positions = network.getPositions();
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      Object.values(positions).forEach(p => {
        minX = Math.min(minX, p.x); minY = Math.min(minY, p.y);
        maxX = Math.max(maxX, p.x); maxY = Math.max(maxY, p.y);
      });
      const pad = 80;
      const width = Math.max(600, (maxX - minX) + pad * 2);
      const height = Math.max(400, (maxY - minY) + pad * 2);
      const offsetX = -minX + pad;
      const offsetY = -minY + pad;

      let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" style="background:#0d1117;">\\n`;
      svg += `<defs><marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#58a6ff"/></marker></defs>\\n`;

      visEdges.forEach(e => {
        const p1 = positions[e.from], p2 = positions[e.to];
        if (p1 && p2) {
          svg += `<line x1="${p1.x + offsetX}" y1="${p1.y + offsetY}" x2="${p2.x + offsetX}" y2="${p2.y + offsetY}" stroke="${e.color.color}" stroke-width="${e.width || 2}" marker-end="url(#arrow)" ${e.dashes ? 'stroke-dasharray="4,4"' : ''}/>\\n`;
        }
      });

      Object.entries(positions).forEach(([name, p]) => {
        const isGod = godSet.has(name);
        svg += `<g transform="translate(${p.x + offsetX - 50}, ${p.y + offsetY - 18})">\\n`;
        svg += `  <rect width="100" height="36" rx="6" fill="${isGod ? '#382800' : '#161b22'}" stroke="${isGod ? '#d29922' : '#58a6ff'}" stroke-width="${isGod ? 2 : 1}"/>\\n`;
        svg += `  <text x="50" y="22" text-anchor="middle" fill="${isGod ? '#f1e05a' : '#c9d1d9'}" font-family="monospace" font-size="12">${name}</text>\\n`;
        svg += `</g>\\n`;
      });
      svg += `</svg>`;
      downloadBlob(new Blob([svg], { type: 'image/svg+xml;charset=utf-8' }), 'cig-causal-dag.svg');
    }

    function exportContourAsPNG() {
      document.getElementById('contour-canvas').toBlob(blob => downloadBlob(blob, 'cig-sensitivity-contour.png'));
    }

    function exportContourAsSVG() {
      const canvas = document.getElementById('contour-canvas');
      const w = canvas.width, h = canvas.height;
      const tau = parseFloat(document.getElementById('sens-tau').value) || 2.5;
      const se = parseFloat(document.getElementById('sens-se').value) || 0.4;
      const dof = parseFloat(document.getElementById('sens-dof').value) || 500;
      const pad = { left: 60, bottom: 50, right: 40, top: 30 };
      const plotW = w - pad.left - pad.right;
      const plotH = h - pad.top - pad.bottom;

      let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" style="background:#0d1117;">\\n`;
      svg += `<rect x="${pad.left}" y="${pad.top}" width="${plotW}" height="${plotH}" fill="none" stroke="#30363d" stroke-width="1"/>\\n`;

      const targetRatios = [1.0, 0.75, 0.5, 0.25, 0.0];
      const colors = ['#3fb950', '#58a6ff', '#bc8cff', '#d29922', '#f85149'];

      targetRatios.forEach((ratio, idx) => {
        const targetTau = tau * ratio;
        let points = [];
        for (let px = 1; px <= 100; px++) {
          const r2d = (px / 100) * 0.5;
          const neededBias = Math.abs(tau - targetTau);
          const r2y = (Math.pow(neededBias / (se * Math.sqrt(dof) + 1e-9), 2) * (1 - r2d)) / r2d;
          if (r2y >= 0 && r2y <= 0.5) {
            points.push(`${(pad.left + (r2d / 0.5) * plotW).toFixed(1)},${(pad.top + plotH - (r2y / 0.5) * plotH).toFixed(1)}`);
          }
        }
        if (points.length > 1) {
          svg += `<polyline fill="none" stroke="${colors[idx]}" stroke-width="${ratio === 0.0 ? 2.5 : 1.5}" points="${points.join(' ')}"/>\\n`;
        }
      });
      svg += `</svg>`;
      downloadBlob(new Blob([svg], { type: 'image/svg+xml;charset=utf-8' }), 'cig-sensitivity-contour.svg');
    }

    function renderContour() {
      const canvas = document.getElementById('contour-canvas');
      const ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height;
      const tau = parseFloat(document.getElementById('sens-tau').value) || 2.5;
      const se = parseFloat(document.getElementById('sens-se').value) || 0.4;
      const dof = parseFloat(document.getElementById('sens-dof').value) || 500;
      const q = parseFloat(document.getElementById('sens-q').value) || 1.0;
      const t_stat = Math.abs(tau / (se + 1e-9));

      const f_q = (q * t_stat) / Math.sqrt(dof);
      const rv_q = f_q > 1e-6 ? (Math.sqrt(f_q * f_q + 1.0) - 1.0) / f_q : 0.0;
      const t_crit = 1.96;
      let rv_alpha = 0.0;
      if (t_stat > t_crit) {
        const f_alpha = (t_stat - t_crit) / Math.sqrt(dof);
        rv_alpha = f_alpha > 1e-6 ? (Math.sqrt(f_alpha * f_alpha + 1.0) - 1.0) / f_alpha : 0.0;
      }

      const rr = Math.exp(0.91 * (Math.abs(tau) / (se * Math.sqrt(dof) * 0.2 + 1.0)));
      const e_val = rr >= 1.0 ? rr + Math.sqrt(rr * (rr - 1.0)) : 1.0;

      document.getElementById('val-rv-q').innerText = `${(rv_q * 100).toFixed(1)}%`;
      document.getElementById('val-rv-alpha').innerText = `${(rv_alpha * 100).toFixed(1)}%`;
      document.getElementById('val-e-value').innerText = e_val.toFixed(2);

      ctx.clearRect(0, 0, w, h);
      const pad = { left: 60, bottom: 50, right: 40, top: 30 };
      const plotW = w - pad.left - pad.right;
      const plotH = h - pad.top - pad.bottom;

      ctx.strokeStyle = '#30363d';
      ctx.lineWidth = 1;
      ctx.strokeRect(pad.left, pad.top, plotW, plotH);
      ctx.fillStyle = '#8b949e';
      ctx.font = '11px monospace';

      for (let i = 0; i <= 5; i++) {
        const val = (i * 0.1).toFixed(1);
        const x = pad.left + (i / 5) * plotW;
        const y = pad.top + plotH - (i / 5) * plotH;
        ctx.fillText(val, x - 10, pad.top + plotH + 18);
        ctx.fillText(val, pad.left - 30, y + 4);
        ctx.beginPath();
        ctx.strokeStyle = '#21262d';
        ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH);
        ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y);
        ctx.stroke();
      }

      ctx.fillStyle = '#c9d1d9';
      ctx.font = '12px sans-serif';
      ctx.fillText('Partial R² of Confounder with Treatment (R²_D~Z|X)', pad.left + plotW / 4, h - 12);
      ctx.save();
      ctx.translate(16, pad.top + (3 * plotH) / 4);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Partial R² of Confounder with Outcome (R²_Y~Z|X,D)', 0, 0);
      ctx.restore();

      const targetRatios = [1.0, 0.75, 0.5, 0.25, 0.0];
      const colors = ['#3fb950', '#58a6ff', '#bc8cff', '#d29922', '#f85149'];

      targetRatios.forEach((ratio, idx) => {
        const targetTau = tau * ratio;
        ctx.beginPath();
        ctx.strokeStyle = colors[idx];
        ctx.lineWidth = ratio === 0.0 ? 2.5 : 1.5;
        let first = true;
        for (let px = 1; px <= 100; px++) {
          const r2d = (px / 100) * 0.5;
          const neededBias = Math.abs(tau - targetTau);
          const r2y = (Math.pow(neededBias / (se * Math.sqrt(dof) + 1e-9), 2) * (1 - r2d)) / r2d;
          if (r2y >= 0 && r2y <= 0.5) {
            const canvasX = pad.left + (r2d / 0.5) * plotW;
            const canvasY = pad.top + plotH - (r2y / 0.5) * plotH;
            if (first) { ctx.moveTo(canvasX, canvasY); first = false; }
            else { ctx.lineTo(canvasX, canvasY); }
          }
        }
        ctx.stroke();
        ctx.fillStyle = colors[idx];
        ctx.font = '10px monospace';
        ctx.fillText(`τ=${targetTau.toFixed(2)} (${(ratio*100).toFixed(0)}%)`, pad.left + plotW - 130, pad.top + 20 + idx * 16);
      });
    }

    // ==========================================
    // Pyodide WebAssembly In-Browser Causal Engine
    // ==========================================
    let pyodideInstance = null;
    let loadedCSVContent = null;

    async function ensurePyodideLoaded() {
      if (pyodideInstance) return;
      const statusText = document.getElementById('wasm-status-text');
      const statusDot = document.getElementById('wasm-status-dot');

      try {
        statusText.innerText = 'Loading Pyodide WASM...';
        pyodideInstance = await loadPyodide();
        statusText.innerText = 'Loading numpy, pandas, statsmodels...';
        await pyodideInstance.loadPackage(['numpy', 'pandas', 'scipy', 'statsmodels']);
        statusText.innerText = 'Pyodide Ready';
        statusDot.classList.add('ready');
      } catch (err) {
        statusText.innerText = `WASM Init Failed: ${err.message}`;
        statusDot.style.background = '#f85149';
      }
    }

    function handleCSVUpload(event) {
      const file = event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = e => {
        loadedCSVContent = e.target.result;
        document.getElementById('wasm-console').innerText = `Loaded custom dataset: ${file.name} (${file.size} bytes)`;
      };
      reader.readAsText(file);
    }

    function generateSyntheticDataset() {
      const tName = document.getElementById('wasm-select-t').value;
      const yName = document.getElementById('wasm-select-y').value;
      const wName = 'confounder_w';

      let csv = `${wName},${tName},${yName}\\n`;
      for (let i = 0; i < 1000; i++) {
        const u1 = Math.random(), u2 = Math.random();
        const randN = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        const randN2 = Math.sqrt(-2.0 * Math.log(u2 || 0.01)) * Math.sin(2.0 * Math.PI * u1);

        const w = randN;
        const t = 0.6 * w + randN2 * 0.8;
        const y = 2.5 * t + 1.4 * w + (Math.random() - 0.5);
        csv += `${w.toFixed(4)},${t.toFixed(4)},${y.toFixed(4)}\\n`;
      }
      loadedCSVContent = csv;
      document.getElementById('wasm-input-w').value = wName;
      document.getElementById('wasm-console').innerText = `Generated synthetic dataset (N=1000) with true causal effect τ=2.50.`;
    }

    async function runInBrowserEstimation() {
      await ensurePyodideLoaded();
      if (!loadedCSVContent) {
        generateSyntheticDataset();
      }

      const tVar = document.getElementById('wasm-select-t').value;
      const yVar = document.getElementById('wasm-select-y').value;
      const wInput = document.getElementById('wasm-input-w').value.trim();
      const wList = wInput ? wInput.split(',').map(s => s.trim()).filter(Boolean) : [];

      const pyCode = [
        "import io",
        "import json",
        "import pandas as pd",
        "import statsmodels.api as sm",
        "",
        `csv_data = ${JSON.stringify(loadedCSVContent)}`,
        "df = pd.read_csv(io.StringIO(csv_data))",
        "",
        `t = ${JSON.stringify(tVar)}`,
        `y = ${JSON.stringify(yVar)}`,
        `w_cols = ${JSON.stringify(wList)}`,
        "",
        "features = [t] + [col for col in w_cols if col in df.columns]",
        "X = sm.add_constant(df[features])",
        "y_vec = df[y]",
        "",
        "model = sm.OLS(y_vec, X).fit()",
        "",
        "tau = float(model.params[t])",
        "se = float(model.bse[t])",
        "p_val = float(model.pvalues[t])",
        "ci_low = float(model.conf_int().loc[t][0])",
        "ci_high = float(model.conf_int().loc[t][1])",
        "",
        "res = {",
        '    "tau": round(tau, 4),',
        '    "se": round(se, 4),',
        '    "p_val": round(p_val, 5),',
        '    "ci": [round(ci_low, 4), round(ci_high, 4)],',
        '    "summary": str(model.summary())',
        "}",
        "json.dumps(res)"
      ].join('\\n');

      try {
        const rawJson = await pyodideInstance.runPythonAsync(pyCode);
        const result = JSON.parse(rawJson);

        document.getElementById('wasm-ate-val').innerText = result.tau.toFixed(4);
        document.getElementById('wasm-ci-val').innerText = `[${result.ci[0]}, ${result.ci[1]}]`;
        document.getElementById('wasm-p-val').innerText = result.p_val < 0.0001 ? '< 0.0001' : result.p_val.toFixed(4);
        document.getElementById('wasm-console').innerText = result.summary;

        document.getElementById('sens-tau').value = result.tau;
        document.getElementById('sens-se').value = result.se;
      } catch (err) {
        document.getElementById('wasm-console').innerText = `Estimation Error:\\n${err.message}`;
      }
    }
  </script>
</body>
</html>
"""


class HTMLDAGVisualizer:
    @staticmethod
    def generate_html(
        report: SCMUnificationReport,
        out_path: str | Path = ".cig/cig.html",
    ) -> Path:
        """
        Renders the SCMUnificationReport into a standalone, interactive HTML visualizer.
        """
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        json_data = json.dumps(report.model_dump())
        rendered_content = HTML_TEMPLATE.replace("__CIG_DATA_PLACEHOLDER__", json_data)

        path.write_text(rendered_content, encoding="utf-8")
        return path
