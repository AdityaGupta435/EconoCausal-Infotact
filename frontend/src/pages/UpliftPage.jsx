/**
 * EconoCausal — Uplift Modeling Page (Phase 10)
 * Run uplift analyses, view Qini / AUUC / segments / model comparison.
 */

import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import { uplift } from "../services/api";
import { SectionHeader, ErrorBanner, StateView, KpiCard } from "../components/common";
import "../styles/common.css";

const PC = { responsive: true, displayModeBar: false };
const LB = {
  autosize: true,
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { family: "Inter, sans-serif", size: 12, color: "#34415f" },
  margin: { l: 55, r: 20, t: 24, b: 50 },
};

export default function UpliftPage() {
  const [history, setHistory] = useState([]);
  const [histLoading, setHistLoading] = useState(true);
  const [histError, setHistError] = useState("");

  const [form, setForm] = useState({ treatment_col: "treatment", outcome_col: "purchase" });
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState("");
  const [currentResult, setCurrentResult] = useState(null);

  const [segments, setSegments] = useState(null);
  const [performance, setPerformance] = useState(null);

  async function loadHistory() {
    setHistLoading(true);
    setHistError("");
    try {
      const res = await uplift.history();
      setHistory(Array.isArray(res?.data) ? res.data : []);
    } catch (err) {
      setHistError(err.message);
    } finally {
      setHistLoading(false);
    }
  }

  async function loadSegmentsAndPerf() {
    try {
      const [s, p] = await Promise.all([uplift.segments(), uplift.performance()]);
      setSegments(s?.data);
      setPerformance(p?.data);
    } catch {}
  }

  useEffect(() => {
    loadHistory();
    loadSegmentsAndPerf();
  }, []);

  async function handleAnalyze(e) {
    e.preventDefault();
    setAnalyzeError("");
    setCurrentResult(null);
    setAnalyzing(true);
    try {
      const res = await uplift.analyze({ treatment_col: form.treatment_col, outcome_col: form.outcome_col });
      setCurrentResult(res?.data);
      await loadHistory();
      await loadSegmentsAndPerf();
    } catch (err) {
      setAnalyzeError(err.message);
    } finally {
      setAnalyzing(false);
    }
  }

  // Segment distribution chart
  const segLabels = segments
    ? ["Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs"]
    : [];
  const segValues = segments
    ? [
        segments.persuadables ?? 0,
        segments.sure_things  ?? 0,
        segments.lost_causes  ?? 0,
        segments.sleeping_dogs ?? 0,
      ]
    : [];

  // Model comparison from performance
  const models = performance?.models ?? [];

  return (
    <div>
      <SectionHeader
        eyebrow="UPLIFT MODELING"
        title="Uplift Analysis"
        subtitle="Identify persuadable customers using uplift models. Metrics are on held-out test data."
      />
      <div className="page-body">
        <div className="info-banner">
          Uplift model metrics (Qini, AUUC) are computed on synthetic held-out test data.
          Model skill relative to random targeting may be limited on small datasets.
        </div>

        {/* Run form */}
        <div className="card">
          <h3 className="card__title">Run Uplift Analysis</h3>
          <ErrorBanner message={analyzeError} />
          <form onSubmit={handleAnalyze} style={{ marginTop: 16 }}>
            <div className="form-row">
              <div className="form-group">
                <label className="form-label">Treatment Column</label>
                <input className="form-input" value={form.treatment_col}
                  onChange={e => setForm(f => ({ ...f, treatment_col: e.target.value }))} />
              </div>
              <div className="form-group">
                <label className="form-label">Outcome Column</label>
                <input className="form-input" value={form.outcome_col}
                  onChange={e => setForm(f => ({ ...f, outcome_col: e.target.value }))} />
              </div>
              <div className="form-group" style={{ flex: "0 0 auto", minWidth: "unset" }}>
                <label className="form-label">&nbsp;</label>
                <button className="btn btn-primary" type="submit" disabled={analyzing}>
                  {analyzing ? "Analyzing…" : "▶ Analyze"}
                </button>
              </div>
            </div>
          </form>

          {currentResult && (
            <div style={{ marginTop: 24 }}>
              <div className="kpi-grid">
                <KpiCard label="Best Model" value={currentResult.best_model ?? "—"} accent="#8b5cf6" />
                <KpiCard label="AUUC" value={currentResult.auuc?.toFixed?.(4) ?? "—"} detail="Area Under Uplift Curve" />
                <KpiCard label="Qini Coeff." value={currentResult.qini_coefficient?.toFixed?.(4) ?? "—"} />
                <KpiCard label="Test Rows" value={currentResult.test_rows?.toLocaleString() ?? "—"} />
              </div>
            </div>
          )}
        </div>

        {/* Charts row */}
        <div className="chart-grid-2" style={{ marginTop: 20 }}>
          {/* Customer segment distribution */}
          <div className="chart-card">
            <div className="chart-card__header">
              <h3 className="chart-card__title">Customer Segment Distribution</h3>
              <p className="chart-card__desc">Segments based on ITE sign and baseline probability.</p>
            </div>
            {!segments ? (
              <StateView icon="↗" title="Run an analysis to see segments" />
            ) : (
              <Plot
                data={[{
                  type: "pie",
                  labels: segLabels,
                  values: segValues,
                  hole: 0.4,
                  marker: { colors: ["#3478f6", "#16a394", "#f59e0b", "#e02525"] },
                  hovertemplate: "%{label}: %{value} (%{percent})<extra></extra>",
                }]}
                layout={{
                  ...LB,
                  margin: { l: 20, r: 20, t: 20, b: 20 },
                  showlegend: true,
                  legend: { orientation: "h", y: -0.1 },
                }}
                config={PC}
                useResizeHandler
                style={{ width: "100%", height: "280px" }}
              />
            )}
          </div>

          {/* Model comparison */}
          <div className="chart-card">
            <div className="chart-card__header">
              <h3 className="chart-card__title">Model AUUC Comparison</h3>
              <p className="chart-card__desc">AUUC for all evaluated uplift models.</p>
            </div>
            {models.length === 0 ? (
              <StateView icon="▤" title="No model performance data" desc="Run an uplift analysis to populate." />
            ) : (
              <Plot
                data={[{
                  type: "bar",
                  x: models.map(m => m.model_name ?? m.name),
                  y: models.map(m => m.auuc ?? 0),
                  marker: { color: models.map((_, i) => i === 0 ? "#3478f6" : "#94a3b8") },
                  hovertemplate: "%{x}: %{y:.4f}<extra></extra>",
                }]}
                layout={{
                  ...LB,
                  yaxis: { title: "AUUC", gridcolor: "#e8edf5" },
                }}
                config={PC}
                useResizeHandler
                style={{ width: "100%", height: "280px" }}
              />
            )}
          </div>
        </div>

        {/* History */}
        <div className="card" style={{ marginTop: 20 }}>
          <h3 className="card__title">Analysis History</h3>
          <ErrorBanner message={histError} />
          {histLoading ? (
            <StateView icon="⟳" title="Loading history…" />
          ) : history.length === 0 ? (
            <StateView icon="↗" title="No analyses yet" desc="Run an uplift analysis to see history." />
          ) : (
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Analysis ID</th>
                    <th>Best Model</th>
                    <th>AUUC</th>
                    <th>Qini Coeff.</th>
                    <th>Created</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((h) => (
                    <tr key={h.analysis_id ?? h.id}>
                      <td><code style={{ fontSize: 11 }}>{h.analysis_id ?? h.id}</code></td>
                      <td>{h.best_model ?? "—"}</td>
                      <td>{h.auuc?.toFixed?.(4) ?? "—"}</td>
                      <td>{h.qini_coefficient?.toFixed?.(4) ?? "—"}</td>
                      <td>{h.created_at ? new Date(h.created_at).toLocaleString() : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

