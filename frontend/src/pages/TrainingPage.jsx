/**
 * EconoCausal — ML Training Page (Phase 10)
 * Start training runs, view status, history, metrics, and logs.
 */

import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import { training } from "../services/api";
import { SectionHeader, ErrorBanner, SuccessBanner, StateView, StatusBadge, KpiCard } from "../components/common";
import "../styles/common.css";

const PLOTLY_CONFIG = { responsive: true, displayModeBar: false };

export default function TrainingPage() {
  const [history, setHistory] = useState([]);
  const [histLoading, setHistLoading] = useState(true);
  const [histError, setHistError] = useState("");

  const [form, setForm] = useState({ model_type: "all", version: "" });
  const [starting, setStarting] = useState(false);
  const [startResult, setStartResult] = useState(null);
  const [startError, setStartError] = useState("");

  const [selectedRun, setSelectedRun] = useState(null);  // run id
  const [runMetrics, setRunMetrics] = useState(null);
  const [runLogs, setRunLogs] = useState([]);
  const [runStatus, setRunStatus] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  async function loadHistory() {
    setHistLoading(true);
    setHistError("");
    try {
      const res = await training.history();
      setHistory(Array.isArray(res?.data) ? res.data : []);
    } catch (err) {
      setHistError(err.message);
    } finally {
      setHistLoading(false);
    }
  }

  useEffect(() => { loadHistory(); }, []);

  async function handleStart(e) {
    e.preventDefault();
    setStartError("");
    setStartResult(null);
    setStarting(true);
    try {
      const payload = { model_type: form.model_type };
      if (form.version.trim()) payload.version = form.version.trim();
      const res = await training.start(payload);
      setStartResult(res?.data);
      await loadHistory();
    } catch (err) {
      setStartError(err.message);
    } finally {
      setStarting(false);
    }
  }

  async function openRunDetails(id) {
    if (selectedRun === id) { setSelectedRun(null); return; }
    setSelectedRun(id);
    setDetailLoading(true);
    setRunMetrics(null);
    setRunLogs([]);
    setRunStatus(null);
    try {
      const [st, mt, lg] = await Promise.all([
        training.status(id),
        training.metrics(id),
        training.logs(id),
      ]);
      setRunStatus(st?.data);
      setRunMetrics(mt?.data);
      setRunLogs(Array.isArray(lg?.data) ? lg.data : []);
    } catch {}
    finally { setDetailLoading(false); }
  }

  // Build chart data from metrics
  const metricKeys = runMetrics
    ? Object.keys(runMetrics).filter(k => typeof runMetrics[k] === "number")
    : [];

  return (
    <div>
      <SectionHeader
        eyebrow="ML TRAINING"
        title="Model Training"
        subtitle="Start training runs, track status, and compare evaluation metrics."
      />
      <div className="page-body">

        {/* Start training form */}
        <div className="card">
          <h3 className="card__title">Start Training Run</h3>
          <p className="card__subtitle">Training runs asynchronously in the background.</p>
          <ErrorBanner message={startError} />
          {startResult && (
            <SuccessBanner message={`Training queued — Run ID: ${startResult.run_id}`} />
          )}
          <form onSubmit={handleStart} style={{ marginTop: 16 }}>
            <div className="form-row">
              <div className="form-group">
                <label className="form-label">Model Type</label>
                <select
                  className="form-select"
                  value={form.model_type}
                  onChange={e => setForm(f => ({ ...f, model_type: e.target.value }))}
                >
                  <option value="all">All Models</option>
                  <option value="logistic">Logistic Regression</option>
                  <option value="random_forest">Random Forest</option>
                  <option value="xgboost">XGBoost</option>
                  <option value="lgbm">LightGBM</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Version Tag (optional)</label>
                <input
                  className="form-input"
                  type="text"
                  placeholder="e.g. v2.0.0"
                  value={form.version}
                  onChange={e => setForm(f => ({ ...f, version: e.target.value }))}
                />
              </div>
              <div className="form-group" style={{ flex: "0 0 auto", minWidth: "unset" }}>
                <label className="form-label">&nbsp;</label>
                <button className="btn btn-primary" type="submit" disabled={starting}>
                  {starting ? "Queueing…" : "▶ Start Training"}
                </button>
              </div>
            </div>
          </form>
        </div>

        {/* Training history */}
        <div className="card" style={{ marginTop: 20 }}>
          <h3 className="card__title">Training History</h3>
          <p className="card__subtitle">Select a run to view detailed metrics and logs.</p>
          <ErrorBanner message={histError} />
          {histLoading ? (
            <StateView icon="⟳" title="Loading history…" />
          ) : history.length === 0 ? (
            <StateView icon="▤" title="No training runs" desc="Start a training run to see history here." />
          ) : (
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Run ID</th>
                    <th>Model Type</th>
                    <th>Version</th>
                    <th>Status</th>
                    <th>Started</th>
                    <th>Details</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((run) => {
                    const id = run.run_id ?? run.id;
                    return (
                      <tr key={id}>
                        <td><code style={{ fontSize: 11 }}>{id}</code></td>
                        <td>{run.model_type ?? "—"}</td>
                        <td>{run.version ?? "—"}</td>
                        <td><StatusBadge status={run.status} /></td>
                        <td>{run.started_at ? new Date(run.started_at).toLocaleString() : "—"}</td>
                        <td>
                          <button className="btn btn-secondary btn-sm" onClick={() => openRunDetails(id)}>
                            {selectedRun === id ? "Hide" : "View"}
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Run detail panel */}
        {selectedRun && (
          <div className="card" style={{ marginTop: 20 }}>
            <h3 className="card__title">Run Details</h3>
            <p className="card__subtitle" style={{ fontFamily: "monospace" }}>{selectedRun}</p>

            {detailLoading ? (
              <StateView icon="⟳" title="Loading details…" />
            ) : (
              <>
                {runStatus && (
                  <div className="kpi-grid" style={{ marginBottom: 20 }}>
                    <KpiCard label="Status" value={runStatus.status ?? "—"} />
                    <KpiCard label="Progress" value={runStatus.progress ?? "—"} />
                    <KpiCard label="Current Stage" value={runStatus.current_stage ?? "—"} />
                  </div>
                )}

                {metricKeys.length > 0 && (
                  <div style={{ marginBottom: 20 }}>
                    <Plot
                      data={[{
                        type: "bar",
                        x: metricKeys,
                        y: metricKeys.map(k => runMetrics[k]),
                        marker: { color: "#3478f6" },
                        hovertemplate: "%{x}: %{y:.4f}<extra></extra>",
                      }]}
                      layout={{
                        autosize: true,
                        paper_bgcolor: "transparent",
                        plot_bgcolor: "transparent",
                        font: { family: "Inter, sans-serif", size: 12, color: "#34415f" },
                        margin: { l: 48, r: 20, t: 20, b: 60 },
                        yaxis: { title: "Value", gridcolor: "#e8edf5" },
                        title: { text: "Evaluation Metrics", font: { size: 14 } },
                      }}
                      config={PLOTLY_CONFIG}
                      useResizeHandler
                      style={{ width: "100%", height: "280px" }}
                    />
                  </div>
                )}

                {runLogs.length > 0 && (
                  <div>
                    <h4 style={{ fontSize: 13, fontWeight: 700, color: "#34415f", marginBottom: 8 }}>Training Logs</h4>
                    <div style={{
                      background: "#f8fafd", border: "1px solid #e8edf5", borderRadius: 10,
                      padding: 14, maxHeight: 220, overflowY: "auto", fontFamily: "monospace",
                      fontSize: 11.5, color: "#2a3a5a", lineHeight: 1.7,
                    }}>
                      {runLogs.map((log, i) => <div key={i}>{typeof log === "string" ? log : JSON.stringify(log)}</div>)}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

