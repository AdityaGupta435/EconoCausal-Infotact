/**
 * EconoCausal — Causal Inference Page (Phase 10)
 * Run causal analyses, view ATE / CI / diagnostics / history.
 */

import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import { causal } from "../services/api";
import { SectionHeader, ErrorBanner, SuccessBanner, StateView, StatusBadge, KpiCard } from "../components/common";
import "../styles/common.css";

const PLOTLY_CONFIG = { responsive: true, displayModeBar: false };
const LAYOUT_BASE = {
  autosize: true,
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { family: "Inter, sans-serif", size: 12, color: "#34415f" },
  margin: { l: 55, r: 20, t: 24, b: 50 },
};

export default function CausalPage() {
  const [history, setHistory] = useState([]);
  const [histLoading, setHistLoading] = useState(true);
  const [histError, setHistError] = useState("");

  const [form, setForm] = useState({ treatment_col: "treatment", outcome_col: "purchase" });
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState("");
  const [currentResult, setCurrentResult] = useState(null);

  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [diagnostics, setDiagnostics] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  async function loadHistory() {
    setHistLoading(true);
    setHistError("");
    try {
      const res = await causal.history();
      setHistory(Array.isArray(res?.data) ? res.data : []);
    } catch (err) {
      setHistError(err.message);
    } finally {
      setHistLoading(false);
    }
  }

  useEffect(() => { loadHistory(); }, []);

  async function handleAnalyze(e) {
    e.preventDefault();
    setAnalyzeError("");
    setCurrentResult(null);
    setAnalyzing(true);
    try {
      const res = await causal.analyze({
        treatment_col: form.treatment_col,
        outcome_col: form.outcome_col,
      });
      setCurrentResult(res?.data);
      await loadHistory();
    } catch (err) {
      setAnalyzeError(err.message);
    } finally {
      setAnalyzing(false);
    }
  }

  async function openDetail(id) {
    if (selectedId === id) { setSelectedId(null); setDetail(null); setDiagnostics(null); return; }
    setSelectedId(id);
    setDetailLoading(true);
    setDetail(null);
    setDiagnostics(null);
    try {
      const [r, d] = await Promise.all([causal.results(id), causal.diagnostics(id)]);
      setDetail(r?.data);
      setDiagnostics(d?.data);
    } catch {}
    finally { setDetailLoading(false); }
  }

  // Confidence interval chart data
  function ciChartData(result) {
    if (!result) return null;
    const ate = result.estimated_ate ?? result.summary?.estimated_ate;
    const ci_lo = result.ci_lower ?? result.summary?.ci_lower;
    const ci_hi = result.ci_upper ?? result.summary?.ci_upper;
    if (ate === undefined) return null;
    return { ate, ci_lo, ci_hi };
  }

  const ci = currentResult ? ciChartData(currentResult) : null;
  const histCi = detail ? ciChartData(detail) : null;

  function CIChart({ data, title }) {
    if (!data) return null;
    const { ate, ci_lo, ci_hi } = data;
    return (
      <Plot
        data={[{
          type: "scatter",
          mode: "markers",
          x: [ate],
          y: ["ATE"],
          error_x: {
            type: "data",
            array: [(ci_hi ?? ate) - ate],
            arrayminus: [ate - (ci_lo ?? ate)],
            visible: true,
            color: "#3478f6",
          },
          marker: { color: "#3478f6", size: 10 },
          hovertemplate: `ATE: %{x:.4f}<extra></extra>`,
        }]}
        layout={{
          ...LAYOUT_BASE,
          margin: { l: 60, r: 30, t: 30, b: 50 },
          xaxis: { title: "Effect Size", gridcolor: "#e8edf5", zeroline: true, zerolinecolor: "#c9d4e8" },
          yaxis: { fixedrange: true },
          title: { text: title ?? "ATE with Confidence Interval", font: { size: 13 } },
        }}
        config={PLOTLY_CONFIG}
        useResizeHandler
        style={{ width: "100%", height: "220px" }}
      />
    );
  }

  return (
    <div>
      <SectionHeader
        eyebrow="CAUSAL INFERENCE"
        title="Causal Analysis"
        subtitle="Estimate Average Treatment Effects (ATE) and conditional effects using causal methods."
      />
      <div className="page-body">
        <div className="info-banner">
          Estimated causal effects depend on the validity of identification assumptions (overlap, unconfoundedness).
          Interpret results carefully. Do not conflate estimated effects with true causal effects on new populations.
        </div>

        {/* Run analysis form */}
        <div className="card">
          <h3 className="card__title">Run Causal Analysis</h3>
          <p className="card__subtitle">Specify the treatment and outcome columns.</p>
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
                <KpiCard label="Estimated ATE"
                  value={(currentResult.estimated_ate ?? currentResult.summary?.estimated_ate ?? "—").toFixed?.(4) ?? "—"}
                  detail="Avg Treatment Effect" accent="#3478f6" />
                <KpiCard label="95% CI Lower"
                  value={(currentResult.ci_lower ?? currentResult.summary?.ci_lower ?? "—").toFixed?.(4) ?? "—"}
                  detail="Lower confidence bound" />
                <KpiCard label="95% CI Upper"
                  value={(currentResult.ci_upper ?? currentResult.summary?.ci_upper ?? "—").toFixed?.(4) ?? "—"}
                  detail="Upper confidence bound" />
                <KpiCard label="Method"
                  value={currentResult.method ?? currentResult.summary?.method ?? "—"} />
              </div>
              <CIChart data={ci} title="Current Analysis — ATE with 95% CI" />
            </div>
          )}
        </div>

        {/* History */}
        <div className="card" style={{ marginTop: 20 }}>
          <h3 className="card__title">Analysis History</h3>
          <p className="card__subtitle">Select a past analysis to view full results and diagnostics.</p>
          <ErrorBanner message={histError} />
          {histLoading ? (
            <StateView icon="⟳" title="Loading history…" />
          ) : history.length === 0 ? (
            <StateView icon="♧" title="No causal analyses yet" desc="Run an analysis to populate history." />
          ) : (
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Analysis ID</th>
                    <th>Treatment</th>
                    <th>Outcome</th>
                    <th>ATE</th>
                    <th>Method</th>
                    <th>Created</th>
                    <th>Details</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((h) => {
                    const id = h.analysis_id ?? h.id;
                    return (
                      <tr key={id}>
                        <td><code style={{ fontSize: 11 }}>{id}</code></td>
                        <td>{h.treatment_col ?? "—"}</td>
                        <td>{h.outcome_col ?? "—"}</td>
                        <td>{(h.ate ?? h.estimated_ate)?.toFixed?.(4) ?? "—"}</td>
                        <td>{h.method ?? "—"}</td>
                        <td>{h.created_at ? new Date(h.created_at).toLocaleString() : "—"}</td>
                        <td>
                          <button className="btn btn-secondary btn-sm" onClick={() => openDetail(id)}>
                            {selectedId === id ? "Hide" : "View"}
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

        {/* Detail panel */}
        {selectedId && (
          <div className="card" style={{ marginTop: 20 }}>
            <h3 className="card__title">Analysis Detail</h3>
            {detailLoading ? <StateView icon="⟳" title="Loading…" /> : (
              <>
                {histCi && <CIChart data={histCi} title="ATE with 95% CI" />}
                {diagnostics && (
                  <div style={{ marginTop: 20 }}>
                    <h4 style={{ fontSize: 13, fontWeight: 700, color: "#34415f", marginBottom: 10 }}>
                      Propensity Score Diagnostics
                    </h4>
                    <div className="table-wrap">
                      <table className="data-table">
                        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                        <tbody>
                          {Object.entries(diagnostics).map(([k, v]) => (
                            <tr key={k}>
                              <td>{k.replace(/_/g, " ")}</td>
                              <td>{typeof v === "number" ? v.toFixed(4) : String(v)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
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

