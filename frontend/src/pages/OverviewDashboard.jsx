/**
 * EconoCausal — Overview Dashboard Page (Phase 10)
 * Displays KPIs and charts using the Phase 9 dashboard API endpoints.
 * No hardcoded values — all data sourced from the backend.
 */

import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import { dashboard } from "../services/api";
import { KpiCard, SectionHeader, ErrorBanner, StateView } from "../components/common";
import "../styles/common.css";

const PLOTLY_LAYOUT_BASE = {
  autosize: true,
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { family: "Inter, sans-serif", size: 12, color: "#34415f" },
  margin: { l: 48, r: 20, t: 20, b: 50 },
};

const PLOTLY_CONFIG = { responsive: true, displayModeBar: false };

export default function OverviewDashboard() {
  const [overview, setOverview] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [modelPerf, setModelPerf] = useState(null);
  const [activity, setActivity] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError("");
      try {
        const [ov, mt, mp, ra] = await Promise.all([
          dashboard.overview(),
          dashboard.metrics(),
          dashboard.modelPerformance(),
          dashboard.recentActivity(),
        ]);
        if (!cancelled) {
          setOverview(ov?.data ?? null);
          setMetrics(mt?.data ?? null);
          setModelPerf(mp?.data ?? null);
          setActivity(Array.isArray(ra?.data) ? ra.data : []);
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, []);

  if (loading) {
    return (
      <div className="page-body" style={{ marginTop: 28 }}>
        <StateView icon="⟳" title="Loading dashboard..." desc="Fetching data from backend." />
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <SectionHeader eyebrow="WORKSPACE" title="Overview Dashboard" />
        <div className="page-body">
          <ErrorBanner message={error} />
        </div>
      </div>
    );
  }

  // ── KPI values ───────────────────────────────────
  const kpis = [
    { label: "Total Datasets",    value: overview?.total_datasets ?? "—",             detail: "Registered in MongoDB" },
    { label: "Records Processed", value: (overview?.total_records_processed ?? 0).toLocaleString(), detail: "Across all datasets" },
    { label: "Training Runs",     value: overview?.training_runs ?? "—",              detail: "All time" },
    { label: "Registered Models", value: overview?.registered_models ?? "—",          detail: "In model registry" },
    { label: "Pricing Recs",      value: overview?.pricing_recommendations ?? "—",    detail: "Generated recommendations" },
    { label: "Campaigns",         value: overview?.promotion_campaigns ?? "—",        detail: "Promotion campaigns" },
  ];

  // ── Model performance chart ───────────────────────
  const models = Array.isArray(modelPerf?.models) ? modelPerf.models : [];
  const modelNames = models.map((m) => m.model_name || m.name || "Model");
  const modelAUC   = models.map((m) => m.roc_auc   ?? m.auc  ?? null);
  const modelF1    = models.map((m) => m.f1_score  ?? m.f1   ?? null);

  // ── Activity list ─────────────────────────────────
  const recentRows = activity.slice(0, 8);

  return (
    <div>
      <SectionHeader
        eyebrow="WORKSPACE"
        title="Overview Dashboard"
        subtitle="High-level platform metrics from the EconoCausal backend."
      />

      <div className="page-body">
        {/* KPI Cards */}
        <div className="kpi-grid">
          {kpis.map((k) => (
            <KpiCard key={k.label} label={k.label} value={k.value} detail={k.detail} accent="#3478f6" />
          ))}
        </div>

        {/* Charts */}
        <div className="chart-grid-2">
          {/* Model Performance Chart */}
          <div className="chart-card">
            <div className="chart-card__header">
              <h3 className="chart-card__title">Model Performance</h3>
              <p className="chart-card__desc">ROC-AUC and F1 score for registered models.</p>
            </div>
            {models.length === 0 ? (
              <StateView icon="▤" title="No models registered" desc="Run a training job to populate this chart." />
            ) : (
              <Plot
                data={[
                  { type: "bar", name: "ROC-AUC", x: modelNames, y: modelAUC, marker: { color: "#3478f6" }, hovertemplate: "%{x}: %{y:.4f}<extra></extra>" },
                  { type: "bar", name: "F1 Score", x: modelNames, y: modelF1,  marker: { color: "#16a394" }, hovertemplate: "%{x}: %{y:.4f}<extra></extra>" },
                ]}
                layout={{
                  ...PLOTLY_LAYOUT_BASE,
                  barmode: "group",
                  yaxis: { title: "Score", range: [0, 1], gridcolor: "#e8edf5" },
                  legend: { orientation: "h", y: -0.2 },
                }}
                config={PLOTLY_CONFIG}
                useResizeHandler
                style={{ width: "100%", height: "280px" }}
              />
            )}
          </div>

          {/* Dataset distribution (from metrics) */}
          <div className="chart-card">
            <div className="chart-card__header">
              <h3 className="chart-card__title">Dataset Breakdown</h3>
              <p className="chart-card__desc">Records per registered dataset (top 6).</p>
            </div>
            {!(metrics?.datasets?.length) ? (
              <StateView icon="▥" title="No dataset metrics" desc="Upload datasets to populate this chart." />
            ) : (
              <Plot
                data={[{
                  type: "bar",
                  orientation: "h",
                  x: metrics.datasets.slice(0, 6).map((d) => d.row_count ?? 0),
                  y: metrics.datasets.slice(0, 6).map((d) => d.name ?? "?"),
                  marker: { color: "#8b5cf6" },
                  hovertemplate: "%{y}: %{x:,}<extra></extra>",
                }]}
                layout={{
                  ...PLOTLY_LAYOUT_BASE,
                  margin: { l: 130, r: 20, t: 20, b: 50 },
                  xaxis: { title: "Rows", gridcolor: "#e8edf5" },
                }}
                config={PLOTLY_CONFIG}
                useResizeHandler
                style={{ width: "100%", height: "280px" }}
              />
            )}
          </div>
        </div>

        {/* Recent Activity */}
        <div className="card">
          <h3 className="card__title">Recent Activity</h3>
          <p className="card__subtitle">Last 8 platform events from the audit log.</p>
          {recentRows.length === 0 ? (
            <StateView icon="○" title="No activity yet" desc="Actions performed on the platform will appear here." />
          ) : (
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Action</th>
                    <th>User</th>
                    <th>Resource</th>
                    <th>Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {recentRows.map((row, i) => (
                    <tr key={i}>
                      <td>{row.action ?? row.event ?? "—"}</td>
                      <td>{row.username ?? row.user ?? "—"}</td>
                      <td>{row.resource ?? row.target ?? "—"}</td>
                      <td>{row.timestamp ? new Date(row.timestamp).toLocaleString() : "—"}</td>
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

