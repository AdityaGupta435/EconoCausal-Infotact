/**
 * EconoCausal — Dynamic Pricing Page (Phase 10)
 * Single and batch pricing recommendations via Phase 9 pricing API.
 */

import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import { pricing } from "../services/api";
import { SectionHeader, ErrorBanner, SuccessBanner, StateView, KpiCard } from "../components/common";
import "../styles/common.css";

const PC = { responsive: true, displayModeBar: false };
const LB = {
  autosize: true,
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { family: "Inter, sans-serif", size: 12, color: "#34415f" },
  margin: { l: 55, r: 20, t: 24, b: 50 },
};

const DEFAULT_CANDIDATE = {
  customer_id: "",
  category: "Electronics",
  current_price: "",
  customer_history: "",
  seasonality: "1.0",
  treatment: "0",
};

export default function PricingPage() {
  const [form, setForm] = useState({ ...DEFAULT_CANDIDATE });
  const [recommending, setRecommending] = useState(false);
  const [recResult, setRecResult] = useState(null);
  const [recError, setRecError] = useState("");

  const [history, setHistory] = useState([]);
  const [histLoading, setHistLoading] = useState(true);

  const [elasticity, setElasticity] = useState(null);
  const [performance, setPerformance] = useState(null);

  async function loadData() {
    setHistLoading(true);
    try {
      const [h, e, p] = await Promise.all([
        pricing.history(),
        pricing.elasticity(),
        pricing.performance(),
      ]);
      setHistory(Array.isArray(h?.data) ? h.data : []);
      setElasticity(e?.data);
      setPerformance(p?.data);
    } catch {}
    finally { setHistLoading(false); }
  }

  useEffect(() => { loadData(); }, []);

  async function handleRecommend(e) {
    e.preventDefault();
    setRecError("");
    setRecResult(null);
    setRecommending(true);
    try {
      const payload = {
        customer_id: form.customer_id || `CUST_${Date.now()}`,
        category: form.category,
        current_price: parseFloat(form.current_price) || 50,
        customer_history: parseFloat(form.customer_history) || 100,
        seasonality: parseFloat(form.seasonality) || 1.0,
        treatment: parseInt(form.treatment) || 0,
      };
      const res = await pricing.recommend(payload);
      setRecResult(res?.data);
      await loadData();
    } catch (err) {
      setRecError(err.message);
    } finally {
      setRecommending(false);
    }
  }

  // Recommendation history chart — price changes
  const histPrices = history.slice(0, 20);
  const hasHistChart = histPrices.length > 0;

  // Elasticity summary
  const elasticityEntries = elasticity
    ? Object.entries(elasticity).filter(([, v]) => typeof v === "number")
    : [];

  return (
    <div>
      <SectionHeader
        eyebrow="DYNAMIC PRICING"
        title="Pricing Recommendations"
        subtitle="Generate price recommendations based on the fitted pricing model and business constraints."
      />
      <div className="page-body">
        <div className="info-banner">
          Pricing recommendations are based on estimated price-demand relationships from observational data.
          Elasticity estimates reflect correlational patterns. Exercise caution before making large price changes.
        </div>

        {/* Single recommendation form */}
        <div className="card">
          <h3 className="card__title">Generate Recommendation</h3>
          <p className="card__subtitle">Enter customer and product details to get a recommended price.</p>
          <ErrorBanner message={recError} />
          <form onSubmit={handleRecommend} style={{ marginTop: 16 }}>
            <div className="form-row">
              <div className="form-group">
                <label className="form-label">Customer ID</label>
                <input className="form-input" placeholder="e.g. CUST_001"
                  value={form.customer_id}
                  onChange={e => setForm(f => ({ ...f, customer_id: e.target.value }))} />
              </div>
              <div className="form-group">
                <label className="form-label">Category</label>
                <select className="form-select" value={form.category}
                  onChange={e => setForm(f => ({ ...f, category: e.target.value }))}>
                  <option>Electronics</option>
                  <option>Fashion</option>
                  <option>Home</option>
                  <option>Sports</option>
                  <option>Beauty</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Current Price (\$)</label>
                <input className="form-input" type="number" min="0" step="0.01" placeholder="50.00"
                  value={form.current_price}
                  onChange={e => setForm(f => ({ ...f, current_price: e.target.value }))} />
              </div>
              <div className="form-group">
                <label className="form-label">Customer History (\$)</label>
                <input className="form-input" type="number" min="0" step="0.01" placeholder="100.00"
                  value={form.customer_history}
                  onChange={e => setForm(f => ({ ...f, customer_history: e.target.value }))} />
              </div>
              <div className="form-group">
                <label className="form-label">Seasonality</label>
                <input className="form-input" type="number" min="0" step="0.01" placeholder="1.0"
                  value={form.seasonality}
                  onChange={e => setForm(f => ({ ...f, seasonality: e.target.value }))} />
              </div>
            </div>
            <div style={{ marginTop: 14 }}>
              <button className="btn btn-primary" type="submit" disabled={recommending}>
                {recommending ? "Generating…" : "▶ Get Recommendation"}
              </button>
            </div>
          </form>

          {recResult && (
            <div style={{ marginTop: 24 }}>
              <div className="kpi-grid">
                <KpiCard label="Recommended Price"
                  value={`$${(recResult.recommended_price ?? 0).toFixed(2)}`}
                  accent="#3478f6" />
                <KpiCard label="Current Price"
                  value={`$${(recResult.current_price ?? 0).toFixed(2)}`} />
                <KpiCard label="Price Change"
                  value={recResult.price_change_pct != null
                    ? `${recResult.price_change_pct > 0 ? "+" : ""}${recResult.price_change_pct.toFixed(1)}%`
                    : "—"} />
                <KpiCard label="Arc Elasticity"
                  value={recResult.arc_elasticity?.toFixed?.(3) ?? "—"}
                  detail="Price–demand relationship" />
                <KpiCard label="Expected Demand"
                  value={recResult.expected_demand?.toFixed?.(2) ?? "—"} />
                <KpiCard label="Expected Revenue"
                  value={recResult.expected_revenue != null
                    ? `$${recResult.expected_revenue.toFixed(2)}`
                    : "—"} />
              </div>
              {recResult.extrapolation_warning && (
                <div className="error-banner" style={{ marginTop: 12 }}>
                  ⚠ Extrapolation warning: {recResult.extrapolation_warning}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Charts */}
        <div className="chart-grid-2" style={{ marginTop: 20 }}>
          {/* Price history scatter */}
          <div className="chart-card">
            <div className="chart-card__header">
              <h3 className="chart-card__title">Recommendation History</h3>
              <p className="chart-card__desc">Current vs. recommended price for recent recommendations.</p>
            </div>
            {!hasHistChart ? (
              <StateView icon="▥" title="No recommendations yet" desc="Generate a recommendation to see this chart." />
            ) : (
              <Plot
                data={[
                  {
                    type: "scatter", mode: "markers", name: "Current",
                    x: histPrices.map((_, i) => i + 1),
                    y: histPrices.map(r => r.current_price ?? 0),
                    marker: { color: "#94a3b8", size: 7 },
                  },
                  {
                    type: "scatter", mode: "markers", name: "Recommended",
                    x: histPrices.map((_, i) => i + 1),
                    y: histPrices.map(r => r.recommended_price ?? 0),
                    marker: { color: "#3478f6", size: 7 },
                  },
                ]}
                layout={{
                  ...LB,
                  xaxis: { title: "Recommendation #" },
                  yaxis: { title: "Price ($)", gridcolor: "#e8edf5" },
                  legend: { orientation: "h", y: -0.2 },
                }}
                config={PC}
                useResizeHandler
                style={{ width: "100%", height: "280px" }}
              />
            )}
          </div>

          {/* Elasticity summary */}
          <div className="chart-card">
            <div className="chart-card__header">
              <h3 className="chart-card__title">Price Elasticity Summary</h3>
              <p className="chart-card__desc">Key elasticity metrics from the fitted model.</p>
            </div>
            {elasticityEntries.length === 0 ? (
              <StateView icon="◎" title="No elasticity data" desc="Elasticity is computed after pricing runs." />
            ) : (
              <Plot
                data={[{
                  type: "bar",
                  x: elasticityEntries.map(([k]) => k.replace(/_/g, " ")),
                  y: elasticityEntries.map(([, v]) => v),
                  marker: { color: "#16a394" },
                  hovertemplate: "%{x}: %{y:.4f}<extra></extra>",
                }]}
                layout={{
                  ...LB,
                  yaxis: { title: "Value", gridcolor: "#e8edf5" },
                }}
                config={PC}
                useResizeHandler
                style={{ width: "100%", height: "280px" }}
              />
            )}
          </div>
        </div>

        {/* History table */}
        <div className="card" style={{ marginTop: 20 }}>
          <h3 className="card__title">Recent Recommendations</h3>
          {histLoading ? <StateView icon="⟳" title="Loading…" /> : history.length === 0 ? (
            <StateView icon="▥" title="No recommendation history" />
          ) : (
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Customer</th>
                    <th>Category</th>
                    <th>Current Price</th>
                    <th>Recommended</th>
                    <th>Change %</th>
                    <th>Elasticity</th>
                  </tr>
                </thead>
                <tbody>
                  {history.slice(0, 20).map((r, i) => (
                    <tr key={i}>
                      <td>{r.customer_id ?? "—"}</td>
                      <td>{r.category ?? "—"}</td>
                      <td>${(r.current_price ?? 0).toFixed(2)}</td>
                      <td>${(r.recommended_price ?? 0).toFixed(2)}</td>
                      <td>{r.price_change_pct != null
                        ? `${r.price_change_pct > 0 ? "+" : ""}${r.price_change_pct.toFixed(1)}%`
                        : "—"}</td>
                      <td>{r.arc_elasticity?.toFixed?.(3) ?? "—"}</td>
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

