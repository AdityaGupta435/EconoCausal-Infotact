/**
 * EconoCausal — Promotion Optimization Page (Phase 10)
 * Run budget-constrained promotion allocation and view results.
 */

import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import { promotions } from "../services/api";
import { SectionHeader, ErrorBanner, SuccessBanner, StateView, KpiCard, StatusBadge } from "../components/common";
import "../styles/common.css";

const PC = { responsive: true, displayModeBar: false };
const LB = {
  autosize: true,
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { family: "Inter, sans-serif", size: 12, color: "#34415f" },
  margin: { l: 55, r: 20, t: 24, b: 50 },
};

const createCandidate = () => ({
  _key: Date.now() + Math.random(),
  customer_id: "",
  baseline_price: "",
  predicted_ite: "",
});

export default function PromotionsPage() {
  const [candidates, setCandidates] = useState([createCandidate()]);
  const [budget, setBudget] = useState("1000");
  const [discountPct, setDiscountPct] = useState("10");

  const [optimizing, setOptimizing] = useState(false);
  const [optResult, setOptResult] = useState(null);
  const [optError, setOptError] = useState("");

  const [campaigns, setCampaigns] = useState([]);
  const [results, setResults] = useState([]);
  const [budgetAnalysis, setBudgetAnalysis] = useState(null);
  const [listLoading, setListLoading] = useState(true);

  async function loadData() {
    setListLoading(true);
    try {
      const [c, r, b] = await Promise.all([
        promotions.campaigns(),
        promotions.results(),
        promotions.budgetAnalysis(),
      ]);
      setCampaigns(Array.isArray(c?.data) ? c.data : []);
      setResults(Array.isArray(r?.data) ? r.data : []);
      setBudgetAnalysis(b?.data);
    } catch {}
    finally { setListLoading(false); }
  }

  useEffect(() => { loadData(); }, []);

  function updateCandidate(key, field, value) {
    setCandidates(cs => cs.map(c => c._key === key ? { ...c, [field]: value } : c));
  }

  async function handleOptimize(e) {
    e.preventDefault();
    setOptError("");
    setOptResult(null);

    const parsedBudget = parseFloat(budget);
    const parsedDiscount = parseFloat(discountPct);
    if (!isFinite(parsedBudget) || parsedBudget <= 0) { setOptError("Enter a valid positive budget."); return; }
    if (!isFinite(parsedDiscount) || parsedDiscount <= 0 || parsedDiscount > 100) { setOptError("Discount must be 1–100%."); return; }

    const validCandidates = candidates.filter(c =>
      c.baseline_price !== "" && c.predicted_ite !== ""
    );
    if (validCandidates.length === 0) { setOptError("Add at least one candidate with baseline price and ITE."); return; }

    setOptimizing(true);
    try {
      const payload = {
        candidates: validCandidates.map(c => ({
          customer_id: c.customer_id || `CUST_${c._key.toString().slice(-4)}`,
          baseline_price: parseFloat(c.baseline_price),
          predicted_ite: parseFloat(c.predicted_ite),
        })),
        budget: parsedBudget,
        discount_pct: parsedDiscount,
      };
      const res = await promotions.optimize(payload);
      setOptResult(res?.data);
      await loadData();
    } catch (err) {
      setOptError(err.message);
    } finally {
      setOptimizing(false);
    }
  }

  // Budget utilization chart
  const budgetData = budgetAnalysis
    ? [
        { label: "Allocated", value: budgetAnalysis.allocated_budget ?? 0, color: "#3478f6" },
        { label: "Remaining", value: (budgetAnalysis.total_budget ?? 0) - (budgetAnalysis.allocated_budget ?? 0), color: "#e8edf5" },
      ]
    : [];

  return (
    <div>
      <SectionHeader
        eyebrow="PROMOTION OPTIMIZATION"
        title="Budget Optimizer"
        subtitle="Allocate promotion discounts to maximize incremental conversions under a fixed budget."
      />
      <div className="page-body">
        <div className="info-banner">
          ROI shown is incremental revenue ROI, not profit ROI. Unit cost is not available from this dataset.
          Optimization uses MILP; "optimal" means budget-feasible maximum expected lift, not guaranteed outcomes.
        </div>

        {/* Optimization form */}
        <div className="card">
          <h3 className="card__title">Configure Optimization</h3>
          <ErrorBanner message={optError} />

          <form onSubmit={handleOptimize} style={{ marginTop: 16 }}>
            <div className="form-row" style={{ marginBottom: 16 }}>
              <div className="form-group">
                <label className="form-label">Budget (\$)</label>
                <input className="form-input" type="number" min="0" step="1"
                  value={budget} onChange={e => setBudget(e.target.value)} />
              </div>
              <div className="form-group">
                <label className="form-label">Discount (%)</label>
                <input className="form-input" type="number" min="1" max="100" step="0.5"
                  value={discountPct} onChange={e => setDiscountPct(e.target.value)} />
              </div>
            </div>

            <h4 style={{ fontSize: 13, fontWeight: 700, color: "#34415f", marginBottom: 10 }}>Candidates</h4>
            {candidates.map((c, i) => (
              <div key={c._key} className="form-row" style={{ marginBottom: 10, alignItems: "center" }}>
                <div className="form-group" style={{ flex: "1 1 140px" }}>
                  <label className="form-label">{i === 0 ? "Customer ID" : ""}</label>
                  <input className="form-input" placeholder={`Customer ${i + 1}`}
                    value={c.customer_id}
                    onChange={e => updateCandidate(c._key, "customer_id", e.target.value)} />
                </div>
                <div className="form-group" style={{ flex: "1 1 110px" }}>
                  <label className="form-label">{i === 0 ? "Baseline Price (\$)" : ""}</label>
                  <input className="form-input" type="number" min="0" step="0.01" placeholder="100"
                    value={c.baseline_price}
                    onChange={e => updateCandidate(c._key, "baseline_price", e.target.value)} />
                </div>
                <div className="form-group" style={{ flex: "1 1 110px" }}>
                  <label className="form-label">{i === 0 ? "Predicted ITE" : ""}</label>
                  <input className="form-input" type="number" step="0.001" placeholder="0.10"
                    value={c.predicted_ite}
                    onChange={e => updateCandidate(c._key, "predicted_ite", e.target.value)} />
                </div>
                <div className="form-group" style={{ flex: "0 0 auto", minWidth: "unset" }}>
                  <label className="form-label">{i === 0 ? " " : ""}</label>
                  <button type="button" className="btn btn-danger btn-sm"
                    onClick={() => setCandidates(cs => cs.filter(x => x._key !== c._key))}
                    disabled={candidates.length === 1}>
                    ✕
                  </button>
                </div>
              </div>
            ))}

            <div style={{ display: "flex", gap: 10, marginTop: 10 }}>
              <button type="button" className="btn btn-secondary btn-sm"
                onClick={() => setCandidates(cs => [...cs, createCandidate()])}>
                + Add Candidate
              </button>
              <button type="submit" className="btn btn-primary" disabled={optimizing}>
                {optimizing ? "Optimizing…" : "▶ Run Optimization"}
              </button>
            </div>
          </form>

          {optResult && (
            <div style={{ marginTop: 24 }}>
              <div className="kpi-grid">
                <KpiCard label="Status" value={optResult.status} accent="#3478f6" />
                <KpiCard label="Allocated Budget"
                  value={`$${(optResult.allocated_budget ?? 0).toFixed(2)}`}
                  detail={`of $${budget} budget`} />
                <KpiCard label="Customers Targeted"
                  value={optResult.selected_count ?? optResult.customers_targeted ?? "—"} />
                <KpiCard label="Expected Lift"
                  value={(optResult.expected_incremental_conversions ?? optResult.expected_lift ?? "—").toFixed?.(4) ?? "—"} />
                <KpiCard label="Expected Incremental Revenue"
                  value={optResult.expected_incremental_revenue != null
                    ? `$${optResult.expected_incremental_revenue.toFixed(2)}`
                    : "—"} />
              </div>

              {Array.isArray(optResult.selected_customers) && optResult.selected_customers.length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <h4 style={{ fontSize: 13, fontWeight: 700, color: "#34415f", marginBottom: 8 }}>
                    Selected Customers
                  </h4>
                  <div className="table-wrap">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Customer ID</th>
                          <th>Discount</th>
                          <th>Baseline Price</th>
                          <th>Predicted ITE</th>
                        </tr>
                      </thead>
                      <tbody>
                        {optResult.selected_customers.map((cust, i) => (
                          <tr key={i}>
                            <td>{cust.customer_id ?? "—"}</td>
                            <td>{cust.discount_pct != null ? `${cust.discount_pct}%` : "—"}</td>
                            <td>${(cust.baseline_price ?? 0).toFixed(2)}</td>
                            <td>{(cust.predicted_ite ?? 0).toFixed(4)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Budget analysis chart */}
        {budgetAnalysis && (
          <div className="chart-grid-2" style={{ marginTop: 20 }}>
            <div className="chart-card">
              <div className="chart-card__header">
                <h3 className="chart-card__title">Budget Utilization</h3>
                <p className="chart-card__desc">Allocated vs. remaining budget across campaigns.</p>
              </div>
              <Plot
                data={[{
                  type: "pie",
                  labels: budgetData.map(d => d.label),
                  values: budgetData.map(d => d.value),
                  marker: { colors: budgetData.map(d => d.color) },
                  hovertemplate: "%{label}: $%{value:.2f}<extra></extra>",
                }]}
                layout={{
                  ...LB,
                  margin: { l: 20, r: 20, t: 20, b: 20 },
                  showlegend: true,
                }}
                config={PC}
                useResizeHandler
                style={{ width: "100%", height: "260px" }}
              />
            </div>
            <div className="card">
              <h3 className="card__title">Budget Analysis</h3>
              <div className="table-wrap" style={{ marginTop: 10 }}>
                <table className="data-table">
                  <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                  <tbody>
                    {Object.entries(budgetAnalysis).map(([k, v]) => (
                      <tr key={k}>
                        <td>{k.replace(/_/g, " ")}</td>
                        <td>{typeof v === "number" ? v.toFixed(2) : String(v)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Results history */}
        <div className="card" style={{ marginTop: 20 }}>
          <h3 className="card__title">Optimization History</h3>
          {listLoading ? <StateView icon="⟳" title="Loading…" /> : results.length === 0 ? (
            <StateView icon="◎" title="No optimization results yet" desc="Run an optimization to see results here." />
          ) : (
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Status</th>
                    <th>Budget</th>
                    <th>Allocated</th>
                    <th>Targeted</th>
                    <th>Created</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r, i) => (
                    <tr key={i}>
                      <td><code style={{ fontSize: 11 }}>{r.id ?? r.optimization_id ?? i}</code></td>
                      <td><StatusBadge status={r.status} /></td>
                      <td>${(r.budget ?? 0).toFixed(2)}</td>
                      <td>${(r.allocated_budget ?? 0).toFixed(2)}</td>
                      <td>{r.selected_count ?? "—"}</td>
                      <td>{r.created_at ? new Date(r.created_at).toLocaleString() : "—"}</td>
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

