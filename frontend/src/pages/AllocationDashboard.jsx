
import { useState } from "react";
import "./AllocationDashboard.css";

const createCandidate = () => ({
  id: `C${Date.now().toString().slice(-6)}`,
  baseline_price: "",
  predicted_ite: "",
});

function MetricCard({ label, value, detail }) {
  return (
    <article className="allocation-metric-card">
      <p>{label}</p>
      <strong>{value}</strong>
      <span>{detail}</span>
    </article>
  );
}

export default function AllocationDashboard() {
  const [candidates, setCandidates] = useState([createCandidate()]);
  const [budget, setBudget] = useState("1000");
  const [discountPct, setDiscountPct] = useState("10");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const updateCandidate = (index, field, value) => {
    setCandidates((current) =>
      current.map((candidate, i) =>
        i === index ? { ...candidate, [field]: value } : candidate
      )
    );
    setResult(null);
  };

  const addCandidate = () => {
    setCandidates((current) => [...current, createCandidate()]);
    setResult(null);
  };

  const removeCandidate = (index) => {
    setCandidates((current) =>
      current.filter((_, i) => i !== index)
    );
    setResult(null);
  };

  const runOptimization = async (event) => {
    event.preventDefault();
    setError("");
    setResult(null);

    const parsedBudget = Number(budget);
    const parsedDiscount = Number(discountPct);

    if (!Number.isFinite(parsedBudget) || parsedBudget < 0) {
      setError("Enter a valid non-negative budget.");
      return;
    }

    if (
      !Number.isFinite(parsedDiscount) ||
      parsedDiscount <= 0 ||
      parsedDiscount > 100
    ) {
      setError("Discount must be greater than 0 and at most 100.");
      return;
    }

    if (candidates.length === 0) {
      setError("Add at least one customer candidate.");
      return;
    }

    const customerIds = candidates.map((candidate) =>
      candidate.id.trim()
    );

    if (customerIds.some((id) => !id)) {
      setError("Customer IDs cannot be empty.");
      return;
    }

    if (new Set(customerIds).size !== customerIds.length) {
      setError("Customer IDs must be unique.");
      return;
    }

    const invalidCandidate = candidates.some(
      (candidate) =>
        candidate.baseline_price === "" ||
        candidate.predicted_ite === "" ||
        !Number.isFinite(Number(candidate.baseline_price)) ||
        Number(candidate.baseline_price) <= 0 ||
        !Number.isFinite(Number(candidate.predicted_ite))
    );

    if (invalidCandidate) {
      setError(
        "Enter a positive baseline price and valid predicted ITE for every customer."
      );
      return;
    }

    const payload = {
      candidates: candidates.map((candidate) => ({
        customer_id: candidate.id.trim(),
        baseline_price: Number(candidate.baseline_price),
        predicted_ite: Number(candidate.predicted_ite),
      })),
      budget: parsedBudget,
      discount_pct: parsedDiscount,
    };

    setLoading(true);

    try {
      const response = await fetch("/api/optimization", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        const detail =
          typeof data.detail === "string"
            ? data.detail
            : data.detail?.message || JSON.stringify(data.detail);

        throw new Error(detail || `API error: ${response.status}`);
      }

      if (!Array.isArray(data.allocation)) {
        throw new Error(
          "Invalid API response: customer allocation details are missing."
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err instanceof TypeError
          ? "Cannot connect to the backend. Start the FastAPI server and retry."
          : err.message || "Optimization failed."
      );
    } finally {
      setLoading(false);
    }
  };

  const totalCustomers = candidates.length;
  const selectedCount = result?.selected_count || 0;

  const utilization =
    result && result.budget > 0
      ? (result.total_cost / result.budget) * 100
      : 0;

  const formatCurrency = (value) =>
    `₹${Number(value).toLocaleString("en-IN", {
      maximumFractionDigits: 2,
    })}`;

  return (
    <section className="allocation-page">
      <header className="allocation-heading">
        <div>
          <span className="allocation-eyebrow">
            PROMOTION OPTIMIZATION
          </span>
          <h1>Budget Allocation</h1>
          <p>
            Optimize customer targeting using predicted incremental effects.
          </p>
        </div>

        <span className="allocation-data-label">
          MILP Optimization
        </span>
      </header>

      <div className="allocation-notice">
        Predicted ITE values must come from your causal model or a verified
        evaluation dataset. Manually entered values are treated as supplied
        estimates, not model-generated predictions.
      </div>

      <form className="allocation-form" onSubmit={runOptimization}>
        <div className="allocation-form-heading">
          <div>
            <h2>Optimization Configuration</h2>
            <p>Set the available budget and promotion discount.</p>
          </div>
        </div>

        <div className="allocation-config-grid">
          <label>
            <span>Total Budget (₹)</span>
            <input
              type="number"
              min="0"
              step="0.01"
              value={budget}
              onChange={(event) => {
                setBudget(event.target.value);
                setResult(null);
              }}
              required
            />
          </label>

          <label>
            <span>Discount Percentage (%)</span>
            <input
              type="number"
              min="0.01"
              max="100"
              step="0.01"
              value={discountPct}
              onChange={(event) => {
                setDiscountPct(event.target.value);
                setResult(null);
              }}
              required
            />
          </label>
        </div>

        <div className="allocation-form-heading candidate-heading">
          <div>
            <h2>Customer Candidates</h2>
            <p>
              Enter customer IDs, baseline prices and predicted ITE values.
            </p>
          </div>

          <button
            type="button"
            className="allocation-secondary-button"
            onClick={addCandidate}
          >
            + Add Customer
          </button>
        </div>

        <div className="allocation-table-wrapper">
          <table className="allocation-input-table">
            <thead>
              <tr>
                <th>Customer ID</th>
                <th>Baseline Price (₹)</th>
                <th>Predicted ITE</th>
                <th>Promotion Cost (₹)</th>
                <th>Action</th>
              </tr>
            </thead>

            <tbody>
              {candidates.map((candidate, index) => {
                const estimatedCost =
                  Number(candidate.baseline_price) *
                  (Number(discountPct) / 100);

                return (
                  <tr key={`${candidate.id}-${index}`}>
                    <td>
                      <input
                        aria-label={`Customer ID ${index + 1}`}
                        value={candidate.id}
                        onChange={(event) =>
                          updateCandidate(
                            index,
                            "id",
                            event.target.value
                          )
                        }
                        required
                      />
                    </td>

                    <td>
                      <input
                        aria-label={`Baseline price ${index + 1}`}
                        type="number"
                        min="0.01"
                        step="0.01"
                        placeholder="e.g. 500"
                        value={candidate.baseline_price}
                        onChange={(event) =>
                          updateCandidate(
                            index,
                            "baseline_price",
                            event.target.value
                          )
                        }
                        required
                      />
                    </td>

                    <td>
                      <input
                        aria-label={`Predicted ITE ${index + 1}`}
                        type="number"
                        step="any"
                        placeholder="e.g. 0.08"
                        value={candidate.predicted_ite}
                        onChange={(event) =>
                          updateCandidate(
                            index,
                            "predicted_ite",
                            event.target.value
                          )
                        }
                        required
                      />
                    </td>

                    <td>
                      {candidate.baseline_price !== "" &&
                      Number.isFinite(estimatedCost)
                        ? formatCurrency(estimatedCost)
                        : "—"}
                    </td>

                    <td>
                      <button
                        type="button"
                        className="allocation-remove-button"
                        onClick={() => removeCandidate(index)}
                        disabled={candidates.length === 1 || loading}
                        aria-label={`Remove customer ${index + 1}`}
                      >
                        Remove
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="allocation-submit-row">
          <span>{totalCustomers} customer candidate(s)</span>

          <button
            type="submit"
            className="allocation-primary-button"
            disabled={loading}
          >
            {loading ? "Optimizing..." : "Run Optimization"}
          </button>
        </div>

        {error && (
          <div className="allocation-error" role="alert">
            {error}
          </div>
        )}
      </form>

      {result && (
        <section className="allocation-results">
          <div className="allocation-results-heading">
            <div>
              <span className="allocation-eyebrow">
                OPTIMIZATION RESULTS
              </span>
              <h2>Allocation Summary</h2>
            </div>

            <span className="allocation-success-label">
              {result.solver_status}
            </span>
          </div>

          <div className="allocation-metrics">
            <MetricCard
              label="Customers Selected"
              value={`${selectedCount} / ${result.allocation.length}`}
              detail="Optimized promotion allocation"
            />

            <MetricCard
              label="Total Promotion Cost"
              value={formatCurrency(result.total_cost)}
              detail="Estimated discount cost"
            />

            <MetricCard
              label="Remaining Budget"
              value={formatCurrency(result.unused_budget)}
              detail="Unutilized budget"
            />

            <MetricCard
              label="Expected Incremental Purchases"
              value={Number(
                result.expected_incremental_purchases
              ).toFixed(4)}
              detail="Model-predicted total effect"
            />
          </div>

          <article className="allocation-budget-card">
            <div className="allocation-budget-heading">
              <strong>Budget Utilization</strong>
              <span>{utilization.toFixed(2)}%</span>
            </div>

            <div
              className="allocation-progress-track"
              role="progressbar"
              aria-valuenow={Math.min(utilization, 100)}
              aria-valuemin="0"
              aria-valuemax="100"
              aria-label="Budget utilization"
            >
              <div
                className="allocation-progress-fill"
                style={{
                  width: `${Math.min(utilization, 100)}%`,
                }}
              />
            </div>

            <p>
              {formatCurrency(result.total_cost)} used out of{" "}
              {formatCurrency(result.budget)}
            </p>
          </article>

          <article className="allocation-matrix-card">
            <h2>Customer Allocation Matrix</h2>
            <p>
              Selection decisions and explanations returned directly
              from the backend optimization engine.
            </p>

            <div className="allocation-table-wrapper">
              <table className="allocation-result-table">
                <thead>
                  <tr>
                    <th>Customer ID</th>
                    <th>Baseline Price</th>
                    <th>Predicted ITE</th>
                    <th>Promotion Cost</th>
                    <th>Status</th>
                    <th>Explanation</th>
                  </tr>
                </thead>

                <tbody>
                  {result.allocation.map((candidate) => (
                    <tr key={candidate.customer_id}>
                      <td>{candidate.customer_id}</td>

                      <td>
                        {formatCurrency(candidate.baseline_price)}
                      </td>

                      <td>
                        {Number(candidate.predicted_ite).toFixed(4)}
                      </td>

                      <td>
                        {formatCurrency(candidate.promotion_cost)}
                      </td>

                      <td>
                        <span
                          className={`allocation-status ${
                            candidate.selected
                              ? "selected"
                              : "rejected"
                          }`}
                        >
                          {candidate.status}
                        </span>
                      </td>

                      <td className="allocation-explanation">
                        {candidate.explanation}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>

          <p className="allocation-disclaimer">
            {result.interpretation} Expected incremental purchases are
            predictions, not guaranteed outcomes. Customer-level
            explanations describe the solver decision and supplied inputs;
            they do not establish individual causal certainty.
          </p>
        </section>
      )}
    </section>
  );
}
