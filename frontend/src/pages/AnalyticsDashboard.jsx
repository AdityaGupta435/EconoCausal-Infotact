
import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import "./AnalyticsDashboard.css";

const API_ENDPOINTS = {
  causal: "/api/causal/ate",
  uplift: "/api/uplift",
};

function MetricCard({ label, value, detail }) {
  return (
    <article className="analytics-metric-card">
      <p>{label}</p>
      <strong>{value}</strong>
      <span>{detail}</span>
    </article>
  );
}

export default function AnalyticsDashboard() {
  const [causal, setCausal] = useState(null);
  const [uplift, setUplift] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();

    async function loadAnalytics() {
      setLoading(true);
      setError("");

      try {
        const responses = await Promise.all(
          Object.values(API_ENDPOINTS).map((endpoint) =>
            fetch(endpoint, { signal: controller.signal })
          )
        );

        const failedResponse = responses.find((response) => !response.ok);

        if (failedResponse) {
          throw new Error(
            `Analytics API returned HTTP ${failedResponse.status}.`
          );
        }

        const [causalData, upliftData] = await Promise.all(
          responses.map((response) => response.json())
        );

        setCausal(causalData);
        setUplift(upliftData);
      } catch (err) {
        if (err.name !== "AbortError") {
          setError(
            err instanceof TypeError
              ? "Cannot connect to the backend. Start the FastAPI server and retry."
              : err.message || "Could not load analytics."
          );
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    loadAnalytics();

    return () => controller.abort();
  }, []);

  if (loading) {
    return (
      <section className="analytics-page" aria-busy="true">
        <p role="status">Loading analytics from the backend...</p>
      </section>
    );
  }

  if (error) {
    return (
      <section className="analytics-page">
        <div className="analytics-state analytics-error" role="alert">
          <h2>Analytics unavailable</h2>
          <p>{error}</p>
          <button type="button" onClick={() => window.location.reload()}>
            Retry
          </button>
        </div>
      </section>
    );
  }

  if (!causal || !uplift) {
    return (
      <section className="analytics-page">
        <div className="analytics-state">
          <h2>No analytics available</h2>
          <p>The API did not return the required analytics reports.</p>
        </div>
      </section>
    );
  }

  const qiniComparison = [
    uplift.observed_qini_area,
    uplift.random_qini_area_mean_30_runs,
  ];

  return (
    <section className="analytics-page">
      <header className="analytics-heading">
        <div>
          <span className="analytics-eyebrow">CAUSAL INTELLIGENCE</span>
          <h1>Analytics Dashboard</h1>
          <p>
            Treatment effects and targeting diagnostics from the backend.
          </p>
        </div>
        <span className="analytics-data-label">Synthetic data</span>
      </header>

      <div className="analytics-notice">
        These metrics use synthetic data. They do not establish real-world
        causal validity or guarantee business outcomes.
      </div>

      <div className="analytics-metrics">
        <MetricCard
          label="Estimated ATE"
          value={`${causal.estimated_effect_percentage_points.toFixed(2)} pp`}
          detail="Estimated treatment effect"
        />
        <MetricCard
          label="Training records"
          value={causal.training_rows.toLocaleString()}
          detail={causal.dataset}
        />
        <MetricCard
          label="ITE prediction MAE"
          value={uplift.predicted_ite_mae.toFixed(4)}
          detail="Synthetic held-out test set"
        />
        <MetricCard
          label="Test records"
          value={uplift.test_rows.toLocaleString()}
          detail={`${uplift.treatment_rows} treated · ${uplift.control_rows} control`}
        />
      </div>

      <div className="analytics-chart-grid">
        <article className="analytics-chart-card">
          <h2>Qini area comparison</h2>
          <p>Observed model ranking versus the random-targeting mean.</p>
          <Plot
            data={[
              {
                type: "bar",
                x: ["Model", "Random mean"],
                y: qiniComparison,
                marker: { color: ["#3478f6", "#94a3b8"] },
                hovertemplate: "%{x}: %{y:.3f}<extra></extra>",
              },
            ]}
            layout={{
              autosize: true,
              margin: { l: 55, r: 20, t: 15, b: 55 },
              paper_bgcolor: "transparent",
              plot_bgcolor: "transparent",
              font: { family: "Inter, sans-serif", color: "#34415f" },
              yaxis: { title: "Qini area", gridcolor: "#e8edf5" },
              xaxis: { fixedrange: true },
              showlegend: false,
            }}
            config={{ responsive: true, displayModeBar: false }}
            useResizeHandler
            style={{ width: "100%", height: "320px" }}
          />
          <p className="analytics-chart-note">
            Difference versus random mean:{" "}
            {uplift.qini_area_difference_vs_random.toFixed(3)}. A negative
            value means the observed model Qini is below the random mean.
          </p>
        </article>

        <article className="analytics-chart-card">
          <h2>Top 20% targeting comparison</h2>
          <p>Mean true effect reported by the synthetic evaluation.</p>
          <Plot
            data={[
              {
                type: "bar",
                x: ["Model ranked", "Random", "Oracle"],
                y: [
                  uplift.top_20_percent.model_ranked_mean_true_effect,
                  uplift.top_20_percent.random_targeting_mean_true_effect,
                  uplift.top_20_percent.oracle_ranked_mean_true_effect,
                ],
                marker: { color: ["#3478f6", "#94a3b8", "#16a394"] },
                hovertemplate: "%{x}: %{y:.4f}<extra></extra>",
              },
            ]}
            layout={{
              autosize: true,
              margin: { l: 55, r: 20, t: 15, b: 65 },
              paper_bgcolor: "transparent",
              plot_bgcolor: "transparent",
              font: { family: "Inter, sans-serif", color: "#34415f" },
              yaxis: {
                title: "Mean true effect",
                gridcolor: "#e8edf5",
              },
              xaxis: { fixedrange: true },
              showlegend: false,
            }}
            config={{ responsive: true, displayModeBar: false }}
            useResizeHandler
            style={{ width: "100%", height: "320px" }}
          />
          <p className="analytics-chart-note">
            Oracle is an evaluation-only reference, not a deployable targeting
            strategy. True effects are not model inputs.
          </p>
        </article>
      </div>
    </section>
  );
}
