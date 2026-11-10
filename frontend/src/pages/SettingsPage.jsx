/**
 * EconoCausal — Settings Page (Phase 10)
 * Platform administration, DB sync, and audit log (admin-only actions).
 */

import { useEffect, useState } from "react";
import { useAuth } from "../context/AuthContext";
import { SectionHeader, ErrorBanner, SuccessBanner, StateView } from "../components/common";
import "../styles/common.css";

export default function SettingsPage() {
  const { user, token } = useAuth();
  const role = user?.role?.toLowerCase() ?? "viewer";
  const isAdmin = role === "admin";

  const [syncLoading, setSyncLoading] = useState(false);
  const [syncResult, setSyncResult] = useState(null);
  const [syncError, setSyncError] = useState("");

  const [dbHealth, setDbHealth] = useState(null);
  const [healthLoading, setHealthLoading] = useState(true);

  async function checkHealth() {
    setHealthLoading(true);
    try {
      const res = await fetch("/api/db/health");
      const data = await res.json();
      setDbHealth(data);
    } catch (err) {
      setDbHealth({ status: "error", message: err.message });
    } finally {
      setHealthLoading(false);
    }
  }

  useEffect(() => { checkHealth(); }, []);

  async function handleSync() {
    setSyncError("");
    setSyncResult(null);
    setSyncLoading(true);
    try {
      const res = await fetch("/api/db/sync", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? "Sync failed.");
      setSyncResult(data?.data ?? data);
    } catch (err) {
      setSyncError(err.message);
    } finally {
      setSyncLoading(false);
    }
  }

  return (
    <div>
      <SectionHeader
        eyebrow="PLATFORM"
        title="Settings & Administration"
        subtitle="Database health, artifact synchronization, and account information."
      />
      <div className="page-body">
        {/* Profile info */}
        <div className="card">
          <h3 className="card__title">Account</h3>
          <div className="kpi-grid" style={{ marginTop: 14 }}>
            <div className="kpi-card">
              <span className="kpi-card__label">Username</span>
              <span className="kpi-card__value" style={{ fontSize: 18 }}>{user?.username ?? "—"}</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-card__label">Email</span>
              <span className="kpi-card__value" style={{ fontSize: 16 }}>{user?.email ?? "—"}</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-card__label">Role</span>
              <span className="kpi-card__value" style={{ fontSize: 18, textTransform: "capitalize" }}>{role}</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-card__label">Full Name</span>
              <span className="kpi-card__value" style={{ fontSize: 16 }}>{user?.full_name ?? "—"}</span>
            </div>
          </div>
        </div>

        {/* DB health */}
        <div className="card" style={{ marginTop: 20 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
            <div>
              <h3 className="card__title">Database Health</h3>
              <p className="card__subtitle">MongoDB Atlas connection status.</p>
            </div>
            <button className="btn btn-secondary btn-sm" onClick={checkHealth} disabled={healthLoading}>
              {healthLoading ? "Checking…" : "↻ Refresh"}
            </button>
          </div>
          {healthLoading ? (
            <StateView icon="⟳" title="Checking connection…" />
          ) : dbHealth ? (
            <div className="table-wrap">
              <table className="data-table">
                <thead><tr><th>Property</th><th>Value</th></tr></thead>
                <tbody>
                  {Object.entries(dbHealth).map(([k, v]) => (
                    <tr key={k}>
                      <td>{k.replace(/_/g, " ")}</td>
                      <td style={{ color: k === "status" && v === "connected" ? "#16a34a" : undefined }}>
                        {typeof v === "object" ? JSON.stringify(v) : String(v)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </div>

        {/* Artifact sync — admin/analyst only */}
        {isAdmin && (
          <div className="card" style={{ marginTop: 20 }}>
            <h3 className="card__title">Artifact Synchronization</h3>
            <p className="card__subtitle">
              Scan local Phase 1–7 artifacts and register them in MongoDB. Safe to run multiple times.
            </p>
            <ErrorBanner message={syncError} />
            {syncResult && (
              <SuccessBanner message={`Synchronized: ${JSON.stringify(syncResult)}`} />
            )}
            <div style={{ marginTop: 14 }}>
              <button className="btn btn-primary" onClick={handleSync} disabled={syncLoading}>
                {syncLoading ? "Syncing…" : "↻ Sync Artifacts"}
              </button>
            </div>
          </div>
        )}

        {!isAdmin && (
          <div className="card" style={{ marginTop: 20 }}>
            <StateView
              icon="⚙"
              title="Admin Features"
              desc="Artifact synchronization and user management require Admin role."
            />
          </div>
        )}
      </div>
    </div>
  );
}

