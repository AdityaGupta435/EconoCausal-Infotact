/**
 * EconoCausal — Dataset Management Page (Phase 10)
 * Upload, preview, summarize, and delete datasets via Phase 9 dataset APIs.
 */

import { useEffect, useRef, useState } from "react";
import { datasets } from "../services/api";
import { SectionHeader, ErrorBanner, SuccessBanner, StateView, Badge, StatusBadge } from "../components/common";
import "../styles/common.css";

function formatBytes(b) {
  if (!b) return "—";
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / (1024 * 1024)).toFixed(2)} MB`;
}

export default function DatasetsPage() {
  const [list, setList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [listError, setListError] = useState("");

  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadSuccess, setUploadSuccess] = useState("");

  const [preview, setPreview] = useState(null);    // { datasetId, data }
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState("");

  const [summary, setSummary] = useState(null);    // { datasetId, data }
  const [confirmDelete, setConfirmDelete] = useState(null); // dataset id
  const [deleteError, setDeleteError] = useState("");

  const fileRef = useRef();

  async function loadList() {
    setLoading(true);
    setListError("");
    try {
      const res = await datasets.list();
      setList(Array.isArray(res) ? res : (res?.data ?? []));
    } catch (err) {
      setListError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { loadList(); }, []);

  async function handleUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadError("");
    setUploadSuccess("");

    if (!file.name.toLowerCase().endsWith(".csv")) {
      setUploadError("Only CSV files are accepted.");
      e.target.value = "";
      return;
    }
    if (file.size === 0) {
      setUploadError("The selected file is empty.");
      e.target.value = "";
      return;
    }

    setUploading(true);
    try {
      const res = await datasets.upload(file);
      setUploadSuccess(`"${res?.data?.name ?? file.name}" uploaded successfully.`);
      await loadList();
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function handlePreview(id) {
    if (preview?.datasetId === id) { setPreview(null); return; }
    setPreviewLoading(true);
    setPreviewError("");
    try {
      const res = await datasets.preview(id, 5);
      setPreview({ datasetId: id, data: res?.data ?? res });
    } catch (err) {
      setPreviewError(err.message);
    } finally {
      setPreviewLoading(false);
    }
  }

  async function handleSummary(id) {
    if (summary?.datasetId === id) { setSummary(null); return; }
    try {
      const res = await datasets.summary(id);
      setSummary({ datasetId: id, data: res?.data ?? res });
    } catch (err) {
      setPreviewError(err.message);
    }
  }

  async function handleDelete(id) {
    setDeleteError("");
    try {
      await datasets.delete(id);
      setConfirmDelete(null);
      if (preview?.datasetId === id) setPreview(null);
      if (summary?.datasetId === id) setSummary(null);
      await loadList();
    } catch (err) {
      setDeleteError(err.message);
      setConfirmDelete(null);
    }
  }

  return (
    <div>
      <SectionHeader
        eyebrow="DATA MANAGEMENT"
        title="Datasets"
        subtitle="Upload, preview, and manage datasets registered in the platform."
      >
        <label className="btn btn-primary" style={{ cursor: uploading ? "not-allowed" : "pointer" }}>
          {uploading ? "Uploading…" : "↑ Upload CSV"}
          <input
            ref={fileRef}
            type="file"
            accept=".csv,text/csv"
            disabled={uploading}
            onChange={handleUpload}
            style={{ display: "none" }}
          />
        </label>
      </SectionHeader>

      <div className="page-body">
        <ErrorBanner message={uploadError} />
        <SuccessBanner message={uploadSuccess} />
        <ErrorBanner message={deleteError} />
        <ErrorBanner message={listError} />

        {/* Datasets table */}
        <div className="card">
          <h3 className="card__title">Registered Datasets</h3>
          <p className="card__subtitle">
            Core system datasets are protected from deletion.
          </p>

          {loading ? (
            <StateView icon="⟳" title="Loading datasets…" />
          ) : list.length === 0 ? (
            <StateView icon="▥" title="No datasets found" desc="Upload a CSV file to register a dataset." />
          ) : (
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Rows</th>
                    <th>Columns</th>
                    <th>Size</th>
                    <th>Source</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {list.map((ds) => {
                    const id = ds.id ?? ds._id;
                    const isCore = ds.source === "system" || ds.is_core;
                    return (
                      <tr key={id}>
                        <td>
                          <strong>{ds.name}</strong>
                          {isCore && <span className="badge badge-info" style={{ marginLeft: 8 }}>Core</span>}
                        </td>
                        <td>{(ds.row_count ?? 0).toLocaleString()}</td>
                        <td>{ds.column_count ?? ds.columns?.length ?? "—"}</td>
                        <td>{formatBytes(ds.file_size_bytes)}</td>
                        <td>{ds.source ?? "uploaded"}</td>
                        <td style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                          <button className="btn btn-secondary btn-sm" onClick={() => handlePreview(id)}>
                            {preview?.datasetId === id ? "Hide" : "Preview"}
                          </button>
                          <button className="btn btn-secondary btn-sm" onClick={() => handleSummary(id)}>
                            {summary?.datasetId === id ? "Hide Stats" : "Stats"}
                          </button>
                          {!isCore && (
                            <button className="btn btn-danger btn-sm" onClick={() => setConfirmDelete(id)}>
                              Delete
                            </button>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Preview panel */}
        {previewLoading && <StateView icon="⟳" title="Loading preview…" />}
        {previewError && <ErrorBanner message={previewError} />}

        {preview?.data && (
          <div className="card" style={{ marginTop: 20 }}>
            <h3 className="card__title">Dataset Preview</h3>
            <p className="card__subtitle">
              First {preview.data.rows_previewed ?? preview.data.rows?.length ?? 5} rows.
            </p>
            {Array.isArray(preview.data.columns) && (
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>{preview.data.columns.map((c) => <th key={c}>{c}</th>)}</tr>
                  </thead>
                  <tbody>
                    {(preview.data.rows ?? []).map((row, i) => (
                      <tr key={i}>
                        {preview.data.columns.map((c) => <td key={c}>{row[c] ?? "—"}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* Summary stats panel */}
        {summary?.data && (
          <div className="card" style={{ marginTop: 20 }}>
            <h3 className="card__title">Summary Statistics</h3>
            <p className="card__subtitle">
              {summary.data.total_rows?.toLocaleString()} rows · {summary.data.total_columns} columns
            </p>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Type</th>
                    <th>Non-null</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Mean</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(summary.data.columns ?? {}).map(([col, stat]) => (
                    <tr key={col}>
                      <td><strong>{col}</strong></td>
                      <td>{stat.dtype ?? "—"}</td>
                      <td>{stat.non_null_count?.toLocaleString() ?? "—"}</td>
                      <td>{stat.min !== undefined ? (+stat.min).toFixed(3) : "—"}</td>
                      <td>{stat.max !== undefined ? (+stat.max).toFixed(3) : "—"}</td>
                      <td>{stat.mean !== undefined ? (+stat.mean).toFixed(3) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Delete confirmation modal */}
        {confirmDelete && (
          <div style={{
            position: "fixed", inset: 0, background: "rgba(0,0,0,0.3)",
            display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000,
          }}>
            <div className="card" style={{ maxWidth: 400, width: "90%", padding: 32 }}>
              <h3 className="card__title">Confirm Deletion</h3>
              <p style={{ color: "#64748b", marginTop: 10 }}>
                Are you sure you want to delete this dataset? This action cannot be undone.
              </p>
              <div style={{ display: "flex", gap: 12, marginTop: 24 }}>
                <button className="btn btn-danger" onClick={() => handleDelete(confirmDelete)}>
                  Delete
                </button>
                <button className="btn btn-secondary" onClick={() => setConfirmDelete(null)}>
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

