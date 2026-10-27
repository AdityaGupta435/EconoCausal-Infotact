
import { useState } from "react";
import "./DataUpload.css";

const REQUIRED_COLUMNS = [
  "baseline_price",
  "customer_history",
  "seasonality",
  "category",
  "treatment",
  "purchase",
];

const MAX_FILE_SIZE = 10 * 1024 * 1024;

function parseCSV(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;

  for (let i = 0; i < text.length; i++) {
    const char = text[i];

    if (char === '"' && quoted && text[i + 1] === '"') {
      cell += '"';
      i++;
    } else if (char === '"') {
      quoted = !quoted;
    } else if ((char === "," || char === "\n" || char === "\r") && !quoted) {
      if (char === ",") {
        row.push(cell);
        cell = "";
      } else {
        if (char === "\r" && text[i + 1] === "\n") i++;

        row.push(cell);
        if (row.some((value) => value.trim() !== "")) rows.push(row);

        row = [];
        cell = "";
      }
    } else {
      cell += char;
    }
  }

  if (quoted) {
    throw new Error("CSV contains an unclosed quoted field.");
  }

  if (cell !== "" || row.length > 0) {
    row.push(cell);
    if (row.some((value) => value.trim() !== "")) rows.push(row);
  }

  if (rows.length < 2) {
    throw new Error("CSV must contain a header and at least one data row.");
  }

  const headers = rows[0].map((header) => header.trim());

  if (headers.some((header) => !header)) {
    throw new Error("CSV contains an empty column name.");
  }

  if (new Set(headers).size !== headers.length) {
    throw new Error("CSV contains duplicate column names.");
  }

  return {
    headers,
    records: rows.slice(1).map((values) =>
      Object.fromEntries(
        headers.map((header, index) => [
          header,
          (values[index] ?? "").trim(),
        ])
      )
    ),
  };
}

export default function DataUpload() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleFileChange(event) {
    const selectedFile = event.target.files?.[0];

    setError("");
    setPreview(null);
    setFile(null);

    if (!selectedFile) return;

    if (!selectedFile.name.toLowerCase().endsWith(".csv")) {
      setError("Please select a CSV file.");
      event.target.value = "";
      return;
    }

    if (selectedFile.size === 0) {
      setError("The selected file is empty.");
      event.target.value = "";
      return;
    }

    if (selectedFile.size > MAX_FILE_SIZE) {
      setError("Maximum supported file size is 10 MB.");
      event.target.value = "";
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch("/api/data/upload", {
        method: "POST",
        body: formData,
      });

      let result;

      try {
        result = await response.json();
      } catch {
        throw new Error("The server returned an invalid response.");
      }

      if (!response.ok) {
        const detail = result.detail;

        if (typeof detail === "object" && detail !== null) {
          const message = detail.message || "CSV validation failed.";
          const missing = detail.missing_columns;

          throw new Error(
            missing?.length
              ? `${message} Missing columns: ${missing.join(", ")}`
              : message
          );
        }

        throw new Error(
          typeof detail === "string" ? detail : "CSV upload failed."
        );
      }

      const content = await selectedFile.text();
      const parsed = parseCSV(content);

      const missingColumns = REQUIRED_COLUMNS.filter(
        (column) => !result.columns?.includes(column)
      );

      if (missingColumns.length > 0) {
        throw new Error(
          `Server response is missing required columns: ${missingColumns.join(", ")}`
        );
      }

      setFile(selectedFile);
      setPreview({
        ...parsed,
        missingColumns,
        rowCount: result.row_count,
        serverValidated: true,
      });
    } catch (err) {
      setError(
        err instanceof TypeError
          ? "Could not connect to the backend. Check that the API server is running."
          : err.message || "Could not validate this CSV file."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="upload-page">
      <div className="upload-heading">
        <div>
          <span className="upload-eyebrow">DATA MANAGEMENT</span>
          <h1>Import your dataset</h1>
          <p>
            Preview your pricing data before using it for causal analysis.
          </p>
        </div>

        <span className="local-status">
          {preview?.serverValidated ? "Backend validated" : "CSV validation"}
        </span>
      </div>

      <div className="upload-card">
        <div className="upload-icon" aria-hidden="true">
          ↑
        </div>

        <h2>Select a CSV file</h2>
        <p className="upload-description">
          Choose a dataset from your computer to validate its columns
          and preview the first five records.
        </p>

        <label className="file-picker">
          Choose CSV file
          <input
            type="file"
            accept=".csv,text/csv"
            disabled={loading}
            onChange={handleFileChange}
          />
        </label>

        <p className="file-hint">CSV format · Maximum 10 MB</p>

        {loading && (
          <p role="status" aria-live="polite">
            Uploading and validating CSV with the backend...
          </p>
        )}

        {!loading && error && (
          <p className="upload-error" role="alert">
            {error}
          </p>
        )}

        {!loading && !error && !preview && (
          <p role="status">
            No dataset selected. Choose a CSV file to begin.
          </p>
        )}
      </div>

      <div className="upload-features">
        <article className="upload-feature">
          <div className="upload-feature-icon" aria-hidden="true">▤</div>
          <div>
            <h3>Validate Columns</h3>
            <p>Check if all required columns are present in your dataset.</p>
          </div>
        </article>

        <article className="upload-feature">
          <div className="upload-feature-icon" aria-hidden="true">◉</div>
          <div>
            <h3>Preview Data</h3>
            <p>See the first five records before processing.</p>
          </div>
        </article>

        <article className="upload-feature">
          <div className="upload-feature-icon" aria-hidden="true">✓</div>
          <div>
            <h3>Data Quality</h3>
            <p>Check column consistency before analysis.</p>
          </div>
        </article>
      </div>

      {preview && (
        <div className="preview-card">
          <div className="preview-heading">
            <div>
              <h2>Dataset preview</h2>
              <p>{file.name}</p>
            </div>

            <span className="row-count">
              {preview.rowCount.toLocaleString()} rows
            </span>
          </div>

          {preview.missingColumns.length === 0 ? (
            <p className="validation-success" role="status">
              Required columns are present. Backend validation successful.
            </p>
          ) : (
            <p className="validation-warning" role="alert">
              Missing required columns: {preview.missingColumns.join(", ")}
            </p>
          )}

          <div className="table-scroll">
            <table className="preview-table">
              <thead>
                <tr>
                  {preview.headers.map((header) => (
                    <th key={header}>{header}</th>
                  ))}
                </tr>
              </thead>

              <tbody>
                {preview.records.slice(0, 5).map((record, index) => (
                  <tr key={index}>
                    {preview.headers.map((header) => (
                      <td key={header}>{record[header] || "—"}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="preview-note">
            The file was validated by the backend in memory. It has not been
            saved to disk or a database.
          </p>
        </div>
      )}
    </section>
  );
}
