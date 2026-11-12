/**
 * EconoCausal — Centralized API Service
 * Wraps all backend API calls with auth token injection, error handling,
 * and standard response envelopes. No endpoint logic is duplicated in components.
 */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/+$/, "");

function getToken() {
  return localStorage.getItem("econocausal_token") || null;
}

function authHeaders(extra = {}) {
  const token = getToken();
  const headers = { "Content-Type": "application/json", ...extra };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  return headers;
}

function formHeaders() {
  const token = getToken();
  const headers = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;
  return headers; // Let browser set multipart boundary
}

async function handleResponse(res) {
  if (res.status === 401) {
    // Expired/invalid token — clear storage
    localStorage.removeItem("econocausal_token");
    localStorage.removeItem("econocausal_user");
    throw new Error("Session expired. Please log in again.");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data?.detail || data?.message || `HTTP ${res.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

async function get(path, params = {}) {
  const url = new URL(`${BASE_URL}${path}`, window.location.origin);
  Object.entries(params).forEach(([k, v]) => v !== undefined && url.searchParams.set(k, v));
  const res = await fetch(url.toString(), { headers: authHeaders() });
  return handleResponse(res);
}

async function post(path, body = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify(body),
  });
  return handleResponse(res);
}

async function postForm(path, formData) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: formHeaders(),
    body: formData,
  });
  return handleResponse(res);
}

async function del(path) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
  return handleResponse(res);
}

// ── Dashboard ──────────────────────────────────────────────
export const dashboard = {
  overview: () => get("/api/dashboard/overview"),
  metrics: () => get("/api/dashboard/metrics"),
  modelPerformance: () => get("/api/dashboard/model-performance"),
  recentActivity: () => get("/api/dashboard/recent-activity"),
  dataSummary: () => get("/api/dashboard/data-summary"),
};

// ── Datasets ───────────────────────────────────────────────
export const datasets = {
  list: (skip = 0, limit = 50) => get("/api/datasets", { skip, limit }),
  upload: (file) => {
    const fd = new FormData();
    fd.append("file", file);
    return postForm("/api/datasets/upload", fd);
  },
  getById: (id) => get(`/api/datasets/${id}`),
  preview: (id, n = 5) => get(`/api/datasets/${id}/preview`, { n_rows: n }),
  summary: (id) => get(`/api/datasets/${id}/summary`),
  delete: (id) => del(`/api/datasets/${id}`),
};

// ── Training ───────────────────────────────────────────────
export const training = {
  start: (body) => post("/api/training/start", body),
  status: (id) => get(`/api/training/status/${id}`),
  history: () => get("/api/training/history"),
  metrics: (id) => get(`/api/training/${id}/metrics`),
  logs: (id) => get(`/api/training/${id}/logs`),
};

// ── Causal Inference ───────────────────────────────────────
export const causal = {
  analyze: (body) => post("/api/causal/analyze", body),
  results: (id) => get(`/api/causal/results/${id}`),
  history: () => get("/api/causal/history"),
  diagnostics: (id) => get(`/api/causal/diagnostics/${id}`),
};

// ── Uplift Modeling ────────────────────────────────────────
export const uplift = {
  analyze: (body) => post("/api/uplift/analyze", body),
  results: (id) => get(`/api/uplift/results/${id}`),
  history: () => get("/api/uplift/history"),
  segments: () => get("/api/uplift/customer-segments"),
  performance: () => get("/api/uplift/performance"),
};

// ── Pricing ────────────────────────────────────────────────
export const pricing = {
  recommend: (body) => post("/api/pricing/recommend", body),
  batchRecommend: (body) => post("/api/pricing/batch-recommend", body),
  history: () => get("/api/pricing/recommendations"),
  elasticity: () => get("/api/pricing/elasticity"),
  performance: () => get("/api/pricing/performance"),
};

// ── Promotions ─────────────────────────────────────────────
export const promotions = {
  optimize: (body) => post("/api/promotions/optimize", body),
  campaigns: () => get("/api/promotions/campaigns"),
  results: () => get("/api/promotions/results"),
  budgetAnalysis: () => get("/api/promotions/budget-analysis"),
};

// ── Pipeline ───────────────────────────────────────────────
export const pipeline = {
  run: (body) => post("/api/pipeline/run", body),
  status: (id) => get(`/api/pipeline/status/${id}`),
  history: () => get("/api/pipeline/history"),
  results: (id) => get(`/api/pipeline/results/${id}`),
};

