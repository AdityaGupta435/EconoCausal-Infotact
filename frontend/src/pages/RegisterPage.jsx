import { useState } from "react";
import { useAuth } from "../context/AuthContext";
import "./AuthPages.css";

export default function RegisterPage({ onSwitchToLogin }) {
  const { register } = useAuth();
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [fullName, setFullName] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [role, setRole] = useState("analyst");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!email.trim() || !username.trim() || !password) {
      setError("Please fill in all required fields.");
      return;
    }

    if (password.length < 6) {
      setError("Password must be at least 6 characters long.");
      return;
    }

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setLoading(true);
    try {
      await register({
        email: email.trim(),
        username: username.trim(),
        password,
        fullName: fullName.trim() || undefined,
        role,
      });
    } catch (err) {
      setError(err.message || "Registration failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <div className="auth-logo">
            <svg viewBox="0 0 64 56" role="img" aria-hidden="true">
              <defs>
                <linearGradient id="brandGradientReg" x1="0" y1="1" x2="1" y2="0">
                  <stop offset="0%" stopColor="#1687ff" />
                  <stop offset="100%" stopColor="#8b5cf6" />
                </linearGradient>
              </defs>
              <rect x="3" y="32" width="13" height="21" rx="3" fill="url(#brandGradientReg)" />
              <rect x="22" y="18" width="13" height="35" rx="3" fill="url(#brandGradientReg)" />
              <rect x="41" y="3" width="13" height="50" rx="3" fill="url(#brandGradientReg)" />
            </svg>
          </div>
          <h1 className="auth-title">Create Account</h1>
          <p className="auth-subtitle">Join EconoCausal pricing intelligence platform</p>
        </div>

        {error && (
          <div className="auth-error-alert" role="alert">
            <span>⚠</span>
            <span>{error}</span>
          </div>
        )}

        <form className="auth-form" onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="regEmail">Email Address *</label>
            <input
              id="regEmail"
              type="email"
              className="form-input"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="e.g. analyst@company.com"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="regUsername">Username *</label>
            <input
              id="regUsername"
              type="text"
              className="form-input"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="e.g. aditya_g"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="regFullName">Full Name (optional)</label>
            <input
              id="regFullName"
              type="text"
              className="form-input"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              placeholder="e.g. Aditya Gupta"
            />
          </div>

          <div className="form-group">
            <label htmlFor="regRole">Platform Role *</label>
            <select
              id="regRole"
              className="form-select"
              value={role}
              onChange={(e) => setRole(e.target.value)}
            >
              <option value="analyst">Analyst (Execute Models & Optimization)</option>
              <option value="admin">Admin (Manage Users, Sync & Audit)</option>
              <option value="viewer">Viewer (Read-Only Dashboards)</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="regPassword">Password *</label>
            <input
              id="regPassword"
              type="password"
              className="form-input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Min. 6 characters"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="regConfirmPassword">Confirm Password *</label>
            <input
              id="regConfirmPassword"
              type="password"
              className="form-input"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder="Re-enter password"
              required
            />
          </div>

          <button
            type="submit"
            className="auth-submit-btn"
            disabled={loading}
          >
            {loading ? "Creating Account..." : "Register & Sign In"}
          </button>
        </form>

        <div className="auth-footer">
          <span>Already have an account? </span>
          <button
            type="button"
            className="auth-switch-btn"
            onClick={onSwitchToLogin}
          >
            Sign in
          </button>
        </div>
      </div>
    </div>
  );
}
