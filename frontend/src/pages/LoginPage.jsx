import { useState } from "react";
import { useAuth } from "../context/AuthContext";
import "./AuthPages.css";

export default function LoginPage({ onSwitchToRegister }) {
  const { login } = useAuth();
  const [usernameOrEmail, setUsernameOrEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!usernameOrEmail.trim() || !password) {
      setError("Please fill in both username/email and password.");
      return;
    }

    setLoading(true);
    try {
      await login(usernameOrEmail.trim(), password);
    } catch (err) {
      setError(err.message || "Login failed.");
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
                <linearGradient id="brandGradientLogin" x1="0" y1="1" x2="1" y2="0">
                  <stop offset="0%" stopColor="#1687ff" />
                  <stop offset="100%" stopColor="#8b5cf6" />
                </linearGradient>
              </defs>
              <rect x="3" y="32" width="13" height="21" rx="3" fill="url(#brandGradientLogin)" />
              <rect x="22" y="18" width="13" height="35" rx="3" fill="url(#brandGradientLogin)" />
              <rect x="41" y="3" width="13" height="50" rx="3" fill="url(#brandGradientLogin)" />
            </svg>
          </div>
          <h1 className="auth-title">Welcome to EconoCausal</h1>
          <p className="auth-subtitle">Sign in to access your pricing intelligence dashboard</p>
        </div>

        {error && (
          <div className="auth-error-alert" role="alert">
            <span>⚠</span>
            <span>{error}</span>
          </div>
        )}

        <form className="auth-form" onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="usernameOrEmail">Username or Email</label>
            <input
              id="usernameOrEmail"
              type="text"
              className="form-input"
              value={usernameOrEmail}
              onChange={(e) => setUsernameOrEmail(e.target.value)}
              placeholder="e.g. aditya or user@company.com"
              required
              autoFocus
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              className="form-input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your account password"
              required
            />
          </div>

          <button
            type="submit"
            className="auth-submit-btn"
            disabled={loading}
          >
            {loading ? "Authenticating..." : "Sign In"}
          </button>
        </form>

        <div className="auth-footer">
          <span>Don't have an account yet? </span>
          <button
            type="button"
            className="auth-switch-btn"
            onClick={onSwitchToRegister}
          >
            Register here
          </button>
        </div>
      </div>
    </div>
  );
}
