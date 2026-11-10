import { useAuth } from "../../context/AuthContext";
import "./Header.css";

export default function Header() {
  const { user, logout } = useAuth();

  const displayName = user?.full_name || user?.username || "Guest User";
  const initials = displayName
    .split(" ")
    .map((part) => part[0])
    .filter(Boolean)
    .slice(0, 2)
    .join("")
    .toUpperCase() || "EC";

  const role = user?.role ? user.role.toLowerCase() : "viewer";

  return (
    <header className="dashboard-header">
      <label className="header-search">
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <circle cx="10.8" cy="10.8" r="6.8" stroke="currentColor" strokeWidth="2" />
          <path d="m16 16 4.5 4.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        </svg>
        <input type="search" placeholder="Search anything..." aria-label="Search" />
      </label>

      <div className="header-actions">
        <button className="header-notification" type="button" aria-label="Notifications">
          <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M18 9a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M10 21h4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
          <span />
        </button>

        <div className="header-profile">
          <div className="header-avatar">{initials}</div>
          <div className="header-user-meta">
            <span className="header-user-name">{displayName}</span>
            <span className={`header-role-badge badge-${role}`}>{role}</span>
          </div>
        </div>

        {user && (
          <button
            type="button"
            className="header-logout-btn"
            onClick={logout}
            title="Sign out of EconoCausal"
          >
            <span>Sign Out</span>
          </button>
        )}
      </div>
    </header>
  );
}
