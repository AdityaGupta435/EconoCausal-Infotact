import { useAuth } from "../../context/AuthContext";
import "./Sidebar.css";

const navigationItems = [
  { icon: "⌂", label: "Overview",      roles: ["admin", "analyst", "viewer"], group: "WORKSPACE" },
  { icon: "▤", label: "Causal Effects", roles: ["admin", "analyst", "viewer"], group: "ANALYTICS" },
  { icon: "↗", label: "Uplift Analysis",roles: ["admin", "analyst", "viewer"], group: "ANALYTICS" },
  { icon: "▥", label: "Pricing",        roles: ["admin", "analyst", "viewer"], group: "ANALYTICS" },
  { icon: "◎", label: "Promotions",     roles: ["admin", "analyst"],           group: "ANALYTICS" },
  { icon: "▤", label: "Datasets",       roles: ["admin", "analyst"],           group: "PLATFORM" },
  { icon: "⟳", label: "Training",       roles: ["admin", "analyst"],           group: "PLATFORM" },
  { icon: "⚙", label: "Settings",       roles: ["admin", "analyst", "viewer"], group: "PLATFORM" },
];

export default function Sidebar({ activePage, onNavigate }) {
  const { role } = useAuth();
  const currentRole = (role || "viewer").toLowerCase();

  const allowed = navigationItems.filter(item => item.roles.includes(currentRole));

  // Group items
  const groups = ["WORKSPACE", "ANALYTICS", "PLATFORM"];
  const grouped = groups.reduce((acc, g) => {
    acc[g] = allowed.filter(i => i.group === g);
    return acc;
  }, {});

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-mark" aria-label="EconoCausal logo">
          <svg viewBox="0 0 64 56" role="img" aria-hidden="true">
            <defs>
              <linearGradient id="brandGradient" x1="0" y1="1" x2="1" y2="0">
                <stop offset="0%" stopColor="#1687ff" />
                <stop offset="100%" stopColor="#8b5cf6" />
              </linearGradient>
            </defs>
            <rect x="3" y="32" width="13" height="21" rx="3" fill="url(#brandGradient)" />
            <rect x="22" y="18" width="13" height="35" rx="3" fill="url(#brandGradient)" />
            <rect x="41" y="3" width="13" height="50" rx="3" fill="url(#brandGradient)" />
          </svg>
        </div>
        <div className="brand-copy">
          <h2>Econo<span>Causal</span></h2>
          <p>PRICING INTELLIGENCE</p>
        </div>
      </div>

      <nav className="sidebar-navigation" aria-label="Main navigation">
        {groups.map((group) => {
          const items = grouped[group];
          if (!items || items.length === 0) return null;
          return (
            <div key={group}>
              <div className="sidebar-section-label">{group}</div>
              {items.map((item) => {
                const isActive = item.label === activePage;
                return (
                  <button
                    key={item.label}
                    type="button"
                    className={`sidebar-link ${isActive ? "active" : ""}`}
                    aria-current={isActive ? "page" : undefined}
                    onClick={() => onNavigate(item.label)}
                  >
                    <span className="sidebar-link-icon" aria-hidden="true">{item.icon}</span>
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </div>
          );
        })}
      </nav>

      <div className="sidebar-bottom">
        <span className="sidebar-status-dot" />
        <div>
          <strong>Analytics Engine</strong>
          <span>Role: {currentRole.toUpperCase()}</span>
        </div>
      </div>
    </aside>
  );
}
