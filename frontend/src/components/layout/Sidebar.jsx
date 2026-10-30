
import "./Sidebar.css";

const navigation = [
  { icon: "⌂", label: "Overview" },
  { icon: "▥", label: "Pricing Analysis" },
  { icon: "♧", label: "Causal Effects" },
  { icon: "↗", label: "Uplift Analysis" },
  { icon: "◎", label: "Budget Optimizer" },
  { icon: "▤", label: "Data Management" },
  { icon: "⚙", label: "Settings" },
];

export default function Sidebar({ activePage, onNavigate }) {
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

      <div className="sidebar-section-label">WORKSPACE</div>

      <nav className="sidebar-navigation" aria-label="Main navigation">
        {navigation.map((item) => {
          const isActive = item.label === activePage;

          return (
            <button
              key={item.label}
              type="button"
              className={`sidebar-link ${isActive ? "active" : ""}`}
              aria-current={isActive ? "page" : undefined}
              onClick={() => onNavigate(item.label)}
            >
              <span className="sidebar-link-icon" aria-hidden="true">
                {item.icon}
              </span>
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="sidebar-promo">
        <div className="promo-chart" aria-hidden="true">
          <svg viewBox="0 0 100 65">
            <path d="M10 52 L32 34 L48 40 L76 12 L89 17" fill="none" stroke="#3478ff" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M76 12 L89 17 L86 30" fill="none" stroke="#3478ff" strokeWidth="4" strokeLinecap="round" />
          </svg>
        </div>
        <h3>Turn data into smarter pricing decisions</h3>
        <p>Causal insights for real business impact.</p>
      </div>

      <div className="sidebar-bottom">
        <span className="sidebar-status-dot" />
        <div>
          <strong>Analytics Engine</strong>
          <span>Workspace ready</span>
        </div>
      </div>
    </aside>
  );
}
