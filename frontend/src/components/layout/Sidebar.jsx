
import "./Sidebar.css";

const navigation = [
  { icon: "▦", label: "Overview" },
  { icon: "↗", label: "Pricing Analysis" },
  { icon: "◎", label: "Causal Effects" },
  { icon: "⌁", label: "Uplift Analysis" },
  { icon: "◈", label: "Budget Optimizer" },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-icon">E</div>
        <div>
          <h2>EconoCausal</h2>
          <span>PRICING INTELLIGENCE</span>
        </div>
      </div>

      <div className="sidebar-section-label">WORKSPACE</div>

      <nav className="sidebar-navigation">
        {navigation.map((item, index) => (
          <button
            key={item.label}
            className={`sidebar-link ${index === 0 ? "active" : ""}`}
            type="button"
          >
            <span className="sidebar-link-icon">{item.icon}</span>
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="sidebar-bottom">
        <div className="sidebar-status-dot" />
        <div>
          <strong>Analytics Engine</strong>
          <span>UI preview mode</span>
        </div>
      </div>
    </aside>
  );
}
