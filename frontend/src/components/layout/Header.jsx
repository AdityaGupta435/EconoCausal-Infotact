
import "./Header.css";

export default function Header() {
  return (
    <header className="dashboard-header">
      <div className="header-heading">
        <span className="header-eyebrow">PRICING INTELLIGENCE</span>
        <h1>Overview</h1>
        <p>Understand price impact. Make data-driven decisions.</p>
      </div>

      <div className="header-actions">
        <span className="header-demo-badge">DEMO DATA</span>
        <div className="header-avatar">AG</div>
      </div>
    </header>
  );
}
